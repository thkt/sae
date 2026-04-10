use std::io::{IsTerminal, Read};

use rurico::embed::{ChunkedEmbedding, Embed, Embedder, ModelId};
use sae::config::Config;

use crate::{SaeError, require_db};

macro_rules! warn_user {
    ($user_msg:literal; $($log:tt)+) => {{
        eprintln!("Warning: {}", format_args!($user_msg));
        tracing::warn!($($log)+);
    }};
}

pub(crate) fn run_search(
    config: &Config,
    query: &str,
    team: Option<&str>,
    limit: u32,
    json: bool,
) -> Result<String, SaeError> {
    let team = config.resolve_team(team)?;
    let db = require_db(config, team)?;
    let embedder = try_load_embedder();
    match &embedder {
        ModelLoad::Absent => eprintln!(
            "Hint: run 'sae model download && sae embed <team>' to enable semantic search"
        ),
        ModelLoad::Failed(e) => {
            warn_user!("embedding model not available ({e})"; error = %e, "embedding model not available");
        }
        ModelLoad::Ready(_) => {}
    }
    let embed_info = if let ModelLoad::Ready(ref emb) = embedder {
        match auto_embed_pending(&db, emb) {
            Ok(info) => Some(info),
            Err(e) => {
                warn_user!("auto-embed failed ({e}), continuing with existing embeddings"; error = %e, "auto_embed_pending failed, continuing search");
                None
            }
        }
    } else {
        None
    };
    let query_embedding = if let ModelLoad::Ready(ref e) = embedder {
        match e.embed_query(query) {
            Ok(v) => Some(v),
            Err(e) => {
                warn_user!("embed_query failed ({e}), falling back to FTS"; error = %e, %query, "embed_query failed, falling back to FTS");
                None
            }
        }
    } else {
        None
    };
    let results = sae::storage::hybrid_search(
        db.conn(),
        query,
        query_embedding.as_deref(),
        limit,
        chrono::Utc::now(),
        None,
        None,
        None,
    )?;
    let semantic = query_embedding.is_some();
    let output = crate::output::search(&results, query, json, semantic, embed_info, team)?;
    const PREFETCH_TTL_SECS: u64 = 5 * 60;
    if !sae::storage::sync_harvested_within(db.conn(), PREFETCH_TTL_SECS) {
        spawn_background_harvest(team);
    }
    Ok(output)
}

fn auto_embed_pending(db: &sae::storage::Db, embedder: &Embedder) -> Result<(u32, bool), SaeError> {
    const EMBED_BUDGET: u32 = 128;
    auto_embed_pending_with(db, EMBED_BUDGET, |texts| {
        embedder
            .embed_documents_batch(texts)
            .map_err(|e| SaeError::Other(format!("Batch embedding failed: {e}")))
    })
}

/// Embeds up to `budget` pending chunks in one batch.
/// Returns `(processed, budget_exhausted)` — `budget_exhausted` signals more chunks remain.
fn auto_embed_pending_with<F>(
    db: &sae::storage::Db,
    budget: u32,
    embed_fn: F,
) -> Result<(u32, bool), SaeError>
where
    F: Fn(&[&str]) -> Result<Vec<ChunkedEmbedding>, SaeError>,
{
    let result = super::embed_batch::embed_one_batch(db.conn(), budget, embed_fn)?;
    if result.processed == 0 {
        return Ok((0, false));
    }
    tracing::debug!(
        chunks = result.processed,
        "auto_embed_pending: embedded chunks during search"
    );
    Ok((result.processed, result.budget_exhausted))
}

pub(crate) enum ModelLoad {
    Ready(Box<Embedder>),
    Absent,
    Failed(String),
}

pub(crate) fn try_load_embedder() -> ModelLoad {
    try_load_embedder_with(|| rurico::embed::cached_artifacts(ModelId::default()))
}

fn try_load_embedder_with<E: std::fmt::Display>(
    cache_check: impl FnOnce() -> Result<Option<rurico::embed::Artifacts>, E>,
) -> ModelLoad {
    use rurico::embed::ProbeStatus;

    let paths = match cache_check() {
        Ok(Some(p)) => p,
        Ok(None) => {
            tracing::debug!("embedding model not cached");
            return ModelLoad::Absent;
        }
        Err(e) => {
            tracing::debug!(error = %e, "embedding model cache check failed");
            return ModelLoad::Failed(e.to_string());
        }
    };
    match Embedder::probe(&paths) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            tracing::debug!("MLX backend unavailable");
            return ModelLoad::Failed("MLX backend is unavailable".to_string());
        }
        Err(e) => {
            tracing::debug!(error = %e, "embedding model probe failed");
            return ModelLoad::Failed(e.to_string());
        }
    }
    match Embedder::new(&paths) {
        Ok(e) => {
            tracing::debug!("embedding model loaded");
            ModelLoad::Ready(Box::new(e))
        }
        Err(e) => {
            tracing::debug!(error = %e, "embedding model load failed");
            ModelLoad::Failed(e.to_string())
        }
    }
}

fn spawn_background_harvest(team: &str) {
    let Ok(exe) = std::env::current_exe() else {
        tracing::debug!("background harvest skipped: current_exe() failed");
        return;
    };
    if let Err(e) = std::process::Command::new(exe)
        .args(["harvest", team])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        tracing::debug!(error = %e, "background harvest spawn failed");
    }
}

pub(crate) fn resolve_search_query(query: Option<String>) -> Result<String, SaeError> {
    let stdin = std::io::stdin();
    resolve_value_with_reader(
        query,
        stdin.lock(),
        stdin.is_terminal(),
        "search query",
        "QUERY",
    )
}

pub(crate) fn resolve_value_with_reader(
    value: Option<String>,
    mut stdin: impl Read,
    stdin_is_terminal: bool,
    label: &str,
    placeholder: &str,
) -> Result<String, SaeError> {
    match value {
        Some(value) if value != "-" => Ok(value),
        Some(_) => read_stdin_value(&mut stdin, label, placeholder),
        None if stdin_is_terminal => Err(SaeError::Input(format!(
            "Missing {label}. Pass {placeholder}, pipe it via stdin, or use `-` to read stdin interactively"
        ))),
        None => read_stdin_value(&mut stdin, label, placeholder),
    }
}

fn read_stdin_value(
    mut stdin: impl Read,
    label: &str,
    placeholder: &str,
) -> Result<String, SaeError> {
    let mut buf = String::new();
    stdin.read_to_string(&mut buf)?;

    let value = buf.trim();
    if value.is_empty() {
        return Err(SaeError::Input(format!(
            "No {label} provided. Pass {placeholder}, pipe it via stdin, or use `-` to read stdin interactively"
        )));
    }

    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_test_row(number: u32) -> sae::storage::EsaPostRow {
        sae::storage::EsaPostRow {
            body_md: "# Hello\nWorld".into(),
            category: None,
            ..sae::storage::test_post_row(number)
        }
    }

    // TC-037: auto_embed_pending_with skips embed_fn when no unembedded chunks
    #[test]
    fn auto_embed_skips_when_no_unembedded_chunks() {
        let db = sae::storage::Db::open_memory().unwrap();
        let called = std::cell::Cell::new(false);
        let result = auto_embed_pending_with(&db, 256, |_| {
            called.set(true);
            Ok(vec![])
        });
        assert!(result.is_ok());
        assert!(
            !called.get(),
            "embed_fn must not be called when no chunks pending"
        );
    }

    // TC-038: auto_embed_pending_with embeds chunks within budget
    #[test]
    fn auto_embed_embeds_pending_chunks() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = sae::storage::Db::open_memory().unwrap();
        let row = make_test_row(1);
        sae::storage::upsert_post(db.conn(), &row).unwrap();
        sae::storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        assert_eq!(sae::storage::count_unembedded_chunks(db.conn()).unwrap(), 1);

        let result = auto_embed_pending_with(&db, 256, |texts| {
            Ok(texts
                .iter()
                .map(|_| ChunkedEmbedding {
                    chunks: vec![vec![0.5; EMBEDDING_DIMS]],
                })
                .collect())
        });
        assert!(result.is_ok());
        assert_eq!(sae::storage::count_unembedded_chunks(db.conn()).unwrap(), 0);
    }

    // TC-039: auto_embed_pending_with respects budget and leaves excess chunks unembedded
    #[test]
    fn auto_embed_respects_budget() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = sae::storage::Db::open_memory().unwrap();
        for i in 1u32..=2 {
            let row = make_test_row(i);
            sae::storage::upsert_post(db.conn(), &row).unwrap();
            sae::storage::rechunk_post(db.conn(), i, "# Hello\nWorld").unwrap();
        }

        assert_eq!(sae::storage::count_unembedded_chunks(db.conn()).unwrap(), 2);

        let result = auto_embed_pending_with(&db, 1, |texts| {
            Ok(texts
                .iter()
                .map(|_| ChunkedEmbedding {
                    chunks: vec![vec![0.5; EMBEDDING_DIMS]],
                })
                .collect())
        });
        assert!(result.is_ok());
        assert!(
            result.unwrap().1,
            "budget=1 with 2 chunks should signal budget exhausted"
        );
        assert_eq!(
            sae::storage::count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "budget=1 should leave 1 chunk unembedded"
        );
    }

    // TC-040: auto_embed_pending_with propagates embed_fn error without side effects
    #[test]
    fn auto_embed_propagates_embed_error() {
        let db = sae::storage::Db::open_memory().unwrap();
        let row = make_test_row(1);
        sae::storage::upsert_post(db.conn(), &row).unwrap();
        sae::storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        assert_eq!(sae::storage::count_unembedded_chunks(db.conn()).unwrap(), 1);

        let result =
            auto_embed_pending_with(&db, 256, |_| Err(SaeError::Other("model OOM".into())));
        assert!(result.is_err(), "embed_fn error should propagate");
        assert!(result.unwrap_err().to_string().contains("model OOM"));
        assert_eq!(
            sae::storage::count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "failed embed must not change unembedded count"
        );
    }

    // TC-041: auto_embed_pending_with rejects embedding count mismatch
    #[test]
    fn auto_embed_rejects_count_mismatch() {
        let db = sae::storage::Db::open_memory().unwrap();
        let row = make_test_row(1);
        sae::storage::upsert_post(db.conn(), &row).unwrap();
        sae::storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        let result = auto_embed_pending_with(&db, 256, |_| Ok(vec![]));
        assert!(result.is_err(), "count mismatch should be an error");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Embedding count mismatch"),
            "error should describe the mismatch"
        );
        assert_eq!(
            sae::storage::count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "mismatched embed must not change unembedded count"
        );
    }

    // TC-010: try_load_embedder_with returns Absent when model is not cached
    #[test]
    fn try_load_embedder_with_returns_absent_when_no_model() {
        let result = try_load_embedder_with(|| Ok::<_, &str>(None));
        assert!(matches!(result, ModelLoad::Absent));
    }

    // TC-010: try_load_embedder_with returns Failed on cache check error
    #[test]
    fn try_load_embedder_with_returns_failed_on_cache_error() {
        let result =
            try_load_embedder_with(|| Err::<Option<rurico::embed::Artifacts>, _>("cache error"));
        assert!(matches!(result, ModelLoad::Failed(_)));
    }

    fn resolve_search(
        value: Option<&str>,
        stdin: &str,
        is_terminal: bool,
    ) -> Result<String, SaeError> {
        resolve_value_with_reader(
            value.map(str::to_string),
            Cursor::new(stdin),
            is_terminal,
            "search query",
            "QUERY",
        )
    }

    #[test]
    fn resolve_value_reads_piped_stdin_when_missing() {
        assert_eq!(resolve_search(None, "認証\n", false).unwrap(), "認証");
    }

    #[test]
    fn resolve_value_reads_stdin_when_dash_is_passed() {
        assert_eq!(resolve_search(Some("-"), "認証\n", true).unwrap(), "認証");
    }

    #[test]
    fn resolve_value_returns_value_when_present() {
        assert_eq!(
            resolve_search(Some("認証"), "ignored", true).unwrap(),
            "認証"
        );
    }

    #[test]
    fn resolve_value_rejects_dash_with_empty_stdin() {
        let err = resolve_search(Some("-"), "", true).unwrap_err();
        assert!(
            err.to_string().contains("No search query provided"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_value_rejects_whitespace_only_piped_stdin() {
        let err = resolve_search(None, "  \n  \t  ", false).unwrap_err();
        assert!(
            err.to_string().contains("No search query provided"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_value_rejects_missing_on_terminal() {
        let err = resolve_search(None, "", true).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Missing search query"),
            "unexpected error: {err}"
        );
        assert!(
            msg.contains("Pass QUERY"),
            "error should include placeholder: {err}"
        );
    }
}
