use std::env;
use std::io::{self, IsTerminal, Read};
use std::process::{Command, Stdio};

use crate::config::Config;
use crate::storage;
use amici::model::ModelLoad;
use amici::model::embedder::degraded_reason_user_note;
use rurico::embed::ChunkedEmbedding;
use rurico::reranker::Rerank;

use super::embedder::try_load_embedder;
use super::reranker::try_load_reranker;
use crate::output;
use crate::tools::{SaeError, require_db};

const SEARCH_EMBED_BUDGET: u32 = 128;

#[derive(Debug, clap::Args)]
pub struct SearchArgs {
    /// Search query. Reads piped stdin when omitted, or any stdin with `-`.
    pub query: Option<String>,
    /// Team name
    #[arg(long)]
    pub team: Option<String>,
    /// Max results (1-100)
    #[arg(long, default_value = "10")]
    pub limit: u32,
    /// Filter: updated on or after this date (YYYY-MM-DD)
    #[arg(long, value_name = "DATE")]
    pub after: Option<String>,
    /// Filter: updated on or before this date (YYYY-MM-DD)
    #[arg(long, value_name = "DATE")]
    pub before: Option<String>,
}

fn parse_date_arg(
    s: Option<&str>,
    flag: &str,
) -> Result<Option<chrono::DateTime<chrono::Utc>>, SaeError> {
    let Some(s) = s else {
        return Ok(None);
    };
    let date = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|_| {
        SaeError::Input(format!(
            "Invalid date '--{flag} {s}': expected YYYY-MM-DD (e.g. 2025-01-01)"
        ))
    })?;
    Ok(Some(date.and_time(chrono::NaiveTime::MIN).and_utc()))
}

/// Embeds up to `budget` pending chunks in one batch.
/// Returns `(processed, budget_exhausted)` — `budget_exhausted` signals more chunks remain.
fn auto_embed_pending_with<F>(
    db: &storage::Db,
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

fn spawn_background_harvest(team: &str) {
    let Ok(exe) = env::current_exe() else {
        tracing::debug!("background harvest skipped: current_exe() failed");
        return;
    };
    if let Err(e) = Command::new(exe)
        .args(["harvest", team])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        tracing::debug!(error = %e, "background harvest spawn failed");
    }
}

pub(crate) fn run_search(
    config: &Config,
    query: &str,
    team: Option<&str>,
    limit: u32,
    after: Option<&str>,
    before: Option<&str>,
    json: bool,
) -> Result<String, SaeError> {
    let updated_after = parse_date_arg(after, "after")?;
    let updated_before = parse_date_arg(before, "before")?;
    let team = config.resolve_team(team)?;
    let db = require_db(config, team)?;
    let embedder = try_load_embedder();
    if let Err(reason) = embedder
        && let Some(note) = degraded_reason_user_note(*reason)
    {
        eprintln!("Warning: {note}");
    }
    let embed_info = if let Ok(emb) = embedder {
        match auto_embed_pending_with(&db, SEARCH_EMBED_BUDGET, |texts| {
            emb.embed_documents_batch(texts)
                .map_err(|e| SaeError::Other(format!("Batch embedding failed: {e}")))
        }) {
            Ok((processed, budget_exhausted)) => Some(output::EmbedInfo {
                processed,
                budget_exhausted,
                team: team.to_owned(),
            }),
            Err(e) => {
                eprintln!("Warning: auto-embed failed ({e}), continuing with existing embeddings");
                tracing::warn!(error = %e, "auto_embed_pending failed, continuing search");
                None
            }
        }
    } else {
        None
    };
    let query_embedding = if let Ok(e) = embedder {
        match e.embed_query(query) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Warning: embed_query failed ({e}), falling back to FTS");
                tracing::warn!(error = %e, %query, "embed_query failed, falling back to FTS");
                None
            }
        }
    } else {
        None
    };
    let rerank_enabled = env::var("SAE_RERANK").as_deref() == Ok("1");
    let reranker: Option<ModelLoad<Box<dyn Rerank>>> = rerank_enabled.then(try_load_reranker);
    if let Some(ref r) = reranker {
        r.emit_load_hint(
            "reranker model not cached; unset SAE_RERANK=1 to skip reranking",
            "reranker",
        );
    }
    let reranker_ref: Option<&dyn Rerank> = reranker
        .as_ref()
        .and_then(|r| r.as_ref())
        .map(|b| b.as_ref() as &dyn Rerank);
    let filter = storage::SearchFilter {
        updated_after,
        updated_before,
        ..Default::default()
    };
    let search_output = storage::hybrid_search(
        db.conn(),
        query,
        query_embedding.as_deref(),
        limit,
        chrono::Utc::now(),
        &filter,
        reranker_ref,
    )?;
    for w in &search_output.warnings {
        eprintln!("Warning: {w}");
    }
    let semantic = query_embedding.is_some();
    let output = output::search(&search_output.results, query, json, semantic, embed_info)?;
    const PREFETCH_TTL_SECS: u64 = 5 * 60;
    if !storage::sync_harvested_within(db.conn(), PREFETCH_TTL_SECS) {
        spawn_background_harvest(team);
    }
    Ok(output)
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

    Ok(value.to_owned())
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

pub fn resolve_search_query(query: Option<String>) -> Result<String, SaeError> {
    let stdin = io::stdin();
    resolve_value_with_reader(
        query,
        stdin.lock(),
        stdin.is_terminal(),
        "search query",
        "QUERY",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::Cursor;

    fn make_test_row(number: u32) -> storage::EsaPostRow {
        storage::EsaPostRow {
            body_md: "# Hello\nWorld".into(),
            category: None,
            ..storage::test_post_row(number)
        }
    }

    // T-065: auto_embed_pending_with skips embed_fn when no unembedded chunks
    #[test]
    fn auto_embed_skips_when_no_unembedded_chunks() {
        let db = storage::Db::open_memory().unwrap();
        let called = Cell::new(false);
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

    // T-066: auto_embed_pending_with embeds chunks within budget
    #[test]
    fn auto_embed_embeds_pending_chunks() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = storage::Db::open_memory().unwrap();
        let row = make_test_row(1);
        storage::upsert_post(db.conn(), &row).unwrap();
        storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 1);

        let result = auto_embed_pending_with(&db, 256, |texts| {
            Ok(texts
                .iter()
                .map(|_| ChunkedEmbedding {
                    chunks: vec![vec![0.5; EMBEDDING_DIMS]],
                })
                .collect())
        });
        assert!(result.is_ok());
        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 0);
    }

    // T-067: auto_embed_pending_with respects budget and leaves excess chunks unembedded
    #[test]
    fn auto_embed_respects_budget() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = storage::Db::open_memory().unwrap();
        for i in 1u32..=2 {
            let row = make_test_row(i);
            storage::upsert_post(db.conn(), &row).unwrap();
            storage::rechunk_post(db.conn(), i, "# Hello\nWorld").unwrap();
        }

        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 2);

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
            storage::count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "budget=1 should leave 1 chunk unembedded"
        );
    }

    // T-068: auto_embed_pending_with propagates embed_fn error without side effects
    #[test]
    fn auto_embed_propagates_embed_error() {
        let db = storage::Db::open_memory().unwrap();
        let row = make_test_row(1);
        storage::upsert_post(db.conn(), &row).unwrap();
        storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 1);

        let result =
            auto_embed_pending_with(&db, 256, |_| Err(SaeError::Other("model OOM".into())));
        assert!(result.is_err(), "embed_fn error should propagate");
        assert!(result.unwrap_err().to_string().contains("model OOM"));
        assert_eq!(
            storage::count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "failed embed must not change unembedded count"
        );
    }

    // T-069: auto_embed_pending_with rejects embedding count mismatch
    #[test]
    fn auto_embed_rejects_count_mismatch() {
        let db = storage::Db::open_memory().unwrap();
        let row = make_test_row(1);
        storage::upsert_post(db.conn(), &row).unwrap();
        storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

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
            storage::count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "mismatched embed must not change unembedded count"
        );
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

    // T-144: resolve_value reads query from piped stdin when no argument given
    #[test]
    fn resolve_value_reads_piped_stdin_when_missing() {
        assert_eq!(resolve_search(None, "認証\n", false).unwrap(), "認証");
    }

    // T-145: resolve_value reads query from stdin when "-" is passed as argument
    #[test]
    fn resolve_value_reads_stdin_when_dash_is_passed() {
        assert_eq!(resolve_search(Some("-"), "認証\n", true).unwrap(), "認証");
    }

    // T-146: resolve_value returns the provided argument value directly
    #[test]
    fn resolve_value_returns_value_when_present() {
        assert_eq!(
            resolve_search(Some("認証"), "ignored", true).unwrap(),
            "認証"
        );
    }

    // T-147: resolve_value errors when "-" is passed but stdin is empty
    #[test]
    fn resolve_value_rejects_dash_with_empty_stdin() {
        let err = resolve_search(Some("-"), "", true).unwrap_err();
        assert!(
            err.to_string().contains("No search query provided"),
            "unexpected error: {err}"
        );
    }

    // T-148: resolve_value errors when piped stdin contains only whitespace
    #[test]
    fn resolve_value_rejects_whitespace_only_piped_stdin() {
        let err = resolve_search(None, "  \n  \t  ", false).unwrap_err();
        assert!(
            err.to_string().contains("No search query provided"),
            "unexpected error: {err}"
        );
    }

    // T-149: resolve_value errors with usage hint when argument missing on terminal
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

    // T-070: parse_date_arg returns None when no value provided
    #[test]
    fn parse_date_arg_returns_none_when_absent() {
        assert!(parse_date_arg(None, "after").unwrap().is_none());
    }

    // T-071: parse_date_arg parses valid YYYY-MM-DD into DateTime<Utc>
    #[test]
    fn parse_date_arg_parses_valid_date() {
        let dt = parse_date_arg(Some("2025-06-30"), "after")
            .unwrap()
            .unwrap();
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "2025-06-30");
    }

    // T-072: parse_date_arg returns Input error for non-ISO8601 date
    #[test]
    fn parse_date_arg_rejects_invalid_date() {
        let err = parse_date_arg(Some("30/06/2025"), "after").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Invalid date"),
            "error should describe the problem: {msg}"
        );
        assert!(
            msg.contains("YYYY-MM-DD"),
            "error should show expected format: {msg}"
        );
    }
}
