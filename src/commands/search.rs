use std::env;
use std::io::{self, IsTerminal, Read};
use std::process::{Command, Stdio};

use crate::config::Config;
use crate::envelope::CommandOutput;
use crate::storage;
use amici::model::{ModelLoad, record_degraded};
use jiff::Timestamp;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use rurico::embed::ChunkedEmbedding;
use rurico::reranker::Rerank;

use super::embedder::try_load_embedder;
use super::reranker::try_load_reranker;
use crate::output;
use crate::tools::{SaeError, require_db};

const SEARCH_EMBED_BUDGET: u32 = 128;

fn parse_date_arg(s: Option<&str>, flag: &str) -> Result<Option<Timestamp>, SaeError> {
    let Some(s) = s else {
        return Ok(None);
    };
    let date = Date::strptime("%Y-%m-%d", s).map_err(|_| {
        SaeError::InputData(format!(
            "Invalid date '--{flag} {s}': expected YYYY-MM-DD (e.g. 2025-01-01)"
        ))
    })?;
    let zoned = date
        .to_zoned(TimeZone::UTC)
        .map_err(|e| SaeError::InputData(format!("failed to zone date to UTC: {e}")))?;
    Ok(Some(zoned.timestamp()))
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

fn spawn_background_index(team: &str) {
    let Ok(exe) = env::current_exe() else {
        tracing::warn!("background index skipped: current_exe() failed");
        return;
    };
    if let Err(e) = Command::new(exe)
        .args(["index", team])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        tracing::warn!(error = %e, "background index spawn failed");
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_search(
    config: &Config,
    query: &str,
    team: Option<&str>,
    limit: u32,
    after: Option<&str>,
    before: Option<&str>,
    no_embed: bool,
    rerank_enabled: bool,
) -> Result<CommandOutput, SaeError> {
    let updated_after = parse_date_arg(after, "after")?;
    let updated_before = parse_date_arg(before, "before")?;
    let team = config.resolve_team(team)?;
    let db = require_db(config, team)?;
    // `embedder_load_failed` distinguishes intentional FTS (`no_embed`) from
    // a silent fallback (loader returned `DegradedReason`). Only the latter
    // populates `degraded=true` + notes in the success envelope.
    let (embedder, embedder_load_failed) = if no_embed {
        (None, false)
    } else {
        let result = try_load_embedder();
        if let Err(reason) = result {
            record_degraded(*reason, "search: embedder load");
        }
        (result.as_ref().ok(), result.is_err())
    };
    let embed_info = if let Some(emb) = embedder {
        match auto_embed_pending_with(&db, SEARCH_EMBED_BUDGET, |texts| {
            emb.embed_documents_batch(texts)
                // Other (UNKNOWN): embedder's runtime error is an opaque
                // `anyhow::Error` (no typed variant from amici/rurico).
                // Mirrors `tools::Sae::embed`; promoting requires upstream
                // typed surface first.
                .map_err(|e| SaeError::Other(format!("Batch embedding failed: {e}")))
        }) {
            Ok((processed, budget_exhausted)) => Some(output::EmbedInfo {
                processed,
                budget_exhausted,
                team: team.to_owned(),
            }),
            Err(e) => {
                tracing::warn!(error = %e, "auto-embed failed, continuing with existing embeddings");
                None
            }
        }
    } else {
        None
    };
    let query_embedding = if let Some(e) = embedder {
        match e.embed_query(query) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!(error = %e, %query, "embed_query failed, falling back to FTS");
                None
            }
        }
    } else {
        None
    };
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
        Timestamp::now(),
        &filter,
        reranker_ref,
    )?;
    for w in &search_output.warnings {
        tracing::warn!("{w}");
    }
    let semantic = query_embedding.is_some();
    let output = output::search(
        &search_output.results,
        query,
        semantic,
        embedder_load_failed,
        embed_info,
        &search_output.warnings,
    )?;
    // 5-minute TTL: long enough that repeated user searches within a single
    // workflow do not trigger redundant background harvests (coalescing), but
    // short enough that an idle period (e.g., next day) re-fetches before the
    // next search returns stale results.
    const PREFETCH_TTL_SECS: u64 = 5 * 60;
    if !storage::sync_harvested_within(db.conn(), PREFETCH_TTL_SECS) {
        spawn_background_index(team);
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
        return Err(SaeError::InputUsage(format!(
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
        None if stdin_is_terminal => Err(SaeError::InputUsage(format!(
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
                .map(|_| ChunkedEmbedding::new(vec![vec![0.5; EMBEDDING_DIMS]]))
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
                .map(|_| ChunkedEmbedding::new(vec![vec![0.5; EMBEDDING_DIMS]]))
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
            value.map(str::to_owned),
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
        assert_eq!(dt.strftime("%Y-%m-%d").to_string(), "2025-06-30");
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
