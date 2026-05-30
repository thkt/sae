use std::io::{self, IsTerminal, Read};

use crate::config::Config;
use crate::envelope::CommandOutput;
use crate::storage;
use amici::model::{ModelLoad, record_degraded};
use jiff::Timestamp;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use rurico::reranker::Rerank;

use super::embedder::try_load_embedder;
use super::reranker::try_load_reranker;
use crate::output;
use crate::tools::{SaeError, require_db};

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
    // `semantic` reflects whether vector search actually ran, not just whether
    // the embedder loaded: hybrid_search skips vectors when nothing is embedded
    // yet (e.g. indexed before `model download`), so report FTS — not a
    // misleading "hybrid" — in that case. Mirrors hybrid_search's own guard.
    let semantic = query_embedding.is_some() && storage::has_embeddings(db.conn());
    // Search is read-only — no auto-embed, no background index. Build the
    // index explicitly with `sae index` / `sae rebuild` (mirrors yomu).
    output::search(
        &search_output.results,
        query,
        semantic,
        embedder_load_failed,
        &search_output.warnings,
    )
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
    use std::io::Cursor;

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
