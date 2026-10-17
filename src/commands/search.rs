use std::io::{self, IsTerminal, Read};

use crate::config::Config;
use crate::envelope::CommandOutput;
use crate::storage;
use jiff::Timestamp;
use jiff::civil::Date;
use jiff::tz::TimeZone;

use super::embedder::try_load_embedder;
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
) -> Result<CommandOutput, SaeError> {
    let updated_after = parse_date_arg(after, "after")?;
    let updated_before = parse_date_arg(before, "before")?;
    let team = config.resolve_team(team)?;
    let db = require_db(config, team)?;
    if !no_embed && storage::has_embeddings(db.conn()) {
        storage::validate_embedding_index(db.conn())?;
    }
    let (embedder, mut embedder_load_failed) = if no_embed {
        (None, false)
    } else {
        let result = try_load_embedder();
        if let Err(reason) = result {
            tracing::warn!(?reason, "search: embedding model unavailable; using FTS");
        }
        let failed = result.is_err();
        (result.ok(), failed)
    };
    let query_embedding = if let Some(e) = embedder {
        match e.embed_query(query) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!(error = %e, %query, "embed_query failed, falling back to FTS");
                embedder_load_failed = true;
                None
            }
        }
    } else {
        None
    };
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
    )?;
    for w in &search_output.warnings {
        tracing::warn!("{w}");
    }
    // A loaded model does not imply vector search ran: the index may have no vectors.
    let semantic = query_embedding.is_some() && storage::has_embeddings(db.conn());
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

    #[test]
    fn parse_date_arg_returns_none_when_absent() {
        assert!(parse_date_arg(None, "after").unwrap().is_none());
    }

    #[test]
    fn parse_date_arg_parses_valid_date() {
        let dt = parse_date_arg(Some("2025-06-30"), "after")
            .unwrap()
            .unwrap();
        assert_eq!(dt.strftime("%Y-%m-%d").to_string(), "2025-06-30");
    }

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
