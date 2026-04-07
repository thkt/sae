use std::io::{IsTerminal, Read};

use rurico::embed::{Embed, Embedder, ModelId};
use sae::config::Config;

use crate::{SaeError, require_db};

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
    let query_embedding = embedder.as_ref().and_then(|e| match e.embed_query(query) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("Warning: embed_query failed ({e}), falling back to FTS");
            tracing::warn!(error = %e, %query, "embed_query failed, falling back to FTS");
            None
        }
    });
    let results = sae::storage::hybrid_search(
        db.conn(),
        query,
        query_embedding.as_deref(),
        limit,
        chrono::Utc::now(),
    )?;
    let semantic = query_embedding.is_some();
    crate::output::search(&results, query, json, semantic)
}

pub(crate) fn try_load_embedder() -> Option<Embedder> {
    try_load_embedder_with(|| rurico::embed::cached_artifacts(ModelId::default()))
}

fn try_load_embedder_with<E: std::fmt::Display>(
    cache_check: impl FnOnce() -> Result<Option<rurico::embed::Artifacts>, E>,
) -> Option<Embedder> {
    let paths = match cache_check() {
        Ok(Some(p)) => p,
        Ok(None) => {
            eprintln!(
                "Hint: run 'sae model download && sae embed <team>' to enable semantic search"
            );
            return None;
        }
        Err(e) => {
            eprintln!("Warning: embedding model not available ({e})");
            tracing::warn!(error = %e, "embedding model not available");
            return None;
        }
    };
    match Embedder::new(&paths) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("Warning: failed to load embedding model ({e})");
            tracing::warn!(error = %e, "failed to load embedding model");
            None
        }
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

    // TC-010: try_load_embedder_with returns None when model is not cached
    #[test]
    fn try_load_embedder_with_returns_none_when_no_model() {
        let result = try_load_embedder_with(|| Ok::<_, &str>(None));
        assert!(result.is_none());
    }

    // TC-010: try_load_embedder_with returns None on cache check error
    #[test]
    fn try_load_embedder_with_returns_none_on_cache_error() {
        let result =
            try_load_embedder_with(|| Err::<Option<rurico::embed::Artifacts>, _>("cache error"));
        assert!(result.is_none());
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
