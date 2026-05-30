pub(crate) mod chunker;
pub mod client;
pub mod commands;
pub mod config;
pub mod envelope;
pub mod io;
pub(crate) mod output;
pub(crate) mod redact;
pub mod storage;
pub mod sync;
pub mod tools;

use clap::error::ErrorKind;

pub use envelope::CommandOutput;
pub use tools::{Sae, SaeError};

/// Hidden test seam: builds a [`CommandOutput`] by running `output::search`
/// with a synthetic search result and a non-empty `search_warnings` slice so
/// `tests/cli_integration.rs` can pin the `--json` envelope wiring for #140
/// without seeding a SQLite DB or loading a real reranker. The production
/// binary built with `cargo install` never exposes this symbol.
#[cfg(feature = "test-support")]
pub fn __test_search_with_warnings() -> Result<CommandOutput, SaeError> {
    let warnings = vec!["reranker failed (forced for test), falling back to RRF order".to_owned()];
    let result = storage::SearchResult {
        post_number: 1,
        post_name: "test post".to_owned(),
        post_url: "https://example.com/posts/1".to_owned(),
        section_title: None,
        snippet: "synthetic snippet".to_owned(),
        score: 0.5,
        match_source: storage::MatchSource::Fts,
    };
    output::search(&[result], "test query", true, false, &warnings)
}

/// Renders [`CommandOutput`] for stdout. With `json_mode` set, emits the
/// `SuccessEnvelope`; otherwise the default-mode markdown.
pub fn render_success(out: &CommandOutput, json_mode: bool) -> String {
    if json_mode {
        envelope::render_json_success(out)
    } else {
        out.markdown.clone()
    }
}

/// Renders a `SaeError` for stderr. JSON mode wraps it via `to_error_envelope`.
///
/// `config` feeds [`SaeError::candidates`] for the team-list cases (#139).
/// Pass `None` when the failure occurred before `Config::load()` completed.
pub fn render_error(err: &SaeError, json_mode: bool, config: Option<&config::Config>) -> String {
    if json_mode {
        envelope::render_json_error(&err.to_error_envelope(config))
    } else {
        err.to_string()
    }
}

/// Renders a clap parse error. JSON mode emits a synthetic `UsageError` envelope
/// so agents see the same shape they get for runtime failures.
///
/// Only the first non-empty line of the clap message lands in the envelope; the
/// usage block and subcommand list that follow are noisy for machine consumers.
///
/// Caller supplies `subcommands`: the user-facing subcommand list. Must not
/// include hidden/test seams or deprecated entries — those would mislead an
/// agent's retry. The list feeds the `candidates` array when, and only
/// when, the failure is `ErrorKind::InvalidSubcommand`. Other clap error
/// kinds (`MissingRequiredArgument`, `UnknownArgument`, etc.) keep
/// `candidates: []` so agents do not see irrelevant subcommand lists in
/// missing-arg failures (#148).
pub fn render_parse_error(e: &clap::Error, json_mode: bool, subcommands: &[&str]) -> String {
    if json_mode {
        let raw = e.to_string();
        let message = raw
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_owned();
        let candidates = if e.kind() == ErrorKind::InvalidSubcommand {
            subcommands.iter().map(|&s| s.to_owned()).collect()
        } else {
            vec![]
        };
        let env = envelope::ErrorEnvelope {
            error: envelope::ErrorPayload {
                code: envelope::ErrorCode::UsageError,
                message,
                next_step: None,
                candidates,
                retryable: false,
            },
        };
        envelope::render_json_error(&env)
    } else {
        e.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Arg, Command};

    fn parse_envelope(out: &str) -> serde_json::Value {
        serde_json::from_str(out).expect("envelope must be JSON")
    }

    // T-397: InvalidSubcommand → candidates lists every supplied subcommand
    // verbatim. Closes #148: typo recovery surfaces in `--json` envelope.
    #[test]
    fn render_parse_error_invalid_subcommand_emits_candidates() {
        let cmd = Command::new("sae")
            .subcommand(Command::new("search"))
            .subcommand(Command::new("get"));
        let err = cmd.try_get_matches_from(["sae", "sserach"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidSubcommand, "test premise");

        let out = render_parse_error(&err, true, &["search", "get"]);
        let env = parse_envelope(&out);
        assert_eq!(
            env["error"]["candidates"],
            serde_json::json!(["search", "get"])
        );
        assert_eq!(env["error"]["code"], "USAGE_ERROR");
    }

    /// Envelope's `candidates: Vec<String>` carries `skip_serializing_if =
    /// "Vec::is_empty"`, so an empty list appears as a missing field rather
    /// than `[]`. Agents read both as "no candidates" — this helper accepts
    /// either shape.
    fn assert_candidates_absent(env: &serde_json::Value) {
        let c = &env["error"]["candidates"];
        assert!(
            c.is_null() || c.as_array().is_some_and(Vec::is_empty),
            "candidates must be absent or empty; got: {c}"
        );
    }

    // T-398: MissingRequiredArgument keeps candidates absent so agents do
    // not see an irrelevant subcommand list when the failure is a missing
    // positional. Regression guard against future "any usage error →
    // suggest subcommands" simplifications.
    #[test]
    fn render_parse_error_missing_required_arg_keeps_candidates_empty() {
        let cmd = Command::new("sae")
            .subcommand(Command::new("get").arg(Arg::new("number").required(true)));
        let err = cmd.try_get_matches_from(["sae", "get"]).unwrap_err();
        assert_eq!(
            err.kind(),
            ErrorKind::MissingRequiredArgument,
            "test premise"
        );

        let out = render_parse_error(&err, true, &["search", "get"]);
        let env = parse_envelope(&out);
        assert_candidates_absent(&env);
    }

    // T-399: UnknownArgument also keeps candidates absent. Pairs with T-398
    // to pin the `==` predicate against `match _ => non-empty` drift.
    #[test]
    fn render_parse_error_unknown_argument_keeps_candidates_empty() {
        let cmd = Command::new("sae").subcommand(Command::new("status"));
        let err = cmd
            .try_get_matches_from(["sae", "status", "--unknownflag"])
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnknownArgument, "test premise");

        let out = render_parse_error(&err, true, &["search", "get"]);
        let env = parse_envelope(&out);
        assert_candidates_absent(&env);
    }
}
