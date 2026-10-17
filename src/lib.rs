#[cfg(not(target_endian = "little"))]
compile_error!("sae sqlite-vec indexes require a little-endian target");

pub(crate) mod chunker;
pub mod client;
pub mod commands;
pub mod compat;
pub mod config;
pub mod embedding;
pub mod envelope;
pub mod index;
pub mod io;
pub mod model;
mod native;
pub(crate) mod output;
pub(crate) mod redact;
pub mod storage;
pub mod sync;
pub mod tools;

use clap::error::ErrorKind;

pub use envelope::CommandOutput;
pub use tools::{Sae, SaeError};

#[cfg(feature = "test-support")]
pub fn __test_search_with_warnings() -> Result<CommandOutput, SaeError> {
    let warnings = vec![
        "search component unavailable (forced for test), falling back to text search".to_owned(),
    ];
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

pub fn render_success(out: &CommandOutput, json_mode: bool) -> String {
    if json_mode {
        envelope::render_json_success(out)
    } else {
        out.markdown.clone()
    }
}

/// Pass no config when loading it failed.
pub fn render_error(err: &SaeError, json_mode: bool, config: Option<&config::Config>) -> String {
    if json_mode {
        envelope::render_json_error(&err.to_error_envelope(config))
    } else {
        err.to_string()
    }
}

/// Suggestions must exclude hidden and retired commands.
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

    // Serde may omit empty candidates, so accept either absent or empty.
    fn assert_candidates_absent(env: &serde_json::Value) {
        let c = &env["error"]["candidates"];
        assert!(
            c.is_null() || c.as_array().is_some_and(Vec::is_empty),
            "candidates must be absent or empty; got: {c}"
        );
    }

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
