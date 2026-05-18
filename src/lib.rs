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
    output::search(&[result], "test query", true, false, None, &warnings)
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
pub fn render_parse_error(e: &clap::Error, json_mode: bool) -> String {
    if json_mode {
        let raw = e.to_string();
        let message = raw
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_owned();
        let env = envelope::ErrorEnvelope {
            error: envelope::ErrorPayload {
                code: envelope::ErrorCode::UsageError,
                message,
                next_step: None,
                candidates: vec![],
                retryable: false,
            },
        };
        envelope::render_json_error(&env)
    } else {
        e.to_string()
    }
}
