//! Output envelopes per ADR-0060.
//!
//! `CommandOutput` is the canonical return type for every command runner.
//! It carries both the default-mode rendering (`markdown`) and a machine
//! payload (`data`), plus a `degraded` flag and `notes` describing why
//! the result diverged from the ideal path (e.g. semantic search fell
//! back to FTS).
//!
//! `render_json_success` / `render_json_error` map these into the wire
//! envelopes ([`SuccessEnvelope`] / [`ErrorEnvelope`]) emitted when
//! `--json` is set.

use amici::cli::exit_code::codes;
use serde::Serialize;

/// JSON-serializable error classification per ADR-0060.
///
/// Covers every sysexits code `SaeError` currently produces. `DATA_ERROR (65)`
/// and `NOT_FOUND (66)` are absent because no sae operation maps to them today;
/// they will be reconsidered when this type lifts into amici alongside yomu /
/// recall (ADR-0060 Phase 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum ErrorCode {
    UsageError,
    Software,
    CantCreat,
    IoError,
    TempFailure,
}

impl ErrorCode {
    /// Delegates to `amici::cli::exit_code::codes::*` so the sysexits numbers
    /// are single-sourced in amici; T-EN002 still pins them as a regression net.
    pub(crate) fn exit_code(self) -> u8 {
        match self {
            Self::UsageError => codes::USAGE,
            Self::Software => codes::SOFTWARE,
            Self::CantCreat => codes::CANT_CREAT,
            Self::IoError => codes::IO_ERR,
            Self::TempFailure => codes::TEMP_FAIL,
        }
    }
}

/// Canonical return type for command runners.
///
/// `markdown` is the human-facing rendering surfaced when `--json` is absent.
/// `data` is the machine payload mirrored into [`SuccessEnvelope::data`] when
/// `--json` is set. `degraded` + `notes` surface deviations from the ideal
/// path (e.g. semantic search unavailable → FTS fallback) so agents can react.
pub struct CommandOutput {
    pub markdown: String,
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

impl CommandOutput {
    pub(crate) fn ok(markdown: String, data: serde_json::Value) -> Self {
        Self {
            markdown,
            data,
            degraded: false,
            notes: Vec::new(),
        }
    }

    pub(crate) fn with_notes(
        markdown: String,
        data: serde_json::Value,
        degraded: bool,
        notes: Vec<String>,
    ) -> Self {
        Self {
            markdown,
            data,
            degraded,
            notes,
        }
    }
}

/// Serialized to stdout when `--json` is set.
#[derive(Debug, Serialize)]
pub(crate) struct SuccessEnvelope {
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

/// Serialized to stderr when `--json` is set and the command failed.
/// Wrapping the payload under `error` lets consumers branch on root key.
#[derive(Debug, Serialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorPayload,
}

/// Error payload nested under [`ErrorEnvelope::error`].
///
/// `next_step` and `candidates` are skipped from JSON when absent/empty
/// to keep the envelope compact when no structured guidance applies.
#[derive(Debug, Serialize)]
pub(crate) struct ErrorPayload {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_step: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<String>,
    pub retryable: bool,
}

/// Serializes a successful result to the wire envelope.
pub(crate) fn render_json_success(out: &CommandOutput) -> String {
    let env = SuccessEnvelope {
        data: out.data.clone(),
        degraded: out.degraded,
        notes: out.notes.clone(),
    };
    serde_json::to_string(&env).expect("SuccessEnvelope is always serializable")
}

/// Serializes a prepared error envelope to the wire format.
///
/// The caller assembles [`ErrorEnvelope`] via `SaeError::to_error_envelope` so
/// this module stays free of a `SaeError` import (avoids a tools↔envelope cycle).
pub(crate) fn render_json_error(env: &ErrorEnvelope) -> String {
    serde_json::to_string(env).expect("ErrorEnvelope is always serializable")
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-EN001: error_code_serializes_screaming_snake_case
    #[test]
    fn error_code_serializes_screaming_snake_case() {
        let pairs = [
            (ErrorCode::UsageError, r#""USAGE_ERROR""#),
            (ErrorCode::Software, r#""SOFTWARE""#),
            (ErrorCode::CantCreat, r#""CANT_CREAT""#),
            (ErrorCode::IoError, r#""IO_ERROR""#),
            (ErrorCode::TempFailure, r#""TEMP_FAILURE""#),
        ];
        for (code, expected) in pairs {
            let actual = serde_json::to_string(&code).unwrap();
            assert_eq!(
                actual, expected,
                "code {code:?} should serialize as {expected}"
            );
        }
    }

    // T-EN002: error_code_exit_code_matches_sysexits
    #[test]
    fn error_code_exit_code_matches_sysexits() {
        assert_eq!(ErrorCode::UsageError.exit_code(), 64);
        assert_eq!(ErrorCode::Software.exit_code(), 70);
        assert_eq!(ErrorCode::CantCreat.exit_code(), 73);
        assert_eq!(ErrorCode::IoError.exit_code(), 74);
        assert_eq!(ErrorCode::TempFailure.exit_code(), 75);
    }

    // T-EN003: error_payload_omits_optional_next_step_when_none
    #[test]
    fn error_payload_omits_optional_next_step_when_none() {
        let payload = ErrorPayload {
            code: ErrorCode::UsageError,
            message: "Missing query".into(),
            next_step: None,
            candidates: vec![],
            retryable: false,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            !json.contains("next_step"),
            "next_step should be omitted when None, got: {json}"
        );
    }

    // T-EN004: error_payload_omits_candidates_when_empty
    #[test]
    fn error_payload_omits_candidates_when_empty() {
        let payload = ErrorPayload {
            code: ErrorCode::UsageError,
            message: "invalid".into(),
            next_step: None,
            candidates: vec![],
            retryable: false,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            !json.contains("candidates"),
            "candidates should be omitted when empty, got: {json}"
        );
    }

    // T-EN005: error_payload_serializes_present_optional_fields
    #[test]
    fn error_payload_serializes_present_optional_fields() {
        let payload = ErrorPayload {
            code: ErrorCode::UsageError,
            message: "did you mean".into(),
            next_step: Some("Pass <QUERY>".into()),
            candidates: vec!["query".into(), "queries".into()],
            retryable: false,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            json.contains(r#""next_step":"Pass <QUERY>""#),
            "got: {json}"
        );
        assert!(
            json.contains(r#""candidates":["query","queries"]"#),
            "got: {json}"
        );
    }

    // T-EN006: error_envelope_wraps_payload_under_error_key
    #[test]
    fn error_envelope_wraps_payload_under_error_key() {
        let env = ErrorEnvelope {
            error: ErrorPayload {
                code: ErrorCode::UsageError,
                message: "Missing query".into(),
                next_step: None,
                candidates: vec![],
                retryable: false,
            },
        };
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.starts_with(r#"{"error":"#),
            "envelope must start with `{{\"error\":`, got: {json}"
        );
        assert!(
            json.contains(r#""code":"USAGE_ERROR""#),
            "payload must contain code, got: {json}"
        );
    }

    // T-EN007: success_envelope_serializes_required_fields
    #[test]
    fn success_envelope_serializes_required_fields() {
        let env = SuccessEnvelope {
            data: serde_json::json!({"markdown": "hello"}),
            degraded: false,
            notes: vec![],
        };
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.contains(r#""data":{"markdown":"hello"}"#),
            "got: {json}"
        );
        assert!(json.contains(r#""degraded":false"#), "got: {json}");
        assert!(json.contains(r#""notes":[]"#), "got: {json}");
    }

    // T-EN008: success_envelope_surfaces_degradation_with_notes
    #[test]
    fn success_envelope_surfaces_degradation_with_notes() {
        let env = SuccessEnvelope {
            data: serde_json::json!(null),
            degraded: true,
            notes: vec!["semantic search unavailable, falling back to FTS".into()],
        };
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains(r#""degraded":true"#), "got: {json}");
        assert!(
            json.contains(r#""notes":["semantic search unavailable, falling back to FTS"]"#),
            "got: {json}"
        );
    }

    // T-EN009: render_json_success_serializes_command_output
    #[test]
    fn render_json_success_serializes_command_output() {
        let out = CommandOutput::ok("hello".into(), serde_json::json!({"posts": 5}));
        let json = render_json_success(&out);
        assert!(json.contains(r#""data":{"posts":5}"#), "got: {json}");
        assert!(json.contains(r#""degraded":false"#), "got: {json}");
        assert!(json.contains(r#""notes":[]"#), "got: {json}");
    }

    // T-EN010: render_json_success_surfaces_notes
    #[test]
    fn render_json_success_surfaces_notes() {
        let out = CommandOutput::with_notes(
            "result".into(),
            serde_json::json!([]),
            true,
            vec!["semantic search unavailable, falling back to FTS".into()],
        );
        let json = render_json_success(&out);
        assert!(json.contains(r#""degraded":true"#), "got: {json}");
        assert!(
            json.contains(r#""semantic search unavailable, falling back to FTS""#),
            "got: {json}"
        );
    }

    // T-EN011: render_json_error_wraps_payload
    #[test]
    fn render_json_error_wraps_payload() {
        let env = ErrorEnvelope {
            error: ErrorPayload {
                code: ErrorCode::TempFailure,
                message: "rate limited".into(),
                next_step: Some("Retry after the rate-limit window resets.".into()),
                candidates: vec![],
                retryable: true,
            },
        };
        let json = render_json_error(&env);
        assert!(json.starts_with(r#"{"error":"#), "got: {json}");
        assert!(json.contains(r#""code":"TEMP_FAILURE""#), "got: {json}");
        assert!(json.contains(r#""retryable":true"#), "got: {json}");
    }
}
