//! Output envelopes per ADR-0060.
//!
//! Phase 2.1 introduces the type definitions and `SaeError` accessor methods
//! (`error_code`, `next_step`, `candidates`, `retryable`). Phase 2.2 wires the
//! renderers (`render_json_success` / `render_json_error`) through the `--json`
//! path. This module exposes no observable behavior change on its own.
//!
//! `#![allow(dead_code)]` covers the envelope structs that are unused until
//! Phase 2.2 wires them; remove the allow when wiring lands.
#![allow(dead_code)]

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

/// Serialized to stdout when `--json` is set (Phase 2.2).
#[derive(Debug, Serialize)]
pub(crate) struct SuccessEnvelope {
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

/// Serialized to stderr when `--json` is set and the command failed (Phase 2.2).
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
}
