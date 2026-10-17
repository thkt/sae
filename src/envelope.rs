use crate::compat::codes;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum ErrorCode {
    UsageError,
    DataError,
    Internal,
    CantCreat,
    IoError,
    TempFailure,
    Unknown,
}

impl ErrorCode {
    pub(crate) fn exit_code(self) -> u8 {
        match self {
            Self::UsageError => codes::USAGE,
            Self::DataError => codes::DATA_ERROR,
            Self::Internal => codes::INTERNAL,
            Self::CantCreat => codes::CANT_CREAT,
            Self::IoError => codes::IO_ERR,
            Self::TempFailure => codes::TEMP_FAIL,
            Self::Unknown => codes::UNKNOWN,
        }
    }
}

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

#[derive(Debug, Serialize)]
pub(crate) struct SuccessEnvelope {
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorPayload,
}

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

pub(crate) fn render_json_success(out: &CommandOutput) -> String {
    let env = SuccessEnvelope {
        data: out.data.clone(),
        degraded: out.degraded,
        notes: out.notes.clone(),
    };
    serde_json::to_string(&env).expect("SuccessEnvelope is always serializable")
}

pub(crate) fn render_json_error(env: &ErrorEnvelope) -> String {
    serde_json::to_string(env).expect("ErrorEnvelope is always serializable")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_serializes_screaming_snake_case() {
        let pairs = [
            (ErrorCode::UsageError, r#""USAGE_ERROR""#),
            (ErrorCode::DataError, r#""DATA_ERROR""#),
            (ErrorCode::Internal, r#""INTERNAL""#),
            (ErrorCode::CantCreat, r#""CANT_CREAT""#),
            (ErrorCode::IoError, r#""IO_ERROR""#),
            (ErrorCode::TempFailure, r#""TEMP_FAILURE""#),
            (ErrorCode::Unknown, r#""UNKNOWN""#),
        ];
        for (code, expected) in pairs {
            let actual = serde_json::to_string(&code).unwrap();
            assert_eq!(
                actual, expected,
                "code {code:?} should serialize as {expected}"
            );
        }
    }

    #[test]
    fn error_code_exit_code_matches_sysexits() {
        assert_eq!(ErrorCode::UsageError.exit_code(), 64);
        assert_eq!(ErrorCode::DataError.exit_code(), 65);
        assert_eq!(ErrorCode::Internal.exit_code(), 70);
        assert_eq!(ErrorCode::CantCreat.exit_code(), 73);
        assert_eq!(ErrorCode::IoError.exit_code(), 74);
        assert_eq!(ErrorCode::TempFailure.exit_code(), 75);
        assert_eq!(ErrorCode::Unknown.exit_code(), 104);
    }

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

    #[test]
    fn render_json_success_serializes_command_output() {
        let out = CommandOutput::ok("hello".into(), serde_json::json!({"posts": 5}));
        let json = render_json_success(&out);
        assert!(json.contains(r#""data":{"posts":5}"#), "got: {json}");
        assert!(json.contains(r#""degraded":false"#), "got: {json}");
        assert!(json.contains(r#""notes":[]"#), "got: {json}");
    }

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
