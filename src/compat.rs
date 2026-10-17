//! Small local utilities retained when removing the MLX dependency chain.
//! Pure text/SQL helpers preserve the existing index and CLI contracts.

pub(crate) mod filter;
pub(crate) mod fts;
pub(crate) mod query;
mod query_helpers;
mod shorthand;
mod spinner;
pub(crate) mod text;

pub(crate) use filter::{anon_placeholders, as_sql_params, in_placeholders};
pub(crate) use query_helpers::{collect_rows, fetch_by_in_clause};
pub use shorthand::try_expand_shorthand;
pub(crate) use spinner::embed_with_spinners;

#[cfg(test)]
use std::error::Error;
use std::{env, fmt, io, process::ExitCode};

pub trait CliError {
    fn exit_code(&self) -> ExitCode;
}

pub mod codes {
    pub const SUCCESS: u8 = 0;
    pub const USAGE: u8 = 64;
    pub const DATA_ERROR: u8 = 65;
    pub const SOFTWARE: u8 = 70;
    pub const CANT_CREAT: u8 = 73;
    pub const IO_ERR: u8 = 74;
    pub const TEMP_FAIL: u8 = 75;
    pub const UNKNOWN: u8 = 104;
    pub const INTERNAL: u8 = SOFTWARE;
}

pub(crate) fn env_lookup() -> impl Fn(&str) -> Option<String> {
    |key| env::var(key).ok()
}

pub fn exit_error(message: &str) {
    eprintln!("error: {message}");
}

pub fn hint_arrow<S: AsRef<str>>(items: &[S]) {
    eprintln!(
        "→ {}",
        items
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<_>>()
            .join(" ")
    );
}

pub(crate) fn progress_step<S: AsRef<str>>(items: &[S]) {
    eprintln!(
        "  {}",
        items
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<_>>()
            .join(" — ")
    );
}

pub fn init_subscriber(default_filter: &str) {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .init();
}

// Kept solely as public error compatibility for clients that classify load
// failures. Native Gemma loading does not execute the former MLX probe IPC.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("model probe handler not installed")]
    HandlerNotInstalled,
    #[error("model load failed: {reason}")]
    ModelLoadFailed { reason: String },
    #[error("probe setup rejected ({reason})")]
    SetupRejected { reason: SetupReason },
    #[error("probe error: {0}")]
    SubprocessFailed(String),
}

#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum SetupReason {
    EnvIncomplete = 3,
    CanonicalizeFailed = 4,
    PathOutsideCache = 5,
    CacheRootInvalid = 6,
}
impl SetupReason {
    pub fn code(self) -> i32 {
        self as i32
    }
}
impl fmt::Display for SetupReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::EnvIncomplete => "env incomplete",
            Self::CanonicalizeFailed => "path canonicalize failed",
            Self::PathOutsideCache => "path outside cache",
            Self::CacheRootInvalid => "cache root invalid",
        };
        write!(f, "code {}: {label}", self.code())
    }
}

#[cfg(test)]
#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelInitError {
    #[error("model init failed: {message}")]
    Backend {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    #[error("model load failed: {reason}")]
    ModelCorrupt { reason: String },
}
#[cfg(test)]
impl From<ProbeError> for ModelInitError {
    fn from(error: ProbeError) -> Self {
        match error {
            ProbeError::ModelLoadFailed { reason } => Self::ModelCorrupt { reason },
            ProbeError::SubprocessFailed(message) => Self::Backend {
                message,
                source: None,
            },
            other => Self::Backend {
                message: other.to_string(),
                source: Some(Box::new(other)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelInitError, ProbeError, SetupReason};
    use std::error::Error;

    #[test]
    fn compatibility_setup_errors_preserve_numeric_codes_and_load_failure_reason() {
        for (reason, code, label) in [
            (SetupReason::EnvIncomplete, 3, "env incomplete"),
            (
                SetupReason::CanonicalizeFailed,
                4,
                "path canonicalize failed",
            ),
            (SetupReason::PathOutsideCache, 5, "path outside cache"),
            (SetupReason::CacheRootInvalid, 6, "cache root invalid"),
        ] {
            assert_eq!(reason.code(), code);
            assert_eq!(reason.to_string(), format!("code {code}: {label}"));
            let converted = ModelInitError::from(ProbeError::SetupRejected { reason });
            assert!(
                converted
                    .source()
                    .unwrap()
                    .downcast_ref::<ProbeError>()
                    .is_some()
            );
        }
        let error = ModelInitError::from(ProbeError::ModelLoadFailed {
            reason: "corrupt model bytes".into(),
        });
        assert!(
            matches!(error, ModelInitError::ModelCorrupt { reason } if reason == "corrupt model bytes")
        );
    }
}
