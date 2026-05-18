use std::convert::Infallible;
use std::io;
use std::process::ExitCode;

use amici::cli::embed_with_spinners;
use amici::cli::env_lookup;
use amici::cli::exit_code::CliError;
use amici::model::embedder::{DegradedReason, try_load_embedder_with};
use amici::model::{ModelDownloadError, download_and_verify_model};
use rurico::embed::{Artifacts, ModelId, cached_artifacts};
use rurico::model_init::ModelInitError;
use rurico::model_probe::ProbeError;

use crate::client::{ClientError, EsaClient};
use crate::commands;
use crate::config::{Config, ConfigError};
use crate::envelope::{CommandOutput, ErrorCode, ErrorEnvelope, ErrorPayload};
use crate::output;
use crate::storage::{Db, EmbedResult, StorageError, count_unembedded_chunks};
use crate::sync::{self, SyncError};

pub use crate::commands::search::resolve_search_query;

#[derive(Debug, clap::Args)]
pub struct SearchArgs {
    /// Search query. Reads piped stdin when omitted, or any stdin with `-`.
    pub query: Option<String>,
    /// Team name
    #[arg(long)]
    pub team: Option<String>,
    /// Max results (1-100)
    #[arg(long, default_value = "10")]
    pub limit: u32,
    /// Filter: updated on or after this date (YYYY-MM-DD)
    #[arg(long, value_name = "DATE")]
    pub after: Option<String>,
    /// Filter: updated on or before this date (YYYY-MM-DD)
    #[arg(long, value_name = "DATE")]
    pub before: Option<String>,
    /// Skip embedding lookups; use FTS only. Avoids embedder load cost.
    #[arg(long)]
    pub no_embed: bool,
}

#[derive(Debug, clap::Args)]
pub struct CreateArgs {
    /// Post title
    #[arg(long)]
    pub name: String,
    /// Post body (Markdown)
    #[arg(long, conflicts_with = "body_file")]
    pub body: Option<String>,
    /// Read body from file (use "-" for stdin)
    #[arg(long, conflicts_with = "body")]
    pub body_file: Option<String>,
    /// Category path
    #[arg(long)]
    pub category: Option<String>,
    /// Tags
    #[arg(long)]
    pub tag: Vec<String>,
    /// Mark as WIP
    #[arg(long)]
    pub wip: bool,
    /// Team name
    #[arg(long)]
    pub team: Option<String>,
    /// Preview without creating (no mutation API calls)
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, clap::Args)]
pub struct UpdateArgs {
    /// Post number
    pub number: u32,
    /// New title
    #[arg(long)]
    pub name: Option<String>,
    /// New body (Markdown)
    #[arg(long, conflicts_with = "body_file")]
    pub body: Option<String>,
    /// Read body from file (use "-" for stdin)
    #[arg(long, conflicts_with = "body")]
    pub body_file: Option<String>,
    /// New category path
    #[arg(long)]
    pub category: Option<String>,
    /// New tags (replaces existing)
    #[arg(long)]
    pub tag: Vec<String>,
    /// Team name
    #[arg(long)]
    pub team: Option<String>,
    /// Preview without updating (no mutation API calls)
    #[arg(long)]
    pub dry_run: bool,
}

pub struct Sae {
    config: Config,
    rerank_enabled: bool,
}

impl Sae {
    pub fn new(config: Config) -> Self {
        Self::with_env(config, env_lookup())
    }

    pub(crate) fn with_env(config: Config, get_var: impl Fn(&str) -> Option<String>) -> Self {
        let rerank_enabled = get_var("SAE_RERANK").as_deref() == Some("1");
        Self {
            config,
            rerank_enabled,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub async fn index(&self, team: &str) -> Result<CommandOutput, SaeError> {
        self.fetch_and_index(team, false).await
    }

    pub async fn rebuild(&self, team: &str) -> Result<CommandOutput, SaeError> {
        self.fetch_and_index(team, true).await
    }

    async fn fetch_and_index(&self, team: &str, full: bool) -> Result<CommandOutput, SaeError> {
        let (team, client) = resolve_client(&self.config, Some(team))?;
        let db_path = self.config.team_db_path(team)?;
        let db = Db::open(&db_path)?;
        let result = sync::harvest(&client, &db, team, full).await?;
        output::harvest(&result)
    }

    pub fn embed(&self, team: &str) -> Result<CommandOutput, SaeError> {
        let team = self.config.resolve_team(Some(team))?;
        let db = require_db(&self.config, team)?;
        let paths = require_embed_model()?;
        let pending = count_unembedded_chunks(db.conn())?;

        let result = embed_with_spinners(
            pending,
            |_| {
                // Ok = typed probe variant; Err = fallback Display for the
                // non-ProbeError path; None = callback never fired.
                let mut captured: Option<Result<ProbeError, String>> = None;
                match try_load_embedder_with(
                    || Ok::<_, Infallible>(Some(paths)),
                    |e| tracing::warn!(error = %e, "failed to delete corrupt model files"),
                    |e| {
                        let display = e.to_string();
                        captured = Some(extract_probe_error(e).ok_or(display));
                    },
                ) {
                    Ok(e) => Ok(e),
                    // Mirrors `ModelDownload(BackendUnavailable)` routing via
                    // `SaeError::BackendUnavailable` so embed and model download
                    // surface the same INTERNAL signal for hardware mismatch
                    // (#127 CHX-001; was `Other` → UNKNOWN before).
                    Err(DegradedReason::BackendUnavailable) => Err(SaeError::BackendUnavailable),
                    Err(reason) => match captured {
                        Some(Ok(probe)) => Err(SaeError::Probe(probe)),
                        // Other (UNKNOWN): `extract_probe_error` returned a non-`ProbeError`
                        // payload that we surfaced via `Display`. No typed variant exists
                        // yet for the remaining `DegradedReason` shapes coming from amici.
                        Some(Err(detail)) => Err(SaeError::Other(format!(
                            "Model probe failed: {reason:?}: {detail}"
                        ))),
                        // Other (UNKNOWN): probe callback never fired — amici returned a
                        // `DegradedReason` without invoking the per-error hook. Genuine
                        // unclassified path.
                        None => Err(SaeError::Other(format!("Model probe failed: {reason:?}"))),
                    },
                }
            },
            |r: &commands::embed_batch::EmbedAllResult| {
                format!("Embedded {} chunks", r.total_chunks)
            },
            |embedder, update| {
                commands::embed_batch::embed_all(
                    db.conn(),
                    |texts| {
                        embedder
                            .embed_documents_batch(texts)
                            // Other (UNKNOWN): embedder's runtime error is an opaque
                            // `anyhow::Error` (no typed variant from amici/rurico).
                            // Promoting requires upstream type surface first.
                            .map_err(|e| SaeError::Other(format!("Batch embedding failed: {e}")))
                    },
                    |n| update(&format!("Embedding... {n}/{pending} chunks")),
                )
            },
        )?;

        match result {
            Some(all) => {
                tracing::info!(
                    total_added = all.added,
                    total_chunks = all.total_chunks,
                    "embed complete"
                );
                output::embed(
                    &EmbedResult {
                        chunks_embedded: all.added,
                    },
                    all.total_chunks,
                )
            }
            None => output::embed(&EmbedResult { chunks_embedded: 0 }, 0),
        }
    }

    pub fn model_download() -> Result<CommandOutput, SaeError> {
        download_and_verify_model()?;
        output::model_download()
    }

    pub async fn get(
        &self,
        number: u32,
        team: Option<&str>,
        with_body: bool,
    ) -> Result<CommandOutput, SaeError> {
        commands::post::run_get(&self.config, number, team, with_body).await
    }

    pub async fn create(&self, args: CreateArgs) -> Result<CommandOutput, SaeError> {
        let db = if args.dry_run {
            None
        } else {
            try_open_team_db(&self.config, args.team.as_deref())
        };
        commands::post::run_create(&self.config, db.as_ref(), args).await
    }

    pub async fn update(&self, args: UpdateArgs) -> Result<CommandOutput, SaeError> {
        let db = if args.dry_run {
            None
        } else {
            try_open_team_db(&self.config, args.team.as_deref())
        };
        commands::post::run_update(&self.config, db.as_ref(), args).await
    }

    pub async fn archive(
        &self,
        number: u32,
        team: Option<&str>,
        dry_run: bool,
    ) -> Result<CommandOutput, SaeError> {
        let db = if dry_run {
            None
        } else {
            try_open_team_db(&self.config, team)
        };
        commands::archive::run_archive(&self.config, db.as_ref(), number, team, dry_run).await
    }

    pub async fn ship(
        &self,
        number: u32,
        team: Option<&str>,
        dry_run: bool,
    ) -> Result<CommandOutput, SaeError> {
        let db = if dry_run {
            None
        } else {
            try_open_team_db(&self.config, team)
        };
        commands::archive::run_ship(&self.config, db.as_ref(), number, team, dry_run).await
    }

    pub fn search(&self, args: SearchArgs) -> Result<CommandOutput, SaeError> {
        let query = resolve_search_query(args.query)?;
        commands::search::run_search(
            &self.config,
            &query,
            args.team.as_deref(),
            args.limit,
            args.after.as_deref(),
            args.before.as_deref(),
            args.no_embed,
            self.rerank_enabled,
        )
    }

    pub fn status(&self, team: Option<&str>) -> Result<CommandOutput, SaeError> {
        commands::status::run_status(&self.config, team)
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SaeError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Sync(#[from] SyncError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    ModelDownload(#[from] ModelDownloadError),
    /// Carries the typed `ProbeError` recovered from `ModelInitError`'s source
    /// chain so [`Self::error_code`] can classify per-variant (ADR-0066 Group 2).
    #[error(transparent)]
    Probe(#[from] ProbeError),
    /// User action required (e.g., run `sae index` or `sae model download` first,
    /// or pass a missing flag). Maps to `USAGE_ERROR` (sysexits 64).
    #[error("{0}")]
    InputUsage(String),
    /// User-supplied data malformed (e.g., date string not `YYYY-MM-DD`).
    /// Maps to `DATA_ERROR` (sysexits 65) per ADR-0066 Group 2.
    #[error("{0}")]
    InputData(String),
    /// MLX backend missing in the runtime hardware (e.g., running on a host
    /// without Apple Silicon). Maps to `INTERNAL` (sysexits 70) so `sae embed`
    /// and `sae model download` surface the same hardware-mismatch signal
    /// (was split between 104 and 70 before #127 CHX-001).
    #[error("MLX backend is unavailable")]
    BackendUnavailable,
    /// Programmer-detectable invariant violation (e.g., embedder returned a
    /// vector count that does not match the requested batch). Maps to
    /// `INTERNAL` (sysexits 70) so agents distinguish bug signals from the
    /// `anyhow`-swallow `UNKNOWN` (104) path per ADR-0066 L136 (#127 CHX-001).
    #[error("{0}")]
    Internal(String),
    /// Operational failure with no classified variant. Maps to `UNKNOWN`
    /// (104) so the `anyhow`-style catch-all is distinguishable from genuine
    /// `INTERNAL` (70) classifications per ADR-0066 L136.
    #[error("{0}")]
    Other(String),
}

/// Prefixes shared by the constructors below and [`SaeError::next_step`]
/// lookup. Centralising them removes the implicit contract that constructor
/// and lookup strings stay in sync by convention.
const NO_DATA_PREFIX: &str = "No data for team ";
const MODEL_NOT_FOUND_PREFIX: &str = "Model not found";

/// Shared `next_step` hint for both `SaeError::BackendUnavailable` (embed
/// path) and `SaeError::ModelDownload(ModelDownloadError::BackendUnavailable)`
/// (model download path). Lives as a single const so the unit tests, the
/// integration test, and the production arm cannot drift.
const BACKEND_UNAVAILABLE_HINT: &str =
    "Install the MLX backend (Apple Silicon required), or pass `--no-embed` for FTS-only search.";

/// Shared `next_step` hint for `ClientError::Api { status: 404, .. }`
/// (post-not-found path). Centralised so the production arm and unit tests
/// cannot drift. The integration test (`tests/cli_integration.rs::T-CI009`)
/// intentionally keeps an exact-string literal: it asserts the binary-boundary
/// surface, not the in-process surface (#136).
const POST_NOT_FOUND_HINT: &str =
    "Verify the post number exists in esa, or run `sae search <keyword>` to find it.";

impl SaeError {
    pub(crate) fn input_no_data_for_team(team: &str) -> Self {
        Self::InputUsage(format!(
            "{NO_DATA_PREFIX}'{team}'. Run `sae index {team}` first."
        ))
    }

    pub(crate) fn input_model_not_found() -> Self {
        Self::InputUsage(format!(
            "{MODEL_NOT_FOUND_PREFIX}. Run 'sae model download' first."
        ))
    }

    /// [`CliError::exit_code`] delegates here so the sysexits mapping is single-sourced.
    pub(crate) fn error_code(&self) -> ErrorCode {
        match self {
            Self::InputUsage(_) | Self::Config(_) => ErrorCode::UsageError,
            Self::InputData(_) => ErrorCode::DataError,
            Self::Client(ClientError::TokenNotSet)
            | Self::Sync(SyncError::Client(ClientError::TokenNotSet)) => ErrorCode::UsageError,
            // Retryable sync failures (SQLite WAL contention, transient I/O)
            // route to `TempFailure` (75) instead of the `CantCreat` (73)
            // default so AI agents auto-retry recoverable conditions
            // (#138 subtask 2).
            Self::Sync(e) if e.is_retryable() => ErrorCode::TempFailure,
            Self::Storage(_) | Self::Sync(SyncError::Storage(_)) => ErrorCode::CantCreat,
            // esa API 404 is an input failure (the post number does not exist
            // in the team), not a server-side fault. Route to DATA_ERROR (65)
            // so AI agents distinguish missing inputs from 5xx INTERNAL (#136).
            Self::Client(ClientError::Api { status: 404, .. })
            | Self::Sync(SyncError::Client(ClientError::Api { status: 404, .. })) => {
                ErrorCode::DataError
            }
            Self::Json(_)
            | Self::Client(ClientError::Api { .. } | ClientError::InvalidRequest(_))
            | Self::Sync(SyncError::Client(
                ClientError::Api { .. } | ClientError::InvalidRequest(_),
            )) => ErrorCode::Internal,
            Self::Client(ClientError::Network(e))
            | Self::Sync(SyncError::Client(ClientError::Network(e)))
                if e.is_decode() =>
            {
                ErrorCode::Internal
            }
            Self::Io(_) => ErrorCode::IoError,
            Self::ModelDownload(ModelDownloadError::DownloadFailed(_)) => ErrorCode::TempFailure,
            Self::ModelDownload(
                ModelDownloadError::BackendUnavailable | ModelDownloadError::ProbeFailed(_),
            ) => ErrorCode::Internal,
            // Mirrors `ModelDownload(BackendUnavailable)` so the embed path and
            // model-download path agree on hardware-mismatch routing (#127 CHX-001).
            Self::BackendUnavailable => ErrorCode::Internal,
            // Programmer-detectable invariant violation (embedder bug, etc.).
            // Routed to `INTERNAL` so it stays distinguishable from anyhow-swallow
            // `Other(=UNKNOWN)` per ADR-0066 L136 (#127).
            Self::Internal(_) => ErrorCode::Internal,
            // PROBE_EXIT_* (3–8) wire codes stay sealed inside ProbeError;
            // only the sysexits classification reaches the process exit code.
            Self::Probe(
                ProbeError::HandlerNotInstalled
                | ProbeError::ModelLoadFailed { .. }
                | ProbeError::SetupRejected { .. },
            ) => ErrorCode::Internal,
            Self::Probe(ProbeError::SubprocessFailed(_)) => ErrorCode::IoError,
            // Remaining Client/Sync (Network connect/timeout, MaxRetries) are retry-eligible.
            // Explicit arms (no wildcard) so future amici variants force a compile-time review.
            Self::Client(_) | Self::Sync(_) => ErrorCode::TempFailure,
            // Catch-all: anyhow-style swallowing surfaces as UNKNOWN (104) to
            // distinguish it from classified INTERNAL (70) per ADR-0066 L136.
            Self::Other(_) => ErrorCode::Unknown,
        }
    }

    /// Returns template strings (`Run \`sae index <team>\``).
    /// Concrete values (`Run \`sae index gaji\``) would require parsing the
    /// message back, which is fragile; the agent-facing template is unambiguous.
    pub(crate) fn next_step(&self) -> Option<&'static str> {
        match self {
            Self::InputUsage(s) if s.starts_with(NO_DATA_PREFIX) => {
                Some("Run `sae index <team>` to fetch posts.")
            }
            Self::InputUsage(s) if s.starts_with(MODEL_NOT_FOUND_PREFIX) => {
                Some("Run `sae model download` to fetch the embedding model.")
            }
            Self::Config(ConfigError::NoTeamSpecified) => {
                Some("Pass --team <name> or set `SAE_TEAM=<name>`.")
            }
            Self::Client(ClientError::TokenNotSet)
            | Self::Sync(SyncError::Client(ClientError::TokenNotSet)) => {
                Some("Set `ESA_ACCESS_TOKEN=<token>` and retry.")
            }
            // esa API 404: the requested post number does not exist in the
            // team. Pairs with the DATA_ERROR routing in `error_code()` (#136)
            // so agents see both a 65 exit code and a recovery hint.
            Self::Client(ClientError::Api { status: 404, .. })
            | Self::Sync(SyncError::Client(ClientError::Api { status: 404, .. })) => {
                Some(POST_NOT_FOUND_HINT)
            }
            // Direct API commands (get/create/update/archive/ship) bubble
            // ClientError::MaxRetries as Self::Client(_); the index/rebuild
            // path wraps it under Sync. Both share the same retry guidance.
            Self::Client(ClientError::MaxRetries(_))
            | Self::Sync(SyncError::Client(ClientError::MaxRetries(_))) => {
                Some("Retry after the rate-limit window resets.")
            }
            Self::ModelDownload(ModelDownloadError::DownloadFailed(_)) => {
                Some("Retry `sae model download`; the failure is transient.")
            }
            // Mirrors `BackendUnavailable` below so both `sae embed` and
            // `sae model download` surface the same actionable hint on
            // hardware mismatch (#127 OPS-007).
            Self::ModelDownload(ModelDownloadError::BackendUnavailable) => {
                Some(BACKEND_UNAVAILABLE_HINT)
            }
            Self::BackendUnavailable => Some(BACKEND_UNAVAILABLE_HINT),
            // Explicit catch-all arms per variant. Adding a new SaeError variant
            // must force a compile-time review (no wildcard).
            Self::InputUsage(_)
            | Self::InputData(_)
            | Self::Config(_)
            | Self::Client(_)
            | Self::Storage(_)
            | Self::Sync(_)
            | Self::Json(_)
            | Self::Io(_)
            | Self::ModelDownload(_)
            | Self::Probe(_)
            | Self::Internal(_)
            | Self::Other(_) => None,
        }
    }

    /// Always returns empty for now. Future revisions may populate
    /// config-derived candidates for `NoTeamSpecified`/`UnknownTeam` once
    /// `Config` is threaded into the call site.
    pub(crate) fn candidates(&self) -> Vec<String> {
        Vec::new()
    }

    /// Mirrors the `TempFailure` classification from [`Self::error_code`].
    pub(crate) fn retryable(&self) -> bool {
        matches!(self.error_code(), ErrorCode::TempFailure)
    }

    /// Bundles [`Self::error_code`], [`Self::next_step`], [`Self::candidates`],
    /// [`Self::retryable`], and the formatted message into an [`ErrorEnvelope`]
    /// for `--json` rendering. Kept here so the envelope module avoids a
    /// dependency on `SaeError` (which would form a cycle).
    pub(crate) fn to_error_envelope(&self) -> ErrorEnvelope {
        ErrorEnvelope {
            error: ErrorPayload {
                code: self.error_code(),
                message: self.to_string(),
                next_step: self.next_step().map(String::from),
                candidates: self.candidates(),
                retryable: self.retryable(),
            },
        }
    }
}

pub(crate) fn resolve_client<'a>(
    config: &'a Config,
    team: Option<&'a str>,
) -> Result<(&'a str, EsaClient), SaeError> {
    let team = config.resolve_team(team)?;
    let client = EsaClient::from_env()?;
    Ok((team, client))
}

pub(crate) fn require_db(config: &Config, team: &str) -> Result<Db, SaeError> {
    let db_path = config.team_db_path(team)?;
    if !db_path.exists() {
        return Err(SaeError::input_no_data_for_team(team));
    }
    Ok(Db::open(&db_path)?)
}

/// Best-effort DB open for mutation commands. Returns `None` when the team
/// cannot be resolved, the DB file does not yet exist (no prior `sae index`
/// run), or the DB fails to open — letting the API call proceed regardless.
/// Only the open failure logs a warning; the other paths are silent.
pub(crate) fn try_open_team_db(config: &Config, team: Option<&str>) -> Option<Db> {
    let team = config.resolve_team(team).ok()?;
    let db_path = config.team_db_path(team).ok()?;
    if !db_path.exists() {
        return None;
    }
    match Db::open(&db_path) {
        Ok(db) => Some(db),
        Err(e) => {
            tracing::warn!(
                team,
                error = %e,
                "failed to open local DB; write-through skipped"
            );
            None
        }
    }
}

impl CliError for SaeError {
    fn exit_code(&self) -> ExitCode {
        ExitCode::from(self.error_code().exit_code())
    }
}

fn require_embed_model() -> Result<Artifacts, SaeError> {
    match cached_artifacts(ModelId::default()) {
        Ok(Some(p)) => Ok(p),
        Ok(None) => Err(SaeError::input_model_not_found()),
        // Other (UNKNOWN): `cached_artifacts` returns an opaque `anyhow::Error`
        // covering filesystem / permission / cache-layout edge cases. Promoting
        // requires typed surface from rurico first.
        Err(e) => Err(SaeError::Other(format!("Failed to check model cache: {e}"))),
    }
}

/// Reverse [`From<ProbeError> for ModelInitError`] so per-variant `ProbeError`
/// routing in [`SaeError::error_code`] survives amici's collapse boundary.
/// Returns `None` only when `Backend.source` carries a non-`ProbeError`
/// payload — left open so a future amici/rurico change does not silently
/// misclassify.
fn extract_probe_error(init_err: ModelInitError) -> Option<ProbeError> {
    match init_err {
        ModelInitError::ModelCorrupt { reason } => Some(ProbeError::ModelLoadFailed { reason }),
        ModelInitError::Backend {
            source: Some(boxed),
            ..
        } => boxed.downcast::<ProbeError>().ok().map(|b| *b),
        ModelInitError::Backend {
            source: None,
            message,
        } => Some(ProbeError::SubprocessFailed(message)),
    }
}

#[cfg(test)]
mod tests {
    use amici::cli::exit_code::codes;

    use super::*;

    fn assert_code(err: &SaeError, expected: u8) {
        assert_eq!(err.exit_code(), ExitCode::from(expected));
    }

    // T-237: InputUsage variant maps to USAGE (sysexits 64)
    #[test]
    fn exit_code_input_usage_is_usage() {
        assert_code(&SaeError::InputUsage("bad".into()), codes::USAGE);
    }

    // T-335: InputData variant maps to DATA_ERROR (sysexits 65) per ADR-0066 Group 2
    #[test]
    fn exit_code_input_data_is_data_error() {
        assert_code(
            &SaeError::InputData("Invalid date".into()),
            codes::DATA_ERROR,
        );
    }

    // T-238: Config variant maps to USAGE (sysexits 64)
    #[test]
    fn exit_code_config_is_usage() {
        assert_code(
            &SaeError::Config(ConfigError::NoTeamSpecified),
            codes::USAGE,
        );
    }

    // T-239: Client(TokenNotSet) maps to USAGE (sysexits 64)
    #[test]
    fn exit_code_token_not_set_is_usage() {
        assert_code(&SaeError::Client(ClientError::TokenNotSet), codes::USAGE);
    }

    // T-240: Storage variant maps to CANT_CREAT (sysexits 73)
    #[test]
    fn exit_code_storage_is_cant_creat() {
        assert_code(
            &SaeError::Storage(StorageError::Open("missing".into())),
            codes::CANT_CREAT,
        );
    }

    // T-241: Json variant maps to INTERNAL (sysexits 70)
    #[test]
    fn exit_code_json_is_internal() {
        let err: serde_json::Error = serde_json::from_str::<i32>("not json").unwrap_err();
        assert_code(&SaeError::Json(err), codes::INTERNAL);
    }

    // T-242: Other variant maps to UNKNOWN (104) — anyhow-style catch-all is
    // distinguished from classified INTERNAL (70) per ADR-0066 L136.
    #[test]
    fn exit_code_other_is_unknown() {
        assert_code(&SaeError::Other("unexpected".into()), codes::UNKNOWN);
    }

    // T-243: Io variant maps to IO_ERR (sysexits 74)
    #[test]
    fn exit_code_io_is_io_err() {
        assert_code(&SaeError::Io(io::Error::other("disk full")), codes::IO_ERR);
    }

    // T-244: Client(MaxRetries) maps to TEMP_FAIL (sysexits 75) for retry
    #[test]
    fn exit_code_client_max_retries_is_temp_fail() {
        assert_code(
            &SaeError::Client(ClientError::MaxRetries(5)),
            codes::TEMP_FAIL,
        );
    }

    // T-245: Client(Api) with non-404 status maps to INTERNAL (sysexits 70)
    // — non-retryable HTTP error from the esa API (5xx, 4xx other than 404).
    #[test]
    fn exit_code_client_api_is_internal() {
        assert_code(
            &SaeError::Client(ClientError::Api {
                status: 500,
                body: "Internal Server Error".into(),
            }),
            codes::INTERNAL,
        );
    }

    // T-346: Client(Api { status: 404 }) maps to DATA_ERROR (sysexits 65)
    // — input-side failure (post number does not exist in the team), distinct
    // from server-side 5xx per #136.
    #[test]
    fn exit_code_client_api_404_is_data_error() {
        assert_code(
            &SaeError::Client(ClientError::Api {
                status: 404,
                body: r#"{"error":"not_found","message":"Not Found"}"#.into(),
            }),
            codes::DATA_ERROR,
        );
    }

    // T-347: Client(InvalidRequest) maps to INTERNAL (sysexits 70) — pre-call
    // request construction failure (URL parse, bad token header), a
    // program-detectable bug, not user input.
    #[test]
    fn exit_code_client_invalid_request_is_internal() {
        assert_code(
            &SaeError::Client(ClientError::InvalidRequest("invalid URL".into())),
            codes::INTERNAL,
        );
    }

    // T-303: ModelDownload(DownloadFailed) maps to TEMP_FAIL (sysexits 75) — transient HTTP failure
    #[test]
    fn exit_code_model_download_failed_is_temp_fail() {
        assert_code(
            &SaeError::ModelDownload(ModelDownloadError::DownloadFailed(
                "connection reset".into(),
            )),
            codes::TEMP_FAIL,
        );
    }

    // T-304: ModelDownload(BackendUnavailable) maps to INTERNAL (sysexits 70) — non-retryable hardware mismatch
    #[test]
    fn exit_code_model_download_backend_unavailable_is_internal() {
        assert_code(
            &SaeError::ModelDownload(ModelDownloadError::BackendUnavailable),
            codes::INTERNAL,
        );
    }

    // T-305: ModelDownload(ProbeFailed) maps to INTERNAL (sysexits 70) — verification failure, retry won't help
    #[test]
    fn exit_code_model_download_probe_failed_is_internal() {
        assert_code(
            &SaeError::ModelDownload(ModelDownloadError::ProbeFailed("hash mismatch".into())),
            codes::INTERNAL,
        );
    }

    // T-246: Sync(Client(TokenNotSet)) maps to USAGE (sysexits 64)
    #[test]
    fn exit_code_sync_token_not_set_is_usage() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::TokenNotSet)),
            codes::USAGE,
        );
    }

    // T-247: Sync(Storage) maps to CANT_CREAT (sysexits 73)
    #[test]
    fn exit_code_sync_storage_is_cant_creat() {
        assert_code(
            &SaeError::Sync(SyncError::Storage(StorageError::Open("missing".into()))),
            codes::CANT_CREAT,
        );
    }

    // T-248: Sync(Client(Api)) with non-404 status maps to INTERNAL (sysexits 70)
    // — index/rebuild path mirrors the direct-call routing of T-245.
    #[test]
    fn exit_code_sync_client_api_is_internal() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::Api {
                status: 503,
                body: "Service Unavailable".into(),
            })),
            codes::INTERNAL,
        );
    }

    // T-348: Sync(Client(Api { status: 404 })) maps to DATA_ERROR (sysexits 65)
    // — index/rebuild path mirrors T-346 so the 404 routing is symmetric across
    // direct API commands and bulk sync.
    #[test]
    fn exit_code_sync_client_api_404_is_data_error() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::Api {
                status: 404,
                body: r#"{"error":"not_found","message":"Not Found"}"#.into(),
            })),
            codes::DATA_ERROR,
        );
    }

    // T-249: Sync(Client(MaxRetries)) maps to TEMP_FAIL (sysexits 75) for retry
    #[test]
    fn exit_code_sync_client_max_retries_is_temp_fail() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::MaxRetries(5))),
            codes::TEMP_FAIL,
        );
    }

    // T-388: Sync(Storage(SQLITE_BUSY)) maps to TEMP_FAIL (75) via the new
    // `is_retryable()` discriminator (#138 subtask 2). Without this arm, the
    // CantCreat (73) default catches retryable contention and AI agents do
    // not auto-retry.
    #[test]
    fn exit_code_sync_storage_sqlite_busy_is_temp_fail() {
        use rusqlite::{Error as SqlError, ErrorCode, ffi};
        let busy = SqlError::SqliteFailure(
            ffi::Error {
                code: ErrorCode::DatabaseBusy,
                extended_code: 5,
            },
            None,
        );
        assert_code(
            &SaeError::Sync(SyncError::Storage(StorageError::Db(busy))),
            codes::TEMP_FAIL,
        );
    }

    // T-389: Sync(Storage(Open)) still maps to CANT_CREAT (73) — the new
    // retryability arm is order-sensitive and must not capture non-retryable
    // storage failures (companion to T-388 / T-247).
    #[test]
    fn exit_code_sync_storage_open_is_cant_creat_after_retry_arm() {
        assert_code(
            &SaeError::Sync(SyncError::Storage(StorageError::Open("missing".into()))),
            codes::CANT_CREAT,
        );
    }

    // T-300: Sae::with_env enables rerank when SAE_RERANK="1"
    #[test]
    fn with_env_enables_rerank_when_one() {
        let sae = Sae::with_env(Config::default(), |key| match key {
            "SAE_RERANK" => Some("1".into()),
            _ => None,
        });
        assert!(sae.rerank_enabled);
    }

    // T-301: Sae::with_env disables rerank when SAE_RERANK is absent
    #[test]
    fn with_env_disables_rerank_when_absent() {
        let sae = Sae::with_env(Config::default(), |_| None);
        assert!(!sae.rerank_enabled);
    }

    // T-302: Sae::with_env disables rerank for any non-"1" value (e.g. "0", "true")
    #[test]
    fn with_env_disables_rerank_for_non_one_values() {
        for value in ["0", "true", "yes", ""] {
            let sae = Sae::with_env(Config::default(), |key| match key {
                "SAE_RERANK" => Some(value.into()),
                _ => None,
            });
            assert!(
                !sae.rerank_enabled,
                "SAE_RERANK={value:?} should not enable rerank"
            );
        }
    }

    // T-310: input_no_data_for_team_pins_next_step_index_hint
    #[test]
    fn input_no_data_for_team_pins_next_step_index_hint() {
        let err = SaeError::input_no_data_for_team("gaji");
        assert_eq!(
            err.next_step(),
            Some("Run `sae index <team>` to fetch posts."),
            "constructor and next_step must stay in sync"
        );
    }

    // T-311: input_model_not_found_pins_next_step_download_hint
    #[test]
    fn input_model_not_found_pins_next_step_download_hint() {
        let err = SaeError::input_model_not_found();
        assert_eq!(
            err.next_step(),
            Some("Run `sae model download` to fetch the embedding model."),
            "constructor and next_step must stay in sync"
        );
    }

    // T-312: next_step_config_no_team_specified
    #[test]
    fn next_step_config_no_team_specified() {
        let err = SaeError::Config(ConfigError::NoTeamSpecified);
        assert_eq!(
            err.next_step(),
            Some("Pass --team <name> or set `SAE_TEAM=<name>`.")
        );
    }

    // T-313: next_step_client_token_not_set
    #[test]
    fn next_step_client_token_not_set() {
        let err = SaeError::Client(ClientError::TokenNotSet);
        assert_eq!(
            err.next_step(),
            Some("Set `ESA_ACCESS_TOKEN=<token>` and retry.")
        );
    }

    // T-314: next_step_sync_client_token_not_set
    #[test]
    fn next_step_sync_client_token_not_set() {
        let err = SaeError::Sync(SyncError::Client(ClientError::TokenNotSet));
        assert_eq!(
            err.next_step(),
            Some("Set `ESA_ACCESS_TOKEN=<token>` and retry.")
        );
    }

    // T-315: next_step_sync_client_max_retries
    #[test]
    fn next_step_sync_client_max_retries() {
        let err = SaeError::Sync(SyncError::Client(ClientError::MaxRetries(5)));
        assert_eq!(
            err.next_step(),
            Some("Retry after the rate-limit window resets.")
        );
    }

    // T-325: next_step_client_max_retries_matches_sync_variant
    // Direct API commands (get/create/update/archive/ship) surface MaxRetries
    // as Self::Client; the index/rebuild path wraps it under Sync. Both must
    // give the same hint.
    #[test]
    fn next_step_client_max_retries_matches_sync_variant() {
        let err = SaeError::Client(ClientError::MaxRetries(5));
        assert_eq!(
            err.next_step(),
            Some("Retry after the rate-limit window resets.")
        );
    }

    // T-316: next_step_model_download_failed
    #[test]
    fn next_step_model_download_failed() {
        let err = SaeError::ModelDownload(ModelDownloadError::DownloadFailed("HTTP 503".into()));
        assert_eq!(
            err.next_step(),
            Some("Retry `sae model download`; the failure is transient.")
        );
    }

    // T-317: next_step_input_data_returns_none
    // InputData carries malformed user input (date parse, etc.) — no
    // structured hint is appropriate because the fix is "correct the value".
    #[test]
    fn next_step_input_data_returns_none() {
        let err = SaeError::InputData(
            "Invalid date '--from 2025-xx-xx': expected YYYY-MM-DD (e.g. 2025-01-01)".into(),
        );
        assert_eq!(err.next_step(), None);
    }

    // T-336: next_step_input_usage_without_recognized_prefix_returns_none
    // InputUsage messages with a recognized prefix get a hint; arbitrary
    // strings (e.g. clap-side messages) fall through to None.
    #[test]
    fn next_step_input_usage_without_recognized_prefix_returns_none() {
        let err = SaeError::InputUsage("Missing search query.".into());
        assert_eq!(err.next_step(), None);
    }

    // T-337: to_error_envelope(InputData) composes the ADR-0060 wire payload.
    // Pins the integration of error_code / next_step / retryable / candidates
    // for the new variant — each method has its own test (T-335, T-317,
    // T-323, T-324) but only their composition is the agent-facing contract.
    #[test]
    fn to_error_envelope_input_data_composes_data_error_payload() {
        let err = SaeError::InputData("Invalid date '--after 2025-xx-xx'".into());
        let env = err.to_error_envelope();
        assert_eq!(env.error.code, ErrorCode::DataError);
        assert!(!env.error.retryable);
        assert!(env.error.next_step.is_none());
        assert!(env.error.candidates.is_empty());
        assert_eq!(env.error.message, "Invalid date '--after 2025-xx-xx'");
    }

    // T-318: next_step_other_returns_none
    #[test]
    fn next_step_other_returns_none() {
        let err = SaeError::Other("internal".into());
        assert_eq!(err.next_step(), None);
    }

    // T-319: next_step_storage_returns_none
    #[test]
    fn next_step_storage_returns_none() {
        let err = SaeError::Storage(StorageError::Open("missing".into()));
        assert_eq!(err.next_step(), None);
    }

    // T-320: retryable_true_for_model_download_failed
    #[test]
    fn retryable_true_for_model_download_failed() {
        let err = SaeError::ModelDownload(ModelDownloadError::DownloadFailed("HTTP 503".into()));
        assert!(err.retryable(), "DownloadFailed must be retryable");
    }

    // T-321: retryable_true_for_sync_max_retries
    #[test]
    fn retryable_true_for_sync_max_retries() {
        let err = SaeError::Sync(SyncError::Client(ClientError::MaxRetries(5)));
        assert!(err.retryable(), "MaxRetries must be retryable");
    }

    // T-322: retryable_false_for_storage
    #[test]
    fn retryable_false_for_storage() {
        let err = SaeError::Storage(StorageError::Open("missing".into()));
        assert!(!err.retryable(), "Storage must not be retryable");
    }

    // T-323: retryable_false_for_input
    #[test]
    fn retryable_false_for_input() {
        for err in [
            SaeError::InputUsage("bad".into()),
            SaeError::InputData("bad".into()),
        ] {
            assert!(!err.retryable(), "{err:?} must not be retryable");
        }
    }

    // T-324: candidates_returns_empty_in_phase_2_1
    #[test]
    fn candidates_returns_empty_in_phase_2_1() {
        for err in [
            SaeError::InputUsage("anything".into()),
            SaeError::InputData("anything".into()),
            SaeError::Config(ConfigError::NoTeamSpecified),
            SaeError::Other("internal".into()),
        ] {
            assert_eq!(
                err.candidates(),
                Vec::<String>::new(),
                "Phase 2.1 returns empty Vec for {err:?}"
            );
        }
    }

    // T-326: every ProbeError variant routes per ADR-0066 Group 2
    #[test]
    fn exit_code_probe_variants_route_per_adr_0066() {
        use rurico::model_probe::SetupReason;
        let internal_cases = [
            ProbeError::HandlerNotInstalled,
            ProbeError::ModelLoadFailed {
                reason: "weight tensor missing".into(),
            },
            ProbeError::SetupRejected {
                reason: SetupReason::EnvIncomplete,
            },
            ProbeError::SetupRejected {
                reason: SetupReason::CanonicalizeFailed,
            },
            ProbeError::SetupRejected {
                reason: SetupReason::PathOutsideCache,
            },
            ProbeError::SetupRejected {
                reason: SetupReason::CacheRootInvalid,
            },
        ];
        for probe in internal_cases {
            assert_code(&SaeError::Probe(probe), codes::INTERNAL);
        }
        assert_code(
            &SaeError::Probe(ProbeError::SubprocessFailed(
                "probe spawn failed: ENOENT".into(),
            )),
            codes::IO_ERR,
        );
    }

    // T-330: PROBE_EXIT_* wire format (3-8) cannot leak through SaeError::exit_code.
    // Wire codes encode rurico's internal IPC contract; surfacing them as the
    // process exit code would confuse agents expecting sysexits classifications.
    #[test]
    fn exit_code_probe_setup_rejected_does_not_leak_wire_code() {
        use rurico::model_probe::SetupReason;
        let reason = SetupReason::PathOutsideCache;
        assert_eq!(
            reason.code(),
            5,
            "test premise: PathOutsideCache wire code is 5"
        );
        let err = SaeError::Probe(ProbeError::SetupRejected { reason });
        assert_eq!(
            err.exit_code(),
            ExitCode::from(codes::INTERNAL),
            "wire code 5 must be sealed; exit must be INTERNAL (70)"
        );
        assert_ne!(
            err.exit_code(),
            ExitCode::from(5),
            "PROBE_EXIT_PATH_OUTSIDE_CACHE (5) must not leak as the process exit code"
        );
    }

    // T-331: extract_probe_error recovers ModelLoadFailed from ModelCorrupt.
    #[test]
    fn extract_probe_error_model_corrupt_maps_to_model_load_failed() {
        let init = ModelInitError::ModelCorrupt {
            reason: "checksum mismatch".into(),
        };
        let probe = extract_probe_error(init).expect("ModelCorrupt yields ProbeError");
        match probe {
            ProbeError::ModelLoadFailed { reason } => assert_eq!(reason, "checksum mismatch"),
            other => panic!("expected ModelLoadFailed, got {other:?}"),
        }
    }

    // T-332: extract_probe_error recovers the typed variant from Backend.source
    #[test]
    fn extract_probe_error_backend_with_source_recovers_typed_variant() {
        let init: ModelInitError = ProbeError::HandlerNotInstalled.into();
        let probe = extract_probe_error(init).expect("Backend with source yields ProbeError");
        assert!(
            matches!(probe, ProbeError::HandlerNotInstalled),
            "expected HandlerNotInstalled, got {probe:?}"
        );
    }

    // T-333: extract_probe_error maps source-less Backend to SubprocessFailed
    // (round-trips From<ProbeError> for ModelInitError on SubprocessFailed).
    #[test]
    fn extract_probe_error_backend_without_source_maps_to_subprocess_failed() {
        let init: ModelInitError = ProbeError::SubprocessFailed("spawn failed".into()).into();
        let probe = extract_probe_error(init).expect("Backend message yields ProbeError");
        match probe {
            ProbeError::SubprocessFailed(msg) => assert_eq!(msg, "spawn failed"),
            other => panic!("expected SubprocessFailed, got {other:?}"),
        }
    }

    // T-334: extract_probe_error returns None for non-ProbeError sources so
    // the Sae::embed Other fallback runs instead of misclassifying as probe.
    #[test]
    fn extract_probe_error_backend_with_non_probe_source_returns_none() {
        let init = ModelInitError::Backend {
            message: "MLX kernel launch failed".into(),
            source: Some(Box::new(io::Error::other("MLX kernel launch failed"))),
        };
        assert!(
            extract_probe_error(init).is_none(),
            "non-ProbeError source must fall through to None so the Other fallback runs"
        );
    }

    // T-338: BackendUnavailable variant maps to INTERNAL (70) — mirrors
    // ModelDownload(BackendUnavailable) (T-304) so embed and model download
    // share the same hardware-mismatch exit code (#127 CHX-001).
    #[test]
    fn exit_code_backend_unavailable_is_internal() {
        assert_code(&SaeError::BackendUnavailable, codes::INTERNAL);
    }

    // T-339: Internal variant maps to INTERNAL (70) — distinguishes
    // programmer-detectable invariant violations from anyhow-swallow
    // Other (UNKNOWN, 104) per ADR-0066 L136 (#127 CHX-001).
    #[test]
    fn exit_code_internal_variant_is_internal() {
        assert_code(
            &SaeError::Internal("Embedding count mismatch: expected 5, got 4".into()),
            codes::INTERNAL,
        );
    }

    // T-340: BackendUnavailable's next_step points to install / --no-embed.
    // Constructor and lookup must stay in sync (matches T-310 / T-311 pattern).
    #[test]
    fn next_step_backend_unavailable_returns_install_or_no_embed_hint() {
        assert_eq!(
            SaeError::BackendUnavailable.next_step(),
            Some(BACKEND_UNAVAILABLE_HINT)
        );
    }

    // T-344: ModelDownload(BackendUnavailable)'s next_step shares the same hint
    // as the embed-path variant (#127 OPS-007). Constructor and lookup must
    // stay in sync via the shared `BACKEND_UNAVAILABLE_HINT` const.
    #[test]
    fn next_step_model_download_backend_unavailable_returns_shared_hint() {
        let err = SaeError::ModelDownload(ModelDownloadError::BackendUnavailable);
        assert_eq!(err.next_step(), Some(BACKEND_UNAVAILABLE_HINT));
    }

    // T-341: Internal variant returns no structured hint — the message body
    // already carries the invariant details; no canned next step applies.
    #[test]
    fn next_step_internal_variant_returns_none() {
        assert_eq!(
            SaeError::Internal("Embedding count mismatch".into()).next_step(),
            None
        );
    }

    // T-342: to_error_envelope(BackendUnavailable) composes the ADR-0060 wire
    // payload — code=INTERNAL, next_step set, retryable=false. Pins the
    // composition (T-337 pattern) for the new variant.
    #[test]
    fn to_error_envelope_backend_unavailable_composes_internal_payload() {
        let env = SaeError::BackendUnavailable.to_error_envelope();
        assert_eq!(env.error.code, ErrorCode::Internal);
        assert!(!env.error.retryable);
        assert_eq!(
            env.error.next_step.as_deref(),
            Some(BACKEND_UNAVAILABLE_HINT)
        );
        assert!(env.error.candidates.is_empty());
        assert_eq!(env.error.message, "MLX backend is unavailable");
    }

    // T-343: to_error_envelope(Internal) composes code=INTERNAL with no hint.
    // Mirrors T-342 so both new variants have their composition pinned.
    #[test]
    fn to_error_envelope_internal_variant_composes_internal_payload() {
        let err = SaeError::Internal("Embedding count mismatch: expected 5, got 4".into());
        let env = err.to_error_envelope();
        assert_eq!(env.error.code, ErrorCode::Internal);
        assert!(!env.error.retryable);
        assert!(env.error.next_step.is_none());
        assert!(env.error.candidates.is_empty());
        assert_eq!(
            env.error.message,
            "Embedding count mismatch: expected 5, got 4"
        );
    }

    // T-349: next_step for Client(Api { status: 404 }) returns the post-not-found
    // hint so AI agents recover from missing post numbers (#136). Mirrors T-340
    // (BackendUnavailable) — the hint surface and the routing surface must
    // agree at the source.
    #[test]
    fn next_step_client_api_404_returns_post_not_found_hint() {
        let err = SaeError::Client(ClientError::Api {
            status: 404,
            body: r#"{"error":"not_found"}"#.into(),
        });
        assert_eq!(err.next_step(), Some(POST_NOT_FOUND_HINT));
    }

    // T-350: next_step for Sync(Client(Api { status: 404 })) returns the same hint
    // as T-349 so direct API and bulk sync paths share guidance.
    #[test]
    fn next_step_sync_client_api_404_returns_post_not_found_hint() {
        let err = SaeError::Sync(SyncError::Client(ClientError::Api {
            status: 404,
            body: r#"{"error":"not_found"}"#.into(),
        }));
        assert_eq!(err.next_step(), Some(POST_NOT_FOUND_HINT));
    }

    // T-351: next_step for Client(Api { status: 500 }) returns None — only 404
    // carries a recovery hint, 5xx is server-side and not actionable on the
    // agent side.
    #[test]
    fn next_step_client_api_non_404_returns_none() {
        let err = SaeError::Client(ClientError::Api {
            status: 500,
            body: "Internal Server Error".into(),
        });
        assert_eq!(err.next_step(), None);
    }

    // T-352: to_error_envelope(Client(Api { status: 404 })) composes the
    // ADR-0060 wire payload with code=DATA_ERROR, retryable=false, and the
    // post-not-found next_step. Mirrors T-342 so the composition surface is
    // pinned alongside the individual method tests.
    #[test]
    fn to_error_envelope_client_api_404_composes_data_error_payload() {
        let err = SaeError::Client(ClientError::Api {
            status: 404,
            body: r#"{"error":"not_found","message":"Not Found"}"#.into(),
        });
        let env = err.to_error_envelope();
        assert_eq!(env.error.code, ErrorCode::DataError);
        assert!(!env.error.retryable);
        assert_eq!(env.error.next_step.as_deref(), Some(POST_NOT_FOUND_HINT));
        assert!(env.error.candidates.is_empty());
        assert!(
            env.error.message.contains("HTTP 404"),
            "message should mention HTTP 404; got: {}",
            env.error.message
        );
    }
}
