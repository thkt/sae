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

    pub async fn harvest(&self, team: &str, full: bool) -> Result<CommandOutput, SaeError> {
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
                let mut probe_err: Option<ProbeError> = None;
                let mut probe_detail: Option<String> = None;
                match try_load_embedder_with(
                    || Ok::<_, Infallible>(Some(paths)),
                    |e| tracing::warn!(error = %e, "failed to delete corrupt model files"),
                    |e| {
                        // Capture Display before move so the non-ProbeError
                        // path (e.g. MLX kernel failures inside Embedder::new)
                        // still surfaces the underlying detail to the user.
                        let display = e.to_string();
                        probe_err = extract_probe_error(e);
                        if probe_err.is_none() {
                            probe_detail = Some(display);
                        }
                    },
                ) {
                    Ok(e) => Ok(e),
                    Err(DegradedReason::BackendUnavailable) => {
                        Err(SaeError::Other("MLX backend is unavailable".to_owned()))
                    }
                    Err(reason) => match probe_err {
                        Some(probe) => Err(SaeError::Probe(probe)),
                        None => {
                            let detail = probe_detail.map(|d| format!(": {d}")).unwrap_or_default();
                            Err(SaeError::Other(format!(
                                "Model probe failed: {reason:?}{detail}"
                            )))
                        }
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
    /// Probe subprocess failure surfaced through `try_load_embedder_with`'s
    /// `ModelInitError` source chain. Carries the original `rurico::ProbeError`
    /// so `error_code()` can classify per-variant (ADR-0066 Group 2: 3 variants
    /// route to INTERNAL, `SubprocessFailed` routes to IO_ERROR).
    #[error(transparent)]
    Probe(#[from] ProbeError),
    /// User action required (e.g., run harvest or model download first)
    #[error("{0}")]
    Input(String),
    /// Operational failure (e.g., model load, embedding, download)
    #[error("{0}")]
    Other(String),
}

impl SaeError {
    /// Production sites MUST use this constructor (not `SaeError::Input(format!(...))`)
    /// so the canonical prefix `next_step()` keys off stays in sync.
    pub(crate) fn input_no_data_for_team(team: &str) -> Self {
        Self::Input(format!(
            "No data for team '{team}'. Run `sae harvest {team}` first."
        ))
    }

    /// Production sites MUST use this constructor (not `SaeError::Input(...)`)
    /// so the canonical prefix `next_step()` keys off stays in sync.
    pub(crate) fn input_model_not_found() -> Self {
        Self::Input("Model not found. Run 'sae model download' first.".to_owned())
    }

    /// [`CliError::exit_code`] delegates here so the sysexits mapping is single-sourced.
    pub(crate) fn error_code(&self) -> ErrorCode {
        match self {
            Self::Input(_) | Self::Config(_) => ErrorCode::UsageError,
            Self::Client(ClientError::TokenNotSet)
            | Self::Sync(SyncError::Client(ClientError::TokenNotSet)) => ErrorCode::UsageError,
            Self::Storage(_) | Self::Sync(SyncError::Storage(_)) => ErrorCode::CantCreat,
            Self::Json(_)
            | Self::Other(_)
            | Self::Client(ClientError::Api(_))
            | Self::Sync(SyncError::Client(ClientError::Api(_))) => ErrorCode::Internal,
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
            // ProbeError per-variant routing per ADR-0066 Group 2. The raw
            // PROBE_EXIT_* (3–8) wire codes from rurico stay sealed inside
            // ProbeError; only the classified sysexits value reaches the
            // process exit code.
            Self::Probe(ProbeError::HandlerNotInstalled)
            | Self::Probe(ProbeError::ModelLoadFailed { .. })
            | Self::Probe(ProbeError::SetupRejected { .. }) => ErrorCode::Internal,
            Self::Probe(ProbeError::SubprocessFailed(_)) => ErrorCode::IoError,
            // Remaining Client/Sync (Network connect/timeout, MaxRetries) are retry-eligible.
            // Explicit arms (no wildcard) so future amici variants force a compile-time review.
            Self::Client(_) | Self::Sync(_) => ErrorCode::TempFailure,
        }
    }

    /// Returns template strings (`Run \`sae harvest <team>\``).
    /// Concrete values (`Run \`sae harvest gaji\``) would require parsing the
    /// message back, which is fragile; the agent-facing template is unambiguous.
    pub(crate) fn next_step(&self) -> Option<&'static str> {
        match self {
            Self::Input(s) if s.starts_with("No data for team ") => {
                Some("Run `sae harvest <team>` to fetch posts.")
            }
            Self::Input(s) if s.starts_with("Model not found") => {
                Some("Run `sae model download` to fetch the embedding model.")
            }
            Self::Config(ConfigError::NoTeamSpecified) => {
                Some("Pass --team <name> or set `SAE_TEAM=<name>`.")
            }
            Self::Client(ClientError::TokenNotSet)
            | Self::Sync(SyncError::Client(ClientError::TokenNotSet)) => {
                Some("Set `ESA_ACCESS_TOKEN=<token>` and retry.")
            }
            // Direct API commands (get/create/update/archive/ship) bubble
            // ClientError::MaxRetries as Self::Client(_); harvest wraps it
            // under Sync. Both share the same retry guidance.
            Self::Client(ClientError::MaxRetries(_))
            | Self::Sync(SyncError::Client(ClientError::MaxRetries(_))) => {
                Some("Retry after the rate-limit window resets.")
            }
            Self::ModelDownload(ModelDownloadError::DownloadFailed(_)) => {
                Some("Retry `sae model download`; the failure is transient.")
            }
            // Explicit catch-all arms per variant. Adding a new SaeError variant
            // must force a compile-time review (no wildcard). Probe failures
            // are unactionable from the agent's side (the host binary needs a
            // fix or the model artifacts need re-download), so no hint is
            // emitted — the error message itself carries the diagnosis.
            Self::Input(_)
            | Self::Config(_)
            | Self::Client(_)
            | Self::Storage(_)
            | Self::Sync(_)
            | Self::Json(_)
            | Self::Io(_)
            | Self::ModelDownload(_)
            | Self::Probe(_)
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
/// cannot be resolved, the DB file does not yet exist (no prior harvest), or
/// the DB fails to open — letting the API call proceed regardless. Only the
/// open failure logs a warning; the other paths are silent.
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
        Err(e) => Err(SaeError::Other(format!("Failed to check model cache: {e}"))),
    }
}

/// Recover the originating [`ProbeError`] from a [`ModelInitError`].
///
/// `try_load_embedder_with` collapses every probe failure into one of two
/// `ModelInitError` variants but preserves the typed source on the chain via
/// [`rurico::model_init::ModelInitError::backend`]. This walker reverses the
/// collapse so `SaeError::error_code` can route per ADR-0066 Group 2.
///
/// | `ModelInitError`        | Source            | Returned `ProbeError`              |
/// | ----------------------- | ----------------- | ---------------------------------- |
/// | `ModelCorrupt { reason }` | (none)          | `ModelLoadFailed { reason }`       |
/// | `Backend { source: Some }` | downcastable  | `HandlerNotInstalled` / `SetupRejected` |
/// | `Backend { source: None }` | (none)         | `SubprocessFailed(message)`        |
///
/// Returns `None` only when `Backend { source: Some }` carries a non-`ProbeError`
/// payload — a path that does not exist today in rurico but is left open so a
/// future amici/rurico change does not silently misclassify.
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

    // T-237: Input variant maps to USAGE (sysexits 64)
    #[test]
    fn exit_code_input_is_usage() {
        assert_code(&SaeError::Input("bad".into()), codes::USAGE);
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

    // T-241: Json variant maps to INTERNAL (sysexits 70, ADR-0066 Group 2)
    #[test]
    fn exit_code_json_is_internal() {
        let err: serde_json::Error = serde_json::from_str::<i32>("not json").unwrap_err();
        assert_code(&SaeError::Json(err), codes::INTERNAL);
    }

    // T-242: Other variant maps to INTERNAL (sysexits 70, ADR-0066 Group 2).
    // Other carries model-related operational failures (MLX backend, probe,
    // batch embedding, model-cache lookup) — all classified as "model
    // artifact 不整合" per ADR-0066.
    #[test]
    fn exit_code_other_is_internal() {
        assert_code(&SaeError::Other("unexpected".into()), codes::INTERNAL);
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

    // T-245: Client(Api) maps to INTERNAL (sysexits 70) — non-retryable HTTP 4xx
    #[test]
    fn exit_code_client_api_is_internal() {
        assert_code(
            &SaeError::Client(ClientError::Api("404 Not Found".into())),
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

    // T-248: Sync(Client(Api)) maps to INTERNAL (sysexits 70) — non-retryable HTTP 4xx
    #[test]
    fn exit_code_sync_client_api_is_internal() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::Api("404 Not Found".into()))),
            codes::INTERNAL,
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

    // T-310: input_no_data_for_team_pins_next_step_harvest_hint
    #[test]
    fn input_no_data_for_team_pins_next_step_harvest_hint() {
        let err = SaeError::input_no_data_for_team("gaji");
        assert_eq!(
            err.next_step(),
            Some("Run `sae harvest <team>` to fetch posts."),
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
    // as Self::Client; harvest wraps it under Sync. Both must give the same hint.
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

    // T-317: next_step_input_without_recognized_prefix_returns_none
    #[test]
    fn next_step_input_without_recognized_prefix_returns_none() {
        let err = SaeError::Input(
            "Invalid date '--from 2025-xx-xx': expected YYYY-MM-DD (e.g. 2025-01-01)".into(),
        );
        assert_eq!(err.next_step(), None);
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
        let err = SaeError::Input("bad".into());
        assert!(!err.retryable(), "Input must not be retryable");
    }

    // T-324: candidates_returns_empty_in_phase_2_1
    #[test]
    fn candidates_returns_empty_in_phase_2_1() {
        for err in [
            SaeError::Input("anything".into()),
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

    // T-326: Probe(HandlerNotInstalled) → INTERNAL (70).
    // Host bin forgot to call `rurico::handle_probe_if_needed()` — an invariant
    // violation in the sae binary itself, not retryable by the agent.
    #[test]
    fn exit_code_probe_handler_not_installed_is_internal() {
        assert_code(
            &SaeError::Probe(ProbeError::HandlerNotInstalled),
            codes::INTERNAL,
        );
    }

    // T-327: Probe(ModelLoadFailed) → INTERNAL (70).
    // Model artifact 不整合 (ADR-0066 Group 2).
    #[test]
    fn exit_code_probe_model_load_failed_is_internal() {
        assert_code(
            &SaeError::Probe(ProbeError::ModelLoadFailed {
                reason: "weight tensor missing".into(),
            }),
            codes::INTERNAL,
        );
    }

    // T-328: Probe(SetupRejected) → INTERNAL (70).
    // env / path invariant violation in the probe child setup phase.
    #[test]
    fn exit_code_probe_setup_rejected_is_internal() {
        use rurico::model_probe::SetupReason;
        for reason in [
            SetupReason::EnvIncomplete,
            SetupReason::CanonicalizeFailed,
            SetupReason::PathOutsideCache,
            SetupReason::CacheRootInvalid,
        ] {
            assert_code(
                &SaeError::Probe(ProbeError::SetupRejected { reason }),
                codes::INTERNAL,
            );
        }
    }

    // T-329: Probe(SubprocessFailed) → IO_ERR (74).
    // Subprocess spawn / wait / pipe IO failure — distinct from
    // ModelLoadFailed because the load itself never ran. Sysexits classifies
    // this as IO_ERROR.
    #[test]
    fn exit_code_probe_subprocess_failed_is_io_err() {
        assert_code(
            &SaeError::Probe(ProbeError::SubprocessFailed(
                "probe spawn failed: ENOENT".into(),
            )),
            codes::IO_ERR,
        );
    }

    // T-330: PROBE_EXIT_* wire format (3-8) cannot leak through SaeError::exit_code.
    // The raw probe-child exit codes encode rurico's internal IPC contract;
    // a parent that exits with those numbers would confuse downstream agents
    // expecting sysexits classifications. This test pins that even when the
    // ProbeError carries a SetupReason whose `code()` is 5, the wrapping
    // SaeError emits 70 (INTERNAL), not 5.
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

    // T-332: extract_probe_error walks the source chain to recover the
    // original ProbeError variant from ModelInitError::Backend.
    #[test]
    fn extract_probe_error_backend_with_source_recovers_typed_variant() {
        let init: ModelInitError = ProbeError::HandlerNotInstalled.into();
        let probe = extract_probe_error(init).expect("Backend with source yields ProbeError");
        assert!(
            matches!(probe, ProbeError::HandlerNotInstalled),
            "expected HandlerNotInstalled, got {probe:?}"
        );
    }

    // T-333: extract_probe_error maps source-less Backend to SubprocessFailed.
    // From<ProbeError>::From routes SubprocessFailed → Backend { source: None }
    // (the only variant whose payload is a bare String); the walker must
    // round-trip that back to SubprocessFailed using the Backend.message.
    #[test]
    fn extract_probe_error_backend_without_source_maps_to_subprocess_failed() {
        let init: ModelInitError = ProbeError::SubprocessFailed("spawn failed".into()).into();
        let probe = extract_probe_error(init).expect("Backend message yields ProbeError");
        match probe {
            ProbeError::SubprocessFailed(msg) => assert_eq!(msg, "spawn failed"),
            other => panic!("expected SubprocessFailed, got {other:?}"),
        }
    }

    // T-334: extract_probe_error returns None for non-ProbeError sources.
    // Backend can carry MLX-layer errors from Embedder::new that are not
    // ProbeError instances. The walker must fall through so the Sae::embed
    // fallback emits SaeError::Other with the underlying detail preserved
    // (rather than misclassifying the error as a probe failure).
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
}
