use std::io;
use std::process::ExitCode;
use std::sync::Arc;

use crate::compat::{CliError, embed_with_spinners};
use crate::embedding::Embed;
#[cfg(feature = "test-support")]
use crate::embedding::MockEmbedder;
#[cfg(not(feature = "test-support"))]
use crate::model;
use crate::model::{ModelDownloadError, download_and_verify_model};

use crate::client::{ClientError, EsaClient};
use crate::commands;
use crate::config::{Config, ConfigError};
use crate::envelope::{CommandOutput, ErrorCode, ErrorEnvelope, ErrorPayload};
use crate::output;
use crate::storage::{Db, StorageError, count_unembedded_chunks, validate_embedding_index};
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
}

impl Sae {
    pub fn new(config: Config) -> Self {
        Self { config }
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
        validate_embedding_index(db.conn())?;
        let result = sync::harvest(&client, &db, team, full).await?;
        // Clear earlier embedding backlogs even when no posts changed; harvest is already committed.
        let embedded = embed_pending(&db)?.map(|r| r.added);
        output::index(&result, embedded)
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
    #[error("model load failed: {0}")]
    ModelLoadFailed(String),
    #[error("{0}")]
    InputUsage(String),
    #[error("{0}")]
    InputData(String),
    #[error("Gemma native backend is unavailable")]
    BackendUnavailable,
    #[error("{0}")]
    EmbedderUnavailable(String),
    /// Classified invariant failure (INTERNAL).
    #[error("{0}")]
    Internal(String),
    /// Unclassified runtime failure (UNKNOWN).
    #[error("{0}")]
    Other(String),
}

const NO_DATA_PREFIX: &str = "No data for team ";

const BACKEND_UNAVAILABLE_HINT: &str =
    "Install a build with the Gemma native backend, or pass `--no-embed` for FTS-only search.";

const POST_NOT_FOUND_HINT: &str =
    "Verify the post number exists in esa, or run `sae search <keyword>` to find it.";

impl SaeError {
    pub(crate) fn input_no_data_for_team(team: &str) -> Self {
        Self::InputUsage(format!(
            "{NO_DATA_PREFIX}'{team}'. Run `sae index {team}` first."
        ))
    }

    pub(crate) fn error_code(&self) -> ErrorCode {
        match self {
            Self::InputUsage(_) | Self::Config(_) => ErrorCode::UsageError,
            Self::InputData(_) => ErrorCode::DataError,
            Self::Client(ClientError::TokenNotSet)
            | Self::Sync(SyncError::Client(ClientError::TokenNotSet)) => ErrorCode::UsageError,
            Self::Storage(StorageError::IndexMismatch(_)) => ErrorCode::DataError,
            Self::Storage(e) if e.is_retryable() => ErrorCode::TempFailure,
            Self::Sync(e) if e.is_retryable() => ErrorCode::TempFailure,
            Self::Storage(_) | Self::Sync(SyncError::Storage(_)) => ErrorCode::CantCreat,
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
            )
            | Self::BackendUnavailable
            | Self::Internal(_)
            | Self::ModelLoadFailed(_) => ErrorCode::Internal,
            Self::EmbedderUnavailable(_) => ErrorCode::TempFailure,
            Self::Client(_) | Self::Sync(_) => ErrorCode::TempFailure,
            Self::Other(_) => ErrorCode::Unknown,
        }
    }

    pub(crate) fn next_step(&self) -> Option<&'static str> {
        match self {
            Self::InputUsage(s) if s.starts_with(NO_DATA_PREFIX) => {
                Some("Run `sae index <team>` to fetch posts.")
            }
            Self::Config(ConfigError::NoTeamSpecified) => {
                Some("Pass --team <name> or set `SAE_TEAM=<name>`.")
            }
            Self::Client(ClientError::TokenNotSet)
            | Self::Sync(SyncError::Client(ClientError::TokenNotSet)) => {
                Some("Set `ESA_ACCESS_TOKEN=<token>` and retry.")
            }
            Self::Client(ClientError::Api { status: 404, .. })
            | Self::Sync(SyncError::Client(ClientError::Api { status: 404, .. })) => {
                Some(POST_NOT_FOUND_HINT)
            }
            Self::Client(ClientError::MaxRetries(_))
            | Self::Sync(SyncError::Client(ClientError::MaxRetries(_))) => {
                Some("Retry after the rate-limit window resets.")
            }
            Self::ModelDownload(ModelDownloadError::DownloadFailed(_)) => {
                Some("Retry `sae model download`; the failure is transient.")
            }
            Self::ModelDownload(ModelDownloadError::BackendUnavailable)
            | Self::BackendUnavailable => Some(BACKEND_UNAVAILABLE_HINT),
            Self::EmbedderUnavailable(_) => {
                Some("Run `sae model download` to fetch the embedding model.")
            }
            Self::Storage(StorageError::IndexMismatch(_)) => {
                Some("Run `sae model migrate <team>` to build a compatible Gemma index.")
            }
            Self::InputUsage(_)
            | Self::InputData(_)
            | Self::Config(_)
            | Self::Client(_)
            | Self::Storage(_)
            | Self::Sync(_)
            | Self::Json(_)
            | Self::Io(_)
            | Self::ModelDownload(_)
            | Self::ModelLoadFailed(_)
            | Self::Internal(_)
            | Self::Other(_) => None,
        }
    }

    pub(crate) fn candidates(&self, config: Option<&Config>) -> Vec<String> {
        match self {
            Self::Config(ConfigError::NoTeamSpecified | ConfigError::UnknownTeam(_)) => {
                config.map(|c| c.teams.clone()).unwrap_or_default()
            }
            _ => Vec::new(),
        }
    }

    pub(crate) fn retryable(&self) -> bool {
        matches!(self.error_code(), ErrorCode::TempFailure)
    }

    pub(crate) fn to_error_envelope(&self, config: Option<&Config>) -> ErrorEnvelope {
        ErrorEnvelope {
            error: ErrorPayload {
                code: self.error_code(),
                message: self.to_string(),
                next_step: self.next_step().map(String::from),
                candidates: self.candidates(config),
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
    Ok(Db::open_read_only(&db_path)?)
}

/// Local cache failures must not prevent a successful remote mutation.
pub(crate) fn try_open_team_db(config: &Config, team: Option<&str>) -> Option<Db> {
    let team = config.resolve_team(team).ok()?;
    let db_path = config.team_db_path(team).ok()?;
    if !db_path.exists() {
        return None;
    }
    match Db::open(&db_path) {
        Ok(db) => match validate_embedding_index(db.conn()) {
            Ok(()) => Some(db),
            Err(error) => {
                tracing::warn!(team, %error, "incompatible local index; write-through skipped");
                None
            }
        },
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

#[cfg(not(feature = "test-support"))]
pub(crate) fn load_embedder() -> Result<Arc<dyn Embed>, SaeError> {
    let cached = model::cached_artifacts().map_err(|error| {
        SaeError::ModelLoadFailed(format!("Failed to validate model cache: {error}"))
    })?;
    if cached.is_none() {
        return Err(SaeError::EmbedderUnavailable(
            "embedding model not installed".to_owned(),
        ));
    }
    match model::try_load_embedder_default_logging() {
        Ok(embedder) => Ok(embedder),
        Err(model::DegradedReason::BackendUnavailable) => Err(SaeError::BackendUnavailable),
        Err(model::DegradedReason::NotInstalled) => Err(SaeError::EmbedderUnavailable(
            "embedding model not installed".to_owned(),
        )),
        Err(_) => Err(SaeError::ModelLoadFailed(
            "Gemma model failed validation or isolated loading".to_owned(),
        )),
    }
}

#[cfg(feature = "test-support")]
pub(crate) fn load_embedder() -> Result<Arc<dyn Embed>, SaeError> {
    Ok(Arc::new(MockEmbedder::default()))
}

// Avoid loading the model when no chunks are pending.
fn embed_pending_with<L>(
    db: &Db,
    load: L,
) -> Result<Option<commands::embed_batch::EmbedAllResult>, SaeError>
where
    L: FnOnce() -> Result<Arc<dyn Embed>, SaeError>,
{
    let pending = count_unembedded_chunks(db.conn())?;
    embed_with_spinners(
        pending,
        |_| load(),
        |r: &commands::embed_batch::EmbedAllResult| format!("Embedded {} chunks", r.total_chunks),
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
    )
}

fn embed_pending(db: &Db) -> Result<Option<commands::embed_batch::EmbedAllResult>, SaeError> {
    embed_pending_with(db, load_embedder)
}

#[cfg(test)]
mod tests {
    use crate::compat::codes;

    use super::*;
    use std::cell::Cell;

    use crate::storage::{rechunk_post, sqlite_failure, test_post_row, upsert_post};

    fn assert_code(err: &SaeError, expected: u8) {
        assert_eq!(err.exit_code(), ExitCode::from(expected));
    }

    #[test]
    fn exit_code_input_usage_is_usage() {
        assert_code(&SaeError::InputUsage("bad".into()), codes::USAGE);
    }

    #[test]
    fn exit_code_input_data_is_data_error() {
        assert_code(
            &SaeError::InputData("Invalid date".into()),
            codes::DATA_ERROR,
        );
    }

    #[test]
    fn exit_code_config_is_usage() {
        assert_code(
            &SaeError::Config(ConfigError::NoTeamSpecified),
            codes::USAGE,
        );
    }

    #[test]
    fn exit_code_token_not_set_is_usage() {
        assert_code(&SaeError::Client(ClientError::TokenNotSet), codes::USAGE);
    }

    #[test]
    fn exit_code_storage_is_cant_creat() {
        assert_code(
            &SaeError::Storage(StorageError::Open("missing".into())),
            codes::CANT_CREAT,
        );
    }

    #[test]
    fn exit_code_json_is_internal() {
        let err: serde_json::Error = serde_json::from_str::<i32>("not json").unwrap_err();
        assert_code(&SaeError::Json(err), codes::INTERNAL);
    }

    #[test]
    fn exit_code_other_is_unknown() {
        assert_code(&SaeError::Other("unexpected".into()), codes::UNKNOWN);
    }

    #[test]
    fn exit_code_io_is_io_err() {
        assert_code(&SaeError::Io(io::Error::other("disk full")), codes::IO_ERR);
    }

    #[test]
    fn exit_code_client_max_retries_is_temp_fail() {
        assert_code(
            &SaeError::Client(ClientError::MaxRetries(5)),
            codes::TEMP_FAIL,
        );
    }

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

    #[test]
    fn exit_code_client_invalid_request_is_internal() {
        assert_code(
            &SaeError::Client(ClientError::InvalidRequest("invalid URL".into())),
            codes::INTERNAL,
        );
    }

    #[test]
    fn exit_code_model_download_failed_is_temp_fail() {
        assert_code(
            &SaeError::ModelDownload(ModelDownloadError::DownloadFailed(
                "connection reset".into(),
            )),
            codes::TEMP_FAIL,
        );
    }

    #[test]
    fn exit_code_model_download_backend_unavailable_is_internal() {
        assert_code(
            &SaeError::ModelDownload(ModelDownloadError::BackendUnavailable),
            codes::INTERNAL,
        );
    }

    #[test]
    fn exit_code_model_download_probe_failed_is_internal() {
        assert_code(
            &SaeError::ModelDownload(ModelDownloadError::ProbeFailed("hash mismatch".into())),
            codes::INTERNAL,
        );
    }

    #[test]
    fn exit_code_sync_token_not_set_is_usage() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::TokenNotSet)),
            codes::USAGE,
        );
    }

    #[test]
    fn exit_code_sync_storage_is_cant_creat() {
        assert_code(
            &SaeError::Sync(SyncError::Storage(StorageError::Open("missing".into()))),
            codes::CANT_CREAT,
        );
    }

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

    #[test]
    fn exit_code_sync_client_max_retries_is_temp_fail() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::MaxRetries(5))),
            codes::TEMP_FAIL,
        );
    }

    #[test]
    fn exit_code_sync_storage_sqlite_busy_is_temp_fail() {
        use rusqlite::ErrorCode;
        let busy = sqlite_failure(ErrorCode::DatabaseBusy, 5);
        assert_code(
            &SaeError::Sync(SyncError::Storage(StorageError::Db(busy))),
            codes::TEMP_FAIL,
        );
    }

    #[test]
    fn exit_code_sync_storage_open_is_cant_creat_after_retry_arm() {
        assert_code(
            &SaeError::Sync(SyncError::Storage(StorageError::Open("missing".into()))),
            codes::CANT_CREAT,
        );
    }

    #[test]
    fn exit_code_storage_sqlite_busy_is_temp_fail() {
        use rusqlite::ErrorCode;
        let busy = sqlite_failure(ErrorCode::DatabaseBusy, 5);
        assert_code(&SaeError::Storage(StorageError::Db(busy)), codes::TEMP_FAIL);
    }

    #[test]
    fn input_no_data_for_team_pins_next_step_index_hint() {
        let err = SaeError::input_no_data_for_team("gaji");
        assert_eq!(
            err.next_step(),
            Some("Run `sae index <team>` to fetch posts."),
            "constructor and next_step must stay in sync"
        );
    }

    #[test]
    fn next_step_config_no_team_specified() {
        let err = SaeError::Config(ConfigError::NoTeamSpecified);
        assert_eq!(
            err.next_step(),
            Some("Pass --team <name> or set `SAE_TEAM=<name>`.")
        );
    }

    #[test]
    fn next_step_client_token_not_set() {
        let err = SaeError::Client(ClientError::TokenNotSet);
        assert_eq!(
            err.next_step(),
            Some("Set `ESA_ACCESS_TOKEN=<token>` and retry.")
        );
    }

    #[test]
    fn next_step_sync_client_token_not_set() {
        let err = SaeError::Sync(SyncError::Client(ClientError::TokenNotSet));
        assert_eq!(
            err.next_step(),
            Some("Set `ESA_ACCESS_TOKEN=<token>` and retry.")
        );
    }

    #[test]
    fn next_step_sync_client_max_retries() {
        let err = SaeError::Sync(SyncError::Client(ClientError::MaxRetries(5)));
        assert_eq!(
            err.next_step(),
            Some("Retry after the rate-limit window resets.")
        );
    }

    #[test]
    fn next_step_client_max_retries_matches_sync_variant() {
        let err = SaeError::Client(ClientError::MaxRetries(5));
        assert_eq!(
            err.next_step(),
            Some("Retry after the rate-limit window resets.")
        );
    }

    #[test]
    fn next_step_model_download_failed() {
        let err = SaeError::ModelDownload(ModelDownloadError::DownloadFailed("HTTP 503".into()));
        assert_eq!(
            err.next_step(),
            Some("Retry `sae model download`; the failure is transient.")
        );
    }

    #[test]
    fn next_step_input_data_returns_none() {
        let err = SaeError::InputData(
            "Invalid date '--from 2025-xx-xx': expected YYYY-MM-DD (e.g. 2025-01-01)".into(),
        );
        assert_eq!(err.next_step(), None);
    }

    #[test]
    fn next_step_input_usage_without_recognized_prefix_returns_none() {
        let err = SaeError::InputUsage("Missing search query.".into());
        assert_eq!(err.next_step(), None);
    }

    #[test]
    fn to_error_envelope_input_data_composes_data_error_payload() {
        let err = SaeError::InputData("Invalid date '--after 2025-xx-xx'".into());
        let env = err.to_error_envelope(None);
        assert_eq!(env.error.code, ErrorCode::DataError);
        assert!(!env.error.retryable);
        assert!(env.error.next_step.is_none());
        assert!(env.error.candidates.is_empty());
        assert_eq!(env.error.message, "Invalid date '--after 2025-xx-xx'");
    }

    #[test]
    fn next_step_other_returns_none() {
        let err = SaeError::Other("internal".into());
        assert_eq!(err.next_step(), None);
    }

    #[test]
    fn next_step_storage_returns_none() {
        let err = SaeError::Storage(StorageError::Open("missing".into()));
        assert_eq!(err.next_step(), None);
    }

    #[test]
    fn retryable_true_for_model_download_failed() {
        let err = SaeError::ModelDownload(ModelDownloadError::DownloadFailed("HTTP 503".into()));
        assert!(err.retryable(), "DownloadFailed must be retryable");
    }

    #[test]
    fn retryable_true_for_sync_max_retries() {
        let err = SaeError::Sync(SyncError::Client(ClientError::MaxRetries(5)));
        assert!(err.retryable(), "MaxRetries must be retryable");
    }

    #[test]
    fn retryable_false_for_storage() {
        let err = SaeError::Storage(StorageError::Open("missing".into()));
        assert!(!err.retryable(), "Storage must not be retryable");
    }

    #[test]
    fn retryable_false_for_input() {
        for err in [
            SaeError::InputUsage("bad".into()),
            SaeError::InputData("bad".into()),
        ] {
            assert!(!err.retryable(), "{err:?} must not be retryable");
        }
    }

    #[test]
    fn exit_code_embedder_unavailable_is_temp_fail() {
        assert_code(
            &SaeError::EmbedderUnavailable("model not installed".into()),
            codes::TEMP_FAIL,
        );
    }

    #[test]
    fn retryable_true_for_embedder_unavailable() {
        assert!(
            SaeError::EmbedderUnavailable("x".into()).retryable(),
            "EmbedderUnavailable must be retryable"
        );
    }

    #[test]
    fn next_step_embedder_unavailable_recommends_model_download() {
        let err = SaeError::EmbedderUnavailable("x".into());
        assert_eq!(
            err.next_step(),
            Some("Run `sae model download` to fetch the embedding model.")
        );
    }

    #[test]
    fn embed_pending_with_no_pending_skips_loader() {
        let db = Db::open_memory().unwrap();
        let loader_called = Cell::new(false);
        let result = embed_pending_with(&db, || {
            loader_called.set(true);
            Err(SaeError::Other("loader must not run".to_owned()))
        });
        assert!(
            matches!(result, Ok(None)),
            "no pending chunks must yield Ok(None), got {result:?}"
        );
        assert!(
            !loader_called.get(),
            "loader must not be invoked when nothing is pending"
        );
    }

    #[test]
    fn embed_pending_with_pending_propagates_loader_failure() {
        let db = Db::open_memory().unwrap();
        upsert_post(db.conn(), &test_post_row(1)).unwrap();
        rechunk_post(db.conn(), 1, "# Title\nBody text").unwrap();
        assert_eq!(
            count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "fixture must leave exactly one unembedded chunk"
        );
        let result = embed_pending_with(&db, || {
            Err(SaeError::EmbedderUnavailable("not installed".to_owned()))
        });
        assert!(
            matches!(result, Err(SaeError::EmbedderUnavailable(_))),
            "pending chunks + loader failure must propagate EmbedderUnavailable, got {result:?}"
        );
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn embed_pending_embeds_all_pending_chunks() {
        let db = Db::open_memory().unwrap();
        upsert_post(db.conn(), &test_post_row(1)).unwrap();
        rechunk_post(db.conn(), 1, "# Title\nBody text").unwrap();
        assert_eq!(
            count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "fixture must leave exactly one unembedded chunk"
        );
        let result = embed_pending(&db)
            .expect("embed_pending must succeed with MockEmbedder")
            .expect("pending chunks must yield Some(EmbedAllResult)");
        assert_eq!(result.added, 1, "the single pending chunk must be embedded");
        assert_eq!(
            count_unembedded_chunks(db.conn()).unwrap(),
            0,
            "no chunks must remain unembedded after embed_pending"
        );
    }

    #[test]
    fn candidates_returns_empty_without_config() {
        for err in [
            SaeError::InputUsage("anything".into()),
            SaeError::InputData("anything".into()),
            SaeError::Config(ConfigError::NoTeamSpecified),
            SaeError::Config(ConfigError::UnknownTeam("gajj".into())),
            SaeError::Other("internal".into()),
        ] {
            assert_eq!(
                err.candidates(None),
                Vec::<String>::new(),
                "candidates(None) must be empty for {err:?}"
            );
        }
    }

    fn config_with_teams(teams: &[&str]) -> Config {
        Config {
            teams: teams.iter().map(|&t| t.to_owned()).collect(),
            default_team: None,
            embed_budget: 50,
        }
    }

    #[test]
    fn candidates_returns_teams_for_no_team_specified() {
        let cfg = config_with_teams(&["gaji", "gaji-platform"]);
        let err = SaeError::Config(ConfigError::NoTeamSpecified);
        assert_eq!(err.candidates(Some(&cfg)), vec!["gaji", "gaji-platform"]);
    }

    #[test]
    fn candidates_returns_teams_for_unknown_team() {
        let cfg = config_with_teams(&["alpha", "beta"]);
        let err = SaeError::Config(ConfigError::UnknownTeam("alphaa".into()));
        assert_eq!(err.candidates(Some(&cfg)), vec!["alpha", "beta"]);
    }

    #[test]
    fn candidates_ignores_config_for_unrelated_variants() {
        let cfg = config_with_teams(&["x", "y"]);
        for err in [
            SaeError::InputUsage("bad".into()),
            SaeError::InputData("bad".into()),
            SaeError::Other("oops".into()),
        ] {
            assert_eq!(
                err.candidates(Some(&cfg)),
                Vec::<String>::new(),
                "{err:?} must not leak the team list"
            );
        }
    }

    #[test]
    fn model_load_failure_preserves_internal_exit_and_error_envelope() {
        let err = SaeError::ModelLoadFailed("checksum mismatch".into());
        assert_code(&err, codes::INTERNAL);
        let envelope = err.to_error_envelope(None);
        assert_eq!(envelope.error.code, ErrorCode::Internal);
        assert_eq!(
            envelope.error.message,
            "model load failed: checksum mismatch"
        );
        assert!(!envelope.error.retryable);
        assert!(envelope.error.next_step.is_none());
        assert!(envelope.error.candidates.is_empty());
    }

    #[test]
    fn exit_code_backend_unavailable_is_internal() {
        assert_code(&SaeError::BackendUnavailable, codes::INTERNAL);
    }

    #[test]
    fn exit_code_internal_variant_is_internal() {
        assert_code(
            &SaeError::Internal("Embedding count mismatch: expected 5, got 4".into()),
            codes::INTERNAL,
        );
    }

    #[test]
    fn next_step_backend_unavailable_returns_install_or_no_embed_hint() {
        assert_eq!(
            SaeError::BackendUnavailable.next_step(),
            Some(BACKEND_UNAVAILABLE_HINT)
        );
    }

    #[test]
    fn next_step_model_download_backend_unavailable_returns_shared_hint() {
        let err = SaeError::ModelDownload(ModelDownloadError::BackendUnavailable);
        assert_eq!(err.next_step(), Some(BACKEND_UNAVAILABLE_HINT));
    }

    #[test]
    fn next_step_internal_variant_returns_none() {
        assert_eq!(
            SaeError::Internal("Embedding count mismatch".into()).next_step(),
            None
        );
    }

    #[test]
    fn to_error_envelope_backend_unavailable_composes_internal_payload() {
        let env = SaeError::BackendUnavailable.to_error_envelope(None);
        assert_eq!(env.error.code, ErrorCode::Internal);
        assert!(!env.error.retryable);
        assert_eq!(
            env.error.next_step.as_deref(),
            Some(BACKEND_UNAVAILABLE_HINT)
        );
        assert!(env.error.candidates.is_empty());
        assert_eq!(env.error.message, "Gemma native backend is unavailable");
    }

    #[test]
    fn to_error_envelope_internal_variant_composes_internal_payload() {
        let err = SaeError::Internal("Embedding count mismatch: expected 5, got 4".into());
        let env = err.to_error_envelope(None);
        assert_eq!(env.error.code, ErrorCode::Internal);
        assert!(!env.error.retryable);
        assert!(env.error.next_step.is_none());
        assert!(env.error.candidates.is_empty());
        assert_eq!(
            env.error.message,
            "Embedding count mismatch: expected 5, got 4"
        );
    }

    #[test]
    fn next_step_client_api_404_returns_post_not_found_hint() {
        let err = SaeError::Client(ClientError::Api {
            status: 404,
            body: r#"{"error":"not_found"}"#.into(),
        });
        assert_eq!(err.next_step(), Some(POST_NOT_FOUND_HINT));
    }

    #[test]
    fn next_step_sync_client_api_404_returns_post_not_found_hint() {
        let err = SaeError::Sync(SyncError::Client(ClientError::Api {
            status: 404,
            body: r#"{"error":"not_found"}"#.into(),
        }));
        assert_eq!(err.next_step(), Some(POST_NOT_FOUND_HINT));
    }

    #[test]
    fn next_step_client_api_non_404_returns_none() {
        let err = SaeError::Client(ClientError::Api {
            status: 500,
            body: "Internal Server Error".into(),
        });
        assert_eq!(err.next_step(), None);
    }

    #[test]
    fn to_error_envelope_client_api_404_composes_data_error_payload() {
        let err = SaeError::Client(ClientError::Api {
            status: 404,
            body: r#"{"error":"not_found","message":"Not Found"}"#.into(),
        });
        let env = err.to_error_envelope(None);
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
