use std::convert::Infallible;
use std::io;
use std::process::ExitCode;

use amici::cli::embed_with_spinners;
use amici::cli::env_lookup;
use amici::cli::exit_code::{CliError, codes};
use amici::model::download_and_verify_model;
use amici::model::embedder::{DegradedReason, try_load_embedder_with};
use rurico::embed::{Artifacts, ModelId, cached_artifacts};

use crate::client::{ClientError, EsaClient};
use crate::commands;
use crate::config::{Config, ConfigError};
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

    pub async fn harvest(&self, team: &str, full: bool, json: bool) -> Result<String, SaeError> {
        let (team, client) = resolve_client(&self.config, Some(team))?;
        let db_path = self.config.team_db_path(team)?;
        let db = Db::open(&db_path)?;
        let result = sync::harvest(&client, &db, team, full).await?;
        output::harvest(&result, json)
    }

    pub fn embed(&self, team: &str, json: bool) -> Result<String, SaeError> {
        let team = self.config.resolve_team(Some(team))?;
        let db = require_db(&self.config, team)?;
        let paths = require_embed_model()?;
        let pending = count_unembedded_chunks(db.conn())?;

        let result = embed_with_spinners(
            pending,
            |_| {
                let mut embed_err: Option<String> = None;
                match try_load_embedder_with(
                    || Ok::<_, Infallible>(Some(paths)),
                    |e| tracing::warn!(error = %e, "failed to delete corrupt model files"),
                    |e| {
                        embed_err = Some(e.to_string());
                    },
                ) {
                    Ok(e) => Ok(e),
                    Err(DegradedReason::BackendUnavailable) => {
                        Err(SaeError::Other("MLX backend is unavailable".to_owned()))
                    }
                    Err(reason) => {
                        let detail = embed_err.map(|e| format!(": {e}")).unwrap_or_default();
                        Err(SaeError::Other(format!(
                            "Model probe failed: {reason:?}{detail}"
                        )))
                    }
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
                    json,
                )
            }
            None => output::embed(&EmbedResult { chunks_embedded: 0 }, 0, json),
        }
    }

    pub fn model_download(json: bool) -> Result<String, SaeError> {
        download_and_verify_model().map_err(|e| SaeError::Other(e.to_string()))?;
        output::model_download(json)
    }

    pub async fn get(
        &self,
        number: u32,
        team: Option<&str>,
        with_body: bool,
        json: bool,
    ) -> Result<String, SaeError> {
        commands::post::run_get(&self.config, number, team, with_body, json).await
    }

    pub async fn create(&self, args: CreateArgs, json: bool) -> Result<String, SaeError> {
        commands::post::run_create(&self.config, args, json).await
    }

    pub async fn update(&self, args: UpdateArgs, json: bool) -> Result<String, SaeError> {
        commands::post::run_update(&self.config, args, json).await
    }

    pub async fn archive(
        &self,
        number: u32,
        team: Option<&str>,
        dry_run: bool,
        json: bool,
    ) -> Result<String, SaeError> {
        commands::archive::run_archive(&self.config, number, team, dry_run, json).await
    }

    pub async fn ship(
        &self,
        number: u32,
        team: Option<&str>,
        dry_run: bool,
        json: bool,
    ) -> Result<String, SaeError> {
        commands::archive::run_ship(&self.config, number, team, dry_run, json).await
    }

    pub fn search(&self, args: SearchArgs, json: bool) -> Result<String, SaeError> {
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
            json,
        )
    }

    pub fn status(&self, team: Option<&str>, json: bool) -> Result<String, SaeError> {
        commands::status::run_status(&self.config, team, json)
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
    /// User action required (e.g., run harvest or model download first)
    #[error("{0}")]
    Input(String),
    /// Operational failure (e.g., model load, embedding, download)
    #[error("{0}")]
    Other(String),
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
        return Err(SaeError::Input(format!(
            "No data for team '{team}'. Run `sae harvest {team}` first."
        )));
    }
    Ok(Db::open(&db_path)?)
}

impl CliError for SaeError {
    fn exit_code(&self) -> ExitCode {
        match self {
            Self::Input(_) | Self::Config(_) => ExitCode::from(codes::USAGE),
            Self::Client(ClientError::TokenNotSet)
            | Self::Sync(SyncError::Client(ClientError::TokenNotSet)) => {
                ExitCode::from(codes::USAGE)
            }
            Self::Storage(_) | Self::Sync(SyncError::Storage(_)) => {
                ExitCode::from(codes::CANT_CREAT)
            }
            // Api(_) is non-retryable (HTTP 4xx after retry_wait returned None).
            // Json/Other are internal/data shape errors. All map to SOFTWARE (no retry).
            Self::Json(_)
            | Self::Other(_)
            | Self::Client(ClientError::Api(_))
            | Self::Sync(SyncError::Client(ClientError::Api(_))) => ExitCode::from(codes::SOFTWARE),
            // reqwest decode failures are schema/payload issues, not transient network.
            Self::Client(ClientError::Network(e))
            | Self::Sync(SyncError::Client(ClientError::Network(e)))
                if e.is_decode() =>
            {
                ExitCode::from(codes::SOFTWARE)
            }
            Self::Io(_) => ExitCode::from(codes::IO_ERR),
            // Remaining Client/Sync paths (Network connect/timeout, MaxRetries) are retry-eligible.
            Self::Client(_) | Self::Sync(_) => ExitCode::from(codes::TEMP_FAIL),
        }
    }
}

fn require_embed_model() -> Result<Artifacts, SaeError> {
    match cached_artifacts(ModelId::default()) {
        Ok(Some(p)) => Ok(p),
        Ok(None) => Err(SaeError::Input(
            "Model not found. Run 'sae model download' first.".to_owned(),
        )),
        Err(e) => Err(SaeError::Other(format!("Failed to check model cache: {e}"))),
    }
}

#[cfg(test)]
mod tests {
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

    // T-241: Json variant maps to SOFTWARE (sysexits 70)
    #[test]
    fn exit_code_json_is_software() {
        let err: serde_json::Error = serde_json::from_str::<i32>("not json").unwrap_err();
        assert_code(&SaeError::Json(err), codes::SOFTWARE);
    }

    // T-242: Other variant maps to SOFTWARE (sysexits 70)
    #[test]
    fn exit_code_other_is_software() {
        assert_code(&SaeError::Other("unexpected".into()), codes::SOFTWARE);
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

    // T-245: Client(Api) maps to SOFTWARE (sysexits 70) — non-retryable HTTP 4xx
    #[test]
    fn exit_code_client_api_is_software() {
        assert_code(
            &SaeError::Client(ClientError::Api("404 Not Found".into())),
            codes::SOFTWARE,
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

    // T-248: Sync(Client(Api)) maps to SOFTWARE (sysexits 70) — non-retryable HTTP 4xx
    #[test]
    fn exit_code_sync_client_api_is_software() {
        assert_code(
            &SaeError::Sync(SyncError::Client(ClientError::Api("404 Not Found".into()))),
            codes::SOFTWARE,
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
}
