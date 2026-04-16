use std::convert::Infallible;
use std::io;
use std::process::ExitCode;

use amici::cli::embed_with_spinners;
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

pub fn exit_code_for(e: &SaeError) -> ExitCode {
    match e {
        SaeError::Input(_) | SaeError::Config(_) => ExitCode::from(2),
        SaeError::Client(ClientError::TokenNotSet) => ExitCode::from(2),
        SaeError::Sync(SyncError::Client(ClientError::TokenNotSet)) => ExitCode::from(2),
        SaeError::Storage(_) | SaeError::Sync(SyncError::Storage(_)) | SaeError::Json(_) => {
            ExitCode::from(4)
        }
        SaeError::Client(_) | SaeError::Sync(_) | SaeError::Io(_) | SaeError::Other(_) => {
            ExitCode::FAILURE
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
