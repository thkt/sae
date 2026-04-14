use amici::model::embedder::{DegradedReason, try_load_embedder_with};
use rurico::embed::ModelId;
use sae::config::Config;

use crate::{SaeError, require_db, resolve_client};

pub(crate) async fn run_harvest(
    config: &Config,
    team: &str,
    full: bool,
    json: bool,
) -> Result<String, SaeError> {
    let (team, client) = resolve_client(config, Some(team))?;
    let db_path = config.team_db_path(team)?;
    let db = sae::storage::Db::open(&db_path)?;
    let result = sae::sync::harvest(&client, &db, team, full).await?;
    crate::output::harvest(&result, json)
}

pub(crate) fn run_embed(config: &Config, team: &str, json: bool) -> Result<String, SaeError> {
    let team = config.resolve_team(Some(team))?;
    let db = require_db(config, team)?;

    let paths = require_embed_model()?;
    let spinner = crate::progress::Spinner::new("Loading model...");
    let embed_err: std::cell::Cell<Option<String>> = std::cell::Cell::new(None);
    let embedder = match try_load_embedder_with(
        || Ok::<_, std::convert::Infallible>(Some(paths)),
        |e| tracing::warn!(error = %e, "failed to delete corrupt model files"),
        |e| embed_err.set(Some(e.to_string())),
    ) {
        Ok(embedder) => embedder,
        Err(DegradedReason::BackendUnavailable) => {
            spinner.cancel();
            return Err(SaeError::Other("MLX backend is unavailable".to_string()));
        }
        Err(reason) => {
            spinner.cancel();
            let detail = embed_err.take().map(|e| format!(": {e}")).unwrap_or_default();
            return Err(SaeError::Other(format!("Model probe failed: {reason:?}{detail}")));
        }
    };
    spinner.finish("Model ready");

    const BATCH_SIZE: u32 = 128;
    let pending = sae::storage::count_unembedded_chunks(db.conn())?;
    let mut total_added = 0u32;
    let mut total_chunks = 0u32;
    if pending > 0 {
        let spinner = crate::progress::Spinner::new(&format!("Embedding... 0/{pending} chunks"));
        loop {
            let result = super::embed_batch::embed_one_batch(db.conn(), BATCH_SIZE, |texts| {
                embedder
                    .embed_documents_batch(texts)
                    .map_err(|e| SaeError::Other(format!("Batch embedding failed: {e}")))
            })?;
            if result.processed == 0 {
                break;
            }
            total_added += result.added;
            total_chunks += result.processed;
            spinner.set_message(&format!("Embedding... {total_chunks}/{pending} chunks"));
        }
        spinner.finish(&format!("Embedded {total_chunks} chunks"));
    }
    tracing::info!(total_added, total_chunks, "embed complete");
    let result = sae::storage::EmbedResult {
        chunks_embedded: total_added,
    };
    crate::output::embed(&result, total_chunks, json)
}

pub(crate) fn run_model_download(json: bool) -> Result<String, SaeError> {
    let spinner = crate::progress::Spinner::new("Downloading model...");
    let paths = match rurico::embed::download_model(ModelId::default()) {
        Ok(p) => p,
        Err(e) => {
            spinner.cancel();
            return Err(SaeError::Other(format!("Failed to download model: {e}")));
        }
    };
    let load_err: std::cell::Cell<Option<String>> = std::cell::Cell::new(None);
    match try_load_embedder_with(
        || Ok::<_, std::convert::Infallible>(Some(paths)),
        |e| tracing::warn!(error = %e, "failed to delete corrupt model files"),
        |e| load_err.set(Some(e.to_string())),
    ) {
        Ok(_) => {
            spinner.finish("Model ready");
            crate::output::model_download(json)
        }
        Err(DegradedReason::BackendUnavailable) => {
            spinner.cancel();
            Err(SaeError::Other(
                "Model downloaded but MLX backend is unavailable".to_string(),
            ))
        }
        Err(reason) => {
            spinner.cancel();
            let detail = load_err.take().map(|e| format!(": {e}")).unwrap_or_default();
            Err(SaeError::Other(format!("Model probe failed: {reason:?}{detail}")))
        }
    }
}

pub(crate) fn require_embed_model() -> Result<rurico::embed::Artifacts, SaeError> {
    let auto_download = std::env::var("SAE_AUTO_DOWNLOAD_MODEL").as_deref() == Ok("1");
    require_embed_model_with(
        auto_download,
        || rurico::embed::cached_artifacts(ModelId::default()),
        || rurico::embed::download_model(ModelId::default()),
    )
}

fn require_embed_model_with<CE: std::fmt::Display, DE: std::fmt::Display>(
    auto_download: bool,
    cache_check: impl FnOnce() -> Result<Option<rurico::embed::Artifacts>, CE>,
    download_fn: impl FnOnce() -> Result<rurico::embed::Artifacts, DE>,
) -> Result<rurico::embed::Artifacts, SaeError> {
    match cache_check() {
        Ok(Some(p)) => Ok(p),
        Ok(None) if auto_download => {
            let spinner =
                crate::progress::Spinner::new("Downloading model (SAE_AUTO_DOWNLOAD_MODEL=1)...");
            match download_fn() {
                Ok(paths) => {
                    spinner.finish("Model downloaded");
                    Ok(paths)
                }
                Err(e) => {
                    spinner.cancel();
                    Err(SaeError::Other(format!("Failed to download model: {e}")))
                }
            }
        }
        Ok(None) => Err(SaeError::Input(
            "Model not found. Run 'sae model download' first.".to_string(),
        )),
        Err(e) => Err(SaeError::Other(format!("Failed to check model cache: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-007: model absent → "Model not found" error
    #[test]
    fn require_embed_model_err_when_absent() {
        let result = require_embed_model_with(
            false,
            || Ok::<_, &str>(None),
            || Err::<rurico::embed::Artifacts, _>("should not be called"),
        );
        assert!(result.is_err(), "should fail when model is not cached");
        assert!(
            result.unwrap_err().to_string().contains("Model not found"),
            "error should indicate model not found"
        );
    }

    // T-007: cache check failure → wrapped error
    #[test]
    fn require_embed_model_err_on_cache_failure() {
        let result = require_embed_model_with(
            false,
            || Err::<Option<rurico::embed::Artifacts>, _>("cache broken"),
            || Err::<rurico::embed::Artifacts, _>("should not be called"),
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to check model cache"),
            "error should wrap cache failure"
        );
    }

    // T-007: auto_download=true + cache miss → download_fn invoked and error wrapped
    #[test]
    fn require_embed_model_auto_download_err_on_download_failure() {
        let download_called = std::cell::Cell::new(false);
        let result = require_embed_model_with(
            true,
            || Ok::<_, &str>(None),
            || {
                download_called.set(true);
                Err::<rurico::embed::Artifacts, _>("network error")
            },
        );
        assert!(
            download_called.get(),
            "download_fn should be called when cache misses and auto_download=true"
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to download model"),
            "error should indicate download failure"
        );
    }
}
