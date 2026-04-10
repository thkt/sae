use rurico::embed::{Embedder, ModelId, ProbeStatus};
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
    use rurico::embed::Embed;

    let team = config.resolve_team(Some(team))?;
    let db = require_db(config, team)?;

    let paths = require_embed_model()?;
    let spinner = crate::progress::Spinner::new("Loading model...");
    match Embedder::probe(&paths) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            spinner.cancel();
            return Err(SaeError::Other(
                "MLX backend is unavailable".to_string(),
            ));
        }
        Err(e) => {
            spinner.cancel();
            return Err(SaeError::Other(format!("Model probe failed: {e}")));
        }
    }
    let embedder =
        Embedder::new(&paths).map_err(|e| SaeError::Other(format!("Failed to load model: {e}")))?;
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
    match Embedder::probe(&paths) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            spinner.cancel();
            return Err(SaeError::Other(
                "Model downloaded but MLX backend is unavailable".to_string(),
            ));
        }
        Err(e) => {
            spinner.cancel();
            return Err(SaeError::Other(format!("Model probe failed: {e}")));
        }
    }
    match Embedder::new(&paths) {
        Ok(_) => {
            spinner.finish("Model ready");
            crate::output::model_download(json)
        }
        Err(e) => {
            spinner.cancel();
            Err(SaeError::Other(format!("Failed to verify model: {e}")))
        }
    }
}

pub(crate) fn require_embed_model() -> Result<rurico::embed::Artifacts, SaeError> {
    let auto_download = std::env::var("SAE_AUTO_DOWNLOAD_MODEL").as_deref() == Ok("1");
    require_embed_model_with(auto_download, || {
        rurico::embed::cached_artifacts(ModelId::default())
    })
}

fn require_embed_model_with<E: std::fmt::Display>(
    auto_download: bool,
    cache_check: impl FnOnce() -> Result<Option<rurico::embed::Artifacts>, E>,
) -> Result<rurico::embed::Artifacts, SaeError> {
    if auto_download {
        let spinner =
            crate::progress::Spinner::new("Downloading model (SAE_AUTO_DOWNLOAD_MODEL=1)...");
        return match rurico::embed::download_model(ModelId::default()) {
            Ok(paths) => {
                spinner.finish("Model downloaded");
                Ok(paths)
            }
            Err(e) => {
                spinner.cancel();
                Err(SaeError::Other(format!("Failed to download model: {e}")))
            }
        };
    }
    match cache_check() {
        Ok(Some(p)) => Ok(p),
        Ok(None) => Err(SaeError::Input(
            "Model not found. Run 'sae model download' first.".to_string(),
        )),
        Err(e) => Err(SaeError::Other(format!("Failed to check model cache: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-007: model absent → "Model not found" error (deterministic via DI)
    #[test]
    fn require_embed_model_err_when_absent() {
        let result = require_embed_model_with(false, || Ok::<_, &str>(None));
        assert!(result.is_err(), "should fail when model is not cached");
        assert!(
            result.unwrap_err().to_string().contains("Model not found"),
            "error should indicate model not found"
        );
    }

    // T-007: cache check failure → wrapped error
    #[test]
    fn require_embed_model_err_on_cache_failure() {
        let result = require_embed_model_with(false, || {
            Err::<Option<rurico::embed::Artifacts>, _>("cache broken")
        });
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to check model cache"),
            "error should wrap cache failure"
        );
    }

    // T-007: auto_download=true skips cache_check
    #[test]
    fn require_embed_model_auto_download_skips_cache() {
        let called = std::cell::Cell::new(false);
        // download_model will fail (no network), but cache_check must NOT be invoked
        let _ = require_embed_model_with(true, || {
            called.set(true);
            Ok::<_, &str>(None)
        });
        assert!(
            !called.get(),
            "cache_check should not be called when auto_download=true"
        );
    }
}
