use rurico::embed::{Embedder, ModelId};
use sae::config::Config;

use crate::{AppError, require_db, resolve_client};

pub(crate) async fn run_harvest(
    config: &Config,
    team: &str,
    full: bool,
    json: bool,
) -> Result<(), AppError> {
    let (team, client) = resolve_client(config, Some(team))?;
    let db_path = config.team_db_path(team)?;
    let db = sae::storage::Db::open(&db_path)?;
    let result = sae::sync::harvest(&client, &db, team, full).await?;
    crate::output::harvest(&result, json)?;
    Ok(())
}

pub(crate) fn run_embed(config: &Config, team: &str, json: bool) -> Result<(), AppError> {
    use rurico::embed::Embed;

    let team = config.resolve_team(Some(team))?;
    let db = require_db(config, team)?;

    let paths = require_embed_model()?;
    tracing::info!("Loading model...");
    let embedder = Embedder::new(&paths).map_err(|e| format!("Failed to load model: {e}"))?;
    tracing::info!("Model ready");

    const BATCH_SIZE: u32 = 64;
    let mut total_added = 0u32;
    let mut done = 0u32;
    loop {
        let batch = sae::storage::get_unembedded_chunks(db.conn(), BATCH_SIZE)?;
        if batch.is_empty() {
            break;
        }
        if done == 0 {
            tracing::info!("Embedding chunks...");
        }
        let texts: Vec<&str> = batch.iter().map(|(_, content)| content.as_str()).collect();
        let batch_len = batch.len() as u32;
        let embs = embedder
            .embed_documents_batch(&texts)
            .map_err(|e| format!("Batch embedding failed: {e}"))?;
        let embeddings: Vec<(i64, _)> = batch.iter().map(|(id, _)| *id).zip(embs).collect();
        total_added += sae::storage::add_chunked_embeddings(db.conn(), &embeddings)?;
        done += batch_len;
        tracing::info!("  {done} chunks processed");
    }
    let result = sae::storage::EmbedResult {
        chunks_embedded: total_added,
    };
    crate::output::embed(&result, done, json)?;
    Ok(())
}

pub(crate) fn run_model_download(json: bool) -> Result<(), AppError> {
    eprintln!("Downloading model...");
    let paths = rurico::embed::download_model(ModelId::default())
        .map_err(|e| format!("Failed to download model: {e}"))?;
    let _embedder = Embedder::new(&paths).map_err(|e| format!("Failed to verify model: {e}"))?;
    crate::output::model_download(json)?;
    Ok(())
}

pub(crate) fn require_embed_model() -> Result<rurico::embed::Artifacts, Box<dyn std::error::Error>>
{
    let auto_download = std::env::var("SAE_AUTO_DOWNLOAD_MODEL").as_deref() == Ok("1");
    require_embed_model_with(auto_download, || {
        rurico::embed::cached_artifacts(ModelId::default())
    })
}

fn require_embed_model_with<E: std::fmt::Display>(
    auto_download: bool,
    cache_check: impl FnOnce() -> Result<Option<rurico::embed::Artifacts>, E>,
) -> Result<rurico::embed::Artifacts, Box<dyn std::error::Error>> {
    if auto_download {
        eprintln!("Downloading model (SAE_AUTO_DOWNLOAD_MODEL=1)...");
        return rurico::embed::download_model(ModelId::default())
            .map_err(|e| format!("Failed to download model: {e}").into());
    }
    match cache_check() {
        Ok(Some(p)) => Ok(p),
        Ok(None) => Err("Model not found. Run 'sae model download' first.".into()),
        Err(e) => Err(format!("Failed to check model cache: {e}").into()),
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
