use rurico::embed::{Embedder, ModelId};
use rurico::reranker::{Rerank, Reranker, RerankerModelId};

/// Model load state shared across sae and yomu (amici extraction target).
///
/// Identical definition exists in yomu — intentional DRY violation
/// until amici crate extraction.
pub(crate) enum ModelLoad<T> {
    Ready(T),
    Absent,
    Failed(String),
}

impl<T> ModelLoad<T> {
    pub(crate) fn as_ref(&self) -> Option<&T> {
        match self {
            Self::Ready(v) => Some(v),
            _ => None,
        }
    }

    /// Emits a user-facing hint or warning based on load state.
    /// - `Absent`: prints a hint for how to enable the model.
    /// - `Failed`: prints a warning and logs via tracing.
    /// - `Ready`: no-op.
    pub(crate) fn emit_load_hint(&self, absent_hint: &str, model_label: &str) {
        match self {
            Self::Absent => eprintln!("Hint: {absent_hint}"),
            Self::Failed(e) => {
                eprintln!("Warning: {model_label} not available ({e})");
                tracing::warn!(error = %e, "{} not available", model_label);
            }
            Self::Ready(_) => {}
        }
    }
}

impl<T> std::fmt::Debug for ModelLoad<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(_) => write!(f, "ModelLoad::Ready(..)"),
            Self::Absent => write!(f, "ModelLoad::Absent"),
            Self::Failed(msg) => write!(f, "ModelLoad::Failed({msg:?})"),
        }
    }
}

pub(crate) fn try_load_embedder() -> ModelLoad<Box<Embedder>> {
    try_load_embedder_with(|| rurico::embed::cached_artifacts(ModelId::default()))
}

pub(crate) fn try_load_embedder_with<E: std::fmt::Display>(
    cache_check: impl FnOnce() -> Result<Option<rurico::embed::Artifacts>, E>,
) -> ModelLoad<Box<Embedder>> {
    use rurico::embed::ProbeStatus;

    let paths = match cache_check() {
        Ok(Some(p)) => p,
        Ok(None) => {
            tracing::debug!("embedding model not cached");
            return ModelLoad::Absent;
        }
        Err(e) => {
            tracing::debug!(error = %e, "embedding model cache check failed");
            return ModelLoad::Failed(e.to_string());
        }
    };
    match Embedder::probe(&paths) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            tracing::debug!("MLX backend unavailable");
            return ModelLoad::Failed("MLX backend is unavailable".to_string());
        }
        Err(e) => {
            tracing::debug!(error = %e, "embedding model probe failed");
            return ModelLoad::Failed(e.to_string());
        }
    }
    match Embedder::new(&paths) {
        Ok(e) => {
            tracing::debug!("embedding model loaded");
            ModelLoad::Ready(Box::new(e))
        }
        Err(e) => {
            tracing::debug!(error = %e, "embedding model load failed");
            ModelLoad::Failed(e.to_string())
        }
    }
}

pub(crate) fn try_load_reranker() -> ModelLoad<Box<dyn Rerank>> {
    try_load_reranker_with(|| rurico::reranker::cached_artifacts(RerankerModelId::default()))
}

pub(crate) fn try_load_reranker_with<E: std::fmt::Display>(
    cache_check: impl FnOnce() -> Result<Option<rurico::reranker::Artifacts>, E>,
) -> ModelLoad<Box<dyn Rerank>> {
    use rurico::reranker::ProbeStatus;

    let artifacts = match cache_check() {
        Ok(Some(a)) => a,
        Ok(None) => {
            tracing::debug!("reranker model not cached");
            return ModelLoad::Absent;
        }
        Err(e) => {
            tracing::debug!(error = %e, "reranker model cache check failed");
            return ModelLoad::Failed(e.to_string());
        }
    };
    match Reranker::probe(&artifacts) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            tracing::debug!("MLX backend unavailable for reranker");
            return ModelLoad::Failed("MLX backend is unavailable".to_string());
        }
        Err(e) => {
            tracing::debug!(error = %e, "reranker model probe failed");
            return ModelLoad::Failed(e.to_string());
        }
    }
    match Reranker::new(&artifacts) {
        Ok(r) => {
            tracing::debug!("reranker model loaded");
            ModelLoad::Ready(Box::new(r))
        }
        Err(e) => {
            tracing::debug!(error = %e, "reranker model load failed");
            ModelLoad::Failed(e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TC-042: try_load_reranker_with returns Absent when model is not cached
    #[test]
    fn try_load_reranker_with_returns_absent_when_no_model() {
        let result = try_load_reranker_with(|| Ok::<_, &str>(None));
        assert!(matches!(result, ModelLoad::Absent));
    }

    // TC-043: try_load_reranker_with returns Failed on cache check error
    #[test]
    fn try_load_reranker_with_returns_failed_on_cache_error() {
        let result = try_load_reranker_with(|| {
            Err::<Option<rurico::reranker::Artifacts>, _>("cache broken")
        });
        match result {
            ModelLoad::Failed(msg) => assert!(msg.contains("cache broken")),
            other => panic!("expected ModelLoad::Failed, got {other:?}"),
        }
    }
}
