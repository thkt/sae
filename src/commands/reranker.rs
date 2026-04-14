use std::sync::{Arc, OnceLock};

use amici::model::ModelLoad;
use amici::model::embedder::{DegradedReason, try_load_embedder_with};
use amici::model::reranker::try_load_reranker_with;
use rurico::embed::{Embed, ModelId};
use rurico::reranker::{Rerank, RerankerModelId};

static EMBEDDER: OnceLock<Result<Arc<dyn Embed>, DegradedReason>> = OnceLock::new();

pub(crate) fn try_load_embedder() -> &'static Result<Arc<dyn Embed>, DegradedReason> {
    EMBEDDER.get_or_init(|| {
        try_load_embedder_with(
            || rurico::embed::cached_artifacts(ModelId::default()),
            |e| tracing::warn!(error = %e, "failed to delete corrupt embedder model files"),
        )
    })
}

pub(crate) fn try_load_reranker() -> ModelLoad<Box<dyn Rerank>> {
    try_load_reranker_with(|| rurico::reranker::cached_artifacts(RerankerModelId::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-107: EMBEDDER OnceLock initializes once; both calls return same cached reference
    #[test]
    fn t107_embedder_onclock_returns_same_reference_on_repeated_calls() {
        let r1 = try_load_embedder();
        let r2 = try_load_embedder();
        assert!(
            std::ptr::eq(r1, r2),
            "both calls must return the same static reference (loader runs at most once)"
        );
    }
}
