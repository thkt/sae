use amici::model::ModelLoad;
use amici::model::embedder::DegradedReason;
use amici::model::reranker::try_load_reranker_with;
use rurico::reranker::{Rerank, RerankerModelId, cached_artifacts};

pub(crate) fn try_load_reranker() -> ModelLoad<Box<dyn Rerank>> {
    match try_load_reranker_with(
        || cached_artifacts(RerankerModelId::default()),
        |e| tracing::warn!(error = %e, "failed to delete corrupt reranker model files"),
        |e| tracing::warn!(error = %e, "reranker failed to load"),
    ) {
        Ok(r) => ModelLoad::Ready(r),
        Err(DegradedReason::NotInstalled) => ModelLoad::Absent,
        Err(reason) => ModelLoad::Failed(reason.to_string()),
    }
}
