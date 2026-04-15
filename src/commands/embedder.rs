use std::sync::{Arc, OnceLock};

use amici::model::embedder::{DegradedReason, try_load_embedder_with};
use rurico::embed::{Embed, ModelId, cached_artifacts};

static EMBEDDER: OnceLock<Result<Arc<dyn Embed>, DegradedReason>> = OnceLock::new();

pub(crate) fn try_load_embedder() -> &'static Result<Arc<dyn Embed>, DegradedReason> {
    EMBEDDER.get_or_init(|| {
        try_load_embedder_with(
            || cached_artifacts(ModelId::default()),
            |e| tracing::warn!(error = %e, "failed to delete corrupt embedder model files"),
            |_| {},
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    // T-057: EMBEDDER OnceLock initializes once; both calls return same cached reference
    #[test]
    fn embedder_once_lock_returns_same_reference_on_repeated_calls() {
        let r1 = try_load_embedder();
        let r2 = try_load_embedder();
        assert!(
            ptr::eq(r1, r2),
            "both calls must return the same static reference (loader runs at most once)"
        );
    }
}
