use std::sync::{Arc, OnceLock};

use amici::model::embedder::{DegradedReason, try_load_embedder_default_logging};
use rurico::embed::Embed;

static EMBEDDER: OnceLock<Result<Arc<dyn Embed>, DegradedReason>> = OnceLock::new();

pub(crate) fn try_load_embedder() -> &'static Result<Arc<dyn Embed>, DegradedReason> {
    EMBEDDER.get_or_init(try_load_embedder_default_logging)
}
