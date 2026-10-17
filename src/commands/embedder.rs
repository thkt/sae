use std::sync::Arc;

use crate::embedding::Embed;
use crate::model::{DegradedReason, try_load_embedder_default_logging};

pub(crate) fn try_load_embedder() -> Result<Arc<dyn Embed>, DegradedReason> {
    // llama.cpp's Metal device is finalized at process exit. Models must be
    // dropped first, so an owning Arc cannot live in a never-dropped static.
    try_load_embedder_default_logging()
}
