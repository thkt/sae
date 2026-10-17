//! Provision pinned public model bytes without requiring a GPU on the host.
use sae::model::{ModelDownloadError, download_model_artifacts};

fn main() -> Result<(), ModelDownloadError> {
    download_model_artifacts()
}
