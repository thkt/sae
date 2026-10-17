use std::time::Duration;

pub const EMBEDDING_DIMS: usize = 768;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("empty embedding sequence")]
    EmptySequence,
    #[error("embedding inference failed: {message}")]
    Inference {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },
}

impl From<anyhow::Error> for EmbedError {
    fn from(error: anyhow::Error) -> Self {
        Self::Inference {
            message: error.to_string(),
            source: Some(error),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkedEmbedding {
    pub chunks: Vec<Vec<f32>>,
}
impl ChunkedEmbedding {
    /// Constructs a nonempty embedding for compatibility with persisted chunk callers.
    pub fn new(chunks: Vec<Vec<f32>>) -> Self {
        Self::try_new(chunks).expect("chunked embedding must contain nonempty vectors")
    }

    pub fn try_new(chunks: Vec<Vec<f32>>) -> Result<Self, EmbedError> {
        if chunks.is_empty() || chunks.iter().any(Vec::is_empty) {
            return Err(EmbedError::EmptySequence);
        }
        Ok(Self { chunks })
    }
    pub fn chunks(&self) -> &[Vec<f32>] {
        &self.chunks
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbedOptions {
    pub token_budget: Option<usize>,
    pub forward_pause: Option<Duration>,
}

pub trait Embed: Send + Sync {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError>;
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError>;
    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        _options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.embed_documents_batch(texts)
    }
}

/// Deterministic one-hot embeddings for integration tests without model loading.
#[cfg(any(test, feature = "test-support"))]
pub struct MockEmbedder {
    dims: usize,
}
#[cfg(any(test, feature = "test-support"))]
impl Default for MockEmbedder {
    fn default() -> Self {
        Self {
            dims: EMBEDDING_DIMS,
        }
    }
}
#[cfg(any(test, feature = "test-support"))]
impl MockEmbedder {
    pub fn with_dims(dims: usize) -> Self {
        Self { dims }
    }
    fn vector(&self, index: usize) -> Result<Vec<f32>, EmbedError> {
        if self.dims == 0 {
            return Err(EmbedError::EmptySequence);
        }
        let mut vector = vec![0.; self.dims];
        vector[index % self.dims] = 1.;
        Ok(vector)
    }
}
#[cfg(any(test, feature = "test-support"))]
impl Embed for MockEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        self.vector(0)
    }
    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_query(text)
    }
    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        ChunkedEmbedding::try_new(vec![self.vector(0)?])
    }
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        texts
            .iter()
            .enumerate()
            .map(|(i, _)| ChunkedEmbedding::try_new(vec![self.vector(i)?]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{ChunkedEmbedding, Embed, EmbedError, EmbedOptions, MockEmbedder};
    use std::error::Error;

    #[test]
    fn chunked_embeddings_reject_empty_sequences_and_preserve_nonempty_order() {
        assert!(matches!(
            ChunkedEmbedding::try_new(vec![]),
            Err(EmbedError::EmptySequence)
        ));
        assert!(matches!(
            ChunkedEmbedding::try_new(vec![vec![1.0], vec![]]),
            Err(EmbedError::EmptySequence)
        ));
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            ChunkedEmbedding::try_new(vectors.clone()).unwrap().chunks(),
            vectors
        );
    }

    #[test]
    fn inference_conversion_preserves_original_error_for_diagnostics() {
        let error = EmbedError::from(anyhow::anyhow!("native encoder unavailable"));
        assert_eq!(
            error.to_string(),
            "embedding inference failed: native encoder unavailable"
        );
        assert_eq!(
            error.source().unwrap().to_string(),
            "native encoder unavailable"
        );
    }

    #[test]
    fn deterministic_embedder_honors_dimensions_and_batch_options_fallback() {
        let model = MockEmbedder::with_dims(3);
        assert_eq!(model.embed_query("query").unwrap(), [1.0, 0.0, 0.0]);
        assert_eq!(
            model
                .embed_text("document", "title: none | text: ")
                .unwrap(),
            [1.0, 0.0, 0.0]
        );
        let batch = model
            .embed_documents_batch_with_options(&["a", "b", "c", "d"], &EmbedOptions::default())
            .unwrap();
        assert_eq!(batch.len(), 4);
        assert_eq!(batch[1].chunks()[0], [0.0, 1.0, 0.0]);
        assert_eq!(batch[2].chunks()[0], [0.0, 0.0, 1.0]);
        assert_eq!(batch[3].chunks()[0], [1.0, 0.0, 0.0]);
        let empty = MockEmbedder::with_dims(0);
        assert!(matches!(
            empty.embed_query("query"),
            Err(EmbedError::EmptySequence)
        ));
        assert!(matches!(
            empty.embed_document("document"),
            Err(EmbedError::EmptySequence)
        ));
    }
}
