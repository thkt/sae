use rurico::embed::ChunkedEmbedding;

use crate::SaeError;

pub(crate) struct BatchResult {
    pub added: u32,
    pub processed: u32,
    pub budget_exhausted: bool,
}

pub(crate) fn embed_one_batch<F>(
    conn: &rusqlite::Connection,
    budget: u32,
    embed_fn: F,
) -> Result<BatchResult, SaeError>
where
    F: Fn(&[&str]) -> Result<Vec<ChunkedEmbedding>, SaeError>,
{
    let batch = sae::storage::get_unembedded_chunks(conn, budget)?;
    if batch.is_empty() {
        return Ok(BatchResult {
            added: 0,
            processed: 0,
            budget_exhausted: false,
        });
    }
    let texts: Vec<&str> = batch.iter().map(|(_, c)| c.as_str()).collect();
    let batch_len = batch.len() as u32;
    let embs = embed_fn(&texts)?;
    if embs.len() != batch.len() {
        return Err(SaeError::Other(format!(
            "Embedding count mismatch: expected {}, got {}",
            batch.len(),
            embs.len()
        )));
    }
    let embeddings: Vec<(i64, _)> = batch.iter().map(|(id, _)| *id).zip(embs).collect();
    let added = sae::storage::add_chunked_embeddings(conn, &embeddings)?;
    Ok(BatchResult {
        added,
        processed: batch_len,
        budget_exhausted: batch_len == budget,
    })
}
