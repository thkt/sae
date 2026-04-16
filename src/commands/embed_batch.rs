use rurico::embed::ChunkedEmbedding;

use crate::storage;
use crate::storage::embed::insert_new_embeddings;
use crate::tools::SaeError;

pub(crate) const BATCH_SIZE: u32 = 128;

pub(crate) struct BatchResult {
    pub(crate) added: u32,
    pub(crate) processed: u32,
    pub(crate) budget_exhausted: bool,
}

pub(crate) struct EmbedAllResult {
    pub(crate) added: u32,
    pub(crate) total_chunks: u32,
}

/// Loops [`embed_one_batch`] until no chunks remain, calling `on_progress(total_chunks)` after each batch.
pub(crate) fn embed_all<F, G>(
    conn: &rusqlite::Connection,
    embed_fn: F,
    mut on_progress: G,
) -> Result<EmbedAllResult, SaeError>
where
    F: Fn(&[&str]) -> Result<Vec<ChunkedEmbedding>, SaeError>,
    G: FnMut(u32),
{
    let mut added = 0u32;
    let mut total_chunks = 0u32;
    loop {
        let result = embed_one_batch(conn, BATCH_SIZE, &embed_fn)?;
        if result.processed == 0 {
            break;
        }
        added += result.added;
        total_chunks += result.processed;
        on_progress(total_chunks);
    }
    Ok(EmbedAllResult {
        added,
        total_chunks,
    })
}

pub(crate) fn embed_one_batch<F>(
    conn: &rusqlite::Connection,
    budget: u32,
    embed_fn: F,
) -> Result<BatchResult, SaeError>
where
    F: Fn(&[&str]) -> Result<Vec<ChunkedEmbedding>, SaeError>,
{
    let batch = storage::get_unembedded_chunks(conn, budget)?;
    if batch.is_empty() {
        return Ok(BatchResult {
            added: 0,
            processed: 0,
            budget_exhausted: false,
        });
    }
    let texts: Vec<&str> = batch.iter().map(|(_, c)| c.as_str()).collect();
    let batch_len = u32::try_from(batch.len()).expect("batch size exceeds u32::MAX");
    let embs = embed_fn(&texts)?;
    if embs.len() != batch.len() {
        return Err(SaeError::Other(format!(
            "Embedding count mismatch: expected {}, got {}",
            batch.len(),
            embs.len()
        )));
    }
    let embeddings: Vec<(i64, _)> = batch.iter().map(|(id, _)| *id).zip(embs).collect();
    let added = insert_new_embeddings(conn, &embeddings)?;
    Ok(BatchResult {
        added,
        processed: batch_len,
        budget_exhausted: batch_len == budget,
    })
}
