use rurico::embed::ChunkedEmbedding;

use crate::storage;
use crate::storage::embed::insert_new_embeddings;
use crate::tools::SaeError;

pub(crate) const BATCH_SIZE: u32 = 512;

#[derive(Debug)]
pub(crate) struct BatchResult {
    pub(crate) added: u32,
    pub(crate) processed: u32,
}

#[derive(Debug)]
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
        });
    }
    let texts: Vec<&str> = batch.iter().map(|(_, c)| c.as_str()).collect();
    let batch_len = u32::try_from(batch.len()).expect("batch size exceeds u32::MAX");
    let embs = embed_fn(&texts)?;
    if embs.len() != batch.len() {
        // Programmer-detectable invariant violation: embedder returned a
        // vector count that does not match the requested batch. Surfaces as
        // `INTERNAL` so agents can distinguish bug signals from the
        // `anyhow`-swallow `Other` (=UNKNOWN) path (#127 CHX-001).
        return Err(SaeError::Internal(format!(
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{self, Db};
    use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

    fn db_with_chunks(n: u32) -> Db {
        let db = Db::open_memory().unwrap();
        for i in 1..=n {
            storage::upsert_post(db.conn(), &storage::test_post_row(i)).unwrap();
            storage::rechunk_post(db.conn(), i, "# Hello\nWorld").unwrap();
        }
        db
    }

    fn ok_embed(texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, SaeError> {
        Ok(texts
            .iter()
            .map(|_| ChunkedEmbedding::new(vec![vec![0.5; EMBEDDING_DIMS]]))
            .collect())
    }

    // T-411: embed_one_batch embeds pending chunks within budget and clears them
    #[test]
    fn embed_one_batch_embeds_pending() {
        let db = db_with_chunks(1);
        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 1);
        let result = embed_one_batch(db.conn(), 256, ok_embed).unwrap();
        assert_eq!(result.processed, 1, "one chunk should be processed");
        assert_eq!(result.added, 1, "one embedding should be inserted");
        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 0);
    }

    // T-412: embed_all loops embed_one_batch until every pending chunk is embedded
    #[test]
    fn embed_all_embeds_all_pending() {
        let db = db_with_chunks(2);
        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 2);
        let result = embed_all(db.conn(), ok_embed, |_| {}).unwrap();
        assert_eq!(result.total_chunks, 2, "both chunks should be processed");
        assert_eq!(result.added, 2, "both embeddings should be inserted");
        assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 0);
    }

    // T-413: embed_one_batch rejects an embedding-count mismatch as Internal
    // (#127 invariant: embedder returning the wrong vector count is a
    // programmer-detectable bug, routed to INTERNAL not the anyhow-swallow path).
    #[test]
    fn embed_one_batch_rejects_count_mismatch() {
        let db = db_with_chunks(1);
        let result = embed_one_batch(db.conn(), 256, |_| Ok(vec![]));
        assert!(
            matches!(result, Err(SaeError::Internal(_))),
            "count mismatch must surface as Internal, got {result:?}"
        );
        assert_eq!(
            storage::count_unembedded_chunks(db.conn()).unwrap(),
            1,
            "a rejected batch must not change the unembedded count"
        );
    }
}
