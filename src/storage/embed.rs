use std::collections::HashSet;

use rurico::embed::ChunkedEmbedding;
use rusqlite::Connection;

use amici::storage::fetch_by_in_clause;

use super::{StorageError, collect_storage_rows, f32_as_bytes};

pub fn add_chunked_embeddings(
    conn: &Connection,
    embeddings: &[(i64, ChunkedEmbedding)],
) -> Result<u32, StorageError> {
    if embeddings.is_empty() {
        return Ok(0);
    }
    let chunk_ids: Vec<i64> = embeddings.iter().map(|(id, _)| *id).collect();
    let existing = existing_embedded_ids(conn, &chunk_ids)?;
    let new: Vec<_> = embeddings
        .iter()
        .filter(|(id, _)| !existing.contains(id))
        .collect();
    insert_embeddings(conn, &new)
}

/// Inserts embeddings without checking for duplicates.
///
/// Only safe when the caller guarantees none of the `chunk_ids` are already embedded
/// (e.g., immediately after [`get_unembedded_chunks`]).
pub(crate) fn insert_new_embeddings(
    conn: &Connection,
    embeddings: &[(i64, ChunkedEmbedding)],
) -> Result<u32, StorageError> {
    let refs: Vec<_> = embeddings.iter().collect();
    insert_embeddings(conn, &refs)
}

fn insert_embeddings(
    conn: &Connection,
    embeddings: &[&(i64, ChunkedEmbedding)],
) -> Result<u32, StorageError> {
    if embeddings.is_empty() {
        return Ok(0);
    }
    let tx = conn.unchecked_transaction()?;
    let mut count = 0u32;
    for (chunk_id, chunked_emb) in embeddings {
        for (sub_idx, embedding) in chunked_emb.chunks.iter().enumerate() {
            let bytes: &[u8] = f32_as_bytes(embedding);
            tx.execute(
                "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, ?2, ?3)",
                rusqlite::params![bytes, chunk_id, sub_idx as i64],
            )?;
            let vec_rowid = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO embedded_chunk_ids (chunk_id, sub_idx, vec_rowid) \
                 VALUES (?1, ?2, ?3)",
                rusqlite::params![chunk_id, sub_idx as i64, vec_rowid],
            )?;
        }
        count += 1;
    }
    tx.commit()?;
    Ok(count)
}

fn existing_embedded_ids(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<HashSet<i64>, StorageError> {
    fetch_by_in_clause(
        conn,
        chunk_ids,
        "SELECT chunk_id FROM embedded_chunk_ids WHERE chunk_id IN ({placeholders})",
        |row| row.get::<_, i64>(0),
    )
}

pub fn get_unembedded_chunks(
    conn: &Connection,
    limit: u32,
) -> Result<Vec<(i64, String)>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT c.id, c.content FROM chunks c \
         LEFT JOIN embedded_chunk_ids e ON c.id = e.chunk_id \
         WHERE e.chunk_id IS NULL \
         LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |row| Ok((row.get(0)?, row.get(1)?)))?;
    collect_storage_rows(rows)
}

pub fn count_unembedded_chunks(conn: &Connection) -> Result<u32, StorageError> {
    conn.query_row(
        "SELECT COUNT(*) FROM chunks c \
         LEFT JOIN embedded_chunk_ids e ON c.id = e.chunk_id \
         WHERE e.chunk_id IS NULL",
        [],
        |row| row.get(0),
    )
    .map_err(StorageError::from)
}

pub fn has_embeddings(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedded_chunk_ids)",
        [],
        |row| row.get(0),
    )
    .map_err(|e| {
        tracing::warn!(error = %e, "has_embeddings query failed, assuming no embeddings");
        e
    })
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{self, Db};
    use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

    fn make_chunked(val: f32) -> ChunkedEmbedding {
        ChunkedEmbedding::new(vec![vec![val; EMBEDDING_DIMS]])
    }

    fn make_multi_chunked(vals: &[f32]) -> ChunkedEmbedding {
        ChunkedEmbedding::new(vals.iter().map(|&v| vec![v; EMBEDDING_DIMS]).collect())
    }

    // T-300: existing_embedded_ids does not error on empty input (regression for IN () syntax)
    #[test]
    fn existing_embedded_ids_empty_input_returns_empty_set() {
        let db = Db::open_memory().unwrap();
        let ids = existing_embedded_ids(db.conn(), &[]).unwrap();
        assert!(
            ids.is_empty(),
            "empty input should return empty set without SQL syntax error"
        );
    }

    // T-173: add_chunked_embeddings stores embedding and clears unembedded queue
    #[test]
    fn add_and_query_embeddings() {
        let db = Db::open_memory().unwrap();

        let mut row = storage::test_post_row(1);
        row.body_md = "# Hello\nWorld".into();
        storage::upsert_post(db.conn(), &row).unwrap();
        storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        let unembedded = get_unembedded_chunks(db.conn(), 100).unwrap();
        assert_eq!(unembedded.len(), 1);

        let emb = vec![(unembedded[0].0, make_chunked(0.5))];
        let added = add_chunked_embeddings(db.conn(), &emb).unwrap();
        assert_eq!(added, 1);

        assert!(get_unembedded_chunks(db.conn(), 100).unwrap().is_empty());
        assert!(has_embeddings(db.conn()));
    }

    // T-174: add_chunked_embeddings skips chunks that are already embedded
    #[test]
    fn skip_already_embedded() {
        let db = Db::open_memory().unwrap();
        let mut row = storage::test_post_row(1);
        row.body_md = "# A\nB".into();
        storage::upsert_post(db.conn(), &row).unwrap();
        storage::rechunk_post(db.conn(), 1, "# A\nB").unwrap();

        let chunks = get_unembedded_chunks(db.conn(), 100).unwrap();
        let emb = vec![(chunks[0].0, make_chunked(1.0))];
        add_chunked_embeddings(db.conn(), &emb).unwrap();

        let added = add_chunked_embeddings(db.conn(), &emb).unwrap();
        assert_eq!(added, 0);
    }

    // T-175: add_chunked_embeddings stores multiple sub-embeddings per chunk
    #[test]
    fn multi_chunk_stores_all_sub_embeddings() {
        let db = Db::open_memory().unwrap();
        let mut row = storage::test_post_row(1);
        row.body_md = "# Hello\nWorld".into();
        storage::upsert_post(db.conn(), &row).unwrap();
        storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        let chunks = get_unembedded_chunks(db.conn(), 100).unwrap();
        let chunk_id = chunks[0].0;

        let emb = vec![(chunk_id, make_multi_chunked(&[0.1, 0.9]))];
        let added = add_chunked_embeddings(db.conn(), &emb).unwrap();
        assert_eq!(added, 1);

        assert!(get_unembedded_chunks(db.conn(), 100).unwrap().is_empty());
        assert!(has_embeddings(db.conn()));
    }

    // T-176: count_unembedded_chunks returns zero on a fresh empty database
    #[test]
    fn count_unembedded_returns_zero_on_empty_db() {
        let db = Db::open_memory().unwrap();
        assert_eq!(count_unembedded_chunks(db.conn()).unwrap(), 0);
    }

    // T-177: count_unembedded_chunks decrements after embedding is added
    #[test]
    fn count_unembedded_returns_correct_count_before_and_after_embed() {
        let db = Db::open_memory().unwrap();
        let mut row = storage::test_post_row(1);
        row.body_md = "# Hello\nWorld".into();
        storage::upsert_post(db.conn(), &row).unwrap();
        storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        assert_eq!(count_unembedded_chunks(db.conn()).unwrap(), 1);

        let chunks = get_unembedded_chunks(db.conn(), 100).unwrap();
        let emb = vec![(chunks[0].0, make_chunked(0.5))];
        add_chunked_embeddings(db.conn(), &emb).unwrap();

        assert_eq!(count_unembedded_chunks(db.conn()).unwrap(), 0);
    }
}
