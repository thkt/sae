use rusqlite::Connection;

use super::StorageError;

pub fn add_embeddings(
    conn: &Connection,
    embeddings: &[(i64, Vec<f32>)],
) -> Result<u32, StorageError> {
    if embeddings.is_empty() {
        return Ok(0);
    }
    let existing = existing_chunk_ids(conn, embeddings)?;
    let tx = conn.unchecked_transaction()?;
    let mut count = 0u32;
    for (chunk_id, embedding) in embeddings {
        if existing.contains(chunk_id) {
            continue;
        }
        let bytes: &[u8] = rurico::storage::f32_as_bytes(embedding);
        tx.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
            rusqlite::params![chunk_id, bytes],
        )?;
        count += 1;
    }
    tx.commit()?;
    Ok(count)
}

fn existing_chunk_ids(
    conn: &Connection,
    embeddings: &[(i64, Vec<f32>)],
) -> Result<std::collections::HashSet<i64>, StorageError> {
    let sql = format!(
        "SELECT chunk_id FROM vec_chunks WHERE chunk_id IN ({})",
        super::in_placeholders(embeddings.len())
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = embeddings
        .iter()
        .map(|(id, _)| id as &dyn rusqlite::types::ToSql)
        .collect();
    let ids = stmt
        .query_map(params.as_slice(), |row| row.get::<_, i64>(0))?
        .collect::<Result<std::collections::HashSet<_>, _>>()?;
    Ok(ids)
}

pub fn get_unembedded_chunks(
    conn: &Connection,
    limit: u32,
) -> Result<Vec<(i64, String)>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT c.id, c.content FROM chunks c \
         LEFT JOIN vec_chunks v ON c.id = v.chunk_id \
         WHERE v.chunk_id IS NULL \
         LIMIT ?1",
    )?;
    let rows: Vec<(i64, String)> = stmt
        .query_map([limit], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn has_embeddings(conn: &Connection) -> bool {
    conn.query_row("SELECT EXISTS(SELECT 1 FROM vec_chunks)", [], |row| {
        row.get(0)
    })
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Db;
    use rurico::embed::EMBEDDING_DIMS;

    fn make_embedding(val: f32) -> Vec<f32> {
        vec![val; EMBEDDING_DIMS as usize]
    }

    #[test]
    fn add_and_query_embeddings() {
        let db = Db::open_memory().unwrap();

        let mut row = crate::storage::test_post_row(1);
        row.body_md = "# Hello\nWorld".into();
        crate::storage::upsert_post(db.conn(), &row).unwrap();
        crate::storage::rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        let unembedded = get_unembedded_chunks(db.conn(), 100).unwrap();
        assert_eq!(unembedded.len(), 1);

        let emb = vec![(unembedded[0].0, make_embedding(0.5))];
        let added = add_embeddings(db.conn(), &emb).unwrap();
        assert_eq!(added, 1);

        assert!(get_unembedded_chunks(db.conn(), 100).unwrap().is_empty());
        assert!(has_embeddings(db.conn()));
    }

    #[test]
    fn skip_already_embedded() {
        let db = Db::open_memory().unwrap();
        let mut row = crate::storage::test_post_row(1);
        row.body_md = "# A\nB".into();
        crate::storage::upsert_post(db.conn(), &row).unwrap();
        crate::storage::rechunk_post(db.conn(), 1, "# A\nB").unwrap();

        let chunks = get_unembedded_chunks(db.conn(), 100).unwrap();
        let emb = vec![(chunks[0].0, make_embedding(1.0))];
        add_embeddings(db.conn(), &emb).unwrap();

        let added = add_embeddings(db.conn(), &emb).unwrap();
        assert_eq!(added, 0);
    }
}
