use super::StorageError;
use crate::model::ModelSpec;
use rusqlite::{Connection, OptionalExtension};
use serde_json::json;

/// Full encoder and stored-text contract, rather than dimensionality alone.
pub fn embedding_manifest() -> String {
    json!({
        "version": 1,
        "execution": if cfg!(feature = "test-support") { "deterministic-test-double" } else { "native-gemma" },
        "encoder": ModelSpec::default(),
        "chunk_policy": "sae-markdown-section-16000-bytes-v1",
        "document_input": "stored chunk.content; no additional title",
    })
    .to_string()
}

pub(super) fn initialize(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sae_index_manifest (
        id INTEGER PRIMARY KEY CHECK (id=1), manifest TEXT NOT NULL
    ); CREATE TABLE IF NOT EXISTS embedding_content_hashes (
        chunk_id INTEGER PRIMARY KEY, content_sha256 TEXT NOT NULL
    );",
    )?;
    // An existing nonempty legacy vector space may never be relabeled Gemma.
    let vectors: i64 = conn.query_row("SELECT count(*) FROM vec_chunks", [], |r| r.get(0))?;
    let embedded: i64 =
        conn.query_row("SELECT count(*) FROM embedded_chunk_ids", [], |r| r.get(0))?;
    if vectors == 0 && embedded == 0 {
        conn.execute(
            "INSERT OR IGNORE INTO sae_index_manifest VALUES (1, ?1)",
            [embedding_manifest()],
        )?;
    }
    Ok(())
}

pub fn validate_embedding_index(conn: &Connection) -> Result<(), StorageError> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='sae_index_manifest')", [], |r| r.get(0),
    )?;
    let found: Option<String> = if exists {
        conn.query_row(
            "SELECT manifest FROM sae_index_manifest WHERE id=1",
            [],
            |r| r.get(0),
        )
        .optional()?
    } else {
        None
    };
    match found {
        Some(value) if value == embedding_manifest() => Ok(()),
        Some(_) => Err(StorageError::IndexMismatch(
            "model or chunk manifest differs".into(),
        )),
        None => Err(StorageError::IndexMismatch(
            "index has no encoder identity".into(),
        )),
    }
}
