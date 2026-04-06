pub mod embed;
pub mod search;
pub mod types;
pub use embed::{add_chunked_embeddings, get_unembedded_chunks, has_embeddings};
pub use search::{SearchResult, hybrid_search};
pub use types::*;

use std::path::Path;

use rusqlite::Connection;
use tracing::warn;

use rurico::embed::EMBEDDING_DIMS;

const SCHEMA_VERSION: &str = "5";

pub(crate) fn in_placeholders(len: usize) -> String {
    (1..=len)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn as_sql_params<T: rusqlite::types::ToSql>(
    values: &[T],
) -> Vec<&dyn rusqlite::types::ToSql> {
    values.iter().map(|v| v as &dyn rusqlite::types::ToSql).collect()
}

const DDL: &str = "
    CREATE TABLE IF NOT EXISTS posts (
        number INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        full_name TEXT NOT NULL,
        body_md TEXT NOT NULL DEFAULT '',
        category TEXT,
        tags TEXT NOT NULL DEFAULT '[]',
        wip INTEGER NOT NULL DEFAULT 0,
        kind TEXT NOT NULL DEFAULT 'stock',
        url TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        created_by TEXT NOT NULL,
        updated_by TEXT NOT NULL,
        revision_number INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_posts_updated ON posts(updated_at);

    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        post_number INTEGER NOT NULL,
        section_title TEXT,
        content TEXT NOT NULL,
        chunk_type TEXT NOT NULL DEFAULT 'section'
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_post ON chunks(post_number);

    CREATE TABLE IF NOT EXISTS sync_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        latest_updated_at TEXT,
        total_count INTEGER NOT NULL DEFAULT 0,
        local_count INTEGER NOT NULL DEFAULT 0,
        last_page INTEGER,
        updated_at TEXT NOT NULL DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS index_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS embedded_chunk_ids (
        chunk_id INTEGER NOT NULL,
        sub_idx INTEGER NOT NULL,
        vec_rowid INTEGER NOT NULL,
        PRIMARY KEY (chunk_id, sub_idx)
    );
";

pub struct Db {
    conn: Connection,
}

pub(crate) fn ensure_sqlite_vec() -> Result<(), StorageError> {
    rurico::storage::ensure_sqlite_vec().map_err(StorageError::Open)
}

/// Migrate FTS schema from v0–v3 (1-column) to v4 (2-column with section_title).
pub(crate) fn migrate_fts_v4(conn: &Connection) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;

    tx.execute_batch(
        "DROP TABLE IF EXISTS fts_chunks_vocab;\
         DROP TABLE IF EXISTS fts_chunks;\
         CREATE VIRTUAL TABLE fts_chunks USING fts5(\
             section_title, content, tokenize='trigram'\
         );\
         CREATE VIRTUAL TABLE fts_chunks_vocab \
             USING fts5vocab(fts_chunks, row);",
    )?;

    tx.execute_batch(
        "INSERT INTO fts_chunks(rowid, section_title, content) \
         SELECT id, section_title, content FROM chunks",
    )?;

    tx.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', '4')",
        [],
    )?;

    tx.commit()?;
    Ok(())
}

pub(crate) fn migrate_v5(conn: &Connection) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;

    tx.execute_batch(&format!(
        "DROP TABLE IF EXISTS vec_chunks;\
         CREATE VIRTUAL TABLE vec_chunks USING vec0(\
             embedding FLOAT[{EMBEDDING_DIMS}], \
             +chunk_id INTEGER, \
             +sub_idx INTEGER\
         );\
         CREATE TABLE IF NOT EXISTS embedded_chunk_ids (\
             chunk_id INTEGER NOT NULL, \
             sub_idx INTEGER NOT NULL, \
             vec_rowid INTEGER NOT NULL, \
             PRIMARY KEY (chunk_id, sub_idx)\
         );"
    ))?;

    tx.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', '5')",
        [],
    )?;

    tx.commit()?;
    eprintln!(
        "Migration complete: schema upgraded to v5 (embeddings cleared, please re-run `sae embed`)"
    );
    Ok(())
}

impl Db {
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        ensure_sqlite_vec()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Err(e) =
                    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))
                {
                    warn!(path = %parent.display(), error = %e, "failed to restrict data directory permissions");
                }
            }
        }
        let conn = open_with_wal_recovery(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    pub fn open_memory() -> Result<Self, StorageError> {
        ensure_sqlite_vec()?;
        let conn = Connection::open_in_memory().map_err(|e| StorageError::Open(e.to_string()))?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    fn init_schema(&self) -> Result<(), StorageError> {
        self.conn.execute_batch(DDL)?;

        self.conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(\
                 section_title, content, tokenize='trigram'\
             );\
             CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_vocab \
                 USING fts5vocab(fts_chunks, row);",
        )?;

        self.conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\
                 embedding FLOAT[{EMBEDDING_DIMS}], \
                 +chunk_id INTEGER, \
                 +sub_idx INTEGER\
             )"
        ))?;

        let stored: String = match self.conn.query_row(
            "SELECT value FROM index_meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        ) {
            Ok(v) => v,
            Err(rusqlite::Error::QueryReturnedNoRows) => "0".to_string(),
            Err(e) => return Err(e.into()),
        };

        if stored != SCHEMA_VERSION {
            let ver: u32 = stored
                .parse()
                .map_err(|_| StorageError::Open(format!("corrupt schema version: {stored:?}")))?;
            match ver {
                0..=3 => {
                    migrate_fts_v4(&self.conn)?;
                    migrate_v5(&self.conn)?;
                }
                4 => migrate_v5(&self.conn)?,
                _ => {
                    return Err(StorageError::Open(format!(
                        "database schema version {stored} is newer than supported version {SCHEMA_VERSION}"
                    )));
                }
            }
        }

        Ok(())
    }
}

fn open_with_wal_recovery(path: &Path) -> Result<Connection, StorageError> {
    match Connection::open(path) {
        Ok(c) => Ok(c),
        Err(ref e) if is_recoverable_open_error(e) => {
            warn!(error = %e, "DB open failed, removing WAL/SHM and retrying");
            let p = path.to_string_lossy();
            let _ = std::fs::remove_file(format!("{p}-wal"));
            let _ = std::fs::remove_file(format!("{p}-shm"));
            Ok(Connection::open(path)?)
        }
        Err(e) => Err(e.into()),
    }
}

fn is_recoverable_open_error(err: &rusqlite::Error) -> bool {
    use rusqlite::ffi::ErrorCode;
    matches!(
        err,
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error {
                code: ErrorCode::DatabaseCorrupt | ErrorCode::CannotOpen | ErrorCode::NotADatabase,
                ..
            },
            _,
        )
    )
}

pub fn upsert_post(conn: &Connection, post: &EsaPostRow) -> Result<(), StorageError> {
    conn.execute(
        "INSERT OR REPLACE INTO posts \
         (number, name, full_name, body_md, category, tags, wip, kind, \
          url, created_at, updated_at, created_by, updated_by, revision_number) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        rusqlite::params![
            post.number,
            post.name,
            post.full_name,
            post.body_md,
            post.category,
            post.tags_json(),
            post.wip,
            post.kind,
            post.url,
            post.created_at,
            post.updated_at,
            post.created_by,
            post.updated_by,
            post.revision_number,
        ],
    )?;
    Ok(())
}

pub fn get_sync_state(conn: &Connection) -> Result<Option<SyncState>, StorageError> {
    let result = conn.query_row(
        "SELECT latest_updated_at, total_count, local_count, last_page, updated_at \
         FROM sync_state WHERE id = 1",
        [],
        |row| {
            Ok(SyncState {
                latest_updated_at: row.get(0)?,
                total_count: row.get(1)?,
                local_count: row.get(2)?,
                last_page: row.get(3)?,
                updated_at: row.get(4)?,
            })
        },
    );
    match result {
        Ok(state) => Ok(Some(state)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn save_sync_state(
    conn: &Connection,
    update: &SyncStateUpdate<'_>,
) -> Result<(), StorageError> {
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs() as i64;
    save_sync_state_at(conn, update, epoch)
}

pub(crate) fn save_sync_state_at(
    conn: &Connection,
    update: &SyncStateUpdate<'_>,
    epoch_secs: i64,
) -> Result<(), StorageError> {
    conn.execute(
        "INSERT OR REPLACE INTO sync_state \
         (id, latest_updated_at, total_count, local_count, last_page, updated_at) \
         VALUES (1, ?1, ?2, ?3, ?4, datetime(?5, 'unixepoch'))",
        rusqlite::params![
            update.latest_updated_at,
            update.total_count,
            update.local_count,
            update.last_page,
            epoch_secs
        ],
    )?;
    Ok(())
}

pub fn count_posts(conn: &Connection) -> Result<u32, StorageError> {
    let count: u32 = conn.query_row("SELECT COUNT(*) FROM posts", [], |row| row.get(0))?;
    Ok(count)
}

pub fn count_chunks(conn: &Connection) -> Result<u32, StorageError> {
    let count: u32 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    Ok(count)
}

/// Make post metadata searchable via FTS by prepending it to the chunk body.
pub(crate) fn enrich_body(row: &EsaPostRow) -> String {
    let mut meta = format!("{}\n{}", row.name, row.created_by);
    if let Some(cat) = &row.category {
        meta.push(' ');
        meta.push_str(cat);
    }
    if !row.tags.is_empty() {
        meta.push(' ');
        meta.push_str(&row.tags.join(" "));
    }
    format!("{meta}\n\n{}", row.body_md)
}

pub fn rechunk_post(
    conn: &Connection,
    post_number: u32,
    body_md: &str,
) -> Result<u32, StorageError> {
    use crate::chunker;

    let chunk_ids: Vec<i64> = {
        let mut stmt =
            conn.prepare_cached("SELECT id FROM chunks WHERE post_number = ?1")?;
        let rows = stmt.query_map([post_number], |row| row.get(0))?;
        rows.collect::<Result<_, _>>()?
    };

    // SAVEPOINT works both standalone and within an outer transaction (sync).
    conn.execute_batch("SAVEPOINT rechunk")?;
    let result: Result<u32, StorageError> = (|| {
        if !chunk_ids.is_empty() {
            let ph = in_placeholders(chunk_ids.len());
            let params = as_sql_params(&chunk_ids);
            conn.execute(
                &format!(
                    "DELETE FROM vec_chunks WHERE rowid IN \
                     (SELECT vec_rowid FROM embedded_chunk_ids WHERE chunk_id IN ({ph}))"
                ),
                params.as_slice(),
            )?;
            conn.execute(
                &format!("DELETE FROM embedded_chunk_ids WHERE chunk_id IN ({ph})"),
                params.as_slice(),
            )?;
            conn.execute(
                &format!("DELETE FROM fts_chunks WHERE rowid IN ({ph})"),
                params.as_slice(),
            )?;
        }
        conn.execute("DELETE FROM chunks WHERE post_number = ?1", [post_number])?;

        let chunks = chunker::chunk_markdown(body_md);
        let mut insert_chunk = conn.prepare_cached(
            "INSERT INTO chunks (post_number, section_title, content, chunk_type) \
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        let mut insert_fts = conn.prepare_cached(
            "INSERT INTO fts_chunks(rowid, section_title, content) VALUES (?1, ?2, ?3)",
        )?;
        let mut count = 0u32;
        for chunk in &chunks {
            insert_chunk.execute(rusqlite::params![
                post_number,
                chunk.section_title,
                chunk.content,
                chunk.chunk_type.as_str(),
            ])?;
            let id = conn.last_insert_rowid();
            insert_fts.execute(rusqlite::params![id, chunk.section_title, chunk.content])?;
            count += 1;
        }
        Ok(count)
    })();
    match result {
        Ok(count) => {
            conn.execute_batch("RELEASE rechunk")?;
            Ok(count)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK TO rechunk");
            let _ = conn.execute_batch("RELEASE rechunk");
            Err(e)
        }
    }
}

#[cfg(test)]
pub(crate) fn test_post_row(number: u32) -> EsaPostRow {
    EsaPostRow {
        number,
        name: format!("Post {number}"),
        full_name: format!("dev/Post {number}"),
        body_md: format!("# Post {number}"),
        category: Some("dev".into()),
        tags: vec![],
        wip: false,
        kind: "stock".into(),
        url: format!("https://example.esa.io/posts/{number}"),
        created_at: "2025-01-01T00:00:00+09:00".into(),
        updated_at: "2025-01-02T00:00:00+09:00".into(),
        created_by: "alice".into(),
        updated_by: "bob".into(),
        revision_number: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_and_init_schema() {
        let db = Db::open_memory().unwrap();
        let version: String = db
            .conn()
            .query_row(
                "SELECT value FROM index_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn upsert_and_count() {
        let db = Db::open_memory().unwrap();
        upsert_post(db.conn(), &test_post_row(1)).unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 1);
    }

    #[test]
    fn upsert_multiple() {
        let db = Db::open_memory().unwrap();
        for i in 1..=5 {
            upsert_post(db.conn(), &test_post_row(i)).unwrap();
        }
        assert_eq!(count_posts(db.conn()).unwrap(), 5);
    }

    #[test]
    fn upsert_replaces_on_conflict() {
        let db = Db::open_memory().unwrap();
        let mut post = test_post_row(1);
        upsert_post(db.conn(), &post).unwrap();

        post.name = "Updated".into();
        post.revision_number = 4;
        upsert_post(db.conn(), &post).unwrap();

        assert_eq!(count_posts(db.conn()).unwrap(), 1);
        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(name, "Updated");
    }

    #[test]
    fn sync_state_none_initially() {
        let db = Db::open_memory().unwrap();
        assert!(get_sync_state(db.conn()).unwrap().is_none());
    }

    #[test]
    fn sync_state_roundtrip() {
        let db = Db::open_memory().unwrap();
        // Deterministic timestamp for test assertions
        save_sync_state_at(
            db.conn(),
            &SyncStateUpdate {
                latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
                total_count: 100,
                local_count: 50,
                last_page: Some(3),
            },
            1735689600, // 2025-01-01 00:00:00 UTC
        )
        .unwrap();

        let state = get_sync_state(db.conn()).unwrap().unwrap();
        assert_eq!(
            state.latest_updated_at.as_deref(),
            Some("2025-01-01T00:00:00+09:00")
        );
        assert_eq!(state.total_count, 100);
        assert_eq!(state.local_count, 50);
        assert_eq!(state.last_page, Some(3));
        assert_eq!(state.updated_at, "2025-01-01 00:00:00");
    }

    #[test]
    fn sync_state_clears_checkpoint() {
        let db = Db::open_memory().unwrap();
        save_sync_state(
            db.conn(),
            &SyncStateUpdate {
                latest_updated_at: None,
                total_count: 0,
                local_count: 0,
                last_page: Some(5),
            },
        )
        .unwrap();
        assert_eq!(
            get_sync_state(db.conn()).unwrap().unwrap().last_page,
            Some(5)
        );

        save_sync_state(
            db.conn(),
            &SyncStateUpdate {
                latest_updated_at: None,
                total_count: 10,
                local_count: 10,
                last_page: None,
            },
        )
        .unwrap();
        assert!(
            get_sync_state(db.conn())
                .unwrap()
                .unwrap()
                .last_page
                .is_none()
        );
    }

    #[test]
    fn open_file_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Db::open(&path).unwrap();
        upsert_post(db.conn(), &test_post_row(1)).unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 1);
    }

    #[test]
    fn transaction_upsert_batch() {
        let db = Db::open_memory().unwrap();
        let tx = db.conn().unchecked_transaction().unwrap();
        for i in 1..=10 {
            upsert_post(&tx, &test_post_row(i)).unwrap();
        }
        tx.commit().unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 10);
    }

    #[test]
    fn rechunk_creates_chunks_and_fts() {
        let db = Db::open_memory().unwrap();
        let mut post = test_post_row(1);
        post.body_md = "# Intro\nHello\n# Details\nWorld".into();
        upsert_post(db.conn(), &post).unwrap();

        let n = rechunk_post(db.conn(), 1, &post.body_md).unwrap();
        assert_eq!(n, 2);
        assert_eq!(count_chunks(db.conn()).unwrap(), 2);

        let hits: u32 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH 'Hello'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(hits, 1);
    }

    #[test]
    fn rechunk_replaces_old_chunks() {
        let db = Db::open_memory().unwrap();
        let post = test_post_row(1);
        upsert_post(db.conn(), &post).unwrap();
        rechunk_post(db.conn(), 1, "# Old\nContent").unwrap();
        assert_eq!(count_chunks(db.conn()).unwrap(), 1);

        rechunk_post(db.conn(), 1, "# New A\nFoo\n# New B\nBar").unwrap();
        assert_eq!(count_chunks(db.conn()).unwrap(), 2);

        let hits: u32 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH 'Content'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(hits, 0);
    }

    #[test]
    fn rechunk_cleans_orphaned_vec_chunks() {
        let db = Db::open_memory().unwrap();
        let post = test_post_row(1);
        upsert_post(db.conn(), &post).unwrap();
        rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        let chunks = embed::get_unembedded_chunks(db.conn(), 100).unwrap();
        let emb: Vec<(i64, rurico::embed::ChunkedEmbedding)> = chunks
            .iter()
            .map(|(id, _)| {
                (
                    *id,
                    rurico::embed::ChunkedEmbedding {
                        chunks: vec![vec![0.1; rurico::embed::EMBEDDING_DIMS as usize]],
                    },
                )
            })
            .collect();
        embed::add_chunked_embeddings(db.conn(), &emb).unwrap();
        assert!(embed::has_embeddings(db.conn()));

        rechunk_post(db.conn(), 1, "# New\nContent").unwrap();

        let orphans: u32 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM embedded_chunk_ids", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(orphans, 0, "rechunk should clean orphaned embeddings");
    }

    #[test]
    fn fts_trigram_japanese() {
        let db = Db::open_memory().unwrap();
        let post = test_post_row(1);
        upsert_post(db.conn(), &post).unwrap();
        rechunk_post(db.conn(), 1, "# 認証ガイド\n認証フローの説明").unwrap();

        let hits: u32 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH '認証フ'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(hits, 1);
    }

    // T-005: rechunk 後 fts_chunks 件数 = chunks 件数 (FR-002)
    #[test]
    fn rechunk_fts_count_matches_chunks_count() {
        let db = Db::open_memory().unwrap();
        let body = "# セクション1\n本文A\n# セクション2\n本文B\n# セクション3\n本文C";
        let post = test_post_row(1);
        upsert_post(db.conn(), &post).unwrap();
        let n = rechunk_post(db.conn(), 1, body).unwrap();
        assert_eq!(n, 3);

        let chunks_count: u32 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE post_number = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let fts_count: u32 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM fts_chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            fts_count, chunks_count,
            "[T-005] fts_chunks count ({fts_count}) must equal chunks count ({chunks_count})"
        );
    }

    #[test]
    fn enrich_body_includes_all_metadata() {
        let mut row = test_post_row(1);
        row.name = "Daily 振り返り".into();
        row.body_md = "# やったこと\n実装した".into();
        row.category = Some("チーム/日報/thkt".into());
        row.tags = vec!["日報".into(), "振り返り".into()];
        row.created_by = "thkt".into();
        let enriched = enrich_body(&row);
        assert_eq!(
            enriched,
            "Daily 振り返り\nthkt チーム/日報/thkt 日報 振り返り\n\n# やったこと\n実装した"
        );
    }

    #[test]
    fn enrich_body_no_category_no_tags() {
        let mut row = test_post_row(1);
        row.name = "Untitled".into();
        row.body_md = "text".into();
        row.category = None;
        let enriched = enrich_body(&row);
        assert_eq!(enriched, "Untitled\nalice\n\ntext");
    }

    #[test]
    fn enrich_body_category_only() {
        let mut row = test_post_row(1);
        row.name = "Post".into();
        row.body_md = "body".into();
        let enriched = enrich_body(&row);
        assert_eq!(enriched, "Post\nalice dev\n\nbody");
    }

    #[test]
    fn enrich_body_tags_only() {
        let mut row = test_post_row(1);
        row.name = "Post".into();
        row.body_md = "body".into();
        row.category = None;
        row.tags = vec!["rust".into(), "cli".into()];
        let enriched = enrich_body(&row);
        assert_eq!(enriched, "Post\nalice rust cli\n\nbody");
    }

    // T-006: fresh DB で fts_chunks に section_title + content カラムあり (FR-001)
    #[test]
    fn fresh_db_fts_has_section_title_and_content_columns() {
        let db = Db::open_memory().unwrap();
        let result = db.conn().execute(
            "INSERT INTO fts_chunks(rowid, section_title, content) VALUES (999, '見出し', '本文')",
            [],
        );
        assert!(
            result.is_ok(),
            "[T-006] fts_chunks must accept section_title column, got: {:?}",
            result.err()
        );
        let title: String = db
            .conn()
            .query_row(
                "SELECT section_title FROM fts_chunks WHERE rowid = 999",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(title, "見出し");
    }

    // T-007: migrate_fts_v4 失敗時 rollback 確認 (FR-005)
    #[test]
    fn migrate_fts_v4_rollback_on_failure() {
        ensure_sqlite_vec().unwrap();
        let conn = Connection::open_in_memory().unwrap();

        // 旧スキーマ構築: chunks テーブル + 旧 1-column FTS + data
        conn.execute_batch(
            "CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE chunks (
                 id INTEGER PRIMARY KEY, post_number INTEGER NOT NULL,
                 section_title TEXT, content TEXT NOT NULL,
                 chunk_type TEXT NOT NULL DEFAULT 'section'
             );
             CREATE VIRTUAL TABLE fts_chunks USING fts5(content, tokenize='trigram');
             CREATE VIRTUAL TABLE fts_chunks_vocab USING fts5vocab(fts_chunks, row);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES ('schema_version', '3')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO chunks (id, post_number, section_title, content, chunk_type) \
             VALUES (1, 1, '旧セクション', '旧データの本文テキスト', 'section')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO fts_chunks(rowid, content) VALUES (1, '旧データの本文テキスト')",
            [],
        )
        .unwrap();

        // 前提確認: 旧データが検索可能
        let pre: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH '旧データ'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(pre, 1, "precondition: old FTS data searchable");

        // chunks テーブルを DROP して rebuild を失敗させる
        conn.execute_batch("DROP TABLE chunks").unwrap();

        let result = migrate_fts_v4(&conn);
        assert!(
            result.is_err(),
            "[T-007] migrate_fts_v4 should fail when chunks table is missing"
        );

        // rollback 確認: version は "3" のまま
        let version: String = conn
            .query_row(
                "SELECT value FROM index_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "3", "[T-007] schema_version should remain '3'");

        // rollback 確認: 旧 FTS data が生存（実検索 MATCH）
        let post: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH '旧データ'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(post, 1, "[T-007] old FTS data must survive after rollback");
    }

    // T-004: temp file DB + 旧スキーマ → Db::open で migration (FR-005)
    #[test]
    fn migration_from_old_schema_rebuilds_fts_with_section_title() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("migration_test.db");

        // 旧スキーマ DB を手動構築
        {
            ensure_sqlite_vec().unwrap();
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
                .unwrap();
            conn.execute_batch(
                "CREATE TABLE posts (
                     number INTEGER PRIMARY KEY, name TEXT NOT NULL,
                     full_name TEXT NOT NULL, body_md TEXT NOT NULL DEFAULT '',
                     category TEXT, tags TEXT NOT NULL DEFAULT '[]',
                     wip INTEGER NOT NULL DEFAULT 0, kind TEXT NOT NULL DEFAULT 'stock',
                     url TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                     created_by TEXT NOT NULL, updated_by TEXT NOT NULL,
                     revision_number INTEGER NOT NULL DEFAULT 0
                 );
                 CREATE INDEX idx_posts_updated ON posts(updated_at);
                 CREATE TABLE chunks (
                     id INTEGER PRIMARY KEY, post_number INTEGER NOT NULL,
                     section_title TEXT, content TEXT NOT NULL,
                     chunk_type TEXT NOT NULL DEFAULT 'section'
                 );
                 CREATE INDEX idx_chunks_post ON chunks(post_number);
                 CREATE TABLE sync_state (
                     id INTEGER PRIMARY KEY CHECK (id = 1),
                     latest_updated_at TEXT, total_count INTEGER NOT NULL DEFAULT 0,
                     local_count INTEGER NOT NULL DEFAULT 0, last_page INTEGER,
                     updated_at TEXT NOT NULL DEFAULT ''
                 );
                 CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
            )
            .unwrap();
            // 旧 1-column FTS
            conn.execute_batch(
                "CREATE VIRTUAL TABLE fts_chunks USING fts5(content, tokenize='trigram');
                 CREATE VIRTUAL TABLE fts_chunks_vocab USING fts5vocab(fts_chunks, row);",
            )
            .unwrap();
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE vec_chunks USING vec0(\
                     chunk_id INTEGER PRIMARY KEY, \
                     embedding FLOAT[{}])",
                rurico::embed::EMBEDDING_DIMS,
            ))
            .unwrap();
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES ('schema_version', '3')",
                [],
            )
            .unwrap();
            // テストデータ
            conn.execute(
                "INSERT INTO posts (number, name, full_name, body_md, url, created_at, \
                 updated_at, created_by, updated_by) \
                 VALUES (1, '認証ガイド', 'dev/認証ガイド', '# 認証ガイド\n認証フローの説明', \
                 'https://example.esa.io/posts/1', '2025-01-01T00:00:00+09:00', \
                 '2025-01-01T00:00:00+09:00', 'alice', 'alice')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO chunks (id, post_number, section_title, content, chunk_type) \
                 VALUES (1, 1, '認証ガイド', '認証フローの説明', 'section')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO fts_chunks(rowid, content) VALUES (1, '認証フローの説明')",
                [],
            )
            .unwrap();
        }

        // Db::open で migration 発火
        let db = Db::open(&db_path).unwrap();

        // schema_version = "4"
        let version: String = db
            .conn()
            .query_row(
                "SELECT value FROM index_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            version, "5",
            "[T-004] schema_version should be '5' after full migration"
        );

        // section_title の語で検索可能
        let hits: u32 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH '認証ガイド'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            hits > 0,
            "[T-004] section_title term should match after migration"
        );

        // fts_chunks_vocab が機能
        let vocab: u32 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM fts_chunks_vocab", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(vocab > 0, "[T-004] fts_chunks_vocab should be populated");
    }

    // TC-005: WAL recovery — is_recoverable_open_error recognizes DatabaseCorrupt
    #[test]
    fn is_recoverable_for_database_corrupt() {
        use rusqlite::ffi::ErrorCode;
        let err = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error { code: ErrorCode::DatabaseCorrupt, extended_code: 11 },
            None,
        );
        assert!(is_recoverable_open_error(&err));
    }

    // TC-005: WAL recovery — is_recoverable_open_error recognizes CannotOpen
    #[test]
    fn is_recoverable_for_cannot_open() {
        use rusqlite::ffi::ErrorCode;
        let err = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error { code: ErrorCode::CannotOpen, extended_code: 14 },
            None,
        );
        assert!(is_recoverable_open_error(&err));
    }

    // TC-005: WAL recovery — non-SqliteFailure error is not recoverable
    #[test]
    fn is_not_recoverable_for_non_sqlite_failure() {
        let err = rusqlite::Error::QueryReturnedNoRows;
        assert!(!is_recoverable_open_error(&err));
    }

    // TC-005: WAL recovery — open_with_wal_recovery succeeds on a valid DB file
    #[test]
    fn open_with_wal_recovery_opens_valid_db() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("wal_test.db");
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
        }
        let conn = open_with_wal_recovery(&db_path).unwrap();
        assert!(conn.is_autocommit());
    }

    // TC-002: init_schema rejects a non-numeric (corrupt) schema version string
    #[test]
    fn init_schema_rejects_corrupt_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(DDL).unwrap();
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES ('schema_version', 'not-a-number')",
                [],
            )
            .unwrap();
        }
        match Db::open(&path) {
            Ok(_) => panic!("[TC-002] expected Db::open to fail"),
            Err(StorageError::Open(msg)) => assert!(
                msg.contains("corrupt schema version"),
                "[TC-002] unexpected error message: {msg}"
            ),
            Err(e) => panic!("[TC-002] expected StorageError::Open, got {e:?}"),
        }
    }

    // TC-008: init_schema rejects a schema version number higher than SCHEMA_VERSION
    #[test]
    fn init_schema_rejects_future_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(DDL).unwrap();
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES ('schema_version', '99')",
                [],
            )
            .unwrap();
        }
        match Db::open(&path) {
            Ok(_) => panic!("[TC-008] expected Db::open to fail"),
            Err(StorageError::Open(msg)) => {
                assert!(
                    msg.contains("newer than supported"),
                    "[TC-008] unexpected error message: {msg}"
                );
                assert!(
                    msg.contains("99"),
                    "[TC-008] message should include the invalid version: {msg}"
                );
            }
            Err(e) => panic!("[TC-008] expected StorageError::Open, got {e:?}"),
        }
    }

}
