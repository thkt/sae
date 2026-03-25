pub mod embed;
pub mod search;
pub mod types;
pub use embed::{add_embeddings, get_unembedded_chunks, has_embeddings};
pub use search::{hybrid_search, SearchResult};
pub use types::*;

use std::path::Path;

use rusqlite::Connection;
use tracing::warn;

use rurico::embed::EMBEDDING_DIMS;

const SCHEMA_VERSION: &str = "3";

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
";

pub struct Db {
    conn: Connection,
}

fn ensure_sqlite_vec() -> Result<(), StorageError> {
    rurico::storage::ensure_sqlite_vec().map_err(StorageError::Open)
}

impl Db {
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        ensure_sqlite_vec()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ =
                    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
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
        let conn =
            Connection::open_in_memory().map_err(|e| StorageError::Open(e.to_string()))?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    fn init_schema(&self) -> Result<(), StorageError> {
        self.conn.execute_batch(DDL)?;

        // FTS5 trigram for Japanese full-text search
        self.conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(\
                 content, tokenize='trigram'\
             );\
             CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_vocab \
                 USING fts5vocab(fts_chunks, row);",
        )?;

        self.conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\
                 chunk_id INTEGER PRIMARY KEY, \
                 embedding FLOAT[{EMBEDDING_DIMS}]\
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
            self.conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', ?1)",
                [SCHEMA_VERSION],
            )?;
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
                code: ErrorCode::DatabaseCorrupt
                    | ErrorCode::CannotOpen
                    | ErrorCode::NotADatabase,
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
            post.tags,
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
    latest_updated_at: Option<&str>,
    total_count: u32,
    local_count: u32,
    last_page: Option<u32>,
) -> Result<(), StorageError> {
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs() as i64;
    save_sync_state_at(conn, latest_updated_at, total_count, local_count, last_page, epoch)
}

pub(crate) fn save_sync_state_at(
    conn: &Connection,
    latest_updated_at: Option<&str>,
    total_count: u32,
    local_count: u32,
    last_page: Option<u32>,
    epoch_secs: i64,
) -> Result<(), StorageError> {
    conn.execute(
        "INSERT OR REPLACE INTO sync_state \
         (id, latest_updated_at, total_count, local_count, last_page, updated_at) \
         VALUES (1, ?1, ?2, ?3, ?4, datetime(?5, 'unixepoch'))",
        rusqlite::params![latest_updated_at, total_count, local_count, last_page, epoch_secs],
    )?;
    Ok(())
}

pub fn count_posts(conn: &Connection) -> Result<u32, StorageError> {
    let count: u32 =
        conn.query_row("SELECT COUNT(*) FROM posts", [], |row| row.get(0))?;
    Ok(count)
}

pub fn count_chunks(conn: &Connection) -> Result<u32, StorageError> {
    let count: u32 =
        conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    Ok(count)
}

pub fn rechunk_post(
    conn: &Connection,
    post_number: u32,
    body_md: &str,
) -> Result<u32, StorageError> {
    use crate::chunker;

    conn.execute(
        "DELETE FROM fts_chunks WHERE rowid IN \
         (SELECT id FROM chunks WHERE post_number = ?1)",
        [post_number],
    )?;
    conn.execute("DELETE FROM chunks WHERE post_number = ?1", [post_number])?;

    let chunks = chunker::chunk_markdown(body_md);
    let mut count = 0u32;
    for chunk in &chunks {
        conn.execute(
            "INSERT INTO chunks (post_number, section_title, content, chunk_type) \
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                post_number,
                chunk.section_title,
                chunk.content,
                chunk.chunk_type.as_str(),
            ],
        )?;
        let id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO fts_chunks(rowid, content) VALUES (?1, ?2)",
            rusqlite::params![id, chunk.content],
        )?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_post(number: u32) -> EsaPostRow {
        EsaPostRow {
            number,
            name: format!("Post {number}"),
            full_name: format!("dev/Post {number}"),
            body_md: format!("# Post {number}"),
            category: Some("dev".into()),
            tags: r#"["test"]"#.into(),
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
        upsert_post(db.conn(), &test_post(1)).unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 1);
    }

    #[test]
    fn upsert_multiple() {
        let db = Db::open_memory().unwrap();
        for i in 1..=5 {
            upsert_post(db.conn(), &test_post(i)).unwrap();
        }
        assert_eq!(count_posts(db.conn()).unwrap(), 5);
    }

    #[test]
    fn upsert_replaces_on_conflict() {
        let db = Db::open_memory().unwrap();
        let mut post = test_post(1);
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
            Some("2025-01-01T00:00:00+09:00"),
            100,
            50,
            Some(3),
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
        save_sync_state(db.conn(), None, 0, 0, Some(5)).unwrap();
        assert_eq!(
            get_sync_state(db.conn()).unwrap().unwrap().last_page,
            Some(5)
        );

        save_sync_state(db.conn(), None, 10, 10, None).unwrap();
        assert!(get_sync_state(db.conn())
            .unwrap()
            .unwrap()
            .last_page
            .is_none());
    }

    #[test]
    fn open_file_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Db::open(&path).unwrap();
        upsert_post(db.conn(), &test_post(1)).unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 1);
    }

    #[test]
    fn transaction_upsert_batch() {
        let db = Db::open_memory().unwrap();
        let tx = db.conn().unchecked_transaction().unwrap();
        for i in 1..=10 {
            upsert_post(&tx, &test_post(i)).unwrap();
        }
        tx.commit().unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 10);
    }

    #[test]
    fn rechunk_creates_chunks_and_fts() {
        let db = Db::open_memory().unwrap();
        let mut post = test_post(1);
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
        let post = test_post(1);
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
    fn fts_trigram_japanese() {
        let db = Db::open_memory().unwrap();
        let post = test_post(1);
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
}
