//! Storage layer for sae.
//!
//! Assumes single-process operation. WAL journaling + `busy_timeout=5000ms`
//! handle ad-hoc contention from CLI re-invocations, but daemonized or
//! concurrent CLI runs against the same DB file are unsupported (no advisory
//! lock); moving to multi-process requires schema and lock redesign.

pub(crate) mod embed;
mod search;
mod types;
pub use embed::{
    add_chunked_embeddings, count_unembedded_chunks, get_unembedded_chunks, has_embeddings,
};
pub use search::{MatchSource, SearchFilter, SearchOutput, SearchResult, hybrid_search};
pub use types::*;

use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::Connection;
use tracing::warn;

use rurico::embed::EMBEDDING_DIMS;
use rurico::storage as rurico_storage;
use rurico::storage::{QueryNormalizationConfig, normalize_for_fts};

pub(crate) use amici::storage::{anon_placeholders, as_sql_params, in_placeholders};

use amici::storage::collect_rows;

/// `collect_rows` wrapper that fixes `E = StorageError`, removing the
/// `::<_, _, _, StorageError>` turbofish that the bare helper would otherwise
/// require at every `?`-binding callsite (both `rusqlite::Error` and
/// `StorageError` satisfy `From<rusqlite::Error>`, leaving `E` ambiguous).
pub(crate) fn collect_storage_rows<I, T, C>(rows: I) -> Result<C, StorageError>
where
    I: Iterator<Item = Result<T, rusqlite::Error>>,
    C: FromIterator<T>,
{
    collect_rows(rows)
}

/// Shared `normalize_for_fts` configuration for index- and query-side calls.
/// Divergence between sides makes FTS5 token streams disagree and silently
/// misses matches, so callers must funnel through this single helper.
pub(crate) fn query_norm_config() -> QueryNormalizationConfig {
    QueryNormalizationConfig::default()
}

/// Zero-copy `&[f32] → &[u8]` for binding embedding vectors to sqlite-vec.
/// Replaces `rurico::storage::f32_as_bytes` after its removal upstream;
/// rurico's `compile_error!` for non-little-endian targets is transitive.
pub(crate) fn f32_as_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

const DDL_FTS: &str = "\
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(\
        section_title, content, tokenize='trigram'\
    );\
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_vocab \
        USING fts5vocab(fts_chunks, row);";

const DDL_EMBEDDED_CHUNK_IDS: &str = "\
    CREATE TABLE IF NOT EXISTS embedded_chunk_ids (\
        chunk_id INTEGER NOT NULL, \
        sub_idx INTEGER NOT NULL, \
        vec_rowid INTEGER NOT NULL, \
        PRIMARY KEY (chunk_id, sub_idx)\
    )";

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
        -- Singleton: one team per database, so sync_state has exactly one row.
        -- The CHECK guards against schema changes that would allow multiple rows.
        id INTEGER PRIMARY KEY CHECK (id = 1),
        latest_updated_at TEXT,
        total_count INTEGER NOT NULL DEFAULT 0,
        local_count INTEGER NOT NULL DEFAULT 0,
        last_page INTEGER,
        updated_at TEXT NOT NULL DEFAULT ''
    );
";

pub struct Db {
    conn: Connection,
}

pub(crate) fn ensure_sqlite_vec() -> Result<(), StorageError> {
    rurico_storage::ensure_sqlite_vec().map_err(StorageError::Open)
}

fn ddl_vec_chunks() -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\
             embedding FLOAT[{EMBEDDING_DIMS}], \
             +chunk_id INTEGER, \
             +sub_idx INTEGER\
         )"
    )
}

impl Db {
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        ensure_sqlite_vec()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                // Best-effort: continue on chmod failure (restricted FS, exotic
                // mount options) since DB functionality must not be blocked;
                // the warn surfaces the failure so an operator can notice.
                if let Err(e) = fs::set_permissions(parent, fs::Permissions::from_mode(0o700)) {
                    warn!(path = %parent.display(), error = %e, "failed to restrict data directory permissions");
                }
            }
        }
        let conn = open_with_wal_recovery(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        let db = Self { conn };
        db.init_schema().map_err(|e| match e {
            StorageError::Open(msg) => {
                StorageError::Open(format!("{msg} (database: {})", path.display()))
            }
            StorageError::Db(err) => StorageError::Open(format!(
                "Database error: {err} (database: {})",
                path.display()
            )),
            other => other,
        })?;
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
        self.conn.execute_batch(DDL_EMBEDDED_CHUNK_IDS)?;
        self.conn.execute_batch(DDL_FTS)?;
        self.conn.execute_batch(&ddl_vec_chunks())?;
        Ok(())
    }
}

fn open_with_wal_recovery(path: &Path) -> Result<Connection, StorageError> {
    match Connection::open(path) {
        Ok(c) => Ok(c),
        Err(ref e) if is_recoverable_open_error(e) => {
            warn!(error = %e, "DB open failed, removing WAL/SHM and retrying — uncommitted data may be lost, re-run `sae sync` to rebuild");
            let p = path.to_string_lossy();
            let _ = fs::remove_file(format!("{p}-wal"));
            let _ = fs::remove_file(format!("{p}-shm"));
            Ok(Connection::open(path)?)
        }
        Err(e) => Err(e.into()),
    }
}

fn is_recoverable_open_error(err: &rusqlite::Error) -> bool {
    use rusqlite::ffi::{self, ErrorCode};
    matches!(
        err,
        rusqlite::Error::SqliteFailure(
            ffi::Error {
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
    let epoch = i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_secs(),
    )
    .unwrap_or(i64::MAX);
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
        let mut stmt = conn.prepare_cached("SELECT id FROM chunks WHERE post_number = ?1")?;
        let rows = stmt.query_map([post_number], |row| row.get(0))?;
        collect_storage_rows(rows)?
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
        let config = query_norm_config();
        let mut count = 0u32;
        for chunk in &chunks {
            insert_chunk.execute(rusqlite::params![
                post_number,
                chunk.section_title,
                chunk.content,
                chunk.chunk_type.as_str(),
            ])?;
            let id = conn.last_insert_rowid();
            let fts_section_title = chunk
                .section_title
                .as_deref()
                .map(|t| normalize_for_fts(t, &config));
            let fts_content = normalize_for_fts(&chunk.content, &config);
            insert_fts.execute(rusqlite::params![id, fts_section_title, fts_content])?;
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

#[doc(hidden)]
pub fn test_post_row(number: u32) -> EsaPostRow {
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
    use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};
    use rusqlite::ffi::{self, ErrorCode};

    // T-202: Db::open_memory creates the core tables via init_schema
    #[test]
    fn open_creates_core_tables() {
        let db = Db::open_memory().unwrap();
        let tbl_count: u32 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type='table' AND name IN ('posts', 'chunks', 'sync_state')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tbl_count, 3, "core tables must exist after init_schema");
    }

    // T-203: upsert_post inserts a post and count_posts reflects the change
    #[test]
    fn upsert_and_count() {
        let db = Db::open_memory().unwrap();
        let post = test_post_row(1);
        upsert_post(db.conn(), &post).unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 1);
        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(name, post.name);
    }

    // T-204: upsert_post overwrites an existing post with the same number
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

    // T-205: get_sync_state returns None on a fresh database
    #[test]
    fn sync_state_none_initially() {
        let db = Db::open_memory().unwrap();
        assert!(get_sync_state(db.conn()).unwrap().is_none());
    }

    // T-206: save_sync_state and get_sync_state round-trip all fields correctly
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

    // T-207: save_sync_state with last_page=None clears a previous checkpoint
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

    // T-208: Db::open creates a database file on disk at the specified path
    #[test]
    fn open_file_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Db::open(&path).unwrap();
        assert!(
            path.exists(),
            "Db::open should create the database file on disk"
        );
        upsert_post(db.conn(), &test_post_row(1)).unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 1);
    }

    // T-209: upsert_post inside a transaction batches inserts correctly
    #[test]
    fn transaction_upsert_batch() {
        let db = Db::open_memory().unwrap();
        let tx = db.conn().unchecked_transaction().unwrap();
        for i in 1..=10 {
            upsert_post(&tx, &test_post_row(i)).unwrap();
        }
        tx.commit().unwrap();
        assert_eq!(count_posts(db.conn()).unwrap(), 10);
        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 10", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(name, test_post_row(10).name);
    }

    // T-210: rechunk_post creates chunk rows and FTS entries for each section
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

    // T-211: rechunk_post replaces old chunks and removes stale FTS entries
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

    // T-212: rechunk_post removes orphaned vector embeddings from previous chunks
    #[test]
    fn rechunk_cleans_orphaned_vec_chunks() {
        let db = Db::open_memory().unwrap();
        let post = test_post_row(1);
        upsert_post(db.conn(), &post).unwrap();
        rechunk_post(db.conn(), 1, "# Hello\nWorld").unwrap();

        let chunks = get_unembedded_chunks(db.conn(), 100).unwrap();
        let emb: Vec<(i64, ChunkedEmbedding)> = chunks
            .iter()
            .map(|(id, _)| (*id, ChunkedEmbedding::new(vec![vec![0.1; EMBEDDING_DIMS]])))
            .collect();
        add_chunked_embeddings(db.conn(), &emb).unwrap();
        assert!(has_embeddings(db.conn()));

        rechunk_post(db.conn(), 1, "# New\nContent").unwrap();

        let orphans: u32 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM embedded_chunk_ids", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(orphans, 0, "rechunk should clean orphaned embeddings");
    }

    // T-213: FTS trigram index matches a 3-character Japanese substring
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

    // T-094: rechunk 後 fts_chunks 件数 = chunks 件数 (FR-002)
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
            "fts_chunks count ({fts_count}) must equal chunks count ({chunks_count})"
        );
    }

    // T-214: enrich_body prepends name, author, category, and tags to body
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

    // T-215: enrich_body omits category and tags line when both are absent
    #[test]
    fn enrich_body_no_category_no_tags() {
        let mut row = test_post_row(1);
        row.name = "Untitled".into();
        row.body_md = "text".into();
        row.category = None;
        let enriched = enrich_body(&row);
        assert_eq!(enriched, "Untitled\nalice\n\ntext");
    }

    // T-216: enrich_body includes category but no tags when tags are empty
    #[test]
    fn enrich_body_category_only() {
        let mut row = test_post_row(1);
        row.name = "Post".into();
        row.body_md = "body".into();
        let enriched = enrich_body(&row);
        assert_eq!(enriched, "Post\nalice dev\n\nbody");
    }

    // T-217: enrich_body includes tags but no category when category is None
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
        db.conn()
            .execute(
                "INSERT INTO fts_chunks(rowid, section_title, content) VALUES (999, '見出し', '本文')",
                [],
            )
            .unwrap();
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

    // T-097: WAL recovery — is_recoverable_open_error recognizes DatabaseCorrupt
    #[test]
    fn is_recoverable_for_database_corrupt() {
        let err = rusqlite::Error::SqliteFailure(
            ffi::Error {
                code: ErrorCode::DatabaseCorrupt,
                extended_code: 11,
            },
            None,
        );
        assert!(is_recoverable_open_error(&err));
    }

    // T-097: WAL recovery — is_recoverable_open_error recognizes CannotOpen
    #[test]
    fn is_recoverable_for_cannot_open() {
        let err = rusqlite::Error::SqliteFailure(
            ffi::Error {
                code: ErrorCode::CannotOpen,
                extended_code: 14,
            },
            None,
        );
        assert!(is_recoverable_open_error(&err));
    }

    // T-097: WAL recovery — non-SqliteFailure error is not recoverable
    #[test]
    fn is_not_recoverable_for_non_sqlite_failure() {
        let err = rusqlite::Error::QueryReturnedNoRows;
        assert!(!is_recoverable_open_error(&err));
    }
}
