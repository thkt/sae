//! Local-only index migration. Inference runs against a consistent SQLite
//! snapshot; publication is one fsynced pointer rename after validation.

use crate::{
    commands::embed_batch,
    config::data_dir,
    config::{Config, ConfigError, validate_team_name},
    embedding::EMBEDDING_DIMS,
    embedding::Embed,
    envelope::CommandOutput,
    model::digest_hex,
    storage::{self, Db, StorageError},
    tools::{self, SaeError},
};
use fs2::FileExt;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    version: u32,
    active: String,
    previous: String,
}

fn checked_name(team: &str, name: &str) -> bool {
    name.starts_with(&format!("{team}."))
        && name.ends_with(".db")
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b".-_".contains(&c))
        && !name.contains("..")
}

fn selection_path(dir: &Path, team: &str) -> PathBuf {
    dir.join(format!("{team}.active.json"))
}

fn read_selection(dir: &Path, team: &str) -> Result<Option<Selection>, ConfigError> {
    let path = selection_path(dir, team);
    let data = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(ConfigError::ReadFailed(path, e)),
    };
    let selected: Selection =
        serde_json::from_str(&data).map_err(|e| ConfigError::ParseFailed(path, e))?;
    if selected.version != 1
        || !checked_name(team, &selected.active)
        || !checked_name(team, &selected.previous)
        || selected.active == selected.previous
    {
        return Err(ConfigError::InvalidValue(
            "invalid active index selection".into(),
        ));
    }
    Ok(Some(selected))
}

fn reject_symlink(path: &Path) -> Result<(), ConfigError> {
    match fs::symlink_metadata(path) {
        Ok(meta) if !meta.file_type().is_file() => Err(ConfigError::InvalidValue(
            "index must be a regular file".into(),
        )),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(ConfigError::ReadFailed(path.to_owned(), e)),
    }
}

pub(crate) fn resolve_at(dir: &Path, team: &str) -> Result<PathBuf, ConfigError> {
    validate_team_name(team)?;
    let path = match read_selection(dir, team)? {
        Some(s) => dir.join(s.active),
        None => dir.join(format!("{team}.db")),
    };
    reject_symlink(&path)?;
    Ok(path)
}

fn lock_team(dir: &Path, team: &str) -> Result<File, SaeError> {
    fs::create_dir_all(dir)?;
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(dir.join(format!("{team}.migration.lock")))?;
    file.try_lock_exclusive().map_err(|e| {
        SaeError::InputData(format!("Another migration or rollback is active: {e}"))
    })?;
    Ok(file)
}

/// Content identity includes posts, original chunks, and sync checkpoint.
/// It deliberately excludes old vectors and SQLite's mutable file layout.
fn source_digest(conn: &Connection) -> Result<String, SaeError> {
    let mut hash = Sha256::new();
    for sql in [
        "SELECT json_array(number,name,full_name,body_md,category,tags,wip,kind,url,created_at,updated_at,created_by,updated_by,revision_number) FROM posts ORDER BY number",
        "SELECT json_array(id,post_number,section_title,content,chunk_type) FROM chunks ORDER BY id",
        "SELECT json_array(id,latest_updated_at,total_count,local_count,last_page,updated_at) FROM sync_state ORDER BY id",
    ] {
        hash.update(sql.as_bytes());
        let mut stmt = conn.prepare(sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(StorageError::from)?;
        for row in rows {
            let row = row.map_err(StorageError::from)?;
            hash.update(row.len().to_le_bytes());
            hash.update(row.as_bytes());
        }
    }
    Ok(digest_hex(&hash.finalize()))
}

fn output(action: &str, path: &Path, chunks: u32) -> CommandOutput {
    CommandOutput {
        data: serde_json::json!({"action":action,"index":path,"chunks":chunks,"model":if action == "rolled back" { None } else { Some("EmbeddingGemma 2") }}),
        markdown: format!("{action}: {} ({chunks} chunks)", path.display()),
        degraded: false,
        notes: vec![],
    }
}

fn publish(dir: &Path, team: &str, selected: &Selection) -> Result<(), SaeError> {
    let mut file = tempfile::NamedTempFile::new_in(dir)?;
    serde_json::to_writer(&mut file, selected)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist(selection_path(dir, team))
        .map_err(|e| SaeError::Io(e.error))?;
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn validate_checkpoint(conn: &Connection) -> Result<(), SaeError> {
    storage::validate_embedding_index(conn)?;
    let mut stmt = conn.prepare("SELECT c.id,c.content,h.content_sha256 FROM chunks c JOIN embedding_content_hashes h ON h.chunk_id=c.id").map_err(StorageError::from)?;
    let mut rows = stmt.query([]).map_err(StorageError::from)?;
    while let Some(r) = rows.next().map_err(StorageError::from)? {
        let content: String = r.get(1).map_err(StorageError::from)?;
        let saved: String = r.get(2).map_err(StorageError::from)?;
        if storage::embed::content_hash(&content) != saved {
            return Err(SaeError::InputData(
                "Checkpoint content changed; migration cannot resume".into(),
            ));
        }
    }
    let missing: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM embedded_chunk_ids e LEFT JOIN embedding_content_hashes h ON h.chunk_id=e.chunk_id LEFT JOIN chunks c ON c.id=e.chunk_id WHERE h.chunk_id IS NULL OR c.id IS NULL)", [], |r| r.get(0)).map_err(StorageError::from)?;
    if missing {
        return Err(SaeError::InputData(
            "Incomplete migration checkpoint hashes".into(),
        ));
    }
    Ok(())
}

fn validate_vectors(conn: &Connection) -> Result<(), SaeError> {
    let mut stmt = conn
        .prepare("SELECT embedding FROM vec_chunks")
        .map_err(StorageError::from)?;
    let mut rows = stmt.query([]).map_err(StorageError::from)?;
    while let Some(row) = rows.next().map_err(StorageError::from)? {
        let bytes: Vec<u8> = row.get(0).map_err(StorageError::from)?;
        if bytes.len() != EMBEDDING_DIMS * 4 {
            return Err(SaeError::Internal("Invalid vector dimensionality".into()));
        }
        let mut norm = 0.;
        for word in bytes.as_chunks::<4>().0 {
            let x = f32::from_le_bytes(*word);
            if !x.is_finite() {
                return Err(SaeError::Internal("Nonfinite embedding".into()));
            }
            norm += f64::from(x).powi(2);
        }
        if (norm.sqrt() - 1.).abs() > 1e-3 {
            return Err(SaeError::Internal("Embedding is not L2 normalized".into()));
        }
    }
    let broken: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM embedded_chunk_ids e LEFT JOIN vec_chunks v ON v.rowid=e.vec_rowid WHERE v.rowid IS NULL OR v.chunk_id!=e.chunk_id OR v.sub_idx!=e.sub_idx) OR (SELECT count(*) FROM embedded_chunk_ids)!=(SELECT count(*) FROM vec_chunks)", [], |r| r.get(0)).map_err(StorageError::from)?;
    if broken {
        return Err(SaeError::Internal("Vector mapping mismatch".into()));
    }
    Ok(())
}

pub fn migrate(config: &Config, team: &str) -> Result<CommandOutput, SaeError> {
    let team = config.resolve_team(Some(team))?;
    migrate_at(&data_dir()?, team, tools::load_embedder, |_| Ok(()))
}

pub(crate) fn migrate_at<L, P>(
    dir: &Path,
    team: &str,
    load: L,
    mut on_batch: P,
) -> Result<CommandOutput, SaeError>
where
    L: FnOnce() -> Result<Arc<dyn Embed>, SaeError>,
    P: FnMut(u32) -> Result<(), SaeError>,
{
    validate_team_name(team)?;
    let _lock = lock_team(dir, team)?;
    let source_path = resolve_at(dir, team)?;
    if !source_path.exists() {
        return Err(SaeError::InputUsage(format!(
            "No local index for {team}; run `sae index {team}` first"
        )));
    }
    let source = Db::open_read_only(&source_path)?;
    if storage::validate_embedding_index(source.conn()).is_ok() {
        validate_checkpoint(source.conn())?;
        if storage::count_unembedded_chunks(source.conn())? == 0 {
            validate_vectors(source.conn())?;
            return Ok(output(
                "already current",
                &source_path,
                storage::count_chunks(source.conn())?,
            ));
        }
    }
    let digest = source_digest(source.conn())?;
    let identity = digest_hex(&Sha256::digest(storage::embedding_manifest().as_bytes()));
    let name = format!("{team}.gemma-{}-{}.db", &identity[..16], &digest[..16]);
    let destination = dir.join(&name);
    if destination == source_path {
        return Err(SaeError::InputData(
            "Selected Gemma index is incomplete; run `sae index <team>` to finish pending chunks"
                .into(),
        ));
    }
    reject_symlink(&destination)?;
    if !destination.exists() {
        let temporary = tempfile::tempdir_in(dir)?;
        let snapshot = temporary.path().join("snapshot.db");
        // VACUUM INTO includes committed WAL pages and gives a consistent copy.
        source
            .conn()
            .execute("VACUUM main INTO ?1", [snapshot.to_string_lossy().as_ref()])
            .map_err(StorageError::from)?;
        let snapshot_db = Db::open(&snapshot)?;
        if source_digest(snapshot_db.conn())? != digest {
            return Err(SaeError::InputData(
                "Source changed while snapshotting; retry migration".into(),
            ));
        }
        snapshot_db.conn().execute_batch("BEGIN IMMEDIATE; DROP TABLE vec_chunks; DELETE FROM embedded_chunk_ids; DELETE FROM embedding_content_hashes; DELETE FROM sae_index_manifest;").map_err(StorageError::from)?;
        snapshot_db
            .conn()
            .execute_batch(&storage::ddl_vec_chunks())
            .map_err(StorageError::from)?;
        snapshot_db
            .conn()
            .execute(
                "INSERT INTO sae_index_manifest VALUES(1,?1)",
                [storage::embedding_manifest()],
            )
            .map_err(StorageError::from)?;
        snapshot_db.conn().execute_batch("DROP TABLE IF EXISTS sae_migration; CREATE TABLE sae_migration(id INTEGER PRIMARY KEY CHECK(id=1),source_digest TEXT NOT NULL,source_name TEXT NOT NULL);").map_err(StorageError::from)?;
        let source_name = source_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| SaeError::InputData("Invalid source filename".into()))?;
        snapshot_db
            .conn()
            .execute(
                "INSERT INTO sae_migration VALUES(1,?1,?2)",
                rusqlite::params![digest, source_name],
            )
            .map_err(StorageError::from)?;
        snapshot_db
            .conn()
            .execute_batch("COMMIT; PRAGMA wal_checkpoint(TRUNCATE);")
            .map_err(StorageError::from)?;
        drop(snapshot_db);
        File::open(&snapshot)?.sync_all()?;
        fs::rename(&snapshot, &destination)?;
        File::open(dir)?.sync_all()?;
    }
    let db = Db::open(&destination)?;
    let saved: Option<String> = db
        .conn()
        .query_row(
            "SELECT source_digest FROM sae_migration WHERE id=1",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(StorageError::from)?;
    if saved.as_deref() != Some(digest.as_str()) || source_digest(db.conn())? != digest {
        return Err(SaeError::InputData(
            "Migration snapshot or source identity changed".into(),
        ));
    }
    validate_checkpoint(db.conn())?;
    if storage::count_unembedded_chunks(db.conn())? != 0 {
        let embedder = load()?;
        loop {
            let result = embed_batch::embed_one_batch(db.conn(), 512, |texts| {
                embedder
                    .embed_documents_batch(texts)
                    .map_err(|e| SaeError::Internal(format!("Embedding failed: {e}")))
            })?;
            if result.processed == 0 {
                break;
            }
            on_batch(result.processed)?;
        }
    }
    validate_checkpoint(db.conn())?;
    validate_vectors(db.conn())?;
    if storage::count_unembedded_chunks(db.conn())? != 0 {
        return Err(SaeError::Internal("Migration has pending chunks".into()));
    }
    if source_digest(source.conn())? != digest || resolve_at(dir, team)? != source_path {
        return Err(SaeError::InputData(
            "Source changed during migration; old index remains selected".into(),
        ));
    }
    db.conn()
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
        .map_err(StorageError::from)?;
    let chunks = storage::count_chunks(db.conn())?;
    drop(db);
    File::open(&destination)?.sync_all()?;
    let previous = source_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| SaeError::InputData("Invalid source filename".into()))?
        .to_owned();
    if name == previous {
        return Err(SaeError::Internal(
            "Migration destination equals source".into(),
        ));
    }
    publish(
        dir,
        team,
        &Selection {
            version: 1,
            active: name,
            previous,
        },
    )?;
    Ok(output("migrated", &destination, chunks))
}

pub fn rollback(config: &Config, team: &str) -> Result<CommandOutput, SaeError> {
    let team = config.resolve_team(Some(team))?;
    rollback_at(&data_dir()?, team)
}

pub(crate) fn rollback_at(dir: &Path, team: &str) -> Result<CommandOutput, SaeError> {
    validate_team_name(team)?;
    let _lock = lock_team(dir, team)?;
    let selected = read_selection(dir, team)?
        .ok_or_else(|| SaeError::InputData("No previous index selection".into()))?;
    let path = dir.join(&selected.previous);
    reject_symlink(&path)?;
    let db = Db::open_read_only(&path)?;
    let chunks = storage::count_chunks(db.conn())?;
    let check: String = db
        .conn()
        .query_row("PRAGMA quick_check", [], |r| r.get(0))
        .map_err(StorageError::from)?;
    if check != "ok" {
        return Err(SaeError::InputData(format!(
            "Previous index integrity check failed: {check}"
        )));
    }
    // A legacy Ruri index is useful for FTS and with the retained old binary.
    publish(
        dir,
        team,
        &Selection {
            version: 1,
            active: selected.previous,
            previous: selected.active,
        },
    )?;
    Ok(output("rolled back", &path, chunks))
}

#[cfg(test)]
mod tests {
    use super::{migrate_at, resolve_at, rollback_at};
    use crate::embedding::{ChunkedEmbedding, Embed, EmbedError, MockEmbedder};
    use crate::storage::{self, Db};
    use crate::tools::SaeError;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tempfile::tempdir;

    fn legacy_source(dir: &Path, count: u32) -> PathBuf {
        let path = dir.join("alpha.db");
        let db = Db::open(&path).unwrap();
        for number in 1..=count {
            let mut post = storage::test_post_row(number);
            post.body_md = format!("# 認証API\nRust authentication example {number}");
            storage::upsert_post(db.conn(), &post).unwrap();
            storage::rechunk_post(db.conn(), number, &post.body_md).unwrap();
        }
        let first = storage::get_unembedded_chunks(db.conn(), 1).unwrap()[0].0;
        storage::add_chunked_embeddings(
            db.conn(),
            &[(
                first,
                MockEmbedder::default().embed_document("old input").unwrap(),
            )],
        )
        .unwrap();
        db.conn()
            .execute(
                "UPDATE sae_index_manifest SET manifest='legacy-ruri-manifest'",
                [],
            )
            .unwrap();
        db.conn()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
        path
    }

    fn destination(dir: &Path) -> PathBuf {
        let found: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension().is_some_and(|s| s == "db")
                    && path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .contains(".gemma-")
            })
            .collect();
        assert_eq!(
            found.len(),
            1,
            "migration must retain one reusable snapshot"
        );
        found[0].clone()
    }

    struct CountingEmbedder {
        documents: Arc<AtomicUsize>,
    }
    impl Embed for CountingEmbedder {
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            MockEmbedder::default().embed_query(text)
        }
        fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
            MockEmbedder::default().embed_text(text, prefix)
        }
        fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
            self.documents.fetch_add(1, Ordering::SeqCst);
            MockEmbedder::default().embed_document(text)
        }
        fn embed_documents_batch(
            &self,
            texts: &[&str],
        ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
            self.documents.fetch_add(texts.len(), Ordering::SeqCst);
            MockEmbedder::default().embed_documents_batch(texts)
        }
    }
    fn loader(
        documents: &Arc<AtomicUsize>,
    ) -> impl FnOnce() -> Result<Arc<dyn Embed>, SaeError> + use<> {
        let documents = Arc::clone(documents);
        move || Ok(Arc::new(CountingEmbedder { documents }))
    }

    #[test]
    fn migration_resumes_committed_batches_then_activates_and_rolls_back_original() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 515);
        let source_bytes = fs::read(&source).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let interrupted = migrate_at(dir.path(), "alpha", loader(&counter), |_| {
            Err(SaeError::InputData("simulated interruption".into()))
        });
        assert!(interrupted.is_err());
        let committed = counter.load(Ordering::SeqCst);
        assert!(
            committed > 0 && committed < 515,
            "fixture must stop with both complete and pending work"
        );
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
        assert!(!dir.path().join("alpha.active.json").exists());
        assert_eq!(fs::read(&source).unwrap(), source_bytes);
        let staging = destination(dir.path());
        {
            let db = Db::open_read_only(&staging).unwrap();
            let hashes: u32 = db
                .conn()
                .query_row("SELECT count(*) FROM embedding_content_hashes", [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(
                usize::try_from(hashes).unwrap(),
                committed,
                "checkpoint hashes and vectors commit together"
            );
            assert_eq!(
                storage::count_unembedded_chunks(db.conn()).unwrap() as usize,
                515 - committed
            );
        }
        migrate_at(dir.path(), "alpha", loader(&counter), |_| Ok(())).unwrap();
        assert_eq!(
            counter.load(Ordering::SeqCst),
            515,
            "resume must not embed committed documents again"
        );
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), staging);
        {
            let db = Db::open_read_only(&staging).unwrap();
            storage::validate_embedding_index(db.conn()).unwrap();
            assert_eq!(storage::count_posts(db.conn()).unwrap(), 515);
            assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 0);
            let mappings: u32 = db
                .conn()
                .query_row("SELECT count(*) FROM embedded_chunk_ids", [], |r| r.get(0))
                .unwrap();
            assert_eq!(mappings, 515);
        }
        assert_eq!(fs::read(&source).unwrap(), source_bytes);
        rollback_at(dir.path(), "alpha").unwrap();
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
        assert_eq!(fs::read(&source).unwrap(), source_bytes);
        let old = Db::open_read_only(&source).unwrap();
        let manifest: String = old
            .conn()
            .query_row("SELECT manifest FROM sae_index_manifest", [], |r| r.get(0))
            .unwrap();
        assert_eq!(manifest, "legacy-ruri-manifest");
        assert!(
            staging.exists(),
            "rollback must retain completed Gemma index"
        );
    }

    #[test]
    fn completed_checkpoint_can_activate_without_loading_model_again() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 1);
        let counter = Arc::new(AtomicUsize::new(0));
        assert!(
            migrate_at(dir.path(), "alpha", loader(&counter), |_| Err(
                SaeError::InputData("stop before activation".into())
            ))
            .is_err()
        );
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
        migrate_at(
            dir.path(),
            "alpha",
            || panic!("completed checkpoint must not load model"),
            |_| Ok(()),
        )
        .unwrap();
        assert_ne!(resolve_at(dir.path(), "alpha").unwrap(), source);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn corrupted_checkpoint_is_rejected_without_changing_active_selection() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 1);
        let source_bytes = fs::read(&source).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        assert!(
            migrate_at(dir.path(), "alpha", loader(&counter), |_| Err(
                SaeError::InputData("interrupt".into())
            ))
            .is_err()
        );
        let staging = destination(dir.path());
        let db = Db::open(&staging).unwrap();
        db.conn()
            .execute(
                "UPDATE embedding_content_hashes SET content_sha256='invalid'",
                [],
            )
            .unwrap();
        drop(db);
        let result = migrate_at(
            dir.path(),
            "alpha",
            || panic!("bad checkpoint must be rejected before inference"),
            |_| Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
        assert_eq!(fs::read(&source).unwrap(), source_bytes);
        let db = Db::open_read_only(&staging).unwrap();
        let retained: String = db
            .conn()
            .query_row(
                "SELECT content_sha256 FROM embedding_content_hashes",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            retained, "invalid",
            "failed validation must retain diagnostic checkpoint"
        );
    }

    #[test]
    fn source_edit_after_snapshot_prevents_stale_activation() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 1);
        let counter = Arc::new(AtomicUsize::new(0));
        let result = migrate_at(dir.path(), "alpha", loader(&counter), |_| {
            let source_db = Db::open(&source)?;
            source_db
                .conn()
                .execute(
                    "UPDATE posts SET body_md='edited after snapshot' WHERE number=1",
                    [],
                )
                .map_err(storage::StorageError::from)?;
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "edit occurs after real migration work"
        );
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
        assert!(!dir.path().join("alpha.active.json").exists());
        let old = Db::open_read_only(&source).unwrap();
        let body: String = old
            .conn()
            .query_row("SELECT body_md FROM posts WHERE number=1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(body, "edited after snapshot");
    }

    #[test]
    fn invalid_pointer_cannot_select_another_team_or_escape_data_directory() {
        let dir = tempdir().unwrap();
        assert_eq!(
            resolve_at(dir.path(), "alpha").unwrap(),
            dir.path().join("alpha.db")
        );
        for active in [
            "../outside.db",
            "beta.gemma.db",
            "/tmp/alpha.db",
            "alpha..db",
        ] {
            fs::write(
                dir.path().join("alpha.active.json"),
                serde_json::json!({"version":1,"active":active,"previous":"alpha.db"}).to_string(),
            )
            .unwrap();
            assert!(
                resolve_at(dir.path(), "alpha").is_err(),
                "invalid active name {active}"
            );
        }
    }

    #[test]
    fn rollback_missing_previous_database_preserves_active_pointer() {
        let dir = tempdir().unwrap();
        legacy_source(dir.path(), 1);
        let counter = Arc::new(AtomicUsize::new(0));
        migrate_at(dir.path(), "alpha", loader(&counter), |_| Ok(())).unwrap();
        let pointer = dir.path().join("alpha.active.json");
        let before = fs::read(&pointer).unwrap();
        fs::remove_file(dir.path().join("alpha.db")).unwrap();
        assert!(rollback_at(dir.path(), "alpha").is_err());
        assert_eq!(fs::read(pointer).unwrap(), before);
    }

    #[test]
    fn inference_failure_or_invalid_vector_cannot_publish_new_index() {
        struct FaultyEmbedder {
            non_unit: bool,
        }
        impl Embed for FaultyEmbedder {
            fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
                MockEmbedder::default().embed_query(text)
            }
            fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
                MockEmbedder::default().embed_text(text, prefix)
            }
            fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
                MockEmbedder::default().embed_document(text)
            }
            fn embed_documents_batch(
                &self,
                texts: &[&str],
            ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
                if !self.non_unit {
                    return Err(EmbedError::Inference {
                        message: "injected inference failure".into(),
                        source: None,
                    });
                }
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut vector = vec![0.0; 768];
                        vector[0] = 2.0;
                        ChunkedEmbedding::new(vec![vector])
                    })
                    .collect())
            }
        }
        for non_unit in [false, true] {
            let dir = tempdir().unwrap();
            let source = legacy_source(dir.path(), 1);
            let before = fs::read(&source).unwrap();
            let result = migrate_at(
                dir.path(),
                "alpha",
                || Ok(Arc::new(FaultyEmbedder { non_unit })),
                |_| Ok(()),
            );
            assert!(
                result.is_err(),
                "inference and non-unit vector failures both prevent activation"
            );
            assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
            assert!(!dir.path().join("alpha.active.json").exists());
            assert_eq!(fs::read(source).unwrap(), before);
        }
    }

    #[test]
    fn missing_local_source_and_missing_rollback_leave_selection_absent() {
        let dir = tempdir().unwrap();
        let error = migrate_at(
            dir.path(),
            "alpha",
            || panic!("no input to embed"),
            |_| Ok(()),
        )
        .err()
        .expect("operation must reject the invalid state");
        assert!(error.to_string().contains("No local index"));
        assert!(
            rollback_at(dir.path(), "alpha")
                .err()
                .expect("operation must reject the invalid state")
                .to_string()
                .contains("No previous")
        );
        assert!(!dir.path().join("alpha.db").exists());
        assert!(!dir.path().join("alpha.active.json").exists());
    }

    #[test]
    fn migration_and_rollback_contend_on_the_same_team_lock() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 1);
        let before = fs::read(&source).unwrap();
        let lock = super::lock_team(dir.path(), "alpha").unwrap();
        let error = migrate_at(
            dir.path(),
            "alpha",
            || panic!("lock must precede inference"),
            |_| Ok(()),
        )
        .err()
        .expect("operation must reject the invalid state");
        assert!(error.to_string().contains("Another migration or rollback"));
        assert!(
            rollback_at(dir.path(), "alpha")
                .err()
                .expect("operation must reject the invalid state")
                .to_string()
                .contains("Another migration or rollback")
        );
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
        assert_eq!(fs::read(&source).unwrap(), before);
        assert!(!dir.path().join("alpha.active.json").exists());
        drop(lock);
        let counter = Arc::new(AtomicUsize::new(0));
        migrate_at(dir.path(), "alpha", loader(&counter), |_| Ok(())).unwrap();
        assert_ne!(resolve_at(dir.path(), "alpha").unwrap(), source);
    }

    #[test]
    fn current_index_is_idempotent_without_loading_model_or_republishing() {
        let dir = tempdir().unwrap();
        legacy_source(dir.path(), 1);
        let counter = Arc::new(AtomicUsize::new(0));
        migrate_at(dir.path(), "alpha", loader(&counter), |_| Ok(())).unwrap();
        let active = resolve_at(dir.path(), "alpha").unwrap();
        let pointer = fs::read(dir.path().join("alpha.active.json")).unwrap();
        let before = fs::read(&active).unwrap();
        let result = migrate_at(
            dir.path(),
            "alpha",
            || panic!("already current must avoid inference"),
            |_| panic!("no batches remain"),
        )
        .unwrap();
        assert_eq!(result.data["action"], "already current");
        assert_eq!(result.data["chunks"], 1);
        assert!(!result.degraded);
        assert_eq!(fs::read(&active).unwrap(), before);
        assert_eq!(
            fs::read(dir.path().join("alpha.active.json")).unwrap(),
            pointer
        );
    }

    #[test]
    fn selected_incomplete_snapshot_requires_indexing_and_keeps_pointer() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 515);
        let counter = Arc::new(AtomicUsize::new(0));
        assert!(
            migrate_at(dir.path(), "alpha", loader(&counter), |_| Err(
                SaeError::InputData("stop".into())
            ))
            .is_err()
        );
        let staging = destination(dir.path());
        let pointer = serde_json::json!({"version":1,"active":staging.file_name().unwrap().to_str().unwrap(),"previous":"alpha.db"}).to_string();
        fs::write(dir.path().join("alpha.active.json"), &pointer).unwrap();
        let result = migrate_at(
            dir.path(),
            "alpha",
            || panic!("selected snapshot cannot be overwritten"),
            |_| Ok(()),
        );
        assert!(
            result
                .err()
                .expect("operation must reject the invalid state")
                .to_string()
                .contains("incomplete")
        );
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), staging);
        assert_eq!(
            fs::read_to_string(dir.path().join("alpha.active.json")).unwrap(),
            pointer
        );
        assert!(source.exists());
    }

    #[test]
    fn changed_selection_during_inference_is_never_overwritten() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 1);
        let replacement = dir.path().join("alpha.replacement.db");
        fs::copy(&source, &replacement).unwrap();
        let pointer =
            serde_json::json!({"version":1,"active":"alpha.replacement.db","previous":"alpha.db"})
                .to_string();
        let counter = Arc::new(AtomicUsize::new(0));
        let result = migrate_at(dir.path(), "alpha", loader(&counter), |_| {
            fs::write(dir.path().join("alpha.active.json"), &pointer)?;
            Ok(())
        });
        assert!(
            result
                .err()
                .expect("operation must reject the invalid state")
                .to_string()
                .contains("Source changed")
        );
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), replacement);
        assert_eq!(
            fs::read_to_string(dir.path().join("alpha.active.json")).unwrap(),
            pointer
        );
        assert!(
            destination(dir.path()).exists(),
            "complete snapshot remains resumable"
        );
    }

    #[test]
    fn malformed_or_unreadable_selection_never_falls_back_to_legacy_database() {
        let dir = tempdir().unwrap();
        let source = legacy_source(dir.path(), 1);
        let before = fs::read(&source).unwrap();
        let pointer = dir.path().join("alpha.active.json");
        for contents in [
            "{",
            r#"{"version":2,"active":"alpha.new.db","previous":"alpha.db"}"#,
            r#"{"version":1,"active":"alpha.db","previous":"alpha.db"}"#,
            r#"{"version":1,"active":"alpha.new.db","previous":"beta.db"}"#,
            r#"{"version":1,"active":"alpha.new.db","previous":"alpha.db","unexpected":true}"#,
        ] {
            fs::write(&pointer, contents).unwrap();
            assert!(resolve_at(dir.path(), "alpha").is_err());
            assert!(rollback_at(dir.path(), "alpha").is_err());
            assert_eq!(fs::read_to_string(&pointer).unwrap(), contents);
            assert_eq!(fs::read(&source).unwrap(), before);
        }
        fs::remove_file(&pointer).unwrap();
        fs::create_dir(&pointer).unwrap();
        assert!(resolve_at(dir.path(), "alpha").is_err());
        assert_eq!(fs::read(source).unwrap(), before);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_or_directory_cannot_be_selected_migrated_or_rolled_back() {
        use std::os::unix::fs::symlink;
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let target = legacy_source(outside.path(), 1);
        let before = fs::read(&target).unwrap();
        let selected = dir.path().join("alpha.db");
        symlink(&target, &selected).unwrap();
        assert!(resolve_at(dir.path(), "alpha").is_err());
        assert!(
            migrate_at(
                dir.path(),
                "alpha",
                || panic!("must reject symlink"),
                |_| Ok(())
            )
            .is_err()
        );
        fs::remove_file(&selected).unwrap();
        fs::create_dir(&selected).unwrap();
        assert!(resolve_at(dir.path(), "alpha").is_err());
        fs::remove_dir(&selected).unwrap();
        symlink(&target, &selected).unwrap();
        let pointer = r#"{"version":1,"active":"alpha.new.db","previous":"alpha.db"}"#;
        fs::write(dir.path().join("alpha.active.json"), pointer).unwrap();
        assert!(rollback_at(dir.path(), "alpha").is_err());
        assert_eq!(
            fs::read_to_string(dir.path().join("alpha.active.json")).unwrap(),
            pointer
        );
        assert_eq!(fs::read(target).unwrap(), before);
    }

    #[test]
    fn missing_checkpoint_hash_and_changed_snapshot_identity_cannot_resume() {
        for mutation in [
            "DELETE FROM embedding_content_hashes",
            "UPDATE sae_migration SET source_digest='wrong'",
            "UPDATE posts SET name='edited snapshot'",
        ] {
            let dir = tempdir().unwrap();
            let source = legacy_source(dir.path(), 1);
            let before = fs::read(&source).unwrap();
            let counter = Arc::new(AtomicUsize::new(0));
            assert!(
                migrate_at(dir.path(), "alpha", loader(&counter), |_| Err(
                    SaeError::InputData("stop".into())
                ))
                .is_err()
            );
            let staging = destination(dir.path());
            let db = Db::open(&staging).unwrap();
            db.conn().execute_batch(mutation).unwrap();
            drop(db);
            assert!(
                migrate_at(
                    dir.path(),
                    "alpha",
                    || panic!("corrupt checkpoint must precede inference"),
                    |_| Ok(())
                )
                .is_err()
            );
            assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
            assert_eq!(fs::read(source).unwrap(), before);
            assert!(!dir.path().join("alpha.active.json").exists());
            assert!(staging.exists());
        }
    }

    #[test]
    fn checkpoint_vector_corruption_cannot_be_published() {
        for corruption in ["dimension", "nonfinite", "nonunit", "mapping"] {
            let dir = tempdir().unwrap();
            let source = legacy_source(dir.path(), 1);
            let before = fs::read(&source).unwrap();
            let counter = Arc::new(AtomicUsize::new(0));
            assert!(
                migrate_at(dir.path(), "alpha", loader(&counter), |_| Err(
                    SaeError::InputData("stop".into())
                ))
                .is_err()
            );
            let staging = destination(dir.path());
            let db = Db::open(&staging).unwrap();
            if corruption == "mapping" {
                db.conn()
                    .execute("UPDATE embedded_chunk_ids SET vec_rowid=999999", [])
                    .unwrap();
            } else {
                // A damaged checkpoint can replace vec0 with a plain table; validate bytes before publication.
                let mut vector = vec![0.0_f32; 768];
                vector[0] = match corruption {
                    "nonfinite" => f32::NAN,
                    "nonunit" => 2.0,
                    _ => 1.0,
                };
                if corruption == "dimension" {
                    vector.pop();
                }
                let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                db.conn().execute_batch("DROP TABLE vec_chunks; CREATE TABLE vec_chunks(embedding BLOB, chunk_id INTEGER, sub_idx INTEGER)").unwrap();
                db.conn()
                    .execute("INSERT INTO vec_chunks VALUES(?1,1,0)", [bytes])
                    .unwrap();
            }
            drop(db);
            let error = migrate_at(
                dir.path(),
                "alpha",
                || panic!("complete checkpoint needs no model"),
                |_| Ok(()),
            )
            .err()
            .expect("operation must reject the invalid state");
            assert!(
                error.to_string().contains(match corruption {
                    "dimension" => "dimensionality",
                    "nonfinite" => "Nonfinite",
                    "nonunit" => "normalized",
                    _ => "mapping",
                }),
                "{corruption}: {error}"
            );
            assert_eq!(resolve_at(dir.path(), "alpha").unwrap(), source);
            assert_eq!(fs::read(source).unwrap(), before);
            assert!(!dir.path().join("alpha.active.json").exists());
        }
    }
}
