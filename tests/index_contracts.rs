//! Persistence contracts use only disposable databases and deterministic vectors.
use sae::model::load_verified;
use std::{fs, iter};

use jiff::Timestamp;
use sae::embedding::{ChunkedEmbedding, EMBEDDING_DIMS, EmbedOptions};
use sae::storage::{
    self, Db, EsaPostRow, SearchFilter, add_chunked_embeddings, hybrid_search,
    validate_embedding_index,
};
use tempfile::tempdir;

fn post(number: u32) -> EsaPostRow {
    EsaPostRow {
        number,
        name: "認証とRust".into(),
        full_name: "dev/認証とRust".into(),
        body_md: "# 認証ガイド\nRust API の認証を実装する".into(),
        category: Some("dev".into()),
        tags: vec!["Rust".into()],
        wip: false,
        kind: "stock".into(),
        url: format!("https://example.esa.io/posts/{number}"),
        created_at: "2026-10-01T00:00:00Z".into(),
        updated_at: "2026-10-17T00:00:00Z".into(),
        created_by: "alice".into(),
        updated_by: "alice".into(),
        revision_number: 1,
    }
}

fn seed(db: &Db) -> i64 {
    let row = post(1);
    storage::upsert_post(db.conn(), &row).unwrap();
    storage::rechunk_post(db.conn(), 1, &row.body_md).unwrap();
    storage::get_unembedded_chunks(db.conn(), 1).unwrap()[0].0
}

fn embedding() -> ChunkedEmbedding {
    let mut v = vec![0.0; EMBEDDING_DIMS];
    v[0] = 1.0;
    ChunkedEmbedding::new(vec![v])
}

#[test]
fn new_database_manifest_accepts_matching_query_and_document_vectors() {
    let db = Db::open_memory().unwrap();
    validate_embedding_index(db.conn()).unwrap();
    let id = seed(&db);
    add_chunked_embeddings(db.conn(), &[(id, embedding())]).unwrap();
    validate_embedding_index(db.conn()).unwrap();
    let result = hybrid_search(
        db.conn(),
        "unrelated query",
        Some(&embedding().chunks()[0]),
        10,
        Timestamp::now(),
        &SearchFilter::default(),
    )
    .unwrap();
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].post_number, 1);
}

#[test]
fn unmanifested_vectors_block_semantic_reads_and_appends_but_keep_fts() {
    let db = Db::open_memory().unwrap();
    let id = seed(&db);
    add_chunked_embeddings(db.conn(), &[(id, embedding())]).unwrap();
    db.conn()
        .execute("DELETE FROM sae_index_manifest", [])
        .unwrap();
    assert!(validate_embedding_index(db.conn()).is_err());
    let before: u32 = db
        .conn()
        .query_row("SELECT count(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(add_chunked_embeddings(db.conn(), &[(id, embedding())]).is_err());
    let after: u32 = db
        .conn()
        .query_row("SELECT count(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        after, before,
        "rejected writes must preserve existing vectors"
    );
    assert!(
        hybrid_search(
            db.conn(),
            "認証",
            Some(&embedding().chunks()[0]),
            10,
            Timestamp::now(),
            &SearchFilter::default()
        )
        .is_err()
    );
    let fts = hybrid_search(
        db.conn(),
        "認証",
        None,
        10,
        Timestamp::now(),
        &SearchFilter::default(),
    )
    .unwrap();
    assert_eq!(fts.results.len(), 1);
}

#[test]
fn each_manifest_contract_change_is_incompatible() {
    // Every field contributes to query/document comparability or reproducibility.
    let base: serde_json::Value = serde_json::from_str(&storage::embedding_manifest()).unwrap();
    for field in [
        "version",
        "chunk_policy",
        "document_input",
        "model_id",
        "model_sha256",
        "tokenizer_sha256",
        "query_prefix",
        "document_prefix",
        "pooling",
        "dimensions",
        "max_tokens",
        "overlap_tokens",
        "implementation",
        "backend_revision",
    ] {
        let db = Db::open_memory().unwrap();
        let id = seed(&db);
        add_chunked_embeddings(db.conn(), &[(id, embedding())]).unwrap();
        let mut altered = base.clone();
        let object = altered.as_object_mut().expect("manifest object");
        // Storage may wrap the model spec along with chunk policy.
        let object = if object.contains_key(field) {
            object
        } else {
            object
                .get_mut("encoder")
                .expect("model spec in manifest")
                .as_object_mut()
                .unwrap()
        };
        let value = object
            .get_mut(field)
            .unwrap_or_else(|| panic!("manifest must identify {field}"));
        *value = if value.is_number() {
            serde_json::json!(999)
        } else {
            serde_json::json!("incompatible")
        };
        db.conn()
            .execute(
                "UPDATE sae_index_manifest SET manifest = ?1",
                [altered.to_string()],
            )
            .unwrap();
        assert!(
            validate_embedding_index(db.conn()).is_err(),
            "changed {field} must be rejected"
        );
    }
}

#[test]
fn read_only_database_never_creates_schema_or_allows_writes() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.db");
    assert!(Db::open_read_only(&missing).is_err());
    assert!(!missing.exists());
    let path = dir.path().join("team.db");
    {
        let db = Db::open(&path).unwrap();
        seed(&db);
        db.conn()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    let before = fs::read(&path).unwrap();
    {
        let db = Db::open_read_only(&path).unwrap();
        assert_eq!(storage::count_posts(db.conn()).unwrap(), 1);
        assert!(db.conn().execute("DELETE FROM posts", []).is_err());
        let hits = hybrid_search(
            db.conn(),
            "認証",
            None,
            10,
            Timestamp::now(),
            &SearchFilter::default(),
        )
        .unwrap();
        assert_eq!(hits.results.len(), 1);
    }
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn model_identity_covers_pinned_encoder_and_input_contract() {
    use sae::model::ModelSpec;
    let base = ModelSpec::default();
    base.validate().unwrap();
    assert_eq!(base.dimensions, 768);
    assert_eq!(base.max_tokens, 8192);
    assert_eq!(base.overlap_tokens, 2048);
    assert_eq!(base.query_prefix, "task: search result | query: ");
    assert_eq!(base.document_prefix, "title: none | text: ");
    assert_eq!(base.pooling, "mean-l2-noncausal");
    let original = serde_json::to_value(&base).unwrap();
    for field in [
        "revision",
        "model_sha256",
        "tokenizer_revision",
        "tokenizer_sha256",
        "query_prefix",
        "document_prefix",
        "pooling",
        "dimensions",
        "max_tokens",
        "overlap_tokens",
        "implementation",
        "backend_revision",
        "bindings_revision",
    ] {
        let mut altered = original.clone();
        let value = altered.get_mut(field).unwrap();
        *value = if value.is_number() {
            serde_json::json!(1)
        } else {
            serde_json::json!("different")
        };
        let changed: ModelSpec = serde_json::from_value(altered).unwrap();
        assert_ne!(
            changed.identity(),
            base.identity(),
            "{field} must affect identity"
        );
    }
}

#[test]
fn reopening_legacy_database_does_not_label_existing_vectors_as_gemma() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("legacy.db");
    {
        let db = Db::open(&path).unwrap();
        let id = seed(&db);
        add_chunked_embeddings(db.conn(), &[(id, embedding())]).unwrap();
        db.conn()
            .execute("DROP TABLE sae_index_manifest", [])
            .unwrap();
    }
    let db = Db::open(&path).unwrap();
    assert!(validate_embedding_index(db.conn()).is_err());
    let manifests: u32 = db
        .conn()
        .query_row("SELECT count(*) FROM sae_index_manifest", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        manifests, 0,
        "schema initialization must not relabel legacy vectors"
    );
    let vectors: u32 = db
        .conn()
        .query_row("SELECT count(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vectors, 1);
}

/// Runs only with the pinned model installed into an isolated SAE_MODEL_DIR.
#[test]
#[ignore = "requires the pinned real Gemma artifacts; no network download"]
fn real_gemma_japanese_code_and_long_documents_are_normalized_768_vectors() {
    let embedder = load_verified().expect("install pinned artifacts before opting in");
    let query = embedder
        .embed_query("Rust で認証APIを実装するには？")
        .unwrap();
    let short = embedder.embed_document("# 認証\n```rust\nfn authenticate(token: &str) -> bool { !token.is_empty() }\n```\n日本語の認証ドキュメント").unwrap();
    let samples = [
        "日本語の認証API。Rustでtokenを検証する実装と手順。",
        "```rust\nfn deploy(version: &str) { println!(\"{version}\"); }\n```\nデプロイのロールバック手順。",
    ];
    let singles = samples.map(|text| embedder.embed_document(text).unwrap());
    for texts in [
        vec![samples[0], samples[1]],
        vec![samples[1], samples[0], samples[1]],
    ] {
        for budget in [None, Some(1)] {
            let batch = embedder
                .embed_documents_batch_with_options(
                    &texts,
                    &EmbedOptions {
                        token_budget: budget,
                        forward_pause: None,
                    },
                )
                .unwrap();
            assert_eq!(batch.len(), texts.len());
            for (text, output) in texts.iter().zip(&batch) {
                let reference = &singles[usize::from(*text == samples[1])];
                assert_eq!(output.chunks().len(), reference.chunks().len());
                for (actual, expected) in output.chunks().iter().zip(reference.chunks()) {
                    let max_diff = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        max_diff < 1e-3,
                        "batched output ownership or attention isolation drift: {max_diff}"
                    );
                }
            }
        }
    }
    let long = embedder
        .embed_document(&"日本語の認証APIとRustコード fn authenticate() {}\n".repeat(2000))
        .unwrap();
    assert!(
        long.chunks().len() > 1,
        "long document must produce overlapping token windows"
    );
    for vector in iter::once(query.as_slice())
        .chain(short.chunks().iter().map(Vec::as_slice))
        .chain(long.chunks().iter().map(Vec::as_slice))
    {
        assert_eq!(vector.len(), 768);
        assert!(vector.iter().all(|value| value.is_finite()));
        let norm = vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "L2 norm={norm}");
    }
}

#[test]
fn index_without_manifest_table_is_rejected_without_schema_repair() {
    let db = Db::open_memory().unwrap();
    let id = seed(&db);
    db.conn()
        .execute("DROP TABLE sae_index_manifest", [])
        .unwrap();
    let error = validate_embedding_index(db.conn()).unwrap_err();
    assert!(error.to_string().contains("no encoder identity"));
    assert!(add_chunked_embeddings(db.conn(), &[(id, embedding())]).is_err());
    assert_eq!(storage::count_unembedded_chunks(db.conn()).unwrap(), 1);
    let exists: bool = db
        .conn()
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='sae_index_manifest')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        !exists,
        "validation must not create an identity for unknown vectors"
    );
    let hits = hybrid_search(
        db.conn(),
        "認証",
        None,
        10,
        Timestamp::now(),
        &SearchFilter::default(),
    )
    .unwrap();
    assert_eq!(hits.results.len(), 1);
}
