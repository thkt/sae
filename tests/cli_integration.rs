//! End-to-end checks for the `--json` envelope shape.
//!
//! Spawns the real binary so the envelope rendering, exit code, and stream
//! routing (stdout vs stderr) are all exercised together.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

/// Path to the binary cargo just built for this test run.
fn sae_bin() -> PathBuf {
    let mut path = env::current_exe().unwrap();
    // current_exe → target/<profile>/deps/cli_integration-<hash>
    path.pop(); // → target/<profile>/deps
    path.pop(); // → target/<profile>
    path.push(format!("sae{}", env::consts::EXE_SUFFIX));
    path
}

/// Builds a `Command` with isolated HOME/XDG dirs so the binary sees an
/// empty config (no leakage from the developer's real `~/.config/sae`).
fn sae_command(home: &Path) -> Command {
    sae_command_with_binary(home, &sae_bin())
}

fn sae_command_with_binary(home: &Path, binary: &Path) -> Command {
    let mut cmd = Command::new(binary);
    cmd.env_clear()
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local/share"))
        .env("PATH", env::var_os("PATH").unwrap_or_default());
    // cargo-llvm-cov instruments the spawned binary and directs its profile
    // output via LLVM_PROFILE_FILE; the env_clear() above drops it, leaving the
    // binary unable to write a profraw — so every integration-only path (the
    // `__test_*` seams, search) reads as uncovered. Restore it when present.
    // yomu sidesteps this by never calling env_clear().
    if let Some(profile) = env::var_os("LLVM_PROFILE_FILE") {
        cmd.env("LLVM_PROFILE_FILE", profile);
    }
    cmd
}

/// Extracts the first parseable JSON line from `stderr`. Mirror of the
/// envelope-rendering contract: `--json` failures emit one JSON line on
/// stderr. Panics on missing JSON to give a readable failure message.
fn parse_stderr_envelope(stderr: &str) -> serde_json::Value {
    stderr
        .lines()
        .find_map(|line| serde_json::from_str(line).ok())
        .unwrap_or_else(|| panic!("no JSON line on stderr; got: {stderr}"))
}

// T-CI001: `sae --json` (missing subcommand) → JSON UsageError envelope on stderr
#[test]
fn missing_subcommand_with_json_emits_usage_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(!output.status.success(), "should exit non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(env["error"]["code"], "USAGE_ERROR");
    assert!(
        env["error"]["message"].is_string(),
        "error.message must be present"
    );
}

// T-CI002: `sae --json status` (empty config) → SuccessEnvelope on stdout
#[test]
fn status_with_json_emits_success_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "status"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(
        output.status.success(),
        "status with empty config should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let env: serde_json::Value =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("stdout not JSON: {e}; {stdout}"));
    assert!(env.get("data").is_some(), "envelope must have `data`");
    assert!(
        env.get("degraded").is_some(),
        "envelope must have `degraded`"
    );
    assert!(env.get("notes").is_some(), "envelope must have `notes`");
    assert_eq!(env["degraded"], false);
    assert_eq!(env["notes"], serde_json::json!([]));
}

// T-CI003: default mode (no --json) emits markdown, not JSON
#[test]
fn status_without_json_emits_markdown() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["status"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        serde_json::from_str::<serde_json::Value>(&stdout).is_err() || stdout.trim().is_empty(),
        "default mode should not emit JSON; got: {stdout}"
    );
}

// T-CI004: `sae --json --help` prints help on stdout, exits SUCCESS
// (clap returns Err with use_stderr()=false; --json must NOT override this)
#[test]
fn help_with_json_exits_success_with_help_text() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "--help"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(
        output.status.success(),
        "--help should exit 0 even with --json; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage:"),
        "help text should land on stdout; got: {stdout}"
    );
}

// T-CI005: malformed `--after` value exits 65 (DATA_ERROR) with
// `error.code = "DATA_ERROR"` per ADR-0066 Group 2 baseline.
// Pins the binary-side contract: process-boundary exit code and JSON
// envelope code both reach the consumer for the new DATA_ERROR path.
#[test]
fn malformed_date_with_json_emits_data_error_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "search", "test", "--after", "2025-xx-xx"])
        .output()
        .expect("failed to spawn sae binary");

    assert_eq!(
        output.status.code(),
        Some(65),
        "exit code must be 65 (DATA_ERROR); stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(env["error"]["code"], "DATA_ERROR");
    assert_eq!(env["error"]["retryable"], false);
    let message = env["error"]["message"]
        .as_str()
        .expect("error.message must be a string");
    assert!(
        message.contains("Invalid date"),
        "message must describe the date parse failure; got: {message}"
    );
}

// T-CI006: synthetic UNKNOWN path exits 104 (UNKNOWN) with
// `error.code = "UNKNOWN"`. Pins the process-boundary contract for
// `SaeError::Other` per ADR-0066 L136 (#127 OPS-005). The hidden
// `__test_force_unknown` subcommand is the only hermetic UNKNOWN trigger;
// production sites (MLX backend, opaque embedder errors, model cache check)
// require real hardware / network state we cannot stage in tests.
#[cfg(feature = "test-support")]
#[test]
fn force_unknown_with_json_emits_unknown_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "__test_force_unknown"])
        .output()
        .expect("failed to spawn sae binary");

    assert_eq!(
        output.status.code(),
        Some(104),
        "exit code must be 104 (UNKNOWN); stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(env["error"]["code"], "UNKNOWN");
    assert_eq!(env["error"]["retryable"], false);
    assert!(
        env["error"]["message"].is_string(),
        "error.message must be present"
    );
}

// T-CI007: synthetic BackendUnavailable path exits 70 (INTERNAL) with
// `error.code = "INTERNAL"`. Pins the BREAKING `104 → 70` routing for the
// MLX backend missing path (#127 CHX-001) at the process boundary, mirroring
// T-CI005's role for DATA_ERROR. Without this pin, the contract only lives in
// unit tests (T-338 / T-342) and could regress silently across the
// `error_code()` → `exit_code()` → process exit translation layers.
#[cfg(feature = "test-support")]
#[test]
fn force_backend_unavailable_with_json_emits_internal_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "__test_force_backend_unavailable"])
        .output()
        .expect("failed to spawn sae binary");

    assert_eq!(
        output.status.code(),
        Some(70),
        "exit code must be 70 (INTERNAL); stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(env["error"]["code"], "INTERNAL");
    assert_eq!(env["error"]["retryable"], false);
    assert_eq!(
        env["error"]["message"],
        "Gemma native backend is unavailable"
    );
    // Exact-equality assertion mirrors T-340 unit test so the wire-level hint
    // and the in-process hint cannot drift independently. T-340 / T-344 pin
    // the source-side const; this anchors the binary-boundary surface to the
    // same literal.
    assert_eq!(
        env["error"]["next_step"],
        "Install a build with the Gemma native backend, or pass `--no-embed` for FTS-only search."
    );
    // `candidates` must be present and empty in the envelope (matches the
    // `to_error_envelope` composition pinned by T-342).
    assert!(
        env["error"].get("candidates").is_none()
            || env["error"]["candidates"]
                .as_array()
                .is_some_and(Vec::is_empty),
        "candidates must be absent or empty array; got: {}",
        env["error"]["candidates"]
    );
}

// T-CI008: synthetic Internal path exits 70 (INTERNAL) with
// `error.code = "INTERNAL"`. Pins the `SaeError::Internal` routing at the
// process boundary, parallel to T-CI007 for `BackendUnavailable` (#127
// CHX-001 audit follow-up). Without this pin, the `Internal` ↔ `Other` split
// (70 vs 104) only lives in unit tests (T-339 / T-343) and could regress
// silently across the `error_code()` → `exit_code()` → process exit chain.
#[cfg(feature = "test-support")]
#[test]
fn force_internal_with_json_emits_internal_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "__test_force_internal"])
        .output()
        .expect("failed to spawn sae binary");

    assert_eq!(
        output.status.code(),
        Some(70),
        "exit code must be 70 (INTERNAL); stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(env["error"]["code"], "INTERNAL");
    assert_eq!(env["error"]["retryable"], false);
    // Internal variant returns no canned next_step (T-341 pins this).
    assert!(
        env["error"].get("next_step").is_none() || env["error"]["next_step"].is_null(),
        "next_step must be absent for Internal; got: {}",
        env["error"]["next_step"]
    );
    assert!(
        env["error"]["message"].is_string(),
        "error.message must be present"
    );
}

// T-CI009: synthetic esa API 404 path exits 65 (DATA_ERROR) with
// `error.code = "DATA_ERROR"`. Pins the binary-boundary routing for the
// esa-404 reclassification (#136): the `ClientError::Api { status: 404, .. }`
// → DATA_ERROR mapping must survive the `error_code()` → `exit_code()` →
// process exit chain. Without this pin, the routing only lives in unit tests
// (T-346 / T-348 / T-352) and could regress silently. The hint string is
// exact-equality matched so the wire-level surface cannot drift from the
// in-process surface pinned by T-349 / T-352.
#[cfg(feature = "test-support")]
#[test]
fn force_client_api_404_with_json_emits_data_error_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "__test_force_client_api_404"])
        .output()
        .expect("failed to spawn sae binary");

    assert_eq!(
        output.status.code(),
        Some(65),
        "exit code must be 65 (DATA_ERROR); stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(env["error"]["code"], "DATA_ERROR");
    assert_eq!(env["error"]["retryable"], false);
    assert_eq!(
        env["error"]["next_step"],
        "Verify the post number exists in esa, or run `sae search <keyword>` to find it."
    );
    let message = env["error"]["message"]
        .as_str()
        .expect("error.message must be a string");
    assert!(
        message.contains("HTTP 404"),
        "message must surface the HTTP 404 status; got: {message}"
    );
}

// T-CI010: synthetic search warning emits a success envelope
// with `degraded=true` and the storage-layer warning surfaced in `notes[]`.
// Pins #140 wiring: `SearchOutput.warnings` reaches the `--json` envelope so
// AI agents can detect degraded search. Unit tests T-262 / T-263
// (output transform) cover the pieces; this test pins the
// end-to-end process-boundary shape.
#[cfg(feature = "test-support")]
#[test]
fn force_search_warning_with_json_emits_degraded_envelope() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "__test_force_search_warning"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(
        output.status.success(),
        "synthetic warning path is still a success; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let env: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("stdout not JSON: {e}; got: {stdout}"));
    assert_eq!(
        env["degraded"], true,
        "degraded must be true; got: {stdout}"
    );
    let notes = env["notes"].as_array().expect("notes must be an array");
    assert!(
        notes.iter().any(|n| n
            .as_str()
            .is_some_and(|s| s.contains("search component unavailable"))),
        "notes[] must surface the search warning; got: {notes:?}"
    );
}

/// Writes a `config.json` with the given teams into the test home so the
/// binary's `Config::load()` picks it up. Pairs with [`sae_command`] which
/// already sets `XDG_CONFIG_HOME=home/.config`.
fn write_config_with_teams(home: &Path, teams: &[&str]) {
    let dir = home.join(".config").join("sae");
    fs::create_dir_all(&dir).expect("config dir");
    let body = serde_json::json!({ "teams": teams }).to_string();
    fs::write(dir.join("config.json"), body).expect("write config.json");
}

// T-CI013: `--team unknownteam` against a multi-team config emits a JSON
// envelope whose `error.candidates` lists every team verbatim (config order,
// no fuzzy filter). Closes #139: agents recover from a typo in one
// roundtrip instead of re-running `sae status` for the team list.
#[test]
fn unknown_team_envelope_lists_known_teams_as_candidates() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha", "beta"]);

    let output = sae_command(dir.path())
        .args(["--json", "get", "1", "--team", "unknownteam"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(
        !output.status.success(),
        "unknown team should exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(
        env["error"]["candidates"],
        serde_json::json!(["alpha", "beta"]),
        "candidates must list configured teams in config order; envelope: {env}"
    );
}

// T-CI014: when no `--team` is provided AND the config has multiple teams
// without a default, the resulting `NoTeamSpecified` envelope also surfaces
// the team list. The two recoverable team failures share the same
// candidates contract (#139).
#[test]
fn no_team_envelope_lists_known_teams_as_candidates() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha", "beta"]);

    let output = sae_command(dir.path())
        .args(["--json", "get", "1"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(
        !output.status.success(),
        "no-team-resolved should exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(
        env["error"]["candidates"],
        serde_json::json!(["alpha", "beta"]),
        "candidates must list teams when no team is selectable; envelope: {env}"
    );
}

// T-CI015: end-to-end — a typo subcommand emits a `--json` envelope whose
// `error.candidates` lists the user-facing subcommands without any
// `__test_*` seam leaking. Closes #148.
#[test]
fn invalid_subcommand_envelope_lists_public_subcommands() {
    let dir = tempdir().unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "sserach", "test query"])
        .output()
        .expect("failed to spawn sae binary");

    assert!(
        !output.status.success(),
        "typo subcommand must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let env = parse_stderr_envelope(&stderr);
    assert_eq!(env["error"]["code"], "USAGE_ERROR");

    let candidates = env["error"]["candidates"]
        .as_array()
        .expect("candidates must be an array");
    assert!(
        !candidates.is_empty(),
        "InvalidSubcommand must surface candidates; envelope: {env}"
    );
    assert!(
        candidates
            .iter()
            .all(|c| !c.as_str().is_some_and(|s| s.starts_with("__test_"))),
        "test-seam subcommands must not leak to user-facing candidates; got: {candidates:?}"
    );
    for required in ["search", "get", "status"] {
        assert!(
            candidates.iter().any(|c| c == required),
            "candidates must include {required:?}; got: {candidates:?}"
        );
    }
}

// T-CI011: SSRF guard rejects http://127.0.0.1 base_url at construction time.
// Integration tests cannot see `with_base_url_unchecked` (gated on `cfg(test)`
// in the lib build), so they exercise the strict `with_base_url` path.
// Pins #138 subtask 3: AI agents that read a misconfigured `base_url` from
// config / env should never be able to steer the esa client at a private host.
#[test]
fn esa_client_rejects_private_ip_base_url() {
    use sae::client::{ClientError, EsaClient};

    let result = EsaClient::with_base_url("token".into(), "http://127.0.0.1".into());
    assert!(
        matches!(result, Err(ClientError::InvalidRequest(_))),
        "with_base_url('http://127.0.0.1') must be rejected; got: {result:?}"
    );
}

// T-CI012: SSRF guard rejects arbitrary attacker-controlled hosts even with
// https. Companion to T-CI011 pinning the .esa.io allowlist at the binary
// boundary (strict path, no cfg(test) exemption).
#[test]
fn esa_client_rejects_non_esa_host() {
    use sae::client::{ClientError, EsaClient};

    let result = EsaClient::with_base_url("token".into(), "https://attacker.com".into());
    assert!(
        matches!(result, Err(ClientError::InvalidRequest(_))),
        "with_base_url('https://attacker.com') must be rejected; got: {result:?}"
    );
}

// T-CI012: `search --no-embed` against a seeded DB drives run_search's
// read-only path end to end — covering the semantic-flag decision and the
// success-envelope wiring (search.rs:102,105). Seeds via the public storage
// API because sae cannot run `index` offline (esa fetch), unlike yomu's
// local-file index.
#[test]
fn search_no_embed_against_seeded_db_exits_success() {
    use sae::storage::{Db, rechunk_post, test_post_row, upsert_post};

    let dir = tempdir().unwrap();
    let home = dir.path();
    write_config_with_teams(home, &["myteam"]);

    // Db::open creates the parent data dir; seed one post + its chunks so FTS
    // has something to match.
    let db_path = home.join(".local/share/sae/myteam.db");
    let db = Db::open(&db_path).expect("open seed db");
    upsert_post(db.conn(), &test_post_row(1)).expect("seed post");
    rechunk_post(db.conn(), 1, "# Title\nBody text").expect("seed chunks");
    drop(db);

    let output = sae_command(home)
        .args(["search", "Title", "--no-embed", "--team", "myteam"])
        .output()
        .expect("run search");
    assert!(
        output.status.success(),
        "search --no-embed against a seeded DB must exit 0; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn seed_search_db(home: &Path, legacy: bool) -> PathBuf {
    use sae::embedding::ChunkedEmbedding;
    use sae::storage::{self, Db, EsaPostRow};
    let path = home.join(".local/share/sae/alpha.db");
    let db = Db::open(&path).unwrap();
    let post = EsaPostRow {
        number: 1,
        name: "認証ガイド".into(),
        full_name: "dev/認証ガイド".into(),
        body_md: "Rust API の認証を実装する".into(),
        category: None,
        tags: vec![],
        wip: false,
        kind: "stock".into(),
        url: "https://alpha.esa.io/posts/1".into(),
        created_at: "2026-10-01T00:00:00Z".into(),
        updated_at: "2026-10-17T00:00:00Z".into(),
        created_by: "alice".into(),
        updated_by: "alice".into(),
        revision_number: 1,
    };
    storage::upsert_post(db.conn(), &post).unwrap();
    storage::rechunk_post(db.conn(), 1, &post.body_md).unwrap();
    if legacy {
        let id = storage::get_unembedded_chunks(db.conn(), 1).unwrap()[0].0;
        let vector = ChunkedEmbedding::new(vec![vec![0.5; 768]]);
        storage::add_chunked_embeddings(db.conn(), &[(id, vector)]).unwrap();
        db.conn()
            .execute("DELETE FROM sae_index_manifest", [])
            .unwrap();
    }
    db.conn()
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    path
}

#[test]
fn no_embed_search_reads_legacy_index_without_loading_corrupt_model_or_writing() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha"]);
    let path = seed_search_db(dir.path(), true);
    let before = fs::read(&path).unwrap();
    let model = dir.path().join("corrupt-model");
    fs::create_dir(&model).unwrap();
    fs::write(model.join("model.gguf"), b"corrupt weights").unwrap();
    fs::write(model.join("tokenizer.json"), b"corrupt tokenizer").unwrap();
    let output = sae_command(dir.path())
        .env("SAE_MODEL_DIR", &model)
        .env("SAE_RERANK", "1")
        .args(["--json", "search", "認証", "--no-embed"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["degraded"], false);
    assert_eq!(envelope["notes"], serde_json::json!([]));
    assert!(String::from_utf8_lossy(&output.stdout).contains("認証ガイド"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("checksum"));
    assert_eq!(
        fs::read(path).unwrap(),
        before,
        "search must preserve DB bytes"
    );
}

#[test]
fn semantic_search_rejects_legacy_vectors_with_json_error_and_preserves_db() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha"]);
    let path = seed_search_db(dir.path(), true);
    let before = fs::read(&path).unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "search", "認証"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let envelope = parse_stderr_envelope(&String::from_utf8_lossy(&output.stderr));
    let message = envelope["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("index") || message.contains("Index"),
        "{message}"
    );
    assert!(
        envelope["error"]["next_step"]
            .as_str()
            .is_some_and(|s| s.contains("migrate"))
    );
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn missing_and_corrupt_models_degrade_search_with_success_envelope() {
    for corrupt in [false, true] {
        let dir = tempdir().unwrap();
        write_config_with_teams(dir.path(), &["alpha"]);
        let path = seed_search_db(dir.path(), false);
        let before = fs::read(&path).unwrap();
        let model = dir.path().join("model");
        fs::create_dir(&model).unwrap();
        if corrupt {
            fs::write(model.join("model.gguf"), b"not a valid model").unwrap();
            fs::write(model.join("tokenizer.json"), b"{}").unwrap();
        }
        let output = sae_command(dir.path())
            .env("SAE_MODEL_DIR", &model)
            .args(["--json", "search", "認証"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(envelope["degraded"], true);
        assert!(
            envelope["notes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|note| note.as_str().is_some_and(|s| s.contains("FTS")))
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("認証ガイド"));
        assert_eq!(fs::read(path).unwrap(), before);
    }
}

// test-support intentionally substitutes a model-free migration embedder.
// Run this production-loader contract in the default-feature suite.
#[cfg(not(feature = "test-support"))]
#[test]
fn migrate_without_model_keeps_local_source_and_returns_error_envelope() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha"]);
    let path = seed_search_db(dir.path(), true);
    let before = fs::read(&path).unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "model", "migrate", "alpha"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let envelope = parse_stderr_envelope(&String::from_utf8_lossy(&output.stderr));
    assert!(envelope["error"]["message"].is_string());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!path.with_file_name("alpha.active.json").exists());
}

#[cfg(not(feature = "test-support"))]
#[test]
fn migrate_corrupt_model_emits_internal_error_and_preserves_active_source() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha"]);
    let path = seed_search_db(dir.path(), true);
    let before = fs::read(&path).unwrap();
    let model = dir.path().join("corrupt-model");
    fs::create_dir(&model).unwrap();
    fs::write(model.join("model.gguf"), b"invalid weights").unwrap();
    fs::write(model.join("tokenizer.json"), b"{}").unwrap();
    let output = sae_command(dir.path())
        .env("SAE_MODEL_DIR", model)
        .args(["--json", "model", "migrate", "alpha"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(70));
    let envelope = parse_stderr_envelope(&String::from_utf8_lossy(&output.stderr));
    assert_eq!(envelope["error"]["code"], "INTERNAL");
    assert_eq!(envelope["error"]["retryable"], false);
    let message = envelope["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("checksum"),
        "corruption reason must be visible: {message}"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!path.with_file_name("alpha.active.json").exists());
}

/// A valid envelope is insufficient if native model teardown aborts afterward.
#[test]
#[ignore = "requires pinned Gemma artifacts in SAE_MODEL_DIR; launches the real CLI"]
fn real_semantic_search_exits_successfully_after_emitting_json() {
    use sae::{
        model,
        storage::{self, Db},
    };
    let artifacts = env::var_os("SAE_MODEL_DIR").expect("explicit isolated pinned model directory");
    assert!(
        Path::new(&artifacts).is_absolute(),
        "SAE_MODEL_DIR must be absolute"
    );
    let gpu = model::gpu_available().expect("backend hardware diagnostic");
    let home = tempdir().unwrap();
    write_config_with_teams(home.path(), &["alpha"]);
    let path = seed_search_db(home.path(), false);
    if gpu {
        let db = Db::open(&path).unwrap();
        let chunks = storage::get_unembedded_chunks(db.conn(), 10).unwrap();
        assert_eq!(
            chunks.len(),
            1,
            "fixture reaches one real document embedding"
        );
        let texts: Vec<_> = chunks.iter().map(|(_, content)| content.as_str()).collect();
        let embedder = model::load_verified().expect("pinned real model must be available");
        let vectors = embedder.embed_documents_batch(&texts).unwrap();
        let embeddings: Vec<_> = chunks.iter().map(|(id, _)| *id).zip(vectors).collect();
        storage::add_chunked_embeddings(db.conn(), &embeddings).unwrap();
        storage::validate_embedding_index(db.conn()).unwrap();
        db.conn()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    // The real model used for setup is released before the child acquires it.
    let before = fs::read(&path).unwrap();
    let mut command = sae_command_with_binary(home.path(), Path::new(env!("CARGO_BIN_EXE_sae")));
    #[cfg(feature = "test-support")]
    let probe_trace = home.path().join("probe-trace.jsonl");
    #[cfg(feature = "test-support")]
    command.env("__SAE_TEST_PROBE_TRACE", &probe_trace);
    let output = command
        .env("SAE_MODEL_DIR", artifacts)
        .args(["--json", "search", "Rust API の認証", "--team", "alpha"])
        .output()
        .expect("spawn real sae CLI");
    #[cfg(feature = "test-support")]
    eprintln!(
        "probe trace [real_semantic_search_exits_successfully_after_emitting_json]:\n{}",
        fs::read_to_string(&probe_trace).unwrap_or_else(|error| format!("unavailable: {error}"))
    );
    let envelope: serde_json::Value =
        serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
            panic!(
                "valid search JSON expected: {error}; stdout={}; stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        });
    assert_eq!(envelope["degraded"], !gpu);
    assert_eq!(
        envelope["data"]["search_mode"],
        if gpu { "hybrid" } else { "fts" }
    );
    assert_eq!(
        envelope["notes"],
        if gpu {
            serde_json::json!([])
        } else {
            serde_json::json!(["semantic search unavailable, falling back to FTS"])
        }
    );
    assert_eq!(envelope["data"]["results"][0]["post_number"], 1);
    assert_eq!(
        output.status.code(),
        Some(0),
        "real search must exit normally after JSON; status={}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(path).unwrap(),
        before,
        "semantic search remains read-only"
    );
}

#[test]
fn migrate_current_index_reports_success_without_model_and_preserves_database() {
    use sae::embedding::ChunkedEmbedding;
    use sae::storage::{self, Db};
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha"]);
    let path = seed_search_db(dir.path(), false);
    {
        let db = Db::open(&path).unwrap();
        let id = storage::get_unembedded_chunks(db.conn(), 1).unwrap()[0].0;
        let mut vector = vec![0.0; 768];
        vector[0] = 1.0;
        storage::add_chunked_embeddings(db.conn(), &[(id, ChunkedEmbedding::new(vec![vector]))])
            .unwrap();
        db.conn()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    let before = fs::read(&path).unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "model", "migrate", "alpha"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["data"]["action"], "already current");
    assert_eq!(envelope["data"]["model"], "EmbeddingGemma 2");
    assert_eq!(envelope["data"]["chunks"], 1);
    assert_eq!(envelope["degraded"], false);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!path.with_file_name("alpha.active.json").exists());
}

#[test]
fn rollback_command_selects_retained_database_and_keeps_both_files() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha"]);
    let legacy = seed_search_db(dir.path(), true);
    let current = legacy.with_file_name("alpha.current.db");
    fs::copy(&legacy, &current).unwrap();
    let before_legacy = fs::read(&legacy).unwrap();
    let before_current = fs::read(&current).unwrap();
    let pointer = legacy.with_file_name("alpha.active.json");
    fs::write(
        &pointer,
        r#"{"version":1,"active":"alpha.current.db","previous":"alpha.db"}"#,
    )
    .unwrap();
    let output = sae_command(dir.path())
        .args(["--json", "model", "rollback", "alpha"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["data"]["action"], "rolled back");
    assert_eq!(envelope["data"]["chunks"], 1);
    assert!(envelope["data"]["model"].is_null());
    assert_eq!(envelope["degraded"], false);
    let selected: serde_json::Value = serde_json::from_slice(&fs::read(&pointer).unwrap()).unwrap();
    assert_eq!(selected["active"], "alpha.db");
    assert_eq!(selected["previous"], "alpha.current.db");
    assert_eq!(fs::read(legacy).unwrap(), before_legacy);
    assert_eq!(fs::read(current).unwrap(), before_current);
}

#[test]
fn migration_commands_without_local_data_report_errors_and_create_no_database() {
    let dir = tempdir().unwrap();
    write_config_with_teams(dir.path(), &["alpha"]);
    for command in ["migrate", "rollback"] {
        let output = sae_command(dir.path())
            .args(["--json", "model", command, "alpha"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let envelope = parse_stderr_envelope(&String::from_utf8_lossy(&output.stderr));
        let message = envelope["error"]["message"].as_str().unwrap();
        assert!(
            message.contains(if command == "migrate" {
                "No local index"
            } else {
                "No previous index"
            }),
            "{message}"
        );
        let data = dir.path().join(".local/share/sae");
        assert!(!data.join("alpha.db").exists());
        assert!(!data.join("alpha.active.json").exists());
    }
}

#[test]
#[ignore = "requires pinned Gemma artifacts in SAE_MODEL_DIR; launches the real CLI"]
fn real_model_download_reuses_verified_cache_and_reports_hardware_result() {
    use sae::model;
    use sha2::{Digest, Sha256};
    use std::io::Read;
    fn digest(path: &Path) -> String {
        let mut file = fs::File::open(path).unwrap();
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 65536];
        loop {
            let size = file.read(&mut buffer).unwrap();
            if size == 0 {
                break;
            }
            hash.update(&buffer[..size]);
        }
        hash.finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
    let artifacts = env::var_os("SAE_MODEL_DIR").expect("explicit isolated pinned model directory");
    let model_dir = Path::new(&artifacts);
    assert!(model_dir.is_absolute());
    let spec = model::ModelSpec::default();
    let weights = model_dir.join("model.gguf");
    let tokenizer = model_dir.join("tokenizer.json");
    assert_eq!(digest(&weights), spec.model_sha256);
    assert_eq!(digest(&tokenizer), spec.tokenizer_sha256);
    let gpu = model::gpu_available().expect("backend hardware diagnostic");
    let home = tempdir().unwrap();
    let mut command = sae_command_with_binary(home.path(), Path::new(env!("CARGO_BIN_EXE_sae")));
    #[cfg(feature = "test-support")]
    let probe_trace = home.path().join("probe-trace.jsonl");
    #[cfg(feature = "test-support")]
    command.env("__SAE_TEST_PROBE_TRACE", &probe_trace);
    let output = command
        .env("SAE_MODEL_DIR", &artifacts)
        .args(["--json", "model", "download"])
        .output()
        .unwrap();
    #[cfg(feature = "test-support")]
    eprintln!(
        "probe trace [real_model_download_reuses_verified_cache_and_reports_hardware_result]:\n{}",
        fs::read_to_string(&probe_trace).unwrap_or_else(|error| format!("unavailable: {error}"))
    );
    assert_eq!(
        output.status.code(),
        Some(if gpu { 0 } else { 70 }),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    if gpu {
        let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(envelope["data"]["status"], "ok");
        assert_eq!(envelope["degraded"], false);
    } else {
        let envelope = parse_stderr_envelope(&String::from_utf8_lossy(&output.stderr));
        assert_eq!(envelope["error"]["code"], "INTERNAL");
        assert_eq!(envelope["error"]["retryable"], false);
        assert!(
            envelope["error"]["message"]
                .as_str()
                .unwrap()
                .contains("backend")
        );
    }
    assert_eq!(digest(&weights), spec.model_sha256);
    assert_eq!(digest(&tokenizer), spec.tokenizer_sha256);
}
