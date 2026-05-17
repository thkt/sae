//! End-to-end checks for the `--json` envelope shape.
//!
//! Spawns the real binary so the envelope rendering, exit code, and stream
//! routing (stdout vs stderr) are all exercised together.

use std::env;
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
    let mut cmd = Command::new(sae_bin());
    cmd.env_clear()
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local/share"))
        .env("PATH", env::var_os("PATH").unwrap_or_default());
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
    assert_eq!(env["error"]["message"], "MLX backend is unavailable");
    // Exact-equality assertion mirrors T-340 unit test so the wire-level hint
    // and the in-process hint cannot drift independently. T-340 / T-344 pin
    // the source-side const; this anchors the binary-boundary surface to the
    // same literal.
    assert_eq!(
        env["error"]["next_step"],
        "Install the MLX backend (Apple Silicon required), or pass `--no-embed` for FTS-only search."
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

// T-CI010: synthetic reranker-failure search path emits a success envelope
// with `degraded=true` and the storage-layer warning surfaced in `notes[]`.
// Pins #140 wiring: `SearchOutput.warnings` reaches the `--json` envelope so
// AI agents can detect the reranker fallback. Unit tests T-221 (storage push)
// and T-262 / T-263 (output transform) cover the pieces; this test pins the
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
        notes
            .iter()
            .any(|n| n.as_str().is_some_and(|s| s.contains("reranker failed"))),
        "notes[] must surface the reranker-failure warning; got: {notes:?}"
    );
}

// T-CI011: SSRF guard rejects http://127.0.0.1 base_url at construction time.
// Integration tests build the lib without `cfg(test)`, so the test exemption
// inside `validate_base_url` does NOT apply — the production strict path runs.
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
