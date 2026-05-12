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
    let env: serde_json::Value = stderr
        .lines()
        .find_map(|line| serde_json::from_str(line).ok())
        .unwrap_or_else(|| panic!("no JSON line on stderr; got: {stderr}"));
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
