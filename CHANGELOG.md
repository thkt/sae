# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING**: `sae harvest <team>` および `sae harvest <team> --full` を廃止し、
  `sae index <team>`（増分）と `sae rebuild <team>`（全件再構築）の 2 サブコマンドに
  分割。yomu / recall と命名を揃えるため。`--full` の発見性問題も解消する。
  - 移行: `sae harvest myteam` → `sae index myteam`、
    `sae harvest myteam --full` → `sae rebuild myteam`。
  - エラーメッセージの誘導文言（`Run \`sae harvest <team>\` first.`）も
    `Run \`sae index <team>\` first.` に更新。

## [0.2.0] - 2026-05-12

ADR-0060 agent-friendly CLI envelope (Phase 1 + 2.1 + 2.2).

### Added

- Global `--json` flag emits a stable envelope shape on stdout / stderr.
  - Success: `{"data": ..., "degraded": bool, "notes": [...]}`.
  - Error: `{"error": {"code": ..., "message": ..., "retryable": bool, ...}}`.
- `degraded=true` + `notes=["semantic search unavailable, falling back to FTS"]`
  surface when semantic search silently falls back to FTS because the embedder
  failed to load. `--no-embed` (intentional FTS) stays `degraded=false`.
- `dry_run` output is consistently envelope-wrapped regardless of `--json`.
- `tests/cli_integration.rs` spawns the real binary to verify envelope shape,
  exit codes, and stream routing end-to-end.

### Changed

- **BREAKING**: every `Sae::*` method now returns
  `Result<CommandOutput, SaeError>` instead of `Result<String, SaeError>`.
- `output::*` formatters no longer take `json: bool`; the renderer is decided
  in `main.rs` from the pre-scanned `--json` flag (so it survives clap parse
  failures and gates both success and error paths).
- `--help` / `--version` print the usual text and exit 0 even with `--json`
  set; the synthetic envelope is reserved for actual usage errors.
- Stdout writes survive `SIGPIPE` (`| head -0`, `| true`) without panicking
  (exit `SUCCESS` on `BrokenPipe`).
- JSON usage error messages carry only the first line of the clap blurb;
  the usage block and subcommand list no longer pollute `error.message`.

### Internal

- `SaeError::to_error_envelope()` bundles `error_code`, `next_step`,
  `candidates`, and `retryable` into a single struct for the JSON renderer,
  avoiding a `tools` ↔ `envelope` import cycle.
- `ErrorCode::exit_code()` delegates to `amici::cli::exit_code::codes::*`,
  single-sourcing the sysexits mapping with `T-EN002` as a regression net.

[Unreleased]: https://github.com/thkt/sae/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/thkt/sae/releases/tag/v0.2.0
