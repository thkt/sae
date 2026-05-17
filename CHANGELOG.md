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
- **BREAKING**: ADR-0066 Group 2 baseline に整列。`--json` の `error.code` と
  プロセス exit code が以下のように変化する（amici #102 で `DATA_ERROR` (65)
  baseline 追加を受けて、sae 側 routing を整理）。
  - `--after` / `--before` の日付形式不正 (`YYYY-MM-DD` ではない、または
    UTC 変換失敗): `code: "USAGE_ERROR"` (exit 64) → `code: "DATA_ERROR"`
    (exit 65)。
  - 分類不能な内部失敗 (`SaeError::Other`、分類されない catch-all):
    `code: "INTERNAL"` (exit 70) → `code: "UNKNOWN"` (exit 104)。
    分類済み `INTERNAL` (70) と区別する ADR-0066 L136 の意図に合わせる。
  - 公開 API: `SaeError::Input` を `SaeError::InputUsage` / `SaeError::InputData`
    に分割 (USAGE_ERROR と DATA_ERROR の責務分離)。`SaeError` は `lib.rs` から
    `pub use` されており、`Input` variant を pattern match で参照していた
    downstream crate はコンパイルエラーになる (現状 0 件)。同時に
    `#[non_exhaustive]` を付与し、以後の variant 追加は非破壊変更となる。
- **BREAKING**: `sae embed` で MLX backend が利用不能な場合の exit code が
  `104` (UNKNOWN) から `70` (INTERNAL) に変化 (#127 CHX-001)。
  `--json` の `error.code` も `"UNKNOWN"` → `"INTERNAL"`。
  `sae model download` 側 (`ModelDownloadError::BackendUnavailable` → 70) と
  routing を一致させ、agent が「ハードウェア / 環境セットアップ問題」シグナル
  として一貫して検知できるようにする。同様に `sae embed` / `sae search` の
  embedder バッチ件数不整合 (programmer-detectable invariant 違反) も `104`
  → `70` に変化。
  - **net 状態** (#126 catch-all 70 → 104 化との関係): #126 は分類不能な
    `SaeError::Other` を 70 → 104 に動かし、本変更 #127 は MLX backend 不在と
    embedder invariant 違反を **104 に流さず 70 のまま固定** する例外を作る。
    agent retry policy は `error.code` (= `INTERNAL` か `UNKNOWN`) で分岐すれば
    両変更を吸収できる。
  - 移行: backend missing / embedder invariant 違反で `UNKNOWN` を branch して
    いたスクリプト・agent retry policy は `INTERNAL` を見るように更新する。
- **BREAKING**: esa API が 404 (post not found) を返した場合の exit code が
  `70` (INTERNAL) から `65` (DATA_ERROR) に変化 (#136)。`--json` の
  `error.code` も `"INTERNAL"` → `"DATA_ERROR"`、`next_step` に「Verify the
  post number exists in esa, or run \`sae search <keyword>\` to find it.」
  ヒントが追加される。`sae get` / `sae update` / `sae archive` / `sae ship`
  に存在しない post 番号を渡したケースを、サーバ起因の 5xx と判別可能にする。
  - **net 状態** (#126 / #127 routing との関係): 404 (入力データ起因) は
    65 へ、404 以外の HTTP error (4xx 401/403/422 や 5xx) は引き続き 70。
    agent retry policy は `error.code` で分岐すれば、入力ミスは「該当 post
    の番号を確認」、5xx は「リトライ」と判別できる。
  - 移行: 「post 番号間違い → INTERNAL (70)」を branch していたスクリプト
    は `DATA_ERROR` (65) を見るように更新する。`error.code` で分岐していれば
    変更は自動吸収される。
- 公開 API: `ClientError::Api(String)` を `ClientError::Api { status: u16,
  body: String }` に変更し、HTTP status を保持できるようにする。あわせて
  URL parse / token format error 用の `ClientError::InvalidRequest(String)`
  variant を追加。`ClientError` には `#[non_exhaustive]` を付与し、以後の
  variant 追加は非破壊。`ClientError::Api` を pattern match で参照していた
  downstream crate (sae crate 内 0 件) はコンパイルエラーになる。

### Added

- 新 `SaeError::BackendUnavailable` variant (MLX backend 不在) と
  `SaeError::Internal(String)` variant (programmer-detectable invariant
  違反 — 例: embedder batch count mismatch)。両方とも `INTERNAL` (70) に
  routing し、anyhow-swallow `Other` (=UNKNOWN 104) と区別する (#127 CHX-001)。

### Fixed

- `sae index` の差分同期で `sync_state.total_count` が差分クエリ
  (`q=updated:>X`) のヒット件数で上書きされ、増分実行のたびに表示が
  `remote: 0 | local: N` のように退行していた。`total_count` は full 同期
  でのみ authoritative とし、増分時は prior_total と `local_count` を
  floor として採用するように修正。既存の汚染ステートは次回 `sae index`
  実行時に local_count まで自己治癒する。あわせて `posts_fetched == 0`
  時は `No updates. remote: M | local: N` を表示し、「差分ヒット 0 件」
  を「リモートに何もない」と誤解させないようにする。
- `sae rebuild` で `pagination_limit` を踏んで window を絞り込んだ場合、
  最後の narrowed レスポンスの `total_count` (実 remote より小さい値)
  で state を上書きしていたのも同じ修正で解消 (最初に観測した最大値を
  採用)。

### Internal

- 残存 `SaeError::Other` 構築サイト 5 箇所 (`tools.rs` の model probe
  fallback × 2、batch embedding error × 1、model cache check × 1、および
  `search.rs` の batch embedding error × 1) にインライン文書化を追加。
  「なぜ Other のままか」「上流の typed surface が必要」を明記し、将来の
  typed 化候補を可視化 (#127)。
- `[features] test-support` 下に hidden `__test_force_unknown` subcommand
  を追加 (`#[command(hide = true)]`)。`tests/cli_integration.rs::T-CI006`
  で UNKNOWN (104) envelope を hermetic に pin する (#127 OPS-005)。
  `cargo install` で配布される production binary には含まれない。
- `[features] test-support` 下に hidden `__test_force_client_api_404`
  subcommand を追加。`tests/cli_integration.rs::T-CI009` で esa-404 →
  DATA_ERROR (65) routing を binary-boundary で pin する (#136)。

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
