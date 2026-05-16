# sae

esaのナレッジを手元で検索するCLI。FTS5 trigram + ベクトル検索のハイブリッド検索で、日本語の部分一致にも対応する。

## なぜ sae か

esaの検索は全文検索だが、ローカルで高速に検索したい場面がある。またAIエージェント（Claude Code等）がesaのナレッジを参照・操作するには、構造化出力と安全なプレビュー手段が必要になる。saeはこの両方を1つのCLIで提供する。

## セットアップ

```bash
cargo install --path .
```

### 環境変数

```bash
export ESA_ACCESS_TOKEN="your-token"
```

### 設定ファイル（任意）

`~/.config/sae/config.json`:

```json
{
  "teams": ["myteam"],
  "default_team": "myteam"
}
```

## 使い方

### 投稿の取得・検索

```bash
# チームの投稿をインデックス（増分）
sae index myteam

# 全件再構築（差分追従が破綻したときなど）
sae rebuild myteam

# 検索（shorthand）
sae "認証"

# オプション付き検索
sae search "認証" --team myteam --limit 5

# 投稿を取得
sae get 42
```

### 投稿の作成・更新

```bash
# 作成
sae create --name "タイトル" --body "本文" --team myteam

# ファイルから本文を読み込み
sae create --name "タイトル" --body-file draft.md

# stdin から読み込み
cat body.md | sae create --name "タイトル" --body-file -

# 更新
sae update 42 --name "新タイトル"

# アーカイブ / Ship
sae archive 42
sae ship 42
```

### エージェント向け機能

```bash
# JSON 出力（全コマンド共通）
sae --json search "認証"
sae --json get 42
sae --json status

# get の JSON は body_md を省略（--with-body で含める）
sae --json get 42 --with-body

# dry-run（API を呼ばずにプレビュー）
sae create --name "タイトル" --body "本文" --dry-run
sae archive 42 --dry-run
```

### セマンティック検索

```bash
# 埋め込みモデルをダウンロード（初回のみ）
sae model download

# チャンクの埋め込み
sae embed myteam

# 以降の検索は FTS + ベクトルのハイブリッド
sae search "認証フローの設計方針"

# embedder のロードコストを避けて FTS のみで検索（CI / スクリプト用途）
sae search "認証" --no-embed
```

### 同期状態の確認

```bash
sae status
sae --json status
```

## データ

- DB: `~/.local/share/sae/{team}.db`（SQLite）
- 設定: `~/.config/sae/config.json`

## Exit code

[sysexits.h](https://man.openbsd.org/sysexits.3) 慣例に準拠（amici #34 で全 CLI 共通化）。LLM / shell script はこの数値で retry policy を判別できる。

名前列は `--json` の `error.code` 文字列 (sysexits 数値と分離した、agent 向けの安定名)。
sysexits.h の symbolic name (`EX_USAGE`, `EX_SOFTWARE` 等) は数値の出典であって `error.code` 値とは別。

| Code | error.code     | 意味             | 例                                                       | 推奨 retry  |
| ---- | -------------- | ---------------- | -------------------------------------------------------- | ----------- |
| 0    | (none)         | 正常終了         |                                                          |             |
| 64   | USAGE_ERROR    | 入力エラー       | 不明なチーム、トークン未設定、未 index 実行              | しない      |
| 65   | DATA_ERROR     | データ形式不正   | `--after` / `--before` の日付形式不正 (`YYYY-MM-DD` 以外) | しない      |
| 70   | INTERNAL       | 内部エラー       | JSON parse 失敗、HTTP 4xx (API)、MLX backend 不在 (`sae embed` / `sae model download` 共通)、model download 検証失敗 (`ModelDownloadError::ProbeFailed`)、`ProbeError` (HandlerNotInstalled / ModelLoadFailed / SetupRejected)、embedder invariant 違反 (例: batch count mismatch) | しない      |
| 73   | CANT_CREAT     | データ層エラー   | DB open 失敗、ファイル作成不可                           | 状況による  |
| 74   | IO_ERROR       | I/O エラー       | ファイル読み書き失敗、`ProbeError::SubprocessFailed`     | 状況による  |
| 75   | TEMP_FAILURE   | 一時的エラー     | esa API rate limit、ネットワーク障害、model download 一時的失敗 (429, 5xx, timeout) | する  |
| 104  | UNKNOWN        | 分類不能         | model probe fallback、未型化の embedder ランタイムエラー (`anyhow::Error` 経由)、その他の想定外例外。分類済み 70 と区別する ADR-0066 L136 の意図に対応 | しない |

> ⚠️ 破壊的変更（issue #91 / amici #34 migration）: 旧 schema (`1` / `2` / `4`) からの切り替え。スクリプト / CI で specific number に branch している場合は更新が必要。

> ⚠️ 破壊的変更（issue #126 / ADR-0066 Group 2 baseline）: 日付形式不正が 64 → 65、catch-all (`SaeError::Other`) が 70 → 104 に変化。`--json` の `error.code` も同様に `USAGE_ERROR` → `DATA_ERROR`、`INTERNAL` → `UNKNOWN`。

> ⚠️ 破壊的変更（issue #127 / `SaeError::Other` routing audit）: #126 で catch-all が 70 → 104 に動いた一方、以下の 2 種の失敗は **70 (INTERNAL) に分類される** — `sae embed` / `sae search` での MLX backend 不在 (`SaeError::BackendUnavailable`)、`sae embed` / `sae search` での embedder invariant 違反 (`SaeError::Internal`、例: batch count mismatch)。`sae model download` 側 (既に 70) と routing を揃え、agent が「ハードウェア / 環境不一致 or プログラム不変条件違反」シグナルとして検知できるようにする。**net 状態**: agent retry policy は `error.code` (= `"INTERNAL"` か `"UNKNOWN"`) で分岐すれば #126 / #127 双方の変更を吸収できる。

## ログ

`tracing` ベース。`RUST_LOG` で制御する。

| 環境 | filter | 例 |
|---|---|---|
| 未設定（default） | `sae=info,rurico=warn` | sae の通常ログ + rurico の degraded path warn |
| 任意指定 | `RUST_LOG=<directives>` | env var の値が完全に default を上書き |

degraded path（embedder/reranker fallback、MLX cache 復旧、probe timeout 等）は rurico 側で `tracing::warn!` を発行する。default filter で `rurico=warn` を含めることで operator が観測可能。

> ⚠️ 破壊的変更（issue #85 / amici #36 migration）: 旧挙動「`RUST_LOG` 設定時も常に `sae=info` を layer」を廃止。`RUST_LOG` 設定時は env 優先、未設定時のみ default を使う。`RUST_LOG=sae=info` を export する運用へ移行が必要な場合は明示指定。

## 開発

### セットアップ

clone 後に1度だけ実行:

```sh
git config --local core.hooksPath .githooks
```

`cargo fmt --check` と `cargo clippy --all-targets --all-features -- -D warnings` を commit 前に走らせる pre-commit hook が有効になる。違反があると commit は中止される。1コミットだけスキップしたいときは `git commit --no-verify`。

### よく使うコマンド

```sh
cargo test                                                # 全テスト
cargo clippy --all-targets --all-features -- -D warnings  # lint（CI と同じ）
cargo fmt -- --check                                      # フォーマット確認
```
