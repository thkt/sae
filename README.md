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
# チームの投稿をインデックス
sae harvest myteam

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

| Code | 名前       | 意味             | 例                                       | 推奨 retry  |
| ---- | ---------- | ---------------- | ---------------------------------------- | ----------- |
| 0    | SUCCESS    | 正常終了         |                                          |             |
| 64   | USAGE      | 入力エラー       | 不明なチーム、トークン未設定、未 harvest | しない      |
| 70   | SOFTWARE   | 内部エラー       | JSON parse 失敗、想定外の例外            | しない      |
| 73   | CANT_CREAT | データ層エラー   | DB open 失敗、ファイル作成不可           | 状況による  |
| 74   | IO_ERR     | I/O エラー       | ファイル読み書き失敗                     | 状況による  |
| 75   | TEMP_FAIL  | 一時的エラー     | esa API rate limit、ネットワーク障害     | する        |

> ⚠️ 破壊的変更（issue #91 / amici #34 migration）: 旧 schema (`1` / `2` / `4`) からの切り替え。スクリプト / CI で specific number に branch している場合は更新が必要。

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
