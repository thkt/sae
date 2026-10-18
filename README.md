# sae

esaのナレッジを手元で検索するCLI。FTS5 trigram + ベクトル検索のハイブリッド検索で、日本語の部分一致にも対応する。

## なぜ sae か

esaの検索は全文検索だが、ローカルで高速に検索したい場面がある。またAIエージェント（Claude Code等）がesaのナレッジを参照・操作するには、構造化出力と安全なプレビュー手段が必要になる。saeはこの両方を1つのCLIで提供する。

## セットアップ

初回ビルドにはRust 1.99.1、CMake、C/C++コンパイラが必要。macOSではXcode Command Line Toolsを導入する。Apple Siliconでは利用可能ならMetalを使用し、GPUが検出されない環境ではCPUで実行する。`rust-toolchain.toml`でローカルのRustを指定し、CIも同じ版で検証する。

```bash
git submodule update --init --recursive
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

# 検索（shorthand: 既知サブコマンド以外の単一 positional は `sae search` に展開され、stderr に `→ sae search ...` のヒントが出る）
sae "認証"

# オプション付き検索
sae search "認証" --team myteam --limit 5

# 期間フィルタ（YYYY-MM-DD、updated 基準）
sae search "認証" --after 2025-01-01 --before 2025-06-30

# stdin から query を読む（pipe / 明示 `-`）
echo "認証" | sae search
sae search - < query.txt

# 投稿を取得
sae get 42
```

`--team` を省略した場合は `default_team` → 設定済みチームが1つだけならそれ → どちらでもなければ `USAGE_ERROR` (64) の順で解決される。`index` / `rebuild` だけは team が必須 positional。

### 投稿の作成・更新

```bash
# 作成
sae create --name "タイトル" --body "本文" --team myteam

# ファイルから本文を読み込み
sae create --name "タイトル" --body-file draft.md

# stdin から読み込み
cat body.md | sae create --name "タイトル" --body-file -

# カテゴリ / タグ / WIP（`--tag` は繰り返し指定で複数）
sae create --name "タイトル" --body "本文" \
  --category "design/notes" --tag rfc --tag draft --wip

# 更新（create と同じ content flag が使える。`--tag` は既存タグを置き換え）
sae update 42 --name "新タイトル"

# アーカイブ / Ship
sae archive 42
sae ship 42
```

### エージェント向け機能

```bash
# JSON 出力（global flag、位置自由）
sae --json search "認証"
sae --json get 42
sae --json status

# get の JSON は body_md を省略（--with-body で含める。`--json` 無しでは無視）
sae --json get 42 --with-body

# dry-run（API を呼ばずにプレビュー）
sae create --name "タイトル" --body "本文" --dry-run
sae archive 42 --dry-run
```

`--json` 成功 envelope は `{"data": ..., "degraded": bool, "notes": [...]}`。embedder のロード失敗で FTS にフォールバックした場合は `degraded=true` + `notes=["semantic search unavailable, falling back to FTS"]` を返す。意図的な `--no-embed` は `degraded=false` のまま。

### セマンティック検索

```bash
# 埋め込みモデルをダウンロード（初回のみ）
sae model download

# モデル導入後は sae index / sae rebuild がチャンク生成と埋め込みをまとめて実行
# （未導入で未埋め込みチャンクが残ると EmbedderUnavailable (75) で失敗するため、
#  先に model download。すべて埋め込み済みならモデル無しでも index は成功）
sae index myteam

# 検索は読み取り専用（FTS + ベクトルのハイブリッド）。esa の最新を反映するには
# sae index を再実行する（search は自動で再取得・再埋め込みしない）
sae search "認証フローの設計方針"

# embedder のロードコストを避けて FTS のみで検索（CI / スクリプト用途）
sae search "認証" --no-embed

# 旧Ruri索引をローカルコピーからGemmaへ移行（esa API取得なし）
sae model migrate myteam

# 直前の索引へ戻す
sae model rollback myteam
```

埋め込みは **EmbeddingGemma 2のBF16版（768次元）** を使用し、rerankerは使わない。`SAE_RERANK`は無視する。制限付き環境でMetalデバイスが見えない場合も、同じモデルをCPUで検証・実行する。CPU実行はGPU実行より時間がかかる。モデルとtokenizerは固定版・SHA-256を検証して導入する。キャッシュは通常`~/.cache/sae/models/<identity>/`、`XDG_CACHE_HOME`指定時はその配下、`HF_HOME`指定時は`<HF_HOME>/sae/models/<identity>/`となる。`SAE_MODEL_DIR`で重みとtokenizerを置いたディレクトリを明示できる。

#### 旧索引からの移行

更新後は`sae model download`、`sae model migrate myteam`の順に実行する。既存のRuri索引は同じ768次元でもGemmaと互換性がない。旧索引への埋め込み追記やGemmaによるベクトル検索は拒否し、移行前も`--no-embed`によるFTS検索は使える。`index`/`rebuild`は旧索引を移行しない。

`model migrate`はSQLiteの整合したローカルコピーを別DBに作り、既存のMarkdownチャンクを埋め込み直す。512チャンクごとにベクトルと本文hashを保存し、中断後は同じコマンドで再開する。モデルmanifest、件数、本文hash、ベクトルの次元・有限値・単位長を検証してから参照先を原子的に切り替える。失敗時は元の索引を選択したままにする。処理中に元の投稿・チャンクが変わった場合も切り替えを拒否する。

元のDBとモデルは削除しない。`sae model rollback myteam`は直前のDBを再選択する。Ruri索引に戻した場合、Gemma版ではFTS検索のみ利用できる。`rollback`をもう一度実行すると切り替え前の選択へ戻る。索引への更新と移行・復帰は同時実行しない。

### 同期状態の確認

```bash
sae status                    # 全チーム
sae status --team myteam      # 単独
sae --json status
```

## データ

- 初期・旧DB: `~/.local/share/sae/{team}.db`（SQLite）
- 移行DB: `~/.local/share/sae/{team}.gemma-<manifest>-<snapshot>.db`
- 索引の選択: `~/.local/share/sae/{team}.active.json`（現行・直前のDB名）
- 設定: `~/.config/sae/config.json`

## Exit code

[sysexits.h](https://man.openbsd.org/sysexits.3) 慣例に準拠。LLM / shell script はこの数値で retry policy を判別できる。

名前列は `--json` の `error.code` 文字列 (sysexits 数値と分離した、agent 向けの安定名)。
sysexits.h の symbolic name (`EX_USAGE`, `EX_SOFTWARE` 等) は数値の出典であって `error.code` 値とは別。

| Code | error.code     | 意味             | 例                                                       | 推奨 retry  |
| ---- | -------------- | ---------------- | -------------------------------------------------------- | ----------- |
| 0    | (none)         | 正常終了         |                                                          |             |
| 64   | USAGE_ERROR    | 入力エラー       | 不明なチーム、トークン未設定、未 index 実行              | しない      |
| 65   | DATA_ERROR     | データ形式不正   | `--after` / `--before` の日付形式不正 (`YYYY-MM-DD` 以外)、esa API 404 (指定 post 番号がチームに存在しない)、埋め込み索引のモデル不一致 | しない      |
| 70   | INTERNAL       | 内部エラー       | JSON parse 失敗、esa API の 404 以外の HTTP エラー (5xx および 401/403/422 等の 4xx)、native backend 不在 (`sae index` / `sae model download` 共通)、model download 検証失敗 (`ModelDownloadError::ProbeFailed`)、Gemmaの検証・ロード失敗 (`SaeError::ModelLoadFailed`)、embedder invariant 違反 (例: batch count mismatch) | しない      |
| 73   | CANT_CREAT     | データ層エラー   | DB open 失敗、ファイル作成不可                           | 状況による  |
| 74   | IO_ERROR       | I/O エラー       | ファイル読み書き失敗     | 状況による  |
| 75   | TEMP_FAILURE   | 一時的エラー     | esa API rate limit、ネットワーク障害、model download 一時的失敗 (429, 5xx, timeout)、embedding モデル未インストールで `sae index` / `rebuild` が埋め込み不能 (`EmbedderUnavailable`、`sae model download` で解消) | する  |
| 104  | UNKNOWN        | 分類不能         | 未型化のランタイムエラー (`anyhow::Error` 経由)、その他の想定外例外 | しない |

## ログ

`tracing` ベース。`RUST_LOG` で制御する。

| 環境 | filter | 例 |
|---|---|---|
| 未設定（default） | `sae=info` | saeの通常ログと検索フォールバックの警告 |
| 任意指定 | `RUST_LOG=<directives>` | env var の値が完全に default を上書き |

モデル欠損・破損、隔離したモデルプローブの失敗、ベクトル検索の失敗は警告をstderrに出す。検索のFTSフォールバックはJSONの`degraded`と`notes`でも確認できる。

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
