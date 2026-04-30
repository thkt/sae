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

| Code | 意味               | 例                                       |
| ---- | ------------------ | ---------------------------------------- |
| 0    | 正常終了           |                                          |
| 1    | Operational エラー | API エラー、ネットワーク障害             |
| 2    | Input エラー       | 不明なチーム、トークン未設定、未 harvest |
| 4    | Internal エラー    | DB 破損、JSON シリアライズ失敗           |

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
