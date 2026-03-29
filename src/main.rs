mod output;

use clap::{Parser, Subcommand};
use rurico::embed::{Embed, Embedder};
use sae::client::EsaClient;
use sae::config::Config;

#[derive(Parser)]
#[command(name = "sae", about = "esa semantic search CLI")]
struct Cli {
    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fetch and index esa posts
    #[command(after_help = "\
Examples:
  sae harvest myteam
  sae harvest myteam --full")]
    Harvest {
        /// Team name
        team: String,
        /// Re-fetch all posts (ignore sync state)
        #[arg(long)]
        full: bool,
    },
    /// Semantic search over indexed posts
    #[command(after_help = "\
Examples:
  sae search \"認証\" --team myteam --limit 5
  sae \"認証\"")]
    Search {
        /// Search query
        query: String,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Max results (1-100)
        #[arg(long, default_value = "10")]
        limit: u32,
    },
    /// Get a post by number
    #[command(after_help = "\
Examples:
  sae get 42 --team myteam
  sae --json get 42
  sae --json get 42 --with-body")]
    Get {
        /// Post number
        number: u32,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Include body_md in JSON output (ignored without --json)
        #[arg(long)]
        with_body: bool,
    },
    /// Create a new post
    #[command(after_help = "\
Examples:
  sae create --name \"Title\" --body \"Content\" --team myteam
  sae create --name \"Title\" --body-file draft.md
  cat body.md | sae create --name \"Title\" --body-file -
  sae create --name \"Title\" --dry-run")]
    Create {
        /// Post title
        #[arg(long)]
        name: String,
        /// Post body (Markdown)
        #[arg(long, conflicts_with = "body_file")]
        body: Option<String>,
        /// Read body from file (use "-" for stdin)
        #[arg(long, conflicts_with = "body")]
        body_file: Option<String>,
        /// Category path
        #[arg(long)]
        category: Option<String>,
        /// Tags
        #[arg(long)]
        tag: Vec<String>,
        /// Mark as WIP
        #[arg(long)]
        wip: bool,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Preview without creating (no mutation API calls)
        #[arg(long)]
        dry_run: bool,
    },
    /// Update a post
    #[command(after_help = "\
Examples:
  sae update 42 --name \"New Title\" --team myteam
  sae update 42 --body-file updated.md
  sae update 42 --name \"New Title\" --dry-run")]
    Update {
        /// Post number
        number: u32,
        /// New title
        #[arg(long)]
        name: Option<String>,
        /// New body (Markdown)
        #[arg(long, conflicts_with = "body_file")]
        body: Option<String>,
        /// Read body from file (use "-" for stdin)
        #[arg(long, conflicts_with = "body")]
        body_file: Option<String>,
        /// New category path
        #[arg(long)]
        category: Option<String>,
        /// New tags (replaces existing)
        #[arg(long)]
        tag: Vec<String>,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Preview without updating (no mutation API calls)
        #[arg(long)]
        dry_run: bool,
    },
    /// Archive a post
    #[command(after_help = "\
Examples:
  sae archive 42 --team myteam
  sae archive 42 --dry-run")]
    Archive {
        /// Post number
        number: u32,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Preview without archiving (reads current post, no mutation)
        #[arg(long)]
        dry_run: bool,
    },
    /// Ship a WIP post (set wip=false)
    #[command(after_help = "\
Examples:
  sae ship 42 --team myteam
  sae ship 42 --dry-run")]
    Ship {
        /// Post number
        number: u32,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Preview without shipping (no mutation API calls)
        #[arg(long)]
        dry_run: bool,
    },
    /// Embed all chunks (model must be downloaded first)
    #[command(after_help = "\
Examples:
  sae embed myteam
  SAE_AUTO_DOWNLOAD_MODEL=1 sae embed myteam")]
    Embed {
        /// Team name
        team: String,
    },
    /// Show sync status
    #[command(after_help = "\
Examples:
  sae status
  sae status --team myteam
  sae --json status")]
    Status {
        /// Team name (omit to show all teams)
        #[arg(long)]
        team: Option<String>,
    },
    /// Manage embedding model
    #[command(subcommand_required = true, arg_required_else_help = true, after_help = "\
Examples:
  sae model download")]
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download embedding model from Hugging Face Hub
    Download,
}

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "harvest", "search", "get", "create", "update", "archive", "ship", "embed", "status", "model",
];

const GLOBAL_FLAGS: &[&str] = &["--json"];

fn parse_cli_args<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let args: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();

    // Shorthand: `sae "query"` or `sae --json "query"` → `sae [flags] search "query"`
    // Strip global flags, check if remaining is [binary, query], then re-inject.
    let (flags, rest): (Vec<_>, Vec<_>) = args
        .iter()
        .enumerate()
        .partition(|(i, a)| *i > 0 && a.to_str().is_some_and(|s| GLOBAL_FLAGS.contains(&s)));
    let rest: Vec<&std::ffi::OsString> = rest.into_iter().map(|(_, a)| a).collect();

    if rest.len() == 2
        && let Some(first_arg) = rest[1].to_str()
        && !first_arg.starts_with('-')
        && !KNOWN_SUBCOMMANDS.contains(&first_arg)
    {
        let mut expanded: Vec<std::ffi::OsString> = vec![rest[0].clone()];
        for (_, f) in &flags {
            expanded.push((*f).clone());
        }
        expanded.push("search".into());
        expanded.push(rest[1].clone());
        return Cli::try_parse_from(expanded);
    }

    Cli::try_parse_from(args)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("sae=info".parse()?),
        )
        .init();

    let cli = parse_cli_args(std::env::args_os()).unwrap_or_else(|e| e.exit());
    let config = Config::load()?;
    let json = cli.json;

    match cli.command {
        Command::Harvest { team, full } => {
            let team = config.resolve_team(Some(&team))?;
            let client = EsaClient::from_env()?;
            let db_path = config.team_db_path(team)?;
            let db = sae::storage::Db::open(&db_path)?;
            let result = sae::sync::harvest(&client, &db, team, full).await?;
            output::harvest(&result, json)?;
        }
        Command::Search { query, team, limit } => {
            let team = config.resolve_team(team.as_deref())?;
            let db_path = config.team_db_path(team)?;
            require_db(&db_path, team);
            let db = sae::storage::Db::open(&db_path)?;
            let embedder = try_load_embedder();
            let query_embedding = embedder.as_ref().and_then(|e| match e.embed_query(&query) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("Warning: embed_query failed: {e}");
                    None
                }
            });
            let results = sae::storage::hybrid_search(
                db.conn(),
                &query,
                query_embedding.as_deref(),
                limit,
                chrono::Utc::now(),
            )?;
            output::search(&results, &query, json)?;
        }
        Command::Get {
            number,
            team,
            with_body,
        } => {
            let team = config.resolve_team(team.as_deref())?;
            let client = EsaClient::from_env()?;
            let post = client.get_post(team, number).await?;
            output::get(&post, json, with_body)?;
        }
        Command::Create {
            name,
            body,
            body_file,
            category,
            tag,
            wip,
            team,
            dry_run,
        } => {
            let resolved_body = resolve_body(body.as_deref(), body_file.as_deref())?;
            if dry_run {
                let payload = serde_json::json!({
                    "name": name,
                    "body_md": resolved_body,
                    "category": category,
                    "tags": tag,
                    "wip": wip,
                });
                println!("{}", serde_json::to_string(&payload)?);
            } else {
                let team = config.resolve_team(team.as_deref())?;
                let client = EsaClient::from_env()?;
                let post = client
                    .create_post(
                        team,
                        &name,
                        resolved_body.as_deref(),
                        category.as_deref(),
                        tag,
                        wip,
                    )
                    .await?;
                output::post("Created", &post, json)?;
            }
        }
        Command::Update {
            number,
            name,
            body,
            body_file,
            category,
            tag,
            team,
            dry_run,
        } => {
            let resolved_body = resolve_body(body.as_deref(), body_file.as_deref())?;
            if dry_run {
                let tags = if tag.is_empty() { None } else { Some(&tag) };
                let payload = serde_json::json!({
                    "number": number,
                    "name": name,
                    "body_md": resolved_body,
                    "category": category,
                    "tags": tags,
                });
                println!("{}", serde_json::to_string(&payload)?);
            } else {
                let team = config.resolve_team(team.as_deref())?;
                let client = EsaClient::from_env()?;
                let tags = if tag.is_empty() { None } else { Some(tag) };
                let post = client
                    .update_post(
                        team,
                        number,
                        name.as_deref(),
                        resolved_body.as_deref(),
                        category.as_deref(),
                        tags,
                        None,
                    )
                    .await?;
                output::post("Updated", &post, json)?;
            }
        }
        Command::Embed { team } => {
            let team = config.resolve_team(Some(&team))?;
            let db_path = config.team_db_path(team)?;
            require_db(&db_path, team);
            let db = sae::storage::Db::open(&db_path)?;

            let paths = require_embed_model()?;
            eprintln!("Loading model...");
            let embedder =
                Embedder::new(&paths).map_err(|e| format!("Failed to load model: {e}"))?;
            eprintln!("Model ready");

            const BATCH_SIZE: u32 = 500;
            let mut total_added = 0u32;
            let mut done = 0u32;
            loop {
                let batch = sae::storage::get_unembedded_chunks(db.conn(), BATCH_SIZE)?;
                if batch.is_empty() {
                    break;
                }
                if done == 0 {
                    eprintln!("Embedding chunks...");
                }
                let texts: Vec<&str> = batch.iter().map(|(_, content)| content.as_str()).collect();
                let batch_len = batch.len() as u32;
                match embedder.embed_documents_batch(&texts) {
                    Ok(embs) => {
                        let embeddings: Vec<(i64, Vec<f32>)> =
                            batch.iter().map(|(id, _)| *id).zip(embs).collect();
                        total_added += sae::storage::add_embeddings(db.conn(), &embeddings)?;
                    }
                    Err(e) => {
                        eprintln!("Error: batch embedding failed: {e}. Aborting.");
                        std::process::exit(1);
                    }
                }
                done += batch_len;
                eprintln!("  {done} chunks processed");
            }
            let result = sae::storage::EmbedResult {
                chunks_embedded: total_added,
            };
            output::embed(&result, done, json)?;
        }
        Command::Archive {
            number,
            team,
            dry_run,
        } => {
            let team = config.resolve_team(team.as_deref())?;
            let client = EsaClient::from_env()?;
            let post = client.get_post(team, number).await?;
            let current_category = post.category.as_deref().unwrap_or("");
            if current_category.starts_with("Archived/") || current_category == "Archived" {
                if dry_run {
                    let payload = serde_json::json!({
                        "number": number,
                        "already_archived": true,
                        "category": current_category,
                    });
                    println!("{}", serde_json::to_string(&payload)?);
                } else {
                    output::post("Already archived", &post, json)?;
                }
            } else {
                let archived_category = if current_category.is_empty() {
                    "Archived".to_string()
                } else {
                    format!("Archived/{current_category}")
                };
                if dry_run {
                    let payload = serde_json::json!({
                        "number": number,
                        "from_category": current_category,
                        "to_category": archived_category,
                    });
                    println!("{}", serde_json::to_string(&payload)?);
                } else {
                    let post = client
                        .update_post(
                            team,
                            number,
                            None,
                            None,
                            Some(&archived_category),
                            None,
                            None,
                        )
                        .await?;
                    output::post("Archived", &post, json)?;
                }
            }
        }
        Command::Ship {
            number,
            team,
            dry_run,
        } => {
            if dry_run {
                let payload = serde_json::json!({
                    "number": number,
                    "wip": false,
                });
                println!("{}", serde_json::to_string(&payload)?);
            } else {
                let team = config.resolve_team(team.as_deref())?;
                let client = EsaClient::from_env()?;
                let post = client
                    .update_post(team, number, None, None, None, None, Some(false))
                    .await?;
                output::post("Shipped", &post, json)?;
            }
        }
        Command::Status { team } => {
            let teams: Vec<&str> = if let Some(ref t) = team {
                vec![config.resolve_team(Some(t))?]
            } else {
                config.teams.iter().map(String::as_str).collect()
            };
            let statuses = collect_team_statuses(&config, &teams)?;
            output::status(&statuses, json)?;
        }
        Command::Model { command } => match command {
            ModelCommand::Download => {
                eprintln!("Downloading model...");
                let paths = rurico::embed::download_model()
                    .map_err(|e| format!("Failed to download model: {e}"))?;
                // Verify model files are loadable (catches corrupt downloads)
                let _embedder = Embedder::new(&paths)
                    .map_err(|e| format!("Failed to verify model: {e}"))?;
                output::model_download(json)?;
            }
        },
    }

    Ok(())
}

fn collect_team_statuses(
    config: &Config,
    teams: &[&str],
) -> Result<Vec<sae::storage::TeamStatus>, Box<dyn std::error::Error>> {
    let mut statuses = Vec::new();
    for t in teams {
        let ts = match config.team_db_path(t) {
            Ok(path) if path.exists() => match query_team_status(t, &path) {
                Ok(ts) => ts,
                Err(e) => sae::storage::TeamStatus {
                    team: t.to_string(),
                    status: sae::storage::SyncStatus::Error,
                    posts: 0,
                    sync_state: None,
                    error: Some(e.to_string()),
                    db_path: None,
                },
            },
            Ok(path) => sae::storage::TeamStatus {
                team: t.to_string(),
                status: sae::storage::SyncStatus::NotSynced,
                posts: 0,
                sync_state: None,
                error: None,
                db_path: Some(path.display().to_string()),
            },
            Err(e) => sae::storage::TeamStatus {
                team: t.to_string(),
                status: sae::storage::SyncStatus::Error,
                posts: 0,
                sync_state: None,
                error: Some(e.to_string()),
                db_path: None,
            },
        };
        statuses.push(ts);
    }
    Ok(statuses)
}

fn require_db(db_path: &std::path::Path, team: &str) {
    if !db_path.exists() {
        eprintln!("No data for team '{team}'. Run `sae harvest {team}` first.");
        std::process::exit(1);
    }
}

fn query_team_status(
    team: &str,
    path: &std::path::Path,
) -> Result<sae::storage::TeamStatus, Box<dyn std::error::Error>> {
    let db = sae::storage::Db::open(path)?;
    let count = sae::storage::count_posts(db.conn())?;
    let state = sae::storage::get_sync_state(db.conn())?;
    Ok(sae::storage::TeamStatus {
        team: team.to_string(),
        status: if state.is_some() {
            sae::storage::SyncStatus::Synced
        } else {
            sae::storage::SyncStatus::NotSynced
        },
        posts: count,
        sync_state: state,
        error: None,
        db_path: None,
    })
}

fn resolve_body(
    body: Option<&str>,
    body_file: Option<&str>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    match (body, body_file) {
        (Some(b), None) => Ok(Some(b.to_string())),
        (None, Some("-")) => {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            Ok(Some(buf))
        }
        (None, Some(path)) => {
            let content = std::fs::read_to_string(path)?;
            Ok(Some(content))
        }
        (None, None) => Ok(None),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with prevents this"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-029: shorthand `sae "認証"` → search
    #[test]
    fn shorthand_single_query_becomes_search() {
        let cli = parse_cli_args(["sae", "認証"]).unwrap();
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query, "認証"),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-030: explicit `sae search "認証"` is not double-injected
    #[test]
    fn explicit_search_not_double_injected() {
        let cli = parse_cli_args(["sae", "search", "認証"]).unwrap();
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query, "認証"),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-031: `sae harvest team` stays as harvest
    #[test]
    fn known_subcommand_not_shorthand() {
        let cli = parse_cli_args(["sae", "harvest", "myteam"]).unwrap();
        match cli.command {
            Command::Harvest { team, .. } => assert_eq!(team, "myteam"),
            other => panic!("expected Harvest, got {other:?}"),
        }
    }

    // T-033: `sae search` (missing required arg) → clap error via helper
    #[test]
    fn search_missing_arg_is_clap_error() {
        let result = parse_cli_args(["sae", "search"]);
        assert!(result.is_err(), "search without query should be clap error");
    }

    // T-034: `sae harvest` (missing required arg) → clap error via helper
    #[test]
    fn harvest_missing_arg_is_clap_error() {
        let result = parse_cli_args(["sae", "harvest"]);
        assert!(result.is_err(), "harvest without team should be clap error");
    }

    // T-037: parse_cli_args(["sae", "--json", "query"]) → Command::Search + json=true
    #[test]
    fn shorthand_with_json_flag_becomes_search_with_json() {
        let cli = parse_cli_args(["sae", "--json", "query"]).unwrap();
        assert!(cli.json, "[T-037] global json flag should be true");
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query, "query"),
            other => panic!("[T-037] expected Search, got {other:?}"),
        }
    }

    // T-049: parse_cli_args(["sae", "query"]) → Command::Search (json=false) - regression
    #[test]
    fn shorthand_without_flags_has_json_false() {
        let cli = parse_cli_args(["sae", "query"]).unwrap();
        assert!(!cli.json, "[T-049] json should default to false");
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query, "query"),
            other => panic!("[T-049] expected Search, got {other:?}"),
        }
    }

    // --- Phase 2: --dry-run, --body-file (FR-004, FR-005, FR-006, FR-007) ---

    // T-038: create args + --dry-run → parse succeeds with dry_run=true
    #[test]
    fn create_with_dry_run_parses_dry_run_flag() {
        let cli = parse_cli_args(["sae", "create", "--name", "Test Post", "--dry-run"]).unwrap();
        match cli.command {
            Command::Create { name, dry_run, .. } => {
                assert_eq!(name, "Test Post", "[T-038] name should match");
                assert!(dry_run, "[T-038] dry_run should be true");
            }
            other => panic!("[T-038] expected Create, got {other:?}"),
        }
    }

    // T-039: --body-file <tempfile> with create → file content becomes body
    #[test]
    fn body_file_reads_from_file() {
        let dir = std::env::temp_dir().join("sae_test_t039");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("body.md");
        std::fs::write(&file_path, "# Hello\nBody from file").unwrap();

        let result = resolve_body(None, Some(file_path.to_str().unwrap())).unwrap();
        assert_eq!(
            result.as_deref(),
            Some("# Hello\nBody from file"),
            "[T-039] body should contain file contents"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // T-040: --body-file - with create → stdin content becomes body
    // Note: stdin mocking is non-trivial; test that resolve_body("-") triggers
    // the stdin path. The function should accept a Read trait for testability.
    // For now, test the parse side: --body-file "-" parses correctly.
    #[test]
    fn body_file_dash_parses_as_stdin() {
        let cli =
            parse_cli_args(["sae", "create", "--name", "Stdin Post", "--body-file", "-"]).unwrap();
        match cli.command {
            Command::Create { body_file, .. } => {
                assert_eq!(
                    body_file.as_deref(),
                    Some("-"),
                    "[T-040] body_file should be \"-\" for stdin"
                );
            }
            other => panic!("[T-040] expected Create, got {other:?}"),
        }
    }

    // T-041: --body and --body-file both specified → clap error
    #[test]
    fn body_and_body_file_conflict_is_clap_error() {
        let result = parse_cli_args([
            "sae",
            "create",
            "--name",
            "Conflict Post",
            "--body",
            "inline body",
            "--body-file",
            "some_file.md",
        ]);
        assert!(
            result.is_err(),
            "[T-041] --body and --body-file together should be a clap error"
        );
    }

    // T-046: archive args + --dry-run → parse succeeds with dry_run=true
    #[test]
    fn archive_with_dry_run_parses_dry_run_flag() {
        let cli = parse_cli_args(["sae", "archive", "42", "--dry-run"]).unwrap();
        match cli.command {
            Command::Archive {
                number, dry_run, ..
            } => {
                assert_eq!(number, 42, "[T-046] post number should be 42");
                assert!(dry_run, "[T-046] dry_run should be true");
            }
            other => panic!("[T-046] expected Archive, got {other:?}"),
        }
    }

    // T-042: help output contains Examples section
    #[test]
    fn help_output_contains_examples() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        for sub in cmd.get_subcommands() {
            let after_help = sub
                .get_after_help()
                .map(|h| h.to_string())
                .unwrap_or_default();
            assert!(
                after_help.contains("Examples"),
                "[T-042] subcommand '{}' should have Examples in after_help",
                sub.get_name()
            );
        }
    }

    // TC-008: resolve_body(Some, None) → inline body
    #[test]
    fn resolve_body_inline_text() {
        let result = resolve_body(Some("inline text"), None).unwrap();
        assert_eq!(result.as_deref(), Some("inline text"));
    }

    // TC-008: resolve_body(None, None) → no body
    #[test]
    fn resolve_body_none_returns_none() {
        let result = resolve_body(None, None).unwrap();
        assert_eq!(result, None);
    }

    // TC-009: resolve_body with nonexistent file → error
    #[test]
    fn resolve_body_nonexistent_file_is_error() {
        let result = resolve_body(None, Some("/nonexistent/path.md"));
        assert!(
            result.is_err(),
            "[TC-009] nonexistent file should return error"
        );
    }

    // TC-011: flag-like argument not treated as shorthand query
    #[test]
    fn flag_like_arg_not_shorthand() {
        let result = parse_cli_args(["sae", "--unknown"]);
        assert!(
            result.is_err(),
            "[TC-011] --unknown should be clap error, not search shorthand"
        );
    }

    // --- model subcommand (FR-001, FR-006, FR-007) ---

    // T-001: `sae model download` parses to Command::Model { ModelCommand::Download }
    #[test]
    fn model_download_parses_correctly() {
        let cli = parse_cli_args(["sae", "model", "download"]).unwrap();
        match cli.command {
            Command::Model { command } => match command {
                ModelCommand::Download => {} // pass
            },
            other => panic!("[T-001] expected Model {{ Download }}, got {other:?}"),
        }
    }

    // T-002: `sae model` without subcommand → clap error (arg_required_else_help)
    #[test]
    fn model_without_subcommand_is_clap_error() {
        let result = parse_cli_args(["sae", "model"]);
        assert!(
            result.is_err(),
            "[T-002] model without subcommand should be clap error"
        );
    }

    // T-003: `sae model` is in KNOWN_SUBCOMMANDS, not rewritten to search shorthand
    #[test]
    fn model_is_known_subcommand_not_shorthand() {
        assert!(
            KNOWN_SUBCOMMANDS.contains(&"model"),
            "[T-003] KNOWN_SUBCOMMANDS must contain \"model\""
        );
        // Also verify parse does not rewrite to search
        let result = parse_cli_args(["sae", "model"]);
        // Should be a clap error (missing subcommand), NOT a search for "model"
        assert!(
            result.is_err(),
            "[T-003] sae model should not become search shorthand"
        );
    }

    // T-004: `sae model download --json` → json=true + ModelCommand::Download
    #[test]
    fn model_download_with_json_flag() {
        let cli = parse_cli_args(["sae", "--json", "model", "download"]).unwrap();
        assert!(cli.json, "[T-004] global json flag should be true");
        match cli.command {
            Command::Model { command } => match command {
                ModelCommand::Download => {} // pass
            },
            other => panic!("[T-004] expected Model {{ Download }}, got {other:?}"),
        }
    }

    // T-005: `sae embed myteam` still parses as Command::Embed (regression)
    #[test]
    fn embed_still_parses_after_model_addition() {
        let cli = parse_cli_args(["sae", "embed", "myteam"]).unwrap();
        match cli.command {
            Command::Embed { team } => {
                assert_eq!(team, "myteam", "[T-005] team should be myteam");
            }
            other => panic!("[T-005] expected Embed, got {other:?}"),
        }
    }
}

fn require_embed_model() -> Result<rurico::embed::ModelPaths, Box<dyn std::error::Error>> {
    let auto_download = std::env::var("SAE_AUTO_DOWNLOAD_MODEL").as_deref() == Ok("1");
    if auto_download {
        eprintln!("Downloading model (SAE_AUTO_DOWNLOAD_MODEL=1)...");
        return rurico::embed::download_model()
            .map_err(|e| format!("Failed to download model: {e}").into());
    }
    match rurico::embed::model_paths_if_cached() {
        Ok(Some(p)) => Ok(p),
        Ok(None) => Err("Model not found. Run 'sae model download' first.".into()),
        Err(e) => Err(format!("Failed to check model cache: {e}").into()),
    }
}

fn try_load_embedder() -> Option<Embedder> {
    let paths = match rurico::embed::model_paths_if_cached() {
        Ok(Some(p)) => p,
        Ok(None) => {
            eprintln!(
                "Hint: run 'sae model download && sae embed <team>' to enable semantic search"
            );
            return None;
        }
        Err(e) => {
            tracing::warn!(error = %e, "embedding model not available");
            return None;
        }
    };
    match Embedder::new(&paths) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("Warning: failed to load embedding model: {e}");
            None
        }
    }
}
