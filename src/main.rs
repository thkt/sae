#![warn(unreachable_pub)]

mod commands;
mod output;
mod progress;
mod shorthand;

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use sae::client::EsaClient;
use sae::config::Config;

#[derive(Parser)]
#[command(name = "sae", about = "esa semantic search CLI")]
struct Cli {
    /// Output as JSON
    // NOTE: parse_cli_args の shorthand 判定は long flag のみ対応。
    // short global flag を追加する場合は get_short() 対応が必要。
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
  sae search \"認証\" --after 2025-01-01
  sae search \"認証\" --after 2025-01-01 --before 2025-06-30
  echo \"認証\" | sae search
  sae search -")]
    Search {
        /// Search query. Reads piped stdin when omitted, or any stdin with `-`.
        query: Option<String>,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Max results (1-100)
        #[arg(long, default_value = "10")]
        limit: u32,
        /// Filter: updated on or after this date (YYYY-MM-DD)
        #[arg(long, value_name = "DATE")]
        after: Option<String>,
        /// Filter: updated on or before this date (YYYY-MM-DD)
        #[arg(long, value_name = "DATE")]
        before: Option<String>,
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
        #[command(flatten)]
        args: commands::post::CreateArgs,
    },
    /// Update a post
    #[command(after_help = "\
Examples:
  sae update 42 --name \"New Title\" --team myteam
  sae update 42 --body-file updated.md
  sae update 42 --name \"New Title\" --dry-run")]
    Update {
        #[command(flatten)]
        args: commands::post::UpdateArgs,
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
    #[command(
        subcommand_required = true,
        arg_required_else_help = true,
        after_help = "\
Examples:
  sae model download"
    )]
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
    let expanded = shorthand::try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS);
    if let Some(expanded) = expanded
        && let Ok(cli) = Cli::try_parse_from(&expanded)
    {
        let display: Vec<_> = std::iter::once("sae")
            .chain(expanded[1..].iter().filter_map(|a| a.to_str()))
            .collect();
        eprintln!("→ {}", display.join(" "));
        return Ok(cli);
    }
    Cli::try_parse_from(args)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SaeError {
    #[error(transparent)]
    Config(#[from] sae::config::ConfigError),
    #[error(transparent)]
    Client(#[from] sae::client::ClientError),
    #[error(transparent)]
    Storage(#[from] sae::storage::StorageError),
    #[error(transparent)]
    Sync(#[from] sae::sync::SyncError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// User action required (e.g., run harvest or model download first)
    #[error("{0}")]
    Input(String),
    /// Operational failure (e.g., model load, embedding, download)
    #[error("{0}")]
    Other(String),
}

pub(crate) fn resolve_client<'a>(
    config: &'a Config,
    team: Option<&'a str>,
) -> Result<(&'a str, EsaClient), SaeError> {
    let team = config.resolve_team(team)?;
    let client = EsaClient::from_env()?;
    Ok((team, client))
}

pub(crate) fn require_db(config: &Config, team: &str) -> Result<sae::storage::Db, SaeError> {
    let db_path = config.team_db_path(team)?;
    if !db_path.exists() {
        return Err(SaeError::Input(format!(
            "No data for team '{team}'. Run `sae harvest {team}` first."
        )));
    }
    Ok(sae::storage::Db::open(&db_path)?)
}

fn exit_code_for(e: &SaeError) -> ExitCode {
    use sae::client::ClientError;
    use sae::sync::SyncError;
    match e {
        SaeError::Input(_) | SaeError::Config(_) => ExitCode::from(2),
        SaeError::Client(ClientError::TokenNotSet) => ExitCode::from(2),
        SaeError::Sync(SyncError::Client(ClientError::TokenNotSet)) => ExitCode::from(2),
        SaeError::Storage(_) | SaeError::Sync(SyncError::Storage(_)) | SaeError::Json(_) => {
            ExitCode::from(4)
        }
        SaeError::Client(_) | SaeError::Sync(_) | SaeError::Io(_) | SaeError::Other(_) => {
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli, config: Config) -> Result<String, SaeError> {
    let json = cli.json;
    let output = match cli.command {
        Command::Harvest { team, full } => {
            commands::data::run_harvest(&config, &team, full, json).await?
        }
        Command::Search {
            query,
            team,
            limit,
            after,
            before,
        } => {
            let query = commands::search::resolve_search_query(query)?;
            commands::search::run_search(
                &config,
                &query,
                team.as_deref(),
                limit,
                after.as_deref(),
                before.as_deref(),
                json,
            )?
        }
        Command::Get {
            number,
            team,
            with_body,
        } => commands::post::run_get(&config, number, team.as_deref(), with_body, json).await?,
        Command::Create { args } => commands::post::run_create(&config, args, json).await?,
        Command::Update { args } => commands::post::run_update(&config, args, json).await?,
        Command::Embed { team } => commands::data::run_embed(&config, &team, json)?,
        Command::Archive {
            number,
            team,
            dry_run,
        } => {
            commands::archive::run_archive(&config, number, team.as_deref(), dry_run, json).await?
        }
        Command::Ship {
            number,
            team,
            dry_run,
        } => commands::archive::run_ship(&config, number, team.as_deref(), dry_run, json).await?,
        Command::Status { team } => commands::status::run_status(&config, team.as_deref(), json)?,
        Command::Model { command } => match command {
            ModelCommand::Download => commands::data::run_model_download(json)?,
        },
    };
    Ok(output)
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("sae=info".parse().expect("hardcoded directive is valid")),
        )
        .init();

    let cli = parse_cli_args(std::env::args_os()).unwrap_or_else(|e| e.exit());
    let config = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return exit_code_for(&e.into());
        }
    };
    match run(cli, config).await {
        Ok(output) => {
            if !output.is_empty() {
                println!("{output}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            exit_code_for(&e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn subcommand_after_help(name: &str) -> String {
        let mut command = Cli::command();
        command
            .find_subcommand_mut(name)
            .unwrap()
            .get_after_help()
            .map(|help| help.to_string())
            .unwrap_or_default()
    }

    // T-030: explicit `sae search "認証"` is not double-injected
    #[test]
    fn explicit_search_not_double_injected() {
        let cli = parse_cli_args(["sae", "search", "認証"]).unwrap();
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query.as_deref(), Some("認証")),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-033: `sae search` (missing arg) parses to stdin fallback path
    #[test]
    fn search_missing_arg_parses_for_stdin_fallback() {
        let cli = parse_cli_args(["sae", "search"]).unwrap();
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query, None),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-034: `sae harvest` (missing required arg) → clap error via helper
    #[test]
    fn harvest_missing_arg_is_clap_error() {
        let result = parse_cli_args(["sae", "harvest"]);
        assert!(result.is_err(), "harvest without team should be clap error");
    }

    // T-049: parse_cli_args(["sae", "query"]) → Command::Search (json=false) - regression
    #[test]
    fn shorthand_without_flags_has_json_false() {
        let cli = parse_cli_args(["sae", "query"]).unwrap();
        assert!(!cli.json, "[T-049] json should default to false");
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query.as_deref(), Some("query")),
            other => panic!("[T-049] expected Search, got {other:?}"),
        }
    }

    // T-038: create args + --dry-run → parse succeeds with dry_run=true
    #[test]
    fn create_with_dry_run_parses_dry_run_flag() {
        let cli = parse_cli_args(["sae", "create", "--name", "Test Post", "--dry-run"]).unwrap();
        match cli.command {
            Command::Create { args } => {
                assert_eq!(args.name, "Test Post", "[T-038] name should match");
                assert!(args.dry_run, "[T-038] dry_run should be true");
            }
            other => panic!("[T-038] expected Create, got {other:?}"),
        }
    }

    #[test]
    fn body_file_dash_parses_as_stdin() {
        let cli =
            parse_cli_args(["sae", "create", "--name", "Stdin Post", "--body-file", "-"]).unwrap();
        match cli.command {
            Command::Create { args } => {
                assert_eq!(
                    args.body_file.as_deref(),
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
        let cmd = Cli::command();
        for sub in cmd.get_subcommands() {
            let after_help = subcommand_after_help(sub.get_name());
            assert!(
                after_help.contains("Examples"),
                "[T-042] subcommand '{}' should have Examples in after_help",
                sub.get_name()
            );
        }
    }

    #[test]
    fn create_help_includes_stdin_example() {
        let after_help = subcommand_after_help("create");
        assert!(
            after_help.contains("cat body.md | sae create --name \"Title\" --body-file -"),
            "[T-042] create help should include stdin example"
        );
    }

    #[test]
    fn search_help_includes_stdin_examples() {
        let after_help = subcommand_after_help("search");
        for snippet in ["echo \"認証\" | sae search", "sae search -"] {
            assert!(
                after_help.contains(snippet),
                "[T-042] search help should include stdin example '{snippet}'"
            );
        }
    }

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

    // T-002/T-003: `sae model` → clap error (not rewritten to search shorthand)
    #[test]
    fn model_without_subcommand_is_clap_error() {
        let result = parse_cli_args(["sae", "model"]);
        assert!(
            result.is_err(),
            "[T-002] model without subcommand should be clap error, not search shorthand"
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

    // TC-012: multi-positional args without valid search expansion → clap error
    #[test]
    fn multi_positional_args_not_shorthand() {
        let result = parse_cli_args(["sae", "foo", "bar"]);
        assert!(
            result.is_err(),
            "two positional args should be clap error, not shorthand"
        );
    }

    // T-050: `sae get 42` → Command::Get { number: 42, team: None, with_body: false }
    #[test]
    fn get_basic_parses_correctly() {
        let cli = parse_cli_args(["sae", "get", "42"]).unwrap();
        match cli.command {
            Command::Get {
                number,
                team,
                with_body,
            } => {
                assert_eq!(number, 42, "[T-050] post number should be 42");
                assert_eq!(team, None, "[T-050] team should be None");
                assert!(!with_body, "[T-050] with_body should default to false");
            }
            other => panic!("[T-050] expected Get, got {other:?}"),
        }
    }

    // T-051: `sae get 42 --team myteam --with-body` → all fields set
    #[test]
    fn get_with_team_and_body_flag() {
        let cli = parse_cli_args(["sae", "get", "42", "--team", "myteam", "--with-body"]).unwrap();
        match cli.command {
            Command::Get {
                number,
                team,
                with_body,
            } => {
                assert_eq!(number, 42, "[T-051] post number should be 42");
                assert_eq!(
                    team.as_deref(),
                    Some("myteam"),
                    "[T-051] team should be myteam"
                );
                assert!(with_body, "[T-051] with_body should be true");
            }
            other => panic!("[T-051] expected Get, got {other:?}"),
        }
    }

    // T-052: `sae update 42 --name "New Title"` → Command::Update with name set
    #[test]
    fn update_with_name_parses_correctly() {
        let cli = parse_cli_args(["sae", "update", "42", "--name", "New Title"]).unwrap();
        match cli.command {
            Command::Update { args } => {
                assert_eq!(args.number, 42, "[T-052] post number should be 42");
                assert_eq!(
                    args.name.as_deref(),
                    Some("New Title"),
                    "[T-052] name should match"
                );
            }
            other => panic!("[T-052] expected Update, got {other:?}"),
        }
    }

    // T-053: `sae ship 42` → Command::Ship { number: 42, dry_run: false }
    #[test]
    fn ship_basic_parses_correctly() {
        let cli = parse_cli_args(["sae", "ship", "42"]).unwrap();
        match cli.command {
            Command::Ship {
                number, dry_run, ..
            } => {
                assert_eq!(number, 42, "[T-053] post number should be 42");
                assert!(!dry_run, "[T-053] dry_run should default to false");
            }
            other => panic!("[T-053] expected Ship, got {other:?}"),
        }
    }

    // T-054: `sae ship 42 --dry-run` → dry_run=true
    #[test]
    fn ship_with_dry_run_parses_dry_run_flag() {
        let cli = parse_cli_args(["sae", "ship", "42", "--dry-run"]).unwrap();
        match cli.command {
            Command::Ship {
                number, dry_run, ..
            } => {
                assert_eq!(number, 42, "[T-054] post number should be 42");
                assert!(dry_run, "[T-054] dry_run should be true");
            }
            other => panic!("[T-054] expected Ship, got {other:?}"),
        }
    }

    // T-055: `sae status` → Command::Status { team: None }
    #[test]
    fn status_no_team_parses_correctly() {
        let cli = parse_cli_args(["sae", "status"]).unwrap();
        match cli.command {
            Command::Status { team } => {
                assert_eq!(team, None, "[T-055] team should be None");
            }
            other => panic!("[T-055] expected Status, got {other:?}"),
        }
    }

    // T-056: `sae status --team myteam` → Command::Status { team: Some("myteam") }
    #[test]
    fn status_with_team_parses_correctly() {
        let cli = parse_cli_args(["sae", "status", "--team", "myteam"]).unwrap();
        match cli.command {
            Command::Status { team } => {
                assert_eq!(
                    team.as_deref(),
                    Some("myteam"),
                    "[T-056] team should be myteam"
                );
            }
            other => panic!("[T-056] expected Status, got {other:?}"),
        }
    }

    // TC-014: non-search subcommand names are not rewritten as search shorthand
    #[test]
    fn all_subcommands_not_shorthand() {
        // `search` itself parses as Command::Search via the normal path — excluded.
        for cmd in [
            "harvest", "get", "update", "ship", "archive", "embed", "status", "model",
        ] {
            let result = parse_cli_args(["sae", cmd]);
            assert!(
                !matches!(
                    result.as_ref().map(|c| &c.command),
                    Ok(Command::Search { .. })
                ),
                "[TC-014] subcommand '{cmd}' should not be rewritten as Search shorthand"
            );
        }
    }
}
