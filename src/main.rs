mod commands;
mod output;

use clap::{CommandFactory, Parser, Subcommand};
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

fn parse_cli_args<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let args: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();

    // Shorthand: `sae "query"` or `sae --json "query" --limit 2` → `sae [flags] search "query" --limit 2`
    // Pre-filter: only build command tree when non-flag arg count suggests shorthand is possible.
    let positional_count = args
        .iter()
        .filter(|a| !a.to_str().is_some_and(|s| s.starts_with('-')))
        .count();

    if positional_count >= 2 {
        let cmd = Cli::command();
        let known: Vec<&str> = cmd.get_subcommands().map(|s| s.get_name()).collect();
        let global_flags: Vec<String> = cmd
            .get_arguments()
            .filter(|a| a.is_global_set())
            .filter_map(|a| a.get_long().map(|l| format!("--{l}")))
            .collect();

        let (flags, rest): (Vec<_>, Vec<_>) = args.iter().enumerate().partition(|(i, a)| {
            *i > 0
                && a.to_str()
                    .is_some_and(|s| global_flags.iter().any(|f| f == s))
        });
        let rest: Vec<&std::ffi::OsString> = rest.into_iter().map(|(_, a)| a).collect();

        if rest.len() >= 2
            && let Some(first_arg) = rest[1].to_str()
            && !first_arg.starts_with('-')
            && first_arg != "help"
            && !known.contains(&first_arg)
            && !known
                .iter()
                .any(|k| strsim::osa_distance(first_arg, k) <= 1)
        {
            let mut expanded: Vec<std::ffi::OsString> = vec![rest[0].clone()];
            for (_, f) in &flags {
                expanded.push((*f).clone());
            }
            expanded.push("search".into());
            for arg in &rest[1..] {
                expanded.push((*arg).clone());
            }
            if let Ok(cli) = Cli::try_parse_from(expanded.clone()) {
                let display: Vec<_> = expanded.iter().filter_map(|a| a.to_str()).collect();
                eprintln!("→ {}", display.join(" "));
                return Ok(cli);
            }
        }
    }

    Cli::try_parse_from(args)
}

pub(crate) type AppError = Box<dyn std::error::Error>;

pub(crate) fn resolve_client<'a>(
    config: &'a Config,
    team: Option<&'a str>,
) -> Result<(&'a str, EsaClient), AppError> {
    let team = config.resolve_team(team)?;
    let client = EsaClient::from_env()?;
    Ok((team, client))
}

pub(crate) fn require_db(config: &Config, team: &str) -> Result<sae::storage::Db, AppError> {
    let db_path = config.team_db_path(team)?;
    if !db_path.exists() {
        return Err(format!("No data for team '{team}'. Run `sae harvest {team}` first.").into());
    }
    Ok(sae::storage::Db::open(&db_path)?)
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
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
            commands::data::run_harvest(&config, &team, full, json).await
        }
        Command::Search { query, team, limit } => {
            let query = commands::search::resolve_search_query(query)?;
            commands::search::run_search(&config, &query, team.as_deref(), limit, json)
        }
        Command::Get {
            number,
            team,
            with_body,
        } => commands::post::run_get(&config, number, team.as_deref(), with_body, json).await,
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
            commands::post::run_create(
                &config,
                commands::post::CreateArgs {
                    name,
                    body,
                    body_file,
                    category,
                    tag,
                    wip,
                    team,
                    dry_run,
                },
                json,
            )
            .await
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
            commands::post::run_update(
                &config,
                commands::post::UpdateArgs {
                    number,
                    name,
                    body,
                    body_file,
                    category,
                    tag,
                    team,
                    dry_run,
                },
                json,
            )
            .await
        }
        Command::Embed { team } => commands::data::run_embed(&config, &team, json),
        Command::Archive {
            number,
            team,
            dry_run,
        } => commands::archive::run_archive(&config, number, team.as_deref(), dry_run, json).await,
        Command::Ship {
            number,
            team,
            dry_run,
        } => commands::archive::run_ship(&config, number, team.as_deref(), dry_run, json).await,
        Command::Status { team } => commands::status::run_status(&config, team.as_deref(), json),
        Command::Model { command } => match command {
            ModelCommand::Download => commands::data::run_model_download(json),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subcommand_after_help(name: &str) -> String {
        let mut command = Cli::command();
        command
            .find_subcommand_mut(name)
            .unwrap()
            .get_after_help()
            .map(|help| help.to_string())
            .unwrap_or_default()
    }

    // T-029: shorthand `sae "認証"` → search
    #[test]
    fn shorthand_single_query_becomes_search() {
        let cli = parse_cli_args(["sae", "認証"]).unwrap();
        match cli.command {
            Command::Search { query, .. } => assert_eq!(query.as_deref(), Some("認証")),
            other => panic!("expected Search, got {other:?}"),
        }
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

    // T-031: `sae harvest team` stays as harvest
    #[test]
    fn known_subcommand_not_shorthand() {
        let cli = parse_cli_args(["sae", "harvest", "myteam"]).unwrap();
        match cli.command {
            Command::Harvest { team, .. } => assert_eq!(team, "myteam"),
            other => panic!("expected Harvest, got {other:?}"),
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
            Command::Create { name, dry_run, .. } => {
                assert_eq!(name, "Test Post", "[T-038] name should match");
                assert!(dry_run, "[T-038] dry_run should be true");
            }
            other => panic!("[T-038] expected Create, got {other:?}"),
        }
    }

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

    // TC-011: flag-like argument not treated as shorthand query
    #[test]
    fn flag_like_arg_not_shorthand() {
        let result = parse_cli_args(["sae", "--unknown"]);
        assert!(
            result.is_err(),
            "[TC-011] --unknown should be clap error, not search shorthand"
        );
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

    // T-022: `sae "query" --limit 2` → Command::Search with limit=2
    #[test]
    fn shorthand_with_limit_option() {
        let cli = parse_cli_args(["sae", "query", "--limit", "2"]).unwrap();
        match cli.command {
            Command::Search { query, limit, .. } => {
                assert_eq!(
                    query.as_deref(),
                    Some("query"),
                    "[T-022] query should match"
                );
                assert_eq!(limit, 2, "[T-022] limit should be 2");
            }
            other => panic!("[T-022] expected Search, got {other:?}"),
        }
    }

    // T-023: `sae --json "query" --limit 2` → json=true + Command::Search with limit=2
    #[test]
    fn shorthand_with_json_and_limit() {
        let cli = parse_cli_args(["sae", "--json", "query", "--limit", "2"]).unwrap();
        assert!(cli.json, "[T-023] json should be true");
        match cli.command {
            Command::Search { query, limit, .. } => {
                assert_eq!(
                    query.as_deref(),
                    Some("query"),
                    "[T-023] query should match"
                );
                assert_eq!(limit, 2, "[T-023] limit should be 2");
            }
            other => panic!("[T-023] expected Search, got {other:?}"),
        }
    }

    // T-024: `sae "query" --team myteam` → Command::Search with team=Some("myteam")
    #[test]
    fn shorthand_with_team_option() {
        let cli = parse_cli_args(["sae", "query", "--team", "myteam"]).unwrap();
        match cli.command {
            Command::Search { query, team, .. } => {
                assert_eq!(
                    query.as_deref(),
                    Some("query"),
                    "[T-024] query should match"
                );
                assert_eq!(
                    team.as_deref(),
                    Some("myteam"),
                    "[T-024] team should be myteam"
                );
            }
            other => panic!("[T-024] expected Search, got {other:?}"),
        }
    }

    // T-025: `sae serach` → clap error (OSA distance ≤ 1 guard blocks shorthand)
    #[test]
    fn typo_near_subcommand_is_clap_error() {
        let result = parse_cli_args(["sae", "serach"]);
        assert!(
            result.is_err(),
            "[T-025] typo 'serach' (osa=1 from 'search') should be clap error"
        );
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

    // TC-013: bare dash is not treated as shorthand
    #[test]
    fn bare_dash_not_shorthand() {
        let result = parse_cli_args(["sae", "-"]);
        assert!(
            result.is_err(),
            "`sae -` should be clap error, not shorthand query"
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
            Command::Update { number, name, .. } => {
                assert_eq!(number, 42, "[T-052] post number should be 42");
                assert_eq!(
                    name.as_deref(),
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
