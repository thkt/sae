#![warn(unreachable_pub)]

use std::env;
use std::ffi;
use std::io;
use std::iter;
use std::process::ExitCode;

use amici::cli::try_expand_shorthand;
use clap::{Parser, Subcommand};
use rurico::model_probe;
use sae::config::Config;
use sae::tools::{CreateArgs, Sae, SaeError, SearchArgs, UpdateArgs, exit_code_for};

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
        #[command(flatten)]
        args: SearchArgs,
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
        args: CreateArgs,
    },
    /// Update a post
    #[command(after_help = "\
Examples:
  sae update 42 --name \"New Title\" --team myteam
  sae update 42 --body-file updated.md
  sae update 42 --name \"New Title\" --dry-run")]
    Update {
        #[command(flatten)]
        args: UpdateArgs,
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
  sae embed myteam")]
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
    T: Into<ffi::OsString> + Clone,
{
    let args: Vec<ffi::OsString> = args.into_iter().map(Into::into).collect();
    let expanded = try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS);
    if let Some(expanded) = expanded
        && let Ok(cli) = Cli::try_parse_from(&expanded)
    {
        let display: Vec<_> = iter::once("sae")
            .chain(expanded[1..].iter().filter_map(|a| a.to_str()))
            .collect();
        eprintln!("→ {}", display.join(" "));
        return Ok(cli);
    }
    Cli::try_parse_from(args)
}

async fn run(cli: Cli, config: Config) -> Result<String, SaeError> {
    let json = cli.json;
    if let Command::Model { command } = cli.command {
        return match command {
            ModelCommand::Download => Sae::model_download(json),
        };
    }
    let sae = Sae::new(config);
    match cli.command {
        Command::Harvest { team, full } => sae.harvest(&team, full, json).await,
        Command::Search { args } => sae.search(args, json),
        Command::Get {
            number,
            team,
            with_body,
        } => sae.get(number, team.as_deref(), with_body, json).await,
        Command::Create { args } => sae.create(args, json).await,
        Command::Update { args } => sae.update(args, json).await,
        Command::Embed { team } => sae.embed(&team, json),
        Command::Archive {
            number,
            team,
            dry_run,
        } => sae.archive(number, team.as_deref(), dry_run, json).await,
        Command::Ship {
            number,
            team,
            dry_run,
        } => sae.ship(number, team.as_deref(), dry_run, json).await,
        Command::Status { team } => sae.status(team.as_deref(), json),
        Command::Model { .. } => unreachable!("handled before Sae::new()"),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    model_probe::handle_probe_if_needed();

    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("sae=info".parse().expect("hardcoded directive is valid")),
        )
        .init();

    let cli = parse_cli_args(env::args_os()).unwrap_or_else(|e| e.exit());
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
    use ffi::OsString;

    fn os(s: &[&str]) -> Vec<OsString> {
        s.iter().map(|&a| a.into()).collect()
    }

    // T-029: bare query expands with "search" inserted as subcommand
    #[test]
    fn single_query_expands_to_search() {
        let exp =
            try_expand_shorthand(&os(&["sae", "認証"]), KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "search", "認証"]);
    }

    // T-031: known subcommand as first positional → not expanded
    #[test]
    fn known_subcommand_not_expanded() {
        assert!(
            try_expand_shorthand(
                &os(&["sae", "harvest", "myteam"]),
                KNOWN_SUBCOMMANDS,
                GLOBAL_FLAGS
            )
            .is_none()
        );
    }

    // T-022: trailing options pass through after the inserted "search"
    #[test]
    fn query_with_trailing_option_expanded() {
        let exp = try_expand_shorthand(
            &os(&["sae", "query", "--limit", "2"]),
            KNOWN_SUBCOMMANDS,
            GLOBAL_FLAGS,
        )
        .unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "search", "query", "--limit", "2"]);
    }

    // T-024: non-global option (--team) stays after the inserted "search"
    #[test]
    fn non_global_option_stays_after_search() {
        let exp = try_expand_shorthand(
            &os(&["sae", "query", "--team", "myteam"]),
            KNOWN_SUBCOMMANDS,
            GLOBAL_FLAGS,
        )
        .unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "search", "query", "--team", "myteam"]);
    }

    // T-025: typo within OSA distance 1 → not expanded (typo guard)
    #[test]
    fn typo_within_distance_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "serach"]), KNOWN_SUBCOMMANDS, GLOBAL_FLAGS)
                .is_none(),
            "typo 'serach' (osa=1 from 'search') should not expand"
        );
    }

    // T-092: bare dash counts as flag prefix → positional_count < 2 → not expanded
    #[test]
    fn bare_dash_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "-"]), KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_none(),
            "`sae -` should not expand"
        );
    }

    // T-091: flag-like arg (--) → positional_count < 2 → not expanded
    #[test]
    fn flag_only_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "--unknown"]), KNOWN_SUBCOMMANDS, GLOBAL_FLAGS)
                .is_none(),
            "--unknown should not expand"
        );
    }

    fn subcommand_after_help(name: &str) -> String {
        let mut command = Cli::command();
        command
            .find_subcommand_mut(name)
            .unwrap()
            .get_after_help()
            .map(ToString::to_string)
            .unwrap_or_default()
    }

    // T-030: explicit `sae search "認証"` is not double-injected
    #[test]
    fn explicit_search_not_double_injected() {
        let cli = parse_cli_args(["sae", "search", "認証"]).unwrap();
        match cli.command {
            Command::Search { args } => assert_eq!(args.query.as_deref(), Some("認証")),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-033: `sae search` (missing arg) parses to stdin fallback path
    #[test]
    fn search_missing_arg_parses_for_stdin_fallback() {
        let cli = parse_cli_args(["sae", "search"]).unwrap();
        match cli.command {
            Command::Search { args } => assert_eq!(args.query, None),
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
        assert!(!cli.json, "json should default to false");
        match cli.command {
            Command::Search { args } => assert_eq!(args.query.as_deref(), Some("query")),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-038: create args + --dry-run → parse succeeds with dry_run=true
    #[test]
    fn create_with_dry_run_parses_dry_run_flag() {
        let cli = parse_cli_args(["sae", "create", "--name", "Test Post", "--dry-run"]).unwrap();
        match cli.command {
            Command::Create { args } => {
                assert_eq!(args.name, "Test Post", "name should match");
                assert!(args.dry_run, "dry_run should be true");
            }
            other => panic!("expected Create, got {other:?}"),
        }
    }

    // T-150: --body-file "-" parses correctly with body_file set to "-"
    #[test]
    fn body_file_dash_parses_as_stdin() {
        let cli =
            parse_cli_args(["sae", "create", "--name", "Stdin Post", "--body-file", "-"]).unwrap();
        match cli.command {
            Command::Create { args } => {
                assert_eq!(
                    args.body_file.as_deref(),
                    Some("-"),
                    "body_file should be \"-\" for stdin"
                );
            }
            other => panic!("expected Create, got {other:?}"),
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
            "--body and --body-file together should be a clap error"
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
                assert_eq!(number, 42, "post number should be 42");
                assert!(dry_run, "dry_run should be true");
            }
            other => panic!("expected Archive, got {other:?}"),
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
                "subcommand '{}' should have Examples in after_help",
                sub.get_name()
            );
        }
    }

    // T-151: create subcommand help text includes stdin pipe example
    #[test]
    fn create_help_includes_stdin_example() {
        let after_help = subcommand_after_help("create");
        assert!(
            after_help.contains("cat body.md | sae create --name \"Title\" --body-file -"),
            "create help should include stdin example"
        );
    }

    // T-152: search subcommand help text includes stdin pipe examples
    #[test]
    fn search_help_includes_stdin_examples() {
        let after_help = subcommand_after_help("search");
        for snippet in ["echo \"認証\" | sae search", "sae search -"] {
            assert!(
                after_help.contains(snippet),
                "search help should include stdin example '{snippet}'"
            );
        }
    }

    // T-001: `sae model download` parses to Command::Model { ModelCommand::Download }
    #[test]
    fn model_download_parses_correctly() {
        let cli = parse_cli_args(["sae", "model", "download"]).unwrap();
        match cli.command {
            Command::Model { command } => match command {
                ModelCommand::Download => {}
            },
            other => panic!("expected Model {{ Download }}, got {other:?}"),
        }
    }

    // T-002: `sae model` → clap error (not rewritten to search shorthand)
    #[test]
    fn model_without_subcommand_is_clap_error() {
        let result = parse_cli_args(["sae", "model"]);
        assert!(
            result.is_err(),
            "model without subcommand should be clap error, not search shorthand"
        );
    }

    // T-004: `sae model download --json` → json=true + ModelCommand::Download
    #[test]
    fn model_download_with_json_flag() {
        let cli = parse_cli_args(["sae", "--json", "model", "download"]).unwrap();
        assert!(cli.json, "global json flag should be true");
        match cli.command {
            Command::Model { command } => match command {
                ModelCommand::Download => {}
            },
            other => panic!("expected Model {{ Download }}, got {other:?}"),
        }
    }

    // T-005: `sae embed myteam` still parses as Command::Embed (regression)
    #[test]
    fn embed_still_parses_after_model_addition() {
        let cli = parse_cli_args(["sae", "embed", "myteam"]).unwrap();
        match cli.command {
            Command::Embed { team } => {
                assert_eq!(team, "myteam", "team should be myteam");
            }
            other => panic!("expected Embed, got {other:?}"),
        }
    }

    // T-077: multi-positional args without valid search expansion → clap error
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
                assert_eq!(number, 42, "post number should be 42");
                assert_eq!(team, None, "team should be None");
                assert!(!with_body, "with_body should default to false");
            }
            other => panic!("expected Get, got {other:?}"),
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
                assert_eq!(number, 42, "post number should be 42");
                assert_eq!(team.as_deref(), Some("myteam"), "team should be myteam");
                assert!(with_body, "with_body should be true");
            }
            other => panic!("expected Get, got {other:?}"),
        }
    }

    // T-052: `sae update 42 --name "New Title"` → Command::Update with name set
    #[test]
    fn update_with_name_parses_correctly() {
        let cli = parse_cli_args(["sae", "update", "42", "--name", "New Title"]).unwrap();
        match cli.command {
            Command::Update { args } => {
                assert_eq!(args.number, 42, "post number should be 42");
                assert_eq!(args.name.as_deref(), Some("New Title"), "name should match");
            }
            other => panic!("expected Update, got {other:?}"),
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
                assert_eq!(number, 42, "post number should be 42");
                assert!(!dry_run, "dry_run should default to false");
            }
            other => panic!("expected Ship, got {other:?}"),
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
                assert_eq!(number, 42, "post number should be 42");
                assert!(dry_run, "dry_run should be true");
            }
            other => panic!("expected Ship, got {other:?}"),
        }
    }

    // T-055: `sae status` → Command::Status { team: None }
    #[test]
    fn status_no_team_parses_correctly() {
        let cli = parse_cli_args(["sae", "status"]).unwrap();
        match cli.command {
            Command::Status { team } => {
                assert_eq!(team, None, "team should be None");
            }
            other => panic!("expected Status, got {other:?}"),
        }
    }

    // T-056: `sae status --team myteam` → Command::Status { team: Some("myteam") }
    #[test]
    fn status_with_team_parses_correctly() {
        let cli = parse_cli_args(["sae", "status", "--team", "myteam"]).unwrap();
        match cli.command {
            Command::Status { team } => {
                assert_eq!(team.as_deref(), Some("myteam"), "team should be myteam");
            }
            other => panic!("expected Status, got {other:?}"),
        }
    }

    // T-078: non-search subcommand names are not rewritten as search shorthand
    #[test]
    fn all_subcommands_not_shorthand() {
        for cmd in [
            "harvest", "get", "update", "ship", "archive", "embed", "status", "model",
        ] {
            let result = parse_cli_args(["sae", cmd]);
            assert!(
                !matches!(
                    result.as_ref().map(|c| &c.command),
                    Ok(Command::Search { .. })
                ),
                "subcommand '{cmd}' should not be rewritten as Search shorthand"
            );
        }
    }
}
