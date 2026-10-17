#![warn(unreachable_pub)]

use std::env;
use std::ffi;
use std::io::{ErrorKind, stdout};
use std::iter;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
#[cfg(feature = "test-support")]
use sae::client::ClientError;
use sae::compat::init_subscriber;
use sae::compat::{CliError, codes};
use sae::compat::{exit_error, hint_arrow, try_expand_shorthand};
use sae::config::Config;
use sae::index::{migrate, rollback};
use sae::io::write_output;
use sae::model::handle_probe_if_needed;
use sae::tools::{CreateArgs, Sae, SaeError, SearchArgs, UpdateArgs};
use sae::{CommandOutput, render_error, render_parse_error, render_success};
use tokio::task::spawn_blocking;

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
    /// Fetch new esa posts incrementally and index them (chunks + embeddings)
    #[command(after_help = "\
Examples:
  sae index myteam")]
    Index {
        /// Team name
        team: String,
    },
    /// Re-fetch all esa posts and rebuild the index from scratch (chunks + embeddings)
    #[command(after_help = "\
Examples:
  sae rebuild myteam")]
    Rebuild {
        /// Team name
        team: String,
    },
    /// Semantic search over indexed posts (read-only)
    #[command(after_help = "\
Examples:
  sae search \"認証\" --team myteam --limit 5
  sae search \"認証\" --after 2025-01-01
  sae search \"認証\" --after 2025-01-01 --before 2025-06-30
  echo \"認証\" | sae search
  sae search -

Search is read-only; build the index first with `sae index` (re-run it to refresh).")]
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
    #[cfg(feature = "test-support")]
    #[command(name = "__test_force_unknown", hide = true)]
    TestForceUnknown,
    #[cfg(feature = "test-support")]
    #[command(name = "__test_force_backend_unavailable", hide = true)]
    TestForceBackendUnavailable,
    #[cfg(feature = "test-support")]
    #[command(name = "__test_force_internal", hide = true)]
    TestForceInternal,
    #[cfg(feature = "test-support")]
    #[command(name = "__test_force_client_api_404", hide = true)]
    TestForceClientApi404,
    #[cfg(feature = "test-support")]
    #[command(name = "__test_force_search_warning", hide = true)]
    TestForceSearchWarning,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download embedding model from Hugging Face Hub
    Download,
    /// Re-embed a consistent local snapshot with Gemma, then activate it
    Migrate { team: String },
    /// Restore the previous local index selection
    Rollback { team: String },
}

// Suggestions exclude hidden and retired commands.
const PUBLIC_SUBCOMMANDS: &[&str] = &[
    "index", "rebuild", "search", "get", "create", "update", "archive", "ship", "status", "model",
];

// Retired and hidden commands must reach clap errors, not search shorthand.
const KNOWN_SUBCOMMANDS: &[&str] = &[
    "index",
    "rebuild",
    "search",
    "get",
    "create",
    "update",
    "archive",
    "ship",
    "embed",
    "status",
    "model",
    "harvest",
    "__test_force_unknown",
    "__test_force_backend_unavailable",
    "__test_force_internal",
    "__test_force_client_api_404",
    "__test_force_search_warning",
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
        hint_arrow(&display);
        return Ok(cli);
    }
    Cli::try_parse_from(args)
}

// Read --json before parsing so parse failures use the requested format.
fn json_mode_from_argv<I, T>(args: I) -> bool
where
    I: IntoIterator<Item = T>,
    T: AsRef<ffi::OsStr>,
{
    args.into_iter().any(|a| a.as_ref() == "--json")
}

async fn run(cli: Cli, config: Config) -> Result<CommandOutput, SaeError> {
    if let Command::Model { command } = cli.command {
        return match command {
            ModelCommand::Migrate { team } => spawn_blocking(move || migrate(&config, &team))
                .await
                .map_err(|e| SaeError::Internal(format!("Migration task failed: {e}")))?,
            ModelCommand::Rollback { team } => rollback(&config, &team),
            ModelCommand::Download => {
                spawn_blocking(Sae::model_download).await.map_err(|error| {
                    SaeError::Internal(format!("Model download task failed: {error}"))
                })?
            }
        };
    }
    let sae = Sae::new(config);
    match cli.command {
        Command::Index { team } => sae.index(&team).await,
        Command::Rebuild { team } => sae.rebuild(&team).await,
        Command::Search { args } => sae.search(args),
        Command::Get {
            number,
            team,
            with_body,
        } => sae.get(number, team.as_deref(), with_body).await,
        Command::Create { args } => sae.create(args).await,
        Command::Update { args } => sae.update(args).await,
        Command::Archive {
            number,
            team,
            dry_run,
        } => sae.archive(number, team.as_deref(), dry_run).await,
        Command::Ship {
            number,
            team,
            dry_run,
        } => sae.ship(number, team.as_deref(), dry_run).await,
        Command::Status { team } => sae.status(team.as_deref()),
        #[cfg(feature = "test-support")]
        Command::TestForceUnknown => Err(SaeError::Other(
            "synthetic UNKNOWN for cli_integration test (test-support feature)".to_owned(),
        )),
        #[cfg(feature = "test-support")]
        Command::TestForceBackendUnavailable => Err(SaeError::BackendUnavailable),
        #[cfg(feature = "test-support")]
        Command::TestForceInternal => Err(SaeError::Internal(
            "synthetic INTERNAL for cli_integration test (test-support feature)".to_owned(),
        )),
        #[cfg(feature = "test-support")]
        Command::TestForceClientApi404 => Err(SaeError::Client(ClientError::Api {
            status: 404,
            body: r#"{"error":"not_found","message":"Not Found"}"#.to_owned(),
        })),
        #[cfg(feature = "test-support")]
        Command::TestForceSearchWarning => sae::__test_search_with_warnings(),
        Command::Model { .. } => unreachable!("handled before Sae::new()"),
    }
}

fn emit_success(out: &CommandOutput, json_mode: bool) -> ExitCode {
    let body = render_success(out, json_mode);
    if body.is_empty() {
        return ExitCode::SUCCESS;
    }
    let mut handle = stdout().lock();
    match write_output(&mut handle, &body) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.kind() == ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            exit_error(&format!("write to stdout failed: {e}"));
            ExitCode::from(codes::IO_ERR)
        }
    }
}

fn emit_error(err: &SaeError, json_mode: bool, config: Option<&Config>) {
    if json_mode {
        eprintln!("{}", render_error(err, true, config));
    } else {
        exit_error(&err.to_string());
    }
}

fn handle_parse_error(e: &clap::Error, json_mode: bool) -> ExitCode {
    // clap returns help/version as errors; preserve their successful output.
    if !e.use_stderr() {
        let _ = e.print();
        return ExitCode::SUCCESS;
    }
    if json_mode {
        eprintln!("{}", render_parse_error(e, true, PUBLIC_SUBCOMMANDS));
    } else {
        let _ = e.print();
    }
    ExitCode::from(codes::USAGE)
}

#[tokio::main]
async fn main() -> ExitCode {
    handle_probe_if_needed();

    init_subscriber("sae=info");

    let json_mode = json_mode_from_argv(env::args_os().skip(1));

    let cli = match parse_cli_args(env::args_os()) {
        Ok(cli) => cli,
        Err(e) => return handle_parse_error(&e, json_mode),
    };
    let config = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            let err: SaeError = e.into();
            emit_error(&err, json_mode, None);
            return err.exit_code();
        }
    };
    let config_for_emit = config.clone();
    match run(cli, config).await {
        Ok(out) => emit_success(&out, json_mode),
        Err(e) => {
            emit_error(&e, json_mode, Some(&config_for_emit));
            e.exit_code()
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

    #[test]
    fn single_query_expands_to_search() {
        let exp =
            try_expand_shorthand(&os(&["sae", "認証"]), KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "search", "認証"]);
    }

    #[test]
    fn known_subcommand_not_expanded() {
        assert!(
            try_expand_shorthand(
                &os(&["sae", "index", "myteam"]),
                KNOWN_SUBCOMMANDS,
                GLOBAL_FLAGS
            )
            .is_none()
        );
    }

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

    #[test]
    fn typo_within_distance_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "serach"]), KNOWN_SUBCOMMANDS, GLOBAL_FLAGS)
                .is_none(),
            "typo 'serach' (osa=1 from 'search') should not expand"
        );
    }

    #[test]
    fn bare_dash_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "-"]), KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_none(),
            "`sae -` should not expand"
        );
    }

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

    #[test]
    fn explicit_search_not_double_injected() {
        let cli = parse_cli_args(["sae", "search", "認証"]).unwrap();
        match cli.command {
            Command::Search { args } => assert_eq!(args.query.as_deref(), Some("認証")),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn search_missing_arg_parses_for_stdin_fallback() {
        let cli = parse_cli_args(["sae", "search"]).unwrap();
        match cli.command {
            Command::Search { args } => assert_eq!(args.query, None),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn index_missing_arg_is_clap_error() {
        let result = parse_cli_args(["sae", "index"]);
        assert!(result.is_err(), "index without team should be clap error");
    }

    #[test]
    fn rebuild_missing_arg_is_clap_error() {
        let result = parse_cli_args(["sae", "rebuild"]);
        assert!(result.is_err(), "rebuild without team should be clap error");
    }

    #[test]
    fn deprecated_harvest_not_rewritten_as_search() {
        let result = parse_cli_args(["sae", "harvest"]);
        assert!(
            result.is_err(),
            "deprecated `harvest` must produce a clap error, not be rewritten as search"
        );
    }

    #[test]
    fn shorthand_without_flags_has_json_false() {
        let cli = parse_cli_args(["sae", "query"]).unwrap();
        assert!(!cli.json, "json should default to false");
        match cli.command {
            Command::Search { args } => assert_eq!(args.query.as_deref(), Some("query")),
            other => panic!("expected Search, got {other:?}"),
        }
    }

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

    #[test]
    fn public_subcommands_matches_cli_non_hidden() {
        let cmd = Cli::command();
        let from_cli: Vec<&str> = cmd
            .get_subcommands()
            .filter(|s| !s.is_hide_set())
            .map(clap::Command::get_name)
            .collect();
        assert_eq!(
            from_cli.as_slice(),
            PUBLIC_SUBCOMMANDS,
            "PUBLIC_SUBCOMMANDS drifted from Cli's non-hidden subcommands. \
             Add the new subcommand to PUBLIC_SUBCOMMANDS, hide it from Cli, \
             or reorder to match the enum definition."
        );
    }

    #[test]
    fn help_output_contains_examples() {
        let cmd = Cli::command();
        for sub in cmd.get_subcommands() {
            if sub.is_hide_set() {
                continue;
            }
            let after_help = subcommand_after_help(sub.get_name());
            assert!(
                after_help.contains("Examples"),
                "subcommand '{}' should have Examples in after_help",
                sub.get_name()
            );
        }
    }

    #[test]
    fn create_help_includes_stdin_example() {
        let after_help = subcommand_after_help("create");
        assert!(
            after_help.contains("cat body.md | sae create --name \"Title\" --body-file -"),
            "create help should include stdin example"
        );
    }

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

    #[test]
    fn search_with_no_embed_parses_flag() {
        let cli = parse_cli_args(["sae", "search", "認証", "--no-embed"]).unwrap();
        match cli.command {
            Command::Search { args } => {
                assert_eq!(args.query.as_deref(), Some("認証"), "query should match");
                assert!(args.no_embed, "no_embed should be true");
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn search_without_no_embed_defaults_to_false() {
        let cli = parse_cli_args(["sae", "search", "認証"]).unwrap();
        match cli.command {
            Command::Search { args } => {
                assert!(!args.no_embed, "no_embed should default to false");
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn model_download_parses_correctly() {
        let cli = parse_cli_args(["sae", "model", "download"]).unwrap();
        match cli.command {
            Command::Model { command } => match command {
                ModelCommand::Download => {}
                other => panic!("expected Download, got {other:?}"),
            },
            other => panic!("expected Model {{ Download }}, got {other:?}"),
        }
    }

    #[test]
    fn model_without_subcommand_is_clap_error() {
        let result = parse_cli_args(["sae", "model"]);
        assert!(
            result.is_err(),
            "model without subcommand should be clap error, not search shorthand"
        );
    }

    #[test]
    fn model_download_with_json_flag() {
        let cli = parse_cli_args(["sae", "--json", "model", "download"]).unwrap();
        assert!(cli.json, "global json flag should be true");
        match cli.command {
            Command::Model { command } => match command {
                ModelCommand::Download => {}
                other => panic!("expected Download, got {other:?}"),
            },
            other => panic!("expected Model {{ Download }}, got {other:?}"),
        }
    }

    #[test]
    fn removed_embed_not_rewritten_as_search() {
        let result = parse_cli_args(["sae", "embed", "myteam"]);
        assert!(
            result.is_err(),
            "removed `embed` must produce a clap error, not be rewritten as search"
        );
    }

    #[test]
    fn multi_positional_args_not_shorthand() {
        let result = parse_cli_args(["sae", "foo", "bar"]);
        assert!(
            result.is_err(),
            "two positional args should be clap error, not shorthand"
        );
    }

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

    #[test]
    fn all_subcommands_not_shorthand() {
        for cmd in [
            "index", "rebuild", "get", "update", "ship", "archive", "status", "model",
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

    #[test]
    fn json_mode_detects_flag_in_argv() {
        assert!(json_mode_from_argv(["--json", "status"]));
        assert!(json_mode_from_argv(["status", "--json"]));
        assert!(json_mode_from_argv(["search", "--json", "query"]));
    }

    #[test]
    fn json_mode_returns_false_when_absent() {
        assert!(!json_mode_from_argv(["status"]));
        assert!(!json_mode_from_argv(["search", "query"]));
        assert!(!json_mode_from_argv::<[&str; 0], _>([]));
    }
}
