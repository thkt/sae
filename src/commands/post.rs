use std::fs;
use std::io::{self, Read};

use crate::client::{CreatePostParams, UpdatePostParams};
use crate::config::Config;

use crate::output;
use crate::tools::{CreateArgs, SaeError, UpdateArgs, resolve_client};

pub(crate) async fn run_get(
    config: &Config,
    number: u32,
    team: Option<&str>,
    with_body: bool,
    json: bool,
) -> Result<String, SaeError> {
    let (team, client) = resolve_client(config, team)?;
    let post = client.get_post(team, number).await?;
    output::get(&post, json, with_body)
}

pub(crate) async fn run_create(
    config: &Config,
    args: CreateArgs,
    json: bool,
) -> Result<String, SaeError> {
    let resolved_body = resolve_body(args.body.as_deref(), args.body_file.as_deref())?;
    if args.dry_run {
        return output::dry_run(&serde_json::json!({
            "name": args.name,
            "body_md": resolved_body,
            "category": args.category,
            "tags": args.tag,
            "wip": args.wip,
        }));
    }
    let (team, client) = resolve_client(config, args.team.as_deref())?;
    let params = CreatePostParams {
        name: &args.name,
        body_md: resolved_body.as_deref(),
        category: args.category.as_deref(),
        tags: args.tag,
        wip: args.wip,
    };
    let post = client.create_post(team, &params).await?;
    output::action_result("Created", &post, json)
}

pub(crate) async fn run_update(
    config: &Config,
    args: UpdateArgs,
    json: bool,
) -> Result<String, SaeError> {
    let resolved_body = resolve_body(args.body.as_deref(), args.body_file.as_deref())?;
    let tags: Option<Vec<String>> = if args.tag.is_empty() {
        None
    } else {
        Some(args.tag)
    };
    if args.dry_run {
        return output::dry_run(&serde_json::json!({
            "number": args.number,
            "name": args.name,
            "body_md": resolved_body,
            "category": args.category,
            "tags": tags.as_deref(),
        }));
    }
    let (team, client) = resolve_client(config, args.team.as_deref())?;
    let params = UpdatePostParams {
        name: args.name.as_deref(),
        body_md: resolved_body.as_deref(),
        category: args.category.as_deref(),
        tags,
        ..Default::default()
    };
    let post = client.update_post(team, args.number, &params).await?;
    output::action_result("Updated", &post, json)
}

pub(crate) fn resolve_body(
    body: Option<&str>,
    body_file: Option<&str>,
) -> Result<Option<String>, SaeError> {
    let mut stdin = io::stdin();
    resolve_body_with_reader(body, body_file, &mut stdin)
}

pub(crate) fn resolve_body_with_reader(
    body: Option<&str>,
    body_file: Option<&str>,
    stdin: &mut impl Read,
) -> Result<Option<String>, SaeError> {
    match (body, body_file) {
        (Some(b), None) => Ok(Some(b.to_owned())),
        (None, Some("-")) => {
            let mut buf = String::new();
            stdin.read_to_string(&mut buf)?;
            Ok(Some(buf))
        }
        (None, Some(path)) => {
            let content = fs::read_to_string(path)?;
            Ok(Some(content))
        }
        (None, None) => Ok(None),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with prevents this"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::tempdir;

    // T-060: resolve_body(Some, None) → inline body
    #[test]
    fn resolve_body_inline_text() {
        let result = resolve_body(Some("inline text"), None).unwrap();
        assert_eq!(result.as_deref(), Some("inline text"));
    }

    // T-061: resolve_body(None, None) → no body
    #[test]
    fn resolve_body_none_returns_none() {
        let result = resolve_body(None, None).unwrap();
        assert_eq!(result, None);
    }

    // T-062: resolve_body with nonexistent file → error
    #[test]
    fn resolve_body_nonexistent_file_is_error() {
        let result = resolve_body(None, Some("/nonexistent/path.md"));
        assert!(result.is_err(), "nonexistent file should return error");
    }

    // T-063: resolve_body_with_reader(`-`) reads from stdin
    #[test]
    fn resolve_body_with_reader_reads_stdin_when_dash() {
        let result = resolve_body_with_reader(None, Some("-"), &mut Cursor::new("本文\n")).unwrap();
        assert_eq!(
            result.as_deref(),
            Some("本文\n"),
            "body from stdin should preserve trailing newline"
        );
    }

    // T-064: resolve_body_with_reader(`-`) with empty stdin → Some("")
    #[test]
    fn resolve_body_with_reader_empty_stdin_returns_empty_body() {
        let result = resolve_body_with_reader(None, Some("-"), &mut Cursor::new("")).unwrap();
        assert_eq!(
            result.as_deref(),
            Some(""),
            "empty stdin yields empty body string"
        );
    }

    // T-039: --body-file <tempfile> with create → file content becomes body
    #[test]
    fn body_file_reads_from_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("body.md");
        fs::write(&file_path, "# Hello\nBody from file").unwrap();

        let result = resolve_body(None, Some(file_path.to_str().unwrap())).unwrap();
        assert_eq!(
            result.as_deref(),
            Some("# Hello\nBody from file"),
            "body should contain file contents"
        );
    }

    // T-040: --body-file - with create → stdin content becomes body
    #[test]
    fn body_file_dash_reads_from_stdin() {
        let mut stdin = Cursor::new("# Hello\nBody from stdin\n");
        let result = resolve_body_with_reader(None, Some("-"), &mut stdin).unwrap();
        assert_eq!(
            result.as_deref(),
            Some("# Hello\nBody from stdin\n"),
            "body should contain stdin contents as-is"
        );
    }
}
