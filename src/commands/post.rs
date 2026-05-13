use std::fs;
use std::io::{self, Read};

use crate::client::{CreatePostParams, EsaClient, UpdatePostParams};
use crate::config::Config;
use crate::envelope::CommandOutput;
use crate::output;
use crate::storage::Db;
use crate::sync;
use crate::tools::{CreateArgs, SaeError, UpdateArgs, resolve_client};

pub(crate) async fn run_get(
    config: &Config,
    number: u32,
    team: Option<&str>,
    with_body: bool,
) -> Result<CommandOutput, SaeError> {
    let (team, client) = resolve_client(config, team)?;
    let post = client.get_post(team, number).await?;
    output::get(&post, with_body)
}

pub(crate) async fn run_create(
    config: &Config,
    db: Option<&Db>,
    args: CreateArgs,
) -> Result<CommandOutput, SaeError> {
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
    create_via_client(team, &client, db, &args, resolved_body.as_deref()).await
}

pub(crate) async fn create_via_client(
    team: &str,
    client: &EsaClient,
    db: Option<&Db>,
    args: &CreateArgs,
    body_md: Option<&str>,
) -> Result<CommandOutput, SaeError> {
    let params = CreatePostParams {
        name: &args.name,
        body_md,
        category: args.category.as_deref(),
        tags: args.tag.clone(),
        wip: args.wip,
    };
    let post = client.create_post(team, &params).await?;
    sync::try_upsert_post_locally(db, &post);
    output::action_result("Created", &post)
}

pub(crate) async fn run_update(
    config: &Config,
    db: Option<&Db>,
    args: UpdateArgs,
) -> Result<CommandOutput, SaeError> {
    let resolved_body = resolve_body(args.body.as_deref(), args.body_file.as_deref())?;
    let tags: Option<Vec<String>> = if args.tag.is_empty() {
        None
    } else {
        Some(args.tag.clone())
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
    update_via_client(team, &client, db, &args, resolved_body.as_deref(), tags).await
}

pub(crate) async fn update_via_client(
    team: &str,
    client: &EsaClient,
    db: Option<&Db>,
    args: &UpdateArgs,
    body_md: Option<&str>,
    tags: Option<Vec<String>>,
) -> Result<CommandOutput, SaeError> {
    let params = UpdatePostParams {
        name: args.name.as_deref(),
        body_md,
        category: args.category.as_deref(),
        tags,
        ..Default::default()
    };
    let post = client.update_post(team, args.number, &params).await?;
    sync::try_upsert_post_locally(db, &post);
    output::action_result("Updated", &post)
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

    use crate::sync::tests::esa_post_fixture;

    fn esa_post_json(
        number: u32,
        name: &str,
        body_md: &str,
        updated_at: &str,
    ) -> serde_json::Value {
        esa_post_fixture(number, name, body_md, Some("dev"), false, updated_at)
    }

    fn create_args(name: &str) -> CreateArgs {
        CreateArgs {
            name: name.into(),
            body: None,
            body_file: None,
            category: None,
            tag: vec![],
            wip: false,
            team: None,
            dry_run: false,
        }
    }

    fn update_args(number: u32) -> UpdateArgs {
        UpdateArgs {
            number,
            name: None,
            body: None,
            body_file: None,
            category: None,
            tag: vec![],
            team: None,
            dry_run: false,
        }
    }

    // T-310: create_via_client reflects API response into local DB
    #[tokio::test]
    async fn create_via_client_writes_through_to_local_db() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/teams/myteam/posts"))
            .respond_with(ResponseTemplate::new(201).set_body_json(esa_post_json(
                42,
                "Created Post",
                "# Hello",
                "2025-06-01T00:00:00+09:00",
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();
        let args = create_args("Created Post");

        let out = create_via_client("myteam", &client, Some(&db), &args, Some("# Hello"))
            .await
            .expect("create should succeed");
        assert!(out.markdown.contains("Created"));

        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 42", [], |r| r.get(0))
            .unwrap();
        assert_eq!(name, "Created Post", "post should be persisted locally");
    }

    // T-311: update_via_client overwrites local row with the API response
    #[tokio::test]
    async fn update_via_client_overwrites_local_post() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PATCH"))
            .and(path("/teams/myteam/posts/7"))
            .respond_with(ResponseTemplate::new(200).set_body_json(esa_post_json(
                7,
                "Renamed",
                "# Updated body",
                "2025-07-01T00:00:00+09:00",
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();
        let args = update_args(7);

        update_via_client(
            "myteam",
            &client,
            Some(&db),
            &args,
            Some("# Updated body"),
            None,
        )
        .await
        .expect("update should succeed");

        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 7", [], |r| r.get(0))
            .unwrap();
        assert_eq!(name, "Renamed");
    }

    // T-312: create_via_client succeeds even when db=None (write-through skipped)
    #[tokio::test]
    async fn create_via_client_works_without_db() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/teams/myteam/posts"))
            .respond_with(ResponseTemplate::new(201).set_body_json(esa_post_json(
                1,
                "No DB",
                "",
                "2025-06-01T00:00:00+09:00",
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
        let args = create_args("No DB");
        create_via_client("myteam", &client, None, &args, None)
            .await
            .expect("create should still succeed without a DB");
    }
}
