use crate::client::{EsaClient, UpdatePostParams};
use crate::config::Config;
use crate::envelope::CommandOutput;
use crate::output;
use crate::storage::Db;
use crate::sync;
use crate::tools::{SaeError, resolve_client};

pub(crate) fn archive_category(current: Option<&str>) -> Option<String> {
    let current = current.unwrap_or("");
    if current.starts_with("Archived/") || current == "Archived" {
        return None;
    }
    Some(if current.is_empty() {
        "Archived".to_owned()
    } else {
        format!("Archived/{current}")
    })
}

pub(crate) async fn run_archive(
    config: &Config,
    db: Option<&Db>,
    number: u32,
    team: Option<&str>,
    dry_run: bool,
) -> Result<CommandOutput, SaeError> {
    let (team, client) = resolve_client(config, team)?;
    archive_via_client(team, &client, db, number, dry_run).await
}

pub(crate) async fn archive_via_client(
    team: &str,
    client: &EsaClient,
    db: Option<&Db>,
    number: u32,
    dry_run: bool,
) -> Result<CommandOutput, SaeError> {
    let post = client.get_post(team, number).await?;
    let new_category = match archive_category(post.category.as_deref()) {
        Some(c) => c,
        None if dry_run => {
            return output::dry_run(&serde_json::json!({
                "number": number,
                "already_archived": true,
                "category": post.category.as_deref().unwrap_or(""),
            }));
        }
        None => {
            return output::action_result("Already archived", &post);
        }
    };
    if dry_run {
        return output::dry_run(&serde_json::json!({
            "number": number,
            "from_category": post.category.as_deref().unwrap_or(""),
            "to_category": new_category,
        }));
    }
    let params = UpdatePostParams {
        category: Some(&new_category),
        ..Default::default()
    };
    let post = client.update_post(team, number, &params).await?;
    sync::try_upsert_post_locally(db, &post);
    output::action_result("Archived", &post)
}

pub(crate) async fn run_ship(
    config: &Config,
    db: Option<&Db>,
    number: u32,
    team: Option<&str>,
    dry_run: bool,
) -> Result<CommandOutput, SaeError> {
    if dry_run {
        return output::dry_run(&serde_json::json!({
            "number": number,
            "wip": false,
        }));
    }
    let (team, client) = resolve_client(config, team)?;
    ship_via_client(team, &client, db, number).await
}

pub(crate) async fn ship_via_client(
    team: &str,
    client: &EsaClient,
    db: Option<&Db>,
    number: u32,
) -> Result<CommandOutput, SaeError> {
    let params = UpdatePostParams {
        wip: Some(false),
        ..Default::default()
    };
    let post = client.update_post(team, number, &params).await?;
    sync::try_upsert_post_locally(db, &post);
    output::action_result("Shipped", &post)
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-140: archive_category returns "Archived" when post has no category
    #[test]
    fn archive_no_category() {
        assert_eq!(archive_category(None), Some("Archived".into()));
        assert_eq!(archive_category(Some("")), Some("Archived".into()));
    }

    // T-141: archive_category prefixes existing category with "Archived/"
    #[test]
    fn archive_with_category() {
        assert_eq!(
            archive_category(Some("dev/guide")),
            Some("Archived/dev/guide".into())
        );
    }

    // T-142: archive_category returns None when post is already archived
    #[test]
    fn archive_already_archived() {
        assert_eq!(archive_category(Some("Archived")), None);
        assert_eq!(archive_category(Some("Archived/dev")), None);
    }

    // T-143: archive_category treats "ArchivedData" as not archived (prefix only)
    #[test]
    fn archive_not_prefix_match() {
        assert_eq!(
            archive_category(Some("ArchivedData")),
            Some("Archived/ArchivedData".into())
        );
    }

    use crate::sync::tests::esa_post_fixture;

    fn esa_post_json(
        number: u32,
        category: Option<&str>,
        wip: bool,
        updated_at: &str,
    ) -> serde_json::Value {
        esa_post_fixture(number, "Post", "# x", category, wip, updated_at)
    }

    // T-313: archive_via_client moves category and writes through to local DB
    #[tokio::test]
    async fn archive_via_client_writes_through_to_local_db() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts/5"))
            .respond_with(ResponseTemplate::new(200).set_body_json(esa_post_json(
                5,
                Some("dev/guide"),
                false,
                "2025-01-01T00:00:00+09:00",
            )))
            .mount(&server)
            .await;
        Mock::given(method("PATCH"))
            .and(path("/teams/myteam/posts/5"))
            .respond_with(ResponseTemplate::new(200).set_body_json(esa_post_json(
                5,
                Some("Archived/dev/guide"),
                false,
                "2025-06-01T00:00:00+09:00",
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        archive_via_client("myteam", &client, Some(&db), 5, false)
            .await
            .expect("archive should succeed");

        let category: String = db
            .conn()
            .query_row("SELECT category FROM posts WHERE number = 5", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(category, "Archived/dev/guide");
    }

    // T-314: ship_via_client unsets wip and reflects to local DB
    #[tokio::test]
    async fn ship_via_client_writes_through_to_local_db() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PATCH"))
            .and(path("/teams/myteam/posts/9"))
            .respond_with(ResponseTemplate::new(200).set_body_json(esa_post_json(
                9,
                Some("dev"),
                false,
                "2025-06-01T00:00:00+09:00",
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        ship_via_client("myteam", &client, Some(&db), 9)
            .await
            .expect("ship should succeed");

        let wip: bool = db
            .conn()
            .query_row("SELECT wip FROM posts WHERE number = 9", [], |r| r.get(0))
            .unwrap();
        assert!(!wip, "wip should be false after ship");
    }
}
