use crate::client::UpdatePostParams;
use crate::config::Config;
use crate::output;
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
    number: u32,
    team: Option<&str>,
    dry_run: bool,
    json: bool,
) -> Result<String, SaeError> {
    let (team, client) = resolve_client(config, team)?;
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
            return output::action_result("Already archived", &post, json);
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
    output::action_result("Archived", &post, json)
}

pub(crate) async fn run_ship(
    config: &Config,
    number: u32,
    team: Option<&str>,
    dry_run: bool,
    json: bool,
) -> Result<String, SaeError> {
    if dry_run {
        return output::dry_run(&serde_json::json!({
            "number": number,
            "wip": false,
        }));
    }
    let (team, client) = resolve_client(config, team)?;
    let params = UpdatePostParams {
        wip: Some(false),
        ..Default::default()
    };
    let post = client.update_post(team, number, &params).await?;
    output::action_result("Shipped", &post, json)
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
}
