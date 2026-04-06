use sae::client::UpdatePostParams;
use sae::config::Config;

use crate::{resolve_client, AppError};

pub(crate) fn archive_category(current: Option<&str>) -> Option<String> {
    let current = current.unwrap_or("");
    if current.starts_with("Archived/") || current == "Archived" {
        return None;
    }
    Some(if current.is_empty() {
        "Archived".to_string()
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
) -> Result<(), AppError> {
    let (team, client) = resolve_client(config, team)?;
    let post = client.get_post(team, number).await?;
    let new_category = match archive_category(post.category.as_deref()) {
        Some(c) => c,
        None if dry_run => {
            crate::output::dry_run(&serde_json::json!({
                "number": number,
                "already_archived": true,
                "category": post.category.as_deref().unwrap_or(""),
            }))?;
            return Ok(());
        }
        None => {
            crate::output::action_result("Already archived", &post, json)?;
            return Ok(());
        }
    };
    if dry_run {
        crate::output::dry_run(&serde_json::json!({
            "number": number,
            "from_category": post.category.as_deref().unwrap_or(""),
            "to_category": new_category,
        }))?;
        return Ok(());
    }
    let params = UpdatePostParams {
        category: Some(&new_category),
        ..Default::default()
    };
    let post = client.update_post(team, number, &params).await?;
    crate::output::action_result("Archived", &post, json)?;
    Ok(())
}

pub(crate) async fn run_ship(
    config: &Config,
    number: u32,
    team: Option<&str>,
    dry_run: bool,
    json: bool,
) -> Result<(), AppError> {
    if dry_run {
        crate::output::dry_run(&serde_json::json!({
            "number": number,
            "wip": false,
        }))?;
        return Ok(());
    }
    let (team, client) = resolve_client(config, team)?;
    let params = UpdatePostParams {
        wip: Some(false),
        ..Default::default()
    };
    let post = client.update_post(team, number, &params).await?;
    crate::output::action_result("Shipped", &post, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_no_category() {
        assert_eq!(archive_category(None), Some("Archived".into()));
        assert_eq!(archive_category(Some("")), Some("Archived".into()));
    }

    #[test]
    fn archive_with_category() {
        assert_eq!(
            archive_category(Some("dev/guide")),
            Some("Archived/dev/guide".into())
        );
    }

    #[test]
    fn archive_already_archived() {
        assert_eq!(archive_category(Some("Archived")), None);
        assert_eq!(archive_category(Some("Archived/dev")), None);
    }

    #[test]
    fn archive_not_prefix_match() {
        assert_eq!(
            archive_category(Some("ArchivedData")),
            Some("Archived/ArchivedData".into())
        );
    }
}
