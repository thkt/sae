use sae::client::EsaPost;
use sae::storage::{EmbedResult, SearchResult, SyncStatus, TeamStatus};
use sae::sync::HarvestResult;

fn print_json<T: serde::Serialize + ?Sized>(val: &T) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string(val)?);
    Ok(())
}

pub fn harvest(result: &HarvestResult, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        print_json(result)?;
    } else {
        println!("{result}");
    }
    Ok(())
}

pub fn search(
    results: &[SearchResult],
    query: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        print_json(results)?;
    } else if results.is_empty() {
        println!("No results for '{query}'");
    } else {
        for r in results {
            let section = r
                .section_title
                .as_deref()
                .map(|s| format!(" > {s}"))
                .unwrap_or_default();
            println!(
                "[{:.4}] {}{} (#{})  {}",
                r.score, r.post_name, section, r.post_number, r.post_url
            );
            if !r.snippet.is_empty() {
                println!("  {}", r.snippet.replace('\n', " "));
            }
        }
    }
    Ok(())
}

pub fn get(post: &EsaPost, json: bool, with_body: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        let mut post = post.clone();
        if !with_body {
            post.body_md = None;
        }
        print_json(&post)?;
    } else {
        println!("---");
        println!("title: \"{}\"", post.full_name.replace('"', "\\\""));
        if let Some(ref cat) = post.category
            && !cat.is_empty()
        {
            println!("category: \"{}\"", cat.replace('"', "\\\""));
        }
        if !post.tags.is_empty() {
            let tags: Vec<String> = post
                .tags
                .iter()
                .map(|t| format!("\"{}\"", t.replace('"', "\\\"")))
                .collect();
            println!("tags: [{}]", tags.join(", "));
        }
        println!("author: \"@{}\"", post.created_by.screen_name);
        if post.updated_by.screen_name != post.created_by.screen_name {
            println!("updated_by: \"@{}\"", post.updated_by.screen_name);
        }
        println!("updated_at: \"{}\"", post.updated_at);
        if post.wip {
            println!("wip: true");
        }
        println!("number: {}", post.number);
        println!("url: {}", post.url);
        println!("---");
        println!();
        println!("{}", post.body_md.as_deref().unwrap_or("(empty)"));
    }
    Ok(())
}

pub fn action_result(
    action: &str,
    post: &EsaPost,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        print_json(post)?;
    } else {
        println!("{action}: {} (#{}) {}", post.name, post.number, post.url);
    }
    Ok(())
}

pub fn embed(
    result: &EmbedResult,
    done: u32,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        print_json(result)?;
    } else if result.chunks_embedded == 0 {
        if done == 0 {
            println!("Nothing to embed");
        } else {
            println!("All {done} chunks already embedded");
        }
    } else {
        println!("Embedded {} chunks", result.chunks_embedded);
    }
    Ok(())
}

pub fn dry_run(payload: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    print_json(payload)
}

pub fn model_download(json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        print_json(&serde_json::json!({"status": "ok"}))?;
    } else {
        println!("Model downloaded and verified");
    }
    Ok(())
}

pub fn status(statuses: &[TeamStatus], json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        print_json(statuses)?;
    } else {
        for ts in statuses {
            println!("--- {} ---", ts.team);
            match ts.status {
                SyncStatus::Error => {
                    println!(
                        "  Error: {}",
                        ts.error.as_deref().unwrap_or("unknown error")
                    );
                }
                SyncStatus::NotSynced => {
                    if let Some(ref path) = ts.db_path {
                        println!("  Not yet synced (no DB at {path})");
                    } else {
                        println!("  Not yet synced");
                    }
                }
                SyncStatus::Synced => {
                    println!("  Posts: {}", ts.posts);
                    if let Some(ref s) = ts.sync_state {
                        println!(
                            "  Last sync: {} (total: {}, local: {})",
                            s.updated_at, s.total_count, s.local_count
                        );
                        if let Some(pg) = s.last_page {
                            println!("  Checkpoint: page {pg} (interrupted)");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sae::storage::{SyncState, SyncStatus, TeamStatus};

    fn make_synced(team: &str, posts: u32) -> TeamStatus {
        TeamStatus {
            team: team.to_string(),
            status: SyncStatus::Synced,
            posts,
            sync_state: Some(SyncState {
                latest_updated_at: None,
                total_count: posts,
                local_count: posts,
                last_page: None,
                updated_at: "2025-01-01 00:00:00".to_string(),
            }),
            error: None,
            db_path: None,
        }
    }

    fn make_not_synced(team: &str) -> TeamStatus {
        TeamStatus {
            team: team.to_string(),
            status: SyncStatus::NotSynced,
            posts: 0,
            sync_state: None,
            error: None,
            db_path: Some("/tmp/team.db".to_string()),
        }
    }

    fn make_error(team: &str, msg: &str) -> TeamStatus {
        TeamStatus {
            team: team.to_string(),
            status: SyncStatus::Error,
            posts: 0,
            sync_state: None,
            error: Some(msg.to_string()),
            db_path: None,
        }
    }

    // TC-007: status --json → valid JSON array
    #[test]
    fn status_json_emits_json_array() {
        let statuses = vec![make_synced("team-a", 10), make_not_synced("team-b")];
        // Verify no error and output can be parsed (capture via serde_json directly)
        let json = serde_json::to_string(&statuses).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.is_array());
        assert_eq!(v[0]["team"], "team-a");
        assert_eq!(v[0]["status"], "synced");
        assert_eq!(v[1]["status"], "not_synced");
    }

    // TC-007: status human-readable empty list — no panic
    #[test]
    fn status_human_empty_no_panic() {
        status(&[], false).unwrap();
    }

    // TC-007: embed json → chunks_embedded field
    #[test]
    fn embed_json_contains_chunks_embedded() {
        let result = sae::storage::EmbedResult {
            chunks_embedded: 42,
        };
        let json = serde_json::to_string(&result).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["chunks_embedded"], 42);
    }

    // TC-007: embed human-readable with done=0 → "Nothing to embed"
    #[test]
    fn embed_human_zero_done_no_panic() {
        let result = sae::storage::EmbedResult { chunks_embedded: 0 };
        embed(&result, 0, false).unwrap();
    }

    // TC-007: status error variant — error field displayed
    #[test]
    fn status_error_variant_serializes_message() {
        let statuses = vec![make_error("broken", "db missing")];
        let json = serde_json::to_string(&statuses).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v[0]["status"], "error");
        assert_eq!(v[0]["error"], "db missing");
    }

    // TC-007: status with checkpoint (interrupted sync) — last_page serialized
    #[test]
    fn status_synced_with_checkpoint_serializes_last_page() {
        let ts = TeamStatus {
            team: "team-c".to_string(),
            status: SyncStatus::Synced,
            posts: 5,
            sync_state: Some(SyncState {
                latest_updated_at: None,
                total_count: 100,
                local_count: 5,
                last_page: Some(3),
                updated_at: "2025-06-01 00:00:00".to_string(),
            }),
            error: None,
            db_path: None,
        };
        let json = serde_json::to_string(&ts).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["sync_state"]["last_page"], 3);
    }
}
