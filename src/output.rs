use sae::client::EsaPost;
use sae::storage::{EmbedResult, SearchResult, SyncStatus, TeamStatus};
use sae::sync::HarvestResult;

pub fn harvest(result: &HarvestResult, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        println!("{}", serde_json::to_string(result)?);
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
        println!("{}", serde_json::to_string(results)?);
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
        println!("{}", serde_json::to_string(&post)?);
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

pub fn post(action: &str, post: &EsaPost, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        println!("{}", serde_json::to_string(post)?);
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
        println!("{}", serde_json::to_string(result)?);
    } else if done == 0 {
        println!("All chunks already embedded");
    } else {
        println!("Embedded {} chunks", result.chunks_embedded);
    }
    Ok(())
}

pub fn status(statuses: &[TeamStatus], json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        println!("{}", serde_json::to_string(statuses)?);
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
