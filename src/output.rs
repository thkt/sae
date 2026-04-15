use sae::client::EsaPost;
use sae::storage::{EmbedResult, SearchResult, SyncStatus, TeamStatus};
use sae::sync::HarvestResult;

use crate::SaeError;

fn format_json<T: serde::Serialize + ?Sized>(val: &T) -> Result<String, SaeError> {
    Ok(serde_json::to_string(val)?)
}

pub(crate) fn harvest(result: &HarvestResult, json: bool) -> Result<String, SaeError> {
    if json {
        format_json(result)
    } else {
        Ok(result.to_string())
    }
}

pub(crate) struct EmbedInfo {
    pub processed: u32,
    pub budget_exhausted: bool,
    pub team: String,
}

pub(crate) fn search(
    results: &[SearchResult],
    query: &str,
    json: bool,
    semantic: bool,
    embed_info: Option<EmbedInfo>,
) -> Result<String, SaeError> {
    let search_mode = if semantic { "hybrid" } else { "fts" };
    if json {
        format_json(&serde_json::json!({
            "search_mode": search_mode,
            "results": results,
        }))
    } else if results.is_empty() {
        Ok(format!("No results for '{query}'"))
    } else {
        let mut lines = Vec::new();
        if !semantic {
            lines.push("(semantic search unavailable, showing text-match results)".to_owned());
        }
        for r in results {
            let section = r
                .section_title
                .as_deref()
                .map(|s| format!(" > {s}"))
                .unwrap_or_default();
            lines.push(format!(
                "[{:.4}] {}{} (#{})  {}",
                r.score, r.post_name, section, r.post_number, r.post_url
            ));
            if !r.snippet.is_empty() {
                lines.push(format!("  {}", r.snippet.replace('\n', " ")));
            }
        }
        if let Some(info) = embed_info {
            if info.processed > 0 {
                lines.push(format!("(Embedded {} new chunks)", info.processed));
            }
            if info.budget_exhausted {
                lines.push(format!(
                    "Hint: more chunks pending. Run 'sae embed {}' to index all.",
                    info.team
                ));
            }
        }
        Ok(lines.join("\n"))
    }
}

pub(crate) fn get(post: &EsaPost, json: bool, with_body: bool) -> Result<String, SaeError> {
    if json {
        if with_body {
            format_json(post)
        } else {
            let mut val = serde_json::to_value(post)?;
            if let Some(obj) = val.as_object_mut() {
                obj.remove("body_md");
            }
            format_json(&val)
        }
    } else {
        Ok(format_post_frontmatter(post))
    }
}

fn format_post_frontmatter(post: &EsaPost) -> String {
    let mut lines = Vec::new();
    lines.push("---".to_owned());
    lines.push(format!(
        "title: \"{}\"",
        post.full_name.replace('"', "\\\"")
    ));
    if let Some(ref cat) = post.category
        && !cat.is_empty()
    {
        lines.push(format!("category: \"{}\"", cat.replace('"', "\\\"")));
    }
    if !post.tags.is_empty() {
        let tags: Vec<String> = post
            .tags
            .iter()
            .map(|t| format!("\"{}\"", t.replace('"', "\\\"")))
            .collect();
        lines.push(format!("tags: [{}]", tags.join(", ")));
    }
    lines.push(format!("author: \"@{}\"", post.created_by.screen_name));
    if post.updated_by.screen_name != post.created_by.screen_name {
        lines.push(format!("updated_by: \"@{}\"", post.updated_by.screen_name));
    }
    lines.push(format!("updated_at: \"{}\"", post.updated_at));
    if post.wip {
        lines.push("wip: true".to_owned());
    }
    lines.push(format!("number: {}", post.number));
    lines.push(format!("url: {}", post.url));
    lines.push("---".to_owned());
    lines.push(String::new());
    lines.push(post.body_md.as_deref().unwrap_or("(empty)").to_owned());
    lines.join("\n")
}

pub(crate) fn action_result(action: &str, post: &EsaPost, json: bool) -> Result<String, SaeError> {
    if json {
        format_json(post)
    } else {
        Ok(format!(
            "{action}: {} (#{}) {}",
            post.name, post.number, post.url
        ))
    }
}

pub(crate) fn embed(
    result: &EmbedResult,
    total_chunks: u32,
    json: bool,
) -> Result<String, SaeError> {
    if json {
        format_json(result)
    } else if result.chunks_embedded == 0 {
        if total_chunks == 0 {
            Ok("Nothing to embed".to_owned())
        } else {
            Ok(format!("All {total_chunks} chunks already embedded"))
        }
    } else {
        Ok(format!("Embedded {} chunks", result.chunks_embedded))
    }
}

/// Always emits JSON regardless of `--json` flag. Dry-run output is for machine consumption.
pub(crate) fn dry_run(payload: &serde_json::Value) -> Result<String, SaeError> {
    format_json(payload)
}

pub(crate) fn model_download(json: bool) -> Result<String, SaeError> {
    if json {
        format_json(&serde_json::json!({"status": "ok"}))
    } else {
        Ok("Model downloaded and verified".to_owned())
    }
}

pub(crate) fn status(statuses: &[TeamStatus], json: bool) -> Result<String, SaeError> {
    if json {
        format_json(statuses)
    } else {
        let mut lines = Vec::new();
        for ts in statuses {
            lines.push(format!("--- {} ---", ts.team));
            match ts.status {
                SyncStatus::Error => {
                    lines.push(format!(
                        "  Error: {}",
                        ts.error.as_deref().unwrap_or("unknown error")
                    ));
                }
                SyncStatus::NotSynced => {
                    if let Some(ref path) = ts.db_path {
                        lines.push(format!("  Not yet synced (no DB at {path})"));
                    } else {
                        lines.push("  Not yet synced".to_owned());
                    }
                }
                SyncStatus::Synced => {
                    lines.push(format!("  Posts: {}", ts.posts));
                    if ts.pending_embed > 0 {
                        lines.push(format!(
                            "  Pending embed: {} chunks (run 'sae embed {}' to index)",
                            ts.pending_embed, ts.team
                        ));
                    }
                    if let Some(ref s) = ts.sync_state {
                        lines.push(format!(
                            "  Last sync: {} (total: {}, local: {})",
                            s.updated_at, s.total_count, s.local_count
                        ));
                        if let Some(pg) = s.last_page {
                            lines.push(format!("  Checkpoint: page {pg} (interrupted)"));
                        }
                    }
                }
            }
        }
        Ok(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sae::storage::{EmbedResult, MatchSource, SyncState};

    fn make_synced(team: &str, posts: u32) -> TeamStatus {
        TeamStatus {
            team: team.to_owned(),
            status: SyncStatus::Synced,
            posts,
            pending_embed: 0,
            sync_state: Some(SyncState {
                latest_updated_at: None,
                total_count: posts,
                local_count: posts,
                last_page: None,
                updated_at: "2025-01-01 00:00:00".to_owned(),
            }),
            error: None,
            db_path: None,
        }
    }

    fn make_not_synced(team: &str) -> TeamStatus {
        TeamStatus {
            team: team.to_owned(),
            status: SyncStatus::NotSynced,
            posts: 0,
            pending_embed: 0,
            sync_state: None,
            error: None,
            db_path: Some("/tmp/team.db".to_owned()),
        }
    }

    fn make_error(team: &str, msg: &str) -> TeamStatus {
        TeamStatus {
            team: team.to_owned(),
            status: SyncStatus::Error,
            posts: 0,
            pending_embed: 0,
            sync_state: None,
            error: Some(msg.to_owned()),
            db_path: None,
        }
    }

    // T-079: status human-readable empty list → empty string
    #[test]
    fn status_human_empty_returns_empty_string() {
        let out = status(&[], false).unwrap();
        assert!(out.is_empty(), "empty list should produce empty output");
    }

    // T-080: status human-readable synced team → expected lines
    #[test]
    fn status_human_synced_contains_expected_lines() {
        let out = status(&[make_synced("team-a", 10)], false).unwrap();
        assert!(out.contains("--- team-a ---"));
        assert!(out.contains("Posts: 10"));
        assert!(out.contains("Last sync: 2025-01-01 00:00:00"));
    }

    // T-081: status human-readable not-synced team → expected lines
    #[test]
    fn status_human_not_synced_contains_expected_lines() {
        let out = status(&[make_not_synced("team-b")], false).unwrap();
        assert!(out.contains("--- team-b ---"));
        assert!(out.contains("Not yet synced"));
    }

    // T-082: embed json → chunks_embedded field
    #[test]
    fn embed_json_contains_chunks_embedded() {
        let result = EmbedResult {
            chunks_embedded: 42,
        };
        let out = embed(&result, 42, true).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["chunks_embedded"], 42);
    }

    // T-083: embed human-readable with done=0 → "Nothing to embed"
    #[test]
    fn embed_human_zero_done_returns_nothing_to_embed() {
        let result = EmbedResult { chunks_embedded: 0 };
        let out = embed(&result, 0, false).unwrap();
        assert_eq!(out, "Nothing to embed");
    }

    // T-084: embed human-readable chunks > 0 → "Embedded N chunks"
    #[test]
    fn embed_human_nonzero_returns_embedded_count() {
        let result = EmbedResult { chunks_embedded: 5 };
        let out = embed(&result, 5, false).unwrap();
        assert_eq!(out, "Embedded 5 chunks");
    }

    // T-085: status error variant — error field displayed
    #[test]
    fn status_error_variant_serializes_message() {
        let statuses = vec![make_error("broken", "db missing")];
        let json = serde_json::to_string(&statuses).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v[0]["status"], "error");
        assert_eq!(v[0]["error"], "db missing");
    }

    // T-086: status with checkpoint (interrupted sync) — last_page serialized
    #[test]
    fn status_synced_with_checkpoint_serializes_last_page() {
        let ts = TeamStatus {
            team: "team-c".to_owned(),
            status: SyncStatus::Synced,
            posts: 5,
            pending_embed: 0,
            sync_state: Some(SyncState {
                latest_updated_at: None,
                total_count: 100,
                local_count: 5,
                last_page: Some(3),
                updated_at: "2025-06-01 00:00:00".to_owned(),
            }),
            error: None,
            db_path: None,
        };
        let json = serde_json::to_string(&ts).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["sync_state"]["last_page"], 3);
    }

    // T-087: search json includes search_mode hybrid
    #[test]
    fn search_json_hybrid_includes_search_mode() {
        let out = search(&[], "test", true, true, None).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["search_mode"], "hybrid");
        assert!(v["results"].is_array());
    }

    // T-088: search json includes search_mode fts
    #[test]
    fn search_json_fts_includes_search_mode() {
        let out = search(&[], "test", true, false, None).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["search_mode"], "fts");
    }

    // T-089: search human fts fallback shows notice
    #[test]
    fn search_human_fts_fallback_shows_notice() {
        let results = vec![SearchResult {
            post_number: 1,
            post_name: "Test".to_owned(),
            post_url: "https://example.com".to_owned(),
            section_title: None,
            snippet: String::new(),
            score: 0.5,
            match_source: MatchSource::Fts,
        }];
        let out = search(&results, "q", false, false, None).unwrap();
        assert!(out.contains("semantic search unavailable"));
    }

    // T-090: search human hybrid does not show fallback notice
    #[test]
    fn search_human_hybrid_no_fallback_notice() {
        let results = vec![SearchResult {
            post_number: 1,
            post_name: "Test".to_owned(),
            post_url: "https://example.com".to_owned(),
            section_title: None,
            snippet: String::new(),
            score: 0.5,
            match_source: MatchSource::Fts,
        }];
        let out = search(&results, "q", false, true, None).unwrap();
        assert!(!out.contains("semantic search unavailable"));
    }
}
