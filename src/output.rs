use crate::client::EsaPost;
use crate::envelope::CommandOutput;
use crate::storage::{EmbedResult, SearchResult, SyncStatus, TeamStatus};
use crate::sync::HarvestResult;
use crate::tools::SaeError;

pub(crate) fn harvest(result: &HarvestResult) -> Result<CommandOutput, SaeError> {
    let markdown = result.to_string();
    let data = serde_json::to_value(result)?;
    Ok(CommandOutput::ok(markdown, data))
}

pub(crate) struct EmbedInfo {
    pub(crate) processed: u32,
    pub(crate) budget_exhausted: bool,
    pub(crate) team: String,
}

pub(crate) fn search(
    results: &[SearchResult],
    query: &str,
    semantic: bool,
    embedder_load_failed: bool,
    embed_info: Option<EmbedInfo>,
) -> Result<CommandOutput, SaeError> {
    let search_mode = if semantic { "hybrid" } else { "fts" };
    let data = serde_json::json!({
        "search_mode": search_mode,
        "results": results,
    });
    let markdown = if results.is_empty() {
        format!("No results for '{query}'")
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
        lines.join("\n")
    };
    if embedder_load_failed {
        Ok(CommandOutput::with_notes(
            markdown,
            data,
            true,
            vec!["semantic search unavailable, falling back to FTS".to_owned()],
        ))
    } else {
        Ok(CommandOutput::ok(markdown, data))
    }
}

pub(crate) fn get(post: &EsaPost, with_body: bool) -> Result<CommandOutput, SaeError> {
    let markdown = format_post_frontmatter(post);
    let data = if with_body {
        serde_json::to_value(post)?
    } else {
        let mut val = serde_json::to_value(post)?;
        if let Some(obj) = val.as_object_mut() {
            obj.remove("body_md");
        }
        val
    };
    Ok(CommandOutput::ok(markdown, data))
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

pub(crate) fn action_result(action: &str, post: &EsaPost) -> Result<CommandOutput, SaeError> {
    let markdown = format!("{action}: {} (#{}) {}", post.name, post.number, post.url);
    let data = serde_json::to_value(post)?;
    Ok(CommandOutput::ok(markdown, data))
}

pub(crate) fn embed(result: &EmbedResult, total_chunks: u32) -> Result<CommandOutput, SaeError> {
    let markdown = if result.chunks_embedded == 0 {
        if total_chunks == 0 {
            "Nothing to embed".to_owned()
        } else {
            format!("All {total_chunks} chunks already embedded")
        }
    } else {
        format!("Embedded {} chunks", result.chunks_embedded)
    };
    let data = serde_json::to_value(result)?;
    Ok(CommandOutput::ok(markdown, data))
}

/// Dry-run is always envelope-wrapped (consistent shape regardless of `--json`).
/// `markdown` carries the pretty-printed payload for default-mode visibility.
pub(crate) fn dry_run(payload: &serde_json::Value) -> Result<CommandOutput, SaeError> {
    let markdown = serde_json::to_string_pretty(payload)?;
    Ok(CommandOutput::ok(markdown, payload.clone()))
}

pub(crate) fn model_download() -> Result<CommandOutput, SaeError> {
    Ok(CommandOutput::ok(
        "Model downloaded and verified".to_owned(),
        serde_json::json!({"status": "ok"}),
    ))
}

pub(crate) fn status(statuses: &[TeamStatus]) -> Result<CommandOutput, SaeError> {
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
    let markdown = lines.join("\n");
    let data = serde_json::to_value(statuses)?;
    Ok(CommandOutput::ok(markdown, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{EmbedResult, MatchSource, SyncState};

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

    // T-079: status human-readable empty list → empty markdown
    #[test]
    fn status_human_empty_returns_empty_string() {
        let out = status(&[]).unwrap();
        assert!(
            out.markdown.is_empty(),
            "empty list should produce empty markdown"
        );
    }

    // T-080: status human-readable synced team → expected lines
    #[test]
    fn status_human_synced_contains_expected_lines() {
        let out = status(&[make_synced("team-a", 10)]).unwrap();
        assert!(out.markdown.contains("--- team-a ---"));
        assert!(out.markdown.contains("Posts: 10"));
        assert!(out.markdown.contains("Last sync: 2025-01-01 00:00:00"));
    }

    // T-081: status human-readable not-synced team → expected lines
    #[test]
    fn status_human_not_synced_contains_expected_lines() {
        let out = status(&[make_not_synced("team-b")]).unwrap();
        assert!(out.markdown.contains("--- team-b ---"));
        assert!(out.markdown.contains("Not yet synced"));
    }

    // T-082: embed json data carries chunks_embedded
    #[test]
    fn embed_json_contains_chunks_embedded() {
        let result = EmbedResult {
            chunks_embedded: 42,
        };
        let out = embed(&result, 42).unwrap();
        assert_eq!(out.data["chunks_embedded"], 42);
    }

    // T-083: embed human-readable with done=0 → "Nothing to embed"
    #[test]
    fn embed_human_zero_done_returns_nothing_to_embed() {
        let result = EmbedResult { chunks_embedded: 0 };
        let out = embed(&result, 0).unwrap();
        assert_eq!(out.markdown, "Nothing to embed");
    }

    // T-084: embed human-readable chunks > 0 → "Embedded N chunks"
    #[test]
    fn embed_human_nonzero_returns_embedded_count() {
        let result = EmbedResult { chunks_embedded: 5 };
        let out = embed(&result, 5).unwrap();
        assert_eq!(out.markdown, "Embedded 5 chunks");
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

    // T-087: search data includes search_mode hybrid
    #[test]
    fn search_json_hybrid_includes_search_mode() {
        let out = search(&[], "test", true, false, None).unwrap();
        assert_eq!(out.data["search_mode"], "hybrid");
        assert!(out.data["results"].is_array());
    }

    // T-088: search data includes search_mode fts
    #[test]
    fn search_json_fts_includes_search_mode() {
        let out = search(&[], "test", false, false, None).unwrap();
        assert_eq!(out.data["search_mode"], "fts");
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
        assert!(out.markdown.contains("semantic search unavailable"));
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
        let out = search(&results, "q", true, false, None).unwrap();
        assert!(!out.markdown.contains("semantic search unavailable"));
    }

    // T-260: search degraded fallback populates notes
    #[test]
    fn search_degraded_fallback_populates_notes() {
        let out = search(&[], "q", false, true, None).unwrap();
        assert!(
            out.degraded,
            "embedder_load_failed=true should set degraded"
        );
        assert_eq!(out.notes.len(), 1);
        assert!(out.notes[0].contains("semantic search unavailable"));
    }

    // T-261: search no-embed (FTS without load failure) leaves notes empty
    #[test]
    fn search_no_embed_does_not_degrade() {
        let out = search(&[], "q", false, false, None).unwrap();
        assert!(!out.degraded);
        assert!(out.notes.is_empty());
    }
}
