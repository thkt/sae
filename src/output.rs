use crate::client::EsaPost;
use crate::envelope::CommandOutput;
use crate::storage::{SearchResult, SyncStatus, TeamStatus};
use crate::sync::HarvestResult;
use crate::tools::SaeError;

/// Renders the combined `index` / `rebuild` result: the harvest summary plus
/// the embed outcome. `embedded` is `Some(n)` when chunks were embedded this
/// run (omitted from markdown when zero), `None` when nothing was pending. The
/// count lives under a nested `embed` key so it never collides with harvest
/// fields, keeping the `--json` envelope shape stable.
pub(crate) fn index(
    harvest: &HarvestResult,
    embedded: Option<u32>,
) -> Result<CommandOutput, SaeError> {
    let mut markdown = harvest.to_string();
    if let Some(n) = embedded
        && n > 0
    {
        markdown.push_str(&format!("\nEmbedded {n} chunks"));
    }
    let mut data = serde_json::to_value(harvest)?;
    if let Some(obj) = data.as_object_mut() {
        obj.insert(
            "embed".to_owned(),
            serde_json::json!({ "chunks_embedded": embedded.unwrap_or(0) }),
        );
    }
    Ok(CommandOutput::ok(markdown, data))
}

pub(crate) fn search(
    results: &[SearchResult],
    query: &str,
    semantic: bool,
    embedder_load_failed: bool,
    search_warnings: &[String],
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
        lines.join("\n")
    };
    let mut notes = Vec::new();
    if embedder_load_failed {
        notes.push("semantic search unavailable, falling back to FTS".to_owned());
    }
    notes.extend(search_warnings.iter().cloned());
    if notes.is_empty() {
        Ok(CommandOutput::ok(markdown, data))
    } else {
        Ok(CommandOutput::with_notes(markdown, data, true, notes))
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
                        "  Pending embed: {} chunks (run 'sae index {}' to embed)",
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
    use crate::client::EsaUser;
    use crate::storage::{MatchSource, SyncState};

    fn make_post() -> EsaPost {
        EsaPost {
            number: 42,
            name: "Onboarding".to_owned(),
            full_name: "design/Onboarding".to_owned(),
            body_md: Some("# Welcome\n\nFirst paragraph.".to_owned()),
            category: Some("design".to_owned()),
            tags: vec!["docs".to_owned(), "team".to_owned()],
            wip: false,
            kind: "stock".to_owned(),
            url: "https://example.esa.io/posts/42".to_owned(),
            created_at: "2025-01-01 00:00:00".to_owned(),
            updated_at: "2025-01-15 12:00:00".to_owned(),
            created_by: EsaUser {
                screen_name: "alice".to_owned(),
            },
            updated_by: EsaUser {
                screen_name: "alice".to_owned(),
            },
            revision_number: 3,
        }
    }

    fn make_result(
        post_number: u32,
        name: &str,
        section: Option<&str>,
        score: f32,
        snippet: &str,
        source: MatchSource,
    ) -> SearchResult {
        SearchResult {
            post_number,
            post_name: name.to_owned(),
            post_url: format!("https://example.esa.io/posts/{post_number}"),
            section_title: section.map(str::to_owned),
            snippet: snippet.to_owned(),
            score,
            match_source: source,
        }
    }

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

    fn make_harvest(posts_fetched: u32) -> HarvestResult {
        HarvestResult {
            posts_fetched,
            posts_stored: posts_fetched,
            total_count: 10,
            local_count: 10,
            gap_detected: false,
        }
    }

    // T-407: index merges harvest stats with the embed result — the embed count
    // is appended to markdown and nested under a dedicated `embed` data key,
    // kept separate from harvest fields so the envelope shape stays stable (#3).
    #[test]
    fn index_merges_harvest_and_embed_result() {
        let out = index(&make_harvest(2), Some(3)).unwrap();
        assert!(
            out.markdown.contains("Fetched 2 posts"),
            "harvest line missing: {}",
            out.markdown
        );
        assert!(
            out.markdown.contains("Embedded 3 chunks"),
            "embed line missing: {}",
            out.markdown
        );
        assert_eq!(out.data["posts_fetched"], 2);
        assert_eq!(out.data["embed"]["chunks_embedded"], 3);
    }

    // T-408: index with nothing to embed (pending == 0) omits the embed line
    // but keeps the `embed` key at zero so agents read a stable shape.
    #[test]
    fn index_without_pending_embed_keeps_zero_embed_key() {
        let out = index(&make_harvest(0), None).unwrap();
        assert!(
            out.markdown.contains("No updates"),
            "harvest line missing: {}",
            out.markdown
        );
        assert!(
            !out.markdown.contains("Embedded"),
            "embed line should be absent: {}",
            out.markdown
        );
        assert_eq!(out.data["embed"]["chunks_embedded"], 0);
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
        let out = search(&[], "test", true, false, &[]).unwrap();
        assert_eq!(out.data["search_mode"], "hybrid");
        assert!(out.data["results"].is_array());
    }

    // T-088: search data includes search_mode fts
    #[test]
    fn search_json_fts_includes_search_mode() {
        let out = search(&[], "test", false, false, &[]).unwrap();
        assert_eq!(out.data["search_mode"], "fts");
    }

    // T-089: search human fts fallback shows notice
    #[test]
    fn search_human_fts_fallback_shows_notice() {
        let results = vec![make_result(1, "Test", None, 0.5, "", MatchSource::Fts)];
        let out = search(&results, "q", false, false, &[]).unwrap();
        assert!(out.markdown.contains("semantic search unavailable"));
    }

    // T-090: search human hybrid does not show fallback notice
    #[test]
    fn search_human_hybrid_no_fallback_notice() {
        let results = vec![make_result(1, "Test", None, 0.5, "", MatchSource::Fts)];
        let out = search(&results, "q", true, false, &[]).unwrap();
        assert!(!out.markdown.contains("semantic search unavailable"));
    }

    // T-260: search degraded fallback populates notes
    #[test]
    fn search_degraded_fallback_populates_notes() {
        let out = search(&[], "q", false, true, &[]).unwrap();
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
        let out = search(&[], "q", false, false, &[]).unwrap();
        assert!(!out.degraded);
        assert!(out.notes.is_empty());
    }

    // T-262: non-empty search_warnings alone (no embedder failure) sets degraded
    // and surfaces the warnings as notes. Pins #140 wiring: storage warnings
    // reach the envelope notes[] without depending on embedder_load_failed.
    #[test]
    fn search_warnings_alone_set_degraded_and_populate_notes() {
        let warnings = vec!["reranker failed (boom), falling back to RRF order".to_owned()];
        let out = search(&[], "q", true, false, &warnings).unwrap();
        assert!(
            out.degraded,
            "non-empty warnings should set degraded even with embedder loaded"
        );
        assert_eq!(out.notes.len(), 1);
        assert!(
            out.notes[0].contains("reranker failed"),
            "warning should be surfaced in notes[0]; got: {:?}",
            out.notes
        );
    }

    // T-263: embedder_load_failed + non-empty search_warnings produces ordered
    // notes (embedder note first, then storage warnings). Order pinned so
    // agents that read notes[0] keep the same upstream-first semantics.
    #[test]
    fn search_warnings_with_embedder_failure_preserve_order() {
        let warnings = vec!["reranker failed (boom), falling back to RRF order".to_owned()];
        let out = search(&[], "q", false, true, &warnings).unwrap();
        assert!(out.degraded);
        assert_eq!(out.notes.len(), 2);
        assert!(
            out.notes[0].contains("semantic search unavailable"),
            "embedder note should come first (upstream degradation); got: {:?}",
            out.notes
        );
        assert!(
            out.notes[1].contains("reranker failed"),
            "storage warning should come second (downstream); got: {:?}",
            out.notes
        );
    }

    // T-401: format_post_frontmatter typical post → frontmatter snapshot.
    // Out of scope: wip:true / updated_by != created_by (separate pin if needed).
    #[test]
    fn format_post_frontmatter_typical_post_snapshot() {
        let post = make_post();
        insta::assert_snapshot!(format_post_frontmatter(&post));
    }

    // T-402: search hybrid with section_title present vs absent → markdown snapshot.
    #[test]
    fn search_semantic_hybrid_two_results_snapshot() {
        let results = vec![
            make_result(
                101,
                "Architecture",
                Some("Overview"),
                0.9123,
                "First snippet content",
                MatchSource::Semantic,
            ),
            make_result(
                102,
                "Pricing model",
                None,
                0.7456,
                "Second snippet content",
                MatchSource::Fts,
            ),
        ];
        let out = search(&results, "design", true, false, &[]).unwrap();
        insta::assert_snapshot!(out.markdown);
    }
}
