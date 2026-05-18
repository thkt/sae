//! Bulk sync (`index` / `rebuild`) between esa.io and the local SQLite cache.

use std::fmt;

use amici::cli::progress_step;
use tracing::info;

use crate::client::{ClientError, EsaClient, EsaPost};
use crate::storage::{self, Db, EsaPostRow, StorageError};

#[derive(Debug, serde::Serialize)]
pub struct HarvestResult {
    pub posts_fetched: u32,
    pub posts_stored: u32,
    pub total_count: u32,
    pub local_count: u32,
    pub gap_detected: bool,
}

impl fmt::Display for HarvestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.posts_fetched == 0 {
            write!(
                f,
                "No updates. remote: {} | local: {}",
                self.total_count, self.local_count,
            )?;
        } else {
            write!(
                f,
                "Fetched {} posts, stored {}. remote: {} | local: {}",
                self.posts_fetched, self.posts_stored, self.total_count, self.local_count,
            )?;
        }
        if self.gap_detected {
            write!(
                f,
                " [gap detected: {} missing]",
                self.total_count.saturating_sub(self.local_count)
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    #[error(transparent)]
    Client(#[from] ClientError),

    #[error(transparent)]
    Storage(#[from] StorageError),
}

impl SyncError {
    /// True when the failure is transient enough that retrying might recover.
    ///
    /// `Client` variants always return `false`: their retryability is
    /// classified per-case in [`crate::tools::SaeError::error_code`]
    /// (Network / MaxRetries → `TempFailure`, Api → `Internal`).
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Client(_) => false,
            Self::Storage(e) => e.is_retryable(),
        }
    }
}

pub async fn harvest(
    client: &EsaClient,
    db: &Db,
    team: &str,
    full: bool,
) -> Result<HarvestResult, SyncError> {
    let state = storage::get_sync_state(db.conn())?;
    let prior_total = if full {
        0
    } else {
        state.as_ref().map_or(0, |s| s.total_count)
    };

    if full {
        storage::save_sync_state(
            db.conn(),
            &storage::SyncStateUpdate {
                latest_updated_at: None,
                total_count: 0,
                local_count: 0,
                last_page: None,
            },
        )?;
    }

    let (start_page, base_query) = resolve_start(full, &state);
    let prior_count = if full {
        0
    } else {
        state.as_ref().map_or(0, |s| s.local_count)
    };

    let mut page = start_page;
    let mut total_fetched = 0u32;
    let mut total_stored = 0u32;
    let mut latest = if full {
        None
    } else {
        state.and_then(|s| s.latest_updated_at)
    };
    let mut max_api_total: u32 = 0;

    // Pagination cap (see PAGINATION_LIMIT): on hit, narrow by updated_at and continue.
    let mut window_boundary: Option<String> = None;
    let mut batch_oldest_ts: Option<String> = None;

    'outer: loop {
        let query = build_window_query(base_query.as_deref(), window_boundary.as_deref());

        let resp = match client.list_posts(team, page, query.as_deref()).await {
            Ok(r) => r,
            Err(ClientError::Api {
                status: 400,
                ref body,
            }) if detect_pagination_limit(body, max_api_total) => {
                if let Some(ref oldest) = batch_oldest_ts {
                    if window_boundary.as_ref() == Some(oldest) {
                        progress_step(&["window didn't advance, stopping"]);
                        break;
                    }
                    progress_step(&[
                        "pagination limit",
                        &format!("narrowing to updated:<={oldest}"),
                    ]);
                    window_boundary = Some(oldest.clone());
                    batch_oldest_ts = None;
                    page = 1;
                    continue 'outer;
                }
                break;
            }
            Err(e) => return Err(e.into()),
        };
        let api_total = resp.total_count;
        max_api_total = max_api_total.max(api_total);
        let est_pages = api_total.div_ceil(100);

        let tx = db
            .conn()
            .unchecked_transaction()
            .map_err(StorageError::Db)?;

        for api_post in &resp.posts {
            let row = post_to_row(api_post);
            if latest.as_ref().is_none_or(|l| row.updated_at > *l) {
                latest = Some(row.updated_at.clone());
            }
            if batch_oldest_ts.as_ref().is_none_or(|o| row.updated_at < *o) {
                batch_oldest_ts = Some(row.updated_at.clone());
            }
            storage::upsert_post(&tx, &row)?;
            let enriched_body = storage::enrich_body(&row);
            storage::rechunk_post(&tx, row.number, &enriched_body)?;
            total_stored += 1;
        }
        total_fetched += u32::try_from(resp.posts.len()).unwrap_or(u32::MAX);

        storage::save_sync_state(
            &tx,
            &storage::SyncStateUpdate {
                latest_updated_at: latest.as_deref(),
                total_count: effective_total(full, prior_total, max_api_total, 0),
                local_count: prior_count + total_stored,
                last_page: resp.next_page,
            },
        )?;
        tx.commit().map_err(StorageError::Db)?;

        progress_step(&[
            &format!("page {page}/{est_pages}"),
            &format!("{total_fetched} posts fetched"),
        ]);
        info!(
            page,
            fetched = resp.posts.len(),
            total_fetched,
            "harvested page"
        );

        match resp.next_page {
            Some(np) => page = np,
            None => break,
        }
    }

    let local_count = storage::count_posts(db.conn())?;
    let total_count = effective_total(full, prior_total, max_api_total, local_count);
    let gap_detected = local_count < total_count;

    storage::save_sync_state(
        db.conn(),
        &storage::SyncStateUpdate {
            latest_updated_at: latest.as_deref(),
            total_count,
            local_count,
            last_page: None,
        },
    )?;

    if gap_detected {
        let missing = total_count - local_count;
        progress_step(&[&format!(
            "warning: gap detected — {missing} missing posts (run `sae rebuild {team}` to re-sync)"
        )]);
        info!(local_count, total_count, missing, "gap detected");
    }

    Ok(HarvestResult {
        posts_fetched: total_fetched,
        posts_stored: total_stored,
        total_count,
        local_count,
        gap_detected,
    })
}

/// esa API caps pagination at this many items per query. Used as a structured
/// detection signal for the `400 Bad Request` response: when `max_api_total`
/// observed in prior responses exceeds this constant, a 400 is almost
/// certainly the pagination cap rather than an unrelated bad-request.
const PAGINATION_LIMIT: u32 = 10_000;

/// Detect esa's 10k-item pagination cap from a `400 Bad Request` response.
///
/// Primary structured signal: `max_api_total > PAGINATION_LIMIT`. esa returns
/// 400 to ANY page request when the result set would exceed 10000 items, so
/// the prior pages' `total_count` is a wording-independent witness.
///
/// Fallback: legacy substring match (`"10,000"` / `"10000"`) for the case
/// where the very first request fails (no prior `total_count` observed).
/// The fallback is fragile against esa wording changes; both code paths emit
/// `tracing::warn!` so operators notice when the structured signal stops
/// agreeing with the substring.
fn detect_pagination_limit(body: &str, max_api_total: u32) -> bool {
    let substring_match = body.contains("10,000") || body.contains("10000");
    if max_api_total > PAGINATION_LIMIT {
        if !substring_match {
            tracing::warn!(
                body = %body,
                max_api_total,
                "pagination limit detected by total_count; esa error message wording may have changed (legacy substring missing)"
            );
        }
        return true;
    }
    if substring_match {
        tracing::warn!(
            body = %body,
            "pagination limit detected via legacy substring fallback (no structured signal); fragile path"
        );
    }
    substring_match
}

/// Resolve the remote total to persist in sync_state.
///
/// In full sync, the largest API total seen across responses is authoritative:
/// pagination narrowing can produce a smaller total in later responses that
/// does not reflect the real remote size, so we take the max.
///
/// In incremental sync, every response's `total_count` comes from a diff
/// filter (`q=updated:>X`) and underestimates the real total. Preserve the
/// prior total and apply `local_floor` so previously corrupted state can
/// self-heal once the floor reflects an authoritative local row count.
///
/// Pass `0` for `local_floor` at intermediate checkpoints. A running upsert
/// counter includes updates to existing posts and would otherwise inflate
/// the persisted total — only the post-loop `count_posts` query reports the
/// actual row count.
fn effective_total(full: bool, prior: u32, max_api: u32, local_floor: u32) -> u32 {
    if full {
        max_api
    } else {
        prior.max(max_api).max(local_floor)
    }
}

fn build_window_query(base: Option<&str>, boundary: Option<&str>) -> Option<String> {
    match (base, boundary) {
        (None, None) => None,
        (Some(b), None) => Some(b.to_owned()),
        (None, Some(o)) => Some(format!("updated:<={o}")),
        (Some(b), Some(o)) => Some(format!("{b} updated:<={o}")),
    }
}

fn resolve_start(full: bool, state: &Option<storage::SyncState>) -> (u32, Option<String>) {
    if full {
        return (1, None);
    }
    let Some(s) = state else {
        return (1, None);
    };
    let q = s
        .latest_updated_at
        .as_deref()
        .map(|ts| format!("updated:>{ts}"));
    let page = s.last_page.unwrap_or(1);
    (page, q)
}

/// Write-through: reflect a single `EsaPost` from a mutation API response
/// into the local DB (posts table + chunks + FTS) atomically.
///
/// Does NOT touch `sync_state.latest_updated_at`. Advancing the harvest cursor
/// from a single post would skip concurrent remote updates whose `updated_at`
/// falls between the last harvest and this post — only a full paginated
/// harvest can prove that range is fully covered.
pub(crate) fn upsert_post_locally(db: &Db, post: &EsaPost) -> Result<(), SyncError> {
    let row = post_to_row(post);
    let tx = db
        .conn()
        .unchecked_transaction()
        .map_err(StorageError::Db)?;
    storage::upsert_post(&tx, &row)?;
    let enriched_body = storage::enrich_body(&row);
    storage::rechunk_post(&tx, row.number, &enriched_body)?;
    tx.commit().map_err(StorageError::Db)?;
    Ok(())
}

/// Best-effort wrapper for [`upsert_post_locally`]: when `db` is `None`,
/// or when the write fails, log a warning and continue — never propagate.
/// Used by mutation commands (create/update/archive/ship) to reflect their
/// API response into the local DB without disrupting the user-visible result.
pub(crate) fn try_upsert_post_locally(db: Option<&Db>, post: &EsaPost) {
    let Some(db) = db else { return };
    if let Err(e) = upsert_post_locally(db, post) {
        tracing::warn!(
            error = %e,
            post = post.number,
            "local DB write-through failed; next harvest will reconcile"
        );
    }
}

fn post_to_row(post: &EsaPost) -> EsaPostRow {
    EsaPostRow {
        number: post.number,
        name: post.name.clone(),
        full_name: post.full_name.clone(),
        body_md: post.body_md.clone().unwrap_or_default(),
        category: post.category.clone(),
        tags: post.tags.clone(),
        wip: post.wip,
        kind: post.kind.clone(),
        url: post.url.clone(),
        created_at: post.created_at.clone(),
        updated_at: post.updated_at.clone(),
        created_by: post.created_by.screen_name.clone(),
        updated_by: post.updated_by.screen_name.clone(),
        revision_number: post.revision_number,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use rusqlite::ErrorCode;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::client::EsaUser;
    use crate::storage::sqlite_failure;

    fn make_esa_post(number: u32, name: &str, body_md: &str, updated_at: &str) -> EsaPost {
        EsaPost {
            number,
            name: name.into(),
            full_name: format!("dev/{name}"),
            body_md: Some(body_md.into()),
            category: Some("dev".into()),
            tags: vec!["rust".into()],
            wip: false,
            kind: "stock".into(),
            url: format!("https://example.esa.io/posts/{number}"),
            created_at: "2025-01-01T00:00:00+09:00".into(),
            updated_at: updated_at.into(),
            created_by: EsaUser {
                screen_name: "alice".into(),
            },
            updated_by: EsaUser {
                screen_name: "bob".into(),
            },
            revision_number: 1,
        }
    }

    pub(crate) fn esa_post_fixture(
        number: u32,
        name: &str,
        body_md: &str,
        category: Option<&str>,
        wip: bool,
        updated_at: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "number": number,
            "name": name,
            "full_name": format!("dev/{name}"),
            "body_md": body_md,
            "category": category,
            "tags": ["test"],
            "wip": wip,
            "kind": "stock",
            "url": format!("https://example.esa.io/posts/{number}"),
            "created_at": "2025-01-01T00:00:00+09:00",
            "updated_at": updated_at,
            "created_by": {"screen_name": "alice"},
            "updated_by": {"screen_name": "bob"},
            "revision_number": 1
        })
    }

    fn api_post(number: u32, name: &str, updated_at: &str) -> serde_json::Value {
        esa_post_fixture(
            number,
            name,
            &format!("# {name}"),
            Some("dev"),
            false,
            updated_at,
        )
    }

    fn posts_response(
        posts: &[serde_json::Value],
        next_page: Option<u32>,
        total: u32,
    ) -> serde_json::Value {
        serde_json::json!({
            "posts": posts,
            "next_page": next_page,
            "total_count": total
        })
    }

    #[tokio::test]
    async fn harvest_stores_posts() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("page", "1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[
                    api_post(1, "A", "2025-01-01T00:00:00+09:00"),
                    api_post(2, "B", "2025-01-02T00:00:00+09:00"),
                    api_post(3, "C", "2025-01-03T00:00:00+09:00"),
                ],
                None,
                3,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert_eq!(r.posts_fetched, 3);
        assert_eq!(r.posts_stored, 3);
        assert_eq!(r.local_count, 3);
        assert!(!r.gap_detected);
    }

    #[tokio::test]
    async fn incremental_fetches_updates_only() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[
                    api_post(1, "A", "2025-01-01T00:00:00+09:00"),
                    api_post(2, "B", "2025-01-02T00:00:00+09:00"),
                ],
                None,
                2,
            )))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();
        harvest(&client, &db, "t", false).await.unwrap();

        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("q", "updated:>2025-01-02T00:00:00+09:00"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[api_post(2, "B Updated", "2025-01-03T00:00:00+09:00")],
                None,
                2,
            )))
            .mount(&server)
            .await;

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert_eq!(r.posts_fetched, 1);
        assert_eq!(r.local_count, 2);

        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 2", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(name, "B Updated");
    }

    #[tokio::test]
    async fn gap_detection() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[api_post(1, "A", "2025-01-01T00:00:00+09:00")],
                None,
                5,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert!(r.gap_detected);
        assert_eq!(r.local_count, 1);
        assert_eq!(r.total_count, 5);
    }

    #[tokio::test]
    async fn pagination_across_pages() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[
                    api_post(1, "A", "2025-01-01T00:00:00+09:00"),
                    api_post(2, "B", "2025-01-02T00:00:00+09:00"),
                ],
                Some(2),
                3,
            )))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[api_post(3, "C", "2025-01-03T00:00:00+09:00")],
                None,
                3,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert_eq!(r.posts_fetched, 3);
        assert_eq!(r.local_count, 3);
    }

    #[tokio::test]
    async fn checkpoint_cleared_after_sync() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[api_post(1, "A", "2025-01-01T00:00:00+09:00")],
                None,
                1,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();
        harvest(&client, &db, "t", false).await.unwrap();

        let state = storage::get_sync_state(db.conn()).unwrap().unwrap();
        assert!(state.last_page.is_none());
        assert_eq!(
            state.latest_updated_at.as_deref(),
            Some("2025-01-01T00:00:00+09:00")
        );
        assert_eq!(state.local_count, 1);
    }

    // T-161: resolve_start returns page 1 and no query for a first-time sync
    #[test]
    fn resolve_start_first_sync() {
        let (page, q) = resolve_start(false, &None);
        assert_eq!(page, 1);
        assert!(q.is_none());
    }

    // T-162: resolve_start returns updated: query when prior sync state exists
    #[test]
    fn resolve_start_incremental() {
        let state = Some(storage::SyncState {
            latest_updated_at: Some("2025-01-01T00:00:00+09:00".into()),
            total_count: 10,
            local_count: 10,
            last_page: None,
            updated_at: String::new(),
        });
        let (page, q) = resolve_start(false, &state);
        assert_eq!(page, 1);
        assert_eq!(q.as_deref(), Some("updated:>2025-01-01T00:00:00+09:00"));
    }

    // T-163: resolve_start resumes from last_page when a checkpoint exists
    #[test]
    fn resolve_start_resume_checkpoint() {
        let state = Some(storage::SyncState {
            latest_updated_at: Some("2025-01-01T00:00:00+09:00".into()),
            total_count: 10,
            local_count: 5,
            last_page: Some(3),
            updated_at: String::new(),
        });
        let (page, q) = resolve_start(false, &state);
        assert_eq!(page, 3);
        assert_eq!(q.as_deref(), Some("updated:>2025-01-01T00:00:00+09:00"));
    }

    // T-164: resolve_start ignores saved state when full sync is requested
    #[test]
    fn resolve_start_full_ignores_state() {
        let state = Some(storage::SyncState {
            latest_updated_at: Some("2025-01-01T00:00:00+09:00".into()),
            total_count: 10,
            local_count: 10,
            last_page: None,
            updated_at: String::new(),
        });
        let (page, q) = resolve_start(true, &state);
        assert_eq!(page, 1);
        assert!(q.is_none());
    }

    // T-044: HarvestResult serialize → posts_fetched, posts_stored, total_count fields
    #[test]
    fn harvest_result_serializes_to_json_with_expected_fields() {
        let result = HarvestResult {
            posts_fetched: 10,
            posts_stored: 8,
            total_count: 100,
            local_count: 50,
            gap_detected: true,
        };
        let json_str = serde_json::to_string(&result).expect("HarvestResult should serialize");
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["posts_fetched"], 10);
        assert_eq!(v["posts_stored"], 8);
        assert_eq!(v["total_count"], 100);
        assert_eq!(v["local_count"], 50);
        assert_eq!(v["gap_detected"], true);
    }

    // T-113: substring fallback detects comma form when no structured signal exists
    #[test]
    fn detect_pagination_limit_substring_comma_format() {
        assert!(detect_pagination_limit("exceeds 10,000 items", 0));
    }

    // T-165: substring fallback also accepts plain "10000"; both forms appear historically in esa errors
    #[test]
    fn detect_pagination_limit_substring_plain_format() {
        assert!(detect_pagination_limit("exceeds 10000 items", 0));
    }

    // T-166: substring fallback rejects unrelated numbers (e.g. "1000") and empty bodies
    #[test]
    fn detect_pagination_limit_rejects_other_numbers() {
        assert!(!detect_pagination_limit("1000 items", 0));
        assert!(!detect_pagination_limit("", 0));
    }

    // T-391: structured signal (max_api_total > PAGINATION_LIMIT) recognises the cap regardless of wording (#138 subtask 4)
    #[test]
    fn detect_pagination_limit_structured_signal_dominates() {
        assert!(
            detect_pagination_limit("Result set capped at 10K", 10_001),
            "structured signal must win even when substring is absent"
        );
        assert!(
            detect_pagination_limit("totally unrelated error wording", 50_000),
            "any 400 with prior total > limit must be treated as the cap"
        );
    }

    // T-392: no signal returns false; pins the strict `>` boundary so a drift to `>=` fails the suite
    #[test]
    fn detect_pagination_limit_returns_false_without_signal() {
        assert!(!detect_pagination_limit("invalid q parameter", 0));
        assert!(
            !detect_pagination_limit("invalid q parameter", PAGINATION_LIMIT),
            "max_api_total equal to PAGINATION_LIMIT must not trigger detection (only > triggers)"
        );
    }

    // T-114: post_to_row with body_md: None → empty string
    #[test]
    fn post_to_row_none_body_becomes_empty() {
        let post = EsaPost {
            number: 1,
            name: "Test".into(),
            full_name: "dev/Test".into(),
            body_md: None,
            category: Some("dev".into()),
            tags: vec!["rust".into()],
            wip: false,
            kind: "stock".into(),
            url: "https://example.esa.io/posts/1".into(),
            created_at: "2025-01-01T00:00:00+09:00".into(),
            updated_at: "2025-01-02T00:00:00+09:00".into(),
            created_by: EsaUser {
                screen_name: "alice".into(),
            },
            updated_by: EsaUser {
                screen_name: "bob".into(),
            },
            revision_number: 1,
        };
        let row = post_to_row(&post);
        assert_eq!(row.body_md, "");
        assert_eq!(row.tags, vec!["rust".to_owned()]);
    }

    // T-167: build_window_query returns None when both base and boundary are absent
    #[test]
    fn build_window_query_none_none() {
        assert!(build_window_query(None, None).is_none());
    }

    // T-168: build_window_query returns base query unchanged when boundary is absent
    #[test]
    fn build_window_query_base_only() {
        assert_eq!(
            build_window_query(Some("updated:>2025-01-01"), None),
            Some("updated:>2025-01-01".into())
        );
    }

    // T-169: build_window_query returns upper-bound clause when only boundary is set
    #[test]
    fn build_window_query_boundary_only() {
        assert_eq!(
            build_window_query(None, Some("2025-06-01")),
            Some("updated:<=2025-06-01".into())
        );
    }

    // T-170: build_window_query combines base and boundary into a window query
    #[test]
    fn build_window_query_both() {
        assert_eq!(
            build_window_query(Some("updated:>2025-01-01"), Some("2025-06-01")),
            Some("updated:>2025-01-01 updated:<=2025-06-01".into())
        );
    }

    // T-171: HarvestResult Display omits gap message when gap_detected is false
    #[test]
    fn harvest_result_display_no_gap() {
        let r = HarvestResult {
            posts_fetched: 10,
            posts_stored: 10,
            total_count: 100,
            local_count: 100,
            gap_detected: false,
        };
        let s = r.to_string();
        assert!(s.contains("Fetched 10"));
        assert!(!s.contains("gap"));
    }

    // T-172: HarvestResult Display includes missing count when gap_detected is true
    #[test]
    fn harvest_result_display_with_gap() {
        let r = HarvestResult {
            posts_fetched: 5,
            posts_stored: 5,
            total_count: 100,
            local_count: 90,
            gap_detected: true,
        };
        let s = r.to_string();
        assert!(s.contains("gap detected: 10 missing"));
    }

    #[tokio::test]
    async fn pagination_limit_narrows_window() {
        let server = MockServer::start().await;

        // First request hits 10,000-item pagination limit
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("page", "1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[
                    api_post(1, "A", "2025-01-03T00:00:00+09:00"),
                    api_post(2, "B", "2025-01-02T00:00:00+09:00"),
                ],
                Some(2),
                10001,
            )))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        // Second request (page 2) returns pagination limit error
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("page", "2"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": "bad_request",
                "message": "Pagination limit: exceeds 10,000 items"
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        // Narrowed window request succeeds
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("q", "updated:<=2025-01-02T00:00:00+09:00"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[api_post(3, "C", "2025-01-01T00:00:00+09:00")],
                None,
                1,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert!(
            r.posts_fetched >= 3,
            "should fetch posts across window narrowing"
        );
        assert_eq!(storage::count_posts(db.conn()).unwrap(), 3);
        assert_eq!(
            r.total_count, 10001,
            "the largest total_count seen (before window narrowing) is authoritative; \
             later narrowed responses must not regress it"
        );
        assert!(
            r.gap_detected,
            "local 3 < remote 10001 should surface a gap"
        );
    }

    // T-393: structured signal narrows the window even when the 400 body has no "10,000"/"10000" substring (#138 subtask 4)
    #[tokio::test]
    async fn pagination_limit_narrows_window_via_structured_signal() {
        let server = MockServer::start().await;

        // Page 1: total_count=10_001 → max_api_total > PAGINATION_LIMIT for the next call
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("page", "1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[
                    api_post(1, "A", "2025-01-03T00:00:00+09:00"),
                    api_post(2, "B", "2025-01-02T00:00:00+09:00"),
                ],
                Some(2),
                10_001,
            )))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        // Reworded esa error — no legacy substring; only the structured
        // total_count witness from page 1 saves us.
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("page", "2"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": "bad_request",
                "message": "Result set capped — refine your query"
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("q", "updated:<=2025-01-02T00:00:00+09:00"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                &[api_post(3, "C", "2025-01-01T00:00:00+09:00")],
                None,
                1,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert!(
            r.posts_fetched >= 3,
            "structured-signal detection must narrow without substring match"
        );
        assert_eq!(storage::count_posts(db.conn()).unwrap(), 3);
        assert_eq!(r.total_count, 10_001);
    }

    // T-300: upsert_post_locally inserts a post into an empty DB
    #[test]
    fn upsert_post_locally_inserts_new_post() {
        let db = Db::open_memory().unwrap();
        let post = make_esa_post(42, "Hello", "# Body", "2025-06-01T00:00:00+09:00");
        upsert_post_locally(&db, &post).unwrap();

        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 42", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(name, "Hello");
        let chunks = storage::count_chunks(db.conn()).unwrap();
        assert!(chunks > 0, "expected at least one chunk to be created");
    }

    // T-301: upsert_post_locally replaces an existing post and re-chunks
    #[test]
    fn upsert_post_locally_replaces_existing() {
        let db = Db::open_memory().unwrap();
        upsert_post_locally(
            &db,
            &make_esa_post(7, "Original", "# Old", "2025-01-01T00:00:00+09:00"),
        )
        .unwrap();
        upsert_post_locally(
            &db,
            &make_esa_post(7, "Renamed", "# New body", "2025-02-01T00:00:00+09:00"),
        )
        .unwrap();

        let name: String = db
            .conn()
            .query_row("SELECT name FROM posts WHERE number = 7", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(name, "Renamed", "later upsert should win");
        let total: u32 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM posts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(total, 1, "no duplicate row should be created");
    }

    // T-302: upsert_post_locally must NOT touch sync_state so that the harvest
    //        cursor only advances after a full paginated harvest proves the
    //        intervening updated_at range was fully covered. Single-post
    //        write-through would otherwise skip concurrent remote updates.
    #[test]
    fn upsert_post_locally_does_not_advance_sync_state() {
        let db = Db::open_memory().unwrap();
        storage::save_sync_state_at(
            db.conn(),
            &storage::SyncStateUpdate {
                latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
                total_count: 5,
                local_count: 5,
                last_page: None,
            },
            1_700_000_000,
        )
        .unwrap();
        let before = storage::get_sync_state(db.conn()).unwrap().unwrap();

        let post = make_esa_post(1, "A", "# x", "2025-06-01T00:00:00+09:00");
        upsert_post_locally(&db, &post).unwrap();

        let after = storage::get_sync_state(db.conn()).unwrap().unwrap();
        assert_eq!(after.latest_updated_at, before.latest_updated_at);
        assert_eq!(after.total_count, before.total_count);
        assert_eq!(after.local_count, before.local_count);
        assert_eq!(after.updated_at, before.updated_at);
    }

    // T-304: upsert_post_locally without prior sync_state leaves it absent
    #[test]
    fn upsert_post_locally_skips_sync_state_when_absent() {
        let db = Db::open_memory().unwrap();
        let post = make_esa_post(1, "A", "# x", "2025-06-01T00:00:00+09:00");
        upsert_post_locally(&db, &post).unwrap();

        assert!(
            storage::get_sync_state(db.conn()).unwrap().is_none(),
            "sync_state should not be created when previously absent"
        );
        assert_eq!(storage::count_posts(db.conn()).unwrap(), 1);
    }

    // T-173: Display shows "No updates" when an incremental harvest fetches 0 posts
    #[test]
    fn harvest_result_display_no_updates() {
        let r = HarvestResult {
            posts_fetched: 0,
            posts_stored: 0,
            total_count: 15712,
            local_count: 15712,
            gap_detected: false,
        };
        let s = r.to_string();
        assert!(s.contains("No updates"), "expected 'No updates', got: {s}");
        assert!(s.contains("remote: 15712"));
        assert!(s.contains("local: 15712"));
        assert!(!s.contains("Fetched"));
    }

    fn seed_posts(db: &Db, n: u32) {
        for i in 1..=n {
            upsert_post_locally(
                db,
                &make_esa_post(i, "P", "# body", "2025-01-01T00:00:00+09:00"),
            )
            .unwrap();
        }
    }

    // T-174: incremental harvest with 0 diff results preserves prior total_count
    //        instead of overwriting sync_state with the diff-query total.
    #[tokio::test]
    async fn incremental_with_zero_diff_preserves_prior_total() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(&[], None, 0)))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();
        seed_posts(&db, 5);
        storage::save_sync_state(
            db.conn(),
            &storage::SyncStateUpdate {
                latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
                total_count: 5,
                local_count: 5,
                last_page: None,
            },
        )
        .unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert_eq!(r.posts_fetched, 0);
        assert_eq!(r.local_count, 5);
        assert_eq!(
            r.total_count, 5,
            "prior total_count must not be overwritten by diff-query result (0)"
        );
        assert!(!r.gap_detected);

        let state = storage::get_sync_state(db.conn()).unwrap().unwrap();
        assert_eq!(
            state.total_count, 5,
            "saved state must preserve prior total"
        );
    }

    // T-175: incremental harvest self-heals when state was previously corrupted
    //        (e.g., total_count overwritten with 0 by an earlier buggy run).
    //        local_count acts as a floor on the effective remote total.
    #[tokio::test]
    async fn incremental_self_heals_corrupt_total() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(&[], None, 0)))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url_unchecked("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();
        seed_posts(&db, 7);
        storage::save_sync_state(
            db.conn(),
            &storage::SyncStateUpdate {
                latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
                total_count: 0,
                local_count: 7,
                last_page: None,
            },
        )
        .unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert_eq!(
            r.total_count, 7,
            "local_count floor should recover the total"
        );
        assert!(!r.gap_detected);

        let state = storage::get_sync_state(db.conn()).unwrap().unwrap();
        assert_eq!(state.total_count, 7, "saved state must be self-healed");
    }

    // T-176: effective_total decision table — one row per branch of the
    //        full / incremental / local_floor matrix. See fn rustdoc for the rule.
    #[test]
    fn effective_total_resolver() {
        assert_eq!(
            effective_total(true, 100, 50, 9999),
            50,
            "full ignores prior/floor"
        );
        assert_eq!(effective_total(true, 0, 100, 0), 100);
        assert_eq!(
            effective_total(false, 100, 5, 0),
            100,
            "incremental checkpoint with prior > max_api keeps prior"
        );
        assert_eq!(
            effective_total(false, 100, 5, 100),
            100,
            "incremental final with local_floor == prior is idempotent"
        );
        assert_eq!(
            effective_total(false, 0, 0, 7),
            7,
            "incremental with corrupt prior self-heals via local_floor"
        );
        assert_eq!(
            effective_total(false, 0, 200, 50),
            200,
            "first incremental (no prior) accepts the diff total when it dominates"
        );
    }

    // T-353: SyncError::is_retryable() returns true for transient SQLite busy
    // (WAL contention). Without this, the catch-all CantCreat (73) routing
    // blocks AI-agent auto-retry on recoverable contention (#138 subtask 2).
    #[test]
    fn is_retryable_true_for_sqlite_busy() {
        let busy = sqlite_failure(ErrorCode::DatabaseBusy, 5);
        let err = SyncError::Storage(StorageError::Db(busy));
        assert!(err.is_retryable(), "SQLITE_BUSY must be retryable");
    }

    // T-354: SyncError::is_retryable() returns true for SQLITE_LOCKED.
    #[test]
    fn is_retryable_true_for_sqlite_locked() {
        let locked = sqlite_failure(ErrorCode::DatabaseLocked, 6);
        let err = SyncError::Storage(StorageError::Db(locked));
        assert!(err.is_retryable(), "SQLITE_LOCKED must be retryable");
    }

    // T-355: SyncError::is_retryable() returns true for transient I/O kinds
    // (WouldBlock / Interrupted / TimedOut).
    #[test]
    fn is_retryable_true_for_transient_io() {
        use std::io;
        for kind in [
            io::ErrorKind::WouldBlock,
            io::ErrorKind::Interrupted,
            io::ErrorKind::TimedOut,
        ] {
            let err = SyncError::Storage(StorageError::Io(io::Error::from(kind)));
            assert!(err.is_retryable(), "{kind:?} must be retryable");
        }
    }

    // T-356: SyncError::is_retryable() returns false for non-retryable
    // storage variants (Open failure, non-busy SQLite, permanent I/O).
    #[test]
    fn is_retryable_false_for_non_retryable_storage() {
        use std::io;

        let open_err = SyncError::Storage(StorageError::Open("path missing".into()));
        assert!(!open_err.is_retryable());

        let schema = sqlite_failure(ErrorCode::DatabaseCorrupt, 11);
        let db_err = SyncError::Storage(StorageError::Db(schema));
        assert!(!db_err.is_retryable());

        let perm = SyncError::Storage(StorageError::Io(io::Error::from(
            io::ErrorKind::PermissionDenied,
        )));
        assert!(!perm.is_retryable());
    }

    // T-357: SyncError::is_retryable() always returns false for Client
    // variants — Client retryability is classified separately in
    // SaeError::error_code (per-variant Network / MaxRetries / Api routing).
    #[test]
    fn is_retryable_false_for_client_variants() {
        let token = SyncError::Client(ClientError::TokenNotSet);
        assert!(!token.is_retryable());

        let api = SyncError::Client(ClientError::Api {
            status: 503,
            body: "Service Unavailable".into(),
        });
        assert!(!api.is_retryable());
    }
}
