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

impl std::fmt::Display for HarvestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fetched {} posts, stored {}. remote: {} | local: {}",
            self.posts_fetched, self.posts_stored, self.total_count, self.local_count,
        )?;
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

pub async fn harvest(
    client: &EsaClient,
    db: &Db,
    team: &str,
    full: bool,
) -> Result<HarvestResult, SyncError> {
    let state = storage::get_sync_state(db.conn())?;

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
    let mut api_total: u32 = 0;

    // esa API caps pagination at 10,000 items per query.
    // When hit, we narrow by updated_at window and continue.
    let mut window_boundary: Option<String> = None;
    let mut batch_oldest_ts: Option<String> = None;

    'outer: loop {
        let query = build_window_query(base_query.as_deref(), window_boundary.as_deref());

        let resp = match client.list_posts(team, page, query.as_deref()).await {
            Ok(r) => r,
            Err(ClientError::Api(ref msg)) if is_pagination_limit(msg) => {
                if let Some(ref oldest) = batch_oldest_ts {
                    if window_boundary.as_ref() == Some(oldest) {
                        eprintln!("  window didn't advance, stopping");
                        break;
                    }
                    eprintln!("  pagination limit — narrowing to updated:<={oldest}");
                    window_boundary = Some(oldest.clone());
                    batch_oldest_ts = None;
                    page = 1;
                    continue 'outer;
                }
                break;
            }
            Err(e) => return Err(e.into()),
        };
        api_total = resp.total_count;
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
        total_fetched += resp.posts.len() as u32;

        storage::save_sync_state(
            &tx,
            &storage::SyncStateUpdate {
                latest_updated_at: latest.as_deref(),
                total_count: api_total,
                local_count: prior_count + total_stored,
                last_page: resp.next_page,
            },
        )?;
        tx.commit().map_err(StorageError::Db)?;

        eprintln!("  page {page}/{est_pages} — {total_fetched} posts fetched");
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
    let gap_detected = local_count < api_total;

    storage::save_sync_state(
        db.conn(),
        &storage::SyncStateUpdate {
            latest_updated_at: latest.as_deref(),
            total_count: api_total,
            local_count,
            last_page: None,
        },
    )?;

    if gap_detected {
        eprintln!(
            "  warning: gap detected — {} missing posts (run with --full to re-sync)",
            api_total - local_count
        );
        info!(
            local_count,
            api_total,
            missing = api_total - local_count,
            "gap detected"
        );
    }

    Ok(HarvestResult {
        posts_fetched: total_fetched,
        posts_stored: total_stored,
        total_count: api_total,
        local_count,
        gap_detected,
    })
}

fn is_pagination_limit(msg: &str) -> bool {
    msg.contains("10,000") || msg.contains("10000")
}

fn build_window_query(base: Option<&str>, boundary: Option<&str>) -> Option<String> {
    match (base, boundary) {
        (None, None) => None,
        (Some(b), None) => Some(b.to_string()),
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
mod tests {
    use super::*;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn api_post(number: u32, name: &str, updated_at: &str) -> serde_json::Value {
        serde_json::json!({
            "number": number,
            "name": name,
            "full_name": format!("dev/{name}"),
            "body_md": format!("# {name}"),
            "category": "dev",
            "tags": ["test"],
            "wip": false,
            "kind": "stock",
            "url": format!("https://example.esa.io/posts/{number}"),
            "created_at": "2025-01-01T00:00:00+09:00",
            "updated_at": updated_at,
            "created_by": {"screen_name": "alice"},
            "updated_by": {"screen_name": "bob"},
            "revision_number": 1
        })
    }

    fn posts_response(
        posts: Vec<serde_json::Value>,
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
                vec![
                    api_post(1, "A", "2025-01-01T00:00:00+09:00"),
                    api_post(2, "B", "2025-01-02T00:00:00+09:00"),
                    api_post(3, "C", "2025-01-03T00:00:00+09:00"),
                ],
                None,
                3,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
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
                vec![
                    api_post(1, "A", "2025-01-01T00:00:00+09:00"),
                    api_post(2, "B", "2025-01-02T00:00:00+09:00"),
                ],
                None,
                2,
            )))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();
        harvest(&client, &db, "t", false).await.unwrap();

        Mock::given(method("GET"))
            .and(path("/teams/t/posts"))
            .and(query_param("q", "updated:>2025-01-02T00:00:00+09:00"))
            .respond_with(ResponseTemplate::new(200).set_body_json(posts_response(
                vec![api_post(2, "B Updated", "2025-01-03T00:00:00+09:00")],
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
                vec![api_post(1, "A", "2025-01-01T00:00:00+09:00")],
                None,
                5,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
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
                vec![
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
                vec![api_post(3, "C", "2025-01-03T00:00:00+09:00")],
                None,
                3,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
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
                vec![api_post(1, "A", "2025-01-01T00:00:00+09:00")],
                None,
                1,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
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

    // T-211: resolve_start returns page 1 and no query for a first-time sync
    #[test]
    fn resolve_start_first_sync() {
        let (page, q) = resolve_start(false, &None);
        assert_eq!(page, 1);
        assert!(q.is_none());
    }

    // T-212: resolve_start returns updated: query when prior sync state exists
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

    // T-213: resolve_start resumes from last_page when a checkpoint exists
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

    // T-214: resolve_start ignores saved state when full sync is requested
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

    // T-163: is_pagination_limit
    #[test]
    fn is_pagination_limit_detects_comma_format() {
        assert!(is_pagination_limit("exceeds 10,000 items"));
    }

    // T-215: is_pagination_limit detects error message without comma in number
    #[test]
    fn is_pagination_limit_detects_plain_format() {
        assert!(is_pagination_limit("exceeds 10000 items"));
    }

    // T-216: is_pagination_limit returns false for messages with non-10000 numbers
    #[test]
    fn is_pagination_limit_rejects_other_numbers() {
        assert!(!is_pagination_limit("1000 items"));
        assert!(!is_pagination_limit(""));
    }

    // T-164: post_to_row with body_md: None → empty string
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
            created_by: crate::client::EsaUser {
                screen_name: "alice".into(),
            },
            updated_by: crate::client::EsaUser {
                screen_name: "bob".into(),
            },
            revision_number: 1,
        };
        let row = post_to_row(&post);
        assert_eq!(row.body_md, "");
        assert_eq!(row.tags, vec!["rust".to_string()]);
    }

    // T-217: build_window_query returns None when both base and boundary are absent
    #[test]
    fn build_window_query_none_none() {
        assert!(build_window_query(None, None).is_none());
    }

    // T-218: build_window_query returns base query unchanged when boundary is absent
    #[test]
    fn build_window_query_base_only() {
        assert_eq!(
            build_window_query(Some("updated:>2025-01-01"), None),
            Some("updated:>2025-01-01".into())
        );
    }

    // T-219: build_window_query returns upper-bound clause when only boundary is set
    #[test]
    fn build_window_query_boundary_only() {
        assert_eq!(
            build_window_query(None, Some("2025-06-01")),
            Some("updated:<=2025-06-01".into())
        );
    }

    // T-220: build_window_query combines base and boundary into a window query
    #[test]
    fn build_window_query_both() {
        assert_eq!(
            build_window_query(Some("updated:>2025-01-01"), Some("2025-06-01")),
            Some("updated:>2025-01-01 updated:<=2025-06-01".into())
        );
    }

    // T-221: HarvestResult Display omits gap message when gap_detected is false
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

    // T-222: HarvestResult Display includes missing count when gap_detected is true
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
                vec![
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
                vec![api_post(3, "C", "2025-01-01T00:00:00+09:00")],
                None,
                1,
            )))
            .mount(&server)
            .await;

        let client = EsaClient::with_base_url("tok".into(), server.uri());
        let db = Db::open_memory().unwrap();

        let r = harvest(&client, &db, "t", false).await.unwrap();
        assert!(
            r.posts_fetched >= 3,
            "should fetch posts across window narrowing"
        );
        assert_eq!(storage::count_posts(db.conn()).unwrap(), 3);
    }
}
