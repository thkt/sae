use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::time::Instant;
use tracing::warn;

use crate::redact::{redact_token, truncate_str};

const MAX_RETRIES: u32 = 5;
const PAGE_SIZE: u32 = 100;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_INTERVAL: Duration = Duration::from_secs(12); // 75 req/15min
const ESA_API_BASE: &str = "https://api.esa.io/v1";

#[derive(Debug, Deserialize)]
pub struct PostsResponse {
    pub posts: Vec<EsaPost>,
    pub next_page: Option<u32>,
    pub total_count: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EsaPost {
    pub number: u32,
    pub name: String,
    pub full_name: String,
    pub body_md: Option<String>,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub wip: bool,
    pub kind: String,
    pub url: String,
    pub created_at: String,
    pub updated_at: String,
    pub created_by: EsaUser,
    pub updated_by: EsaUser,
    pub revision_number: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EsaUser {
    pub screen_name: String,
}

pub struct CreatePostParams<'a> {
    pub name: &'a str,
    pub body_md: Option<&'a str>,
    pub category: Option<&'a str>,
    pub tags: Vec<String>,
    pub wip: bool,
}

#[derive(Serialize)]
struct CreatePostRequest {
    post: CreatePostBody,
}

#[derive(Serialize)]
struct CreatePostBody {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    body_md: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    category: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
    wip: bool,
}

#[derive(Default)]
pub struct UpdatePostParams<'a> {
    pub name: Option<&'a str>,
    pub body_md: Option<&'a str>,
    pub category: Option<&'a str>,
    pub tags: Option<Vec<String>>,
    pub wip: Option<bool>,
}

#[derive(Serialize)]
struct UpdatePostRequest {
    post: UpdatePostBody,
}

#[derive(Default, Serialize)]
struct UpdatePostBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    body_md: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wip: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("ESA_ACCESS_TOKEN not set")]
    TokenNotSet,

    #[error("esa API error: {0}")]
    Api(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Max retries ({0}) exceeded")]
    MaxRetries(u32),
}

pub struct EsaClient {
    http: Client,
    token: String,
    base_url: String,
    request_interval: Duration,
    last_request: Mutex<Option<Instant>>,
}

impl std::fmt::Debug for EsaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EsaClient")
            .field("base_url", &self.base_url)
            .field("token", &"[REDACTED]")
            .finish()
    }
}

impl EsaClient {
    pub fn new(token: String) -> Self {
        Self {
            http: Client::new(),
            token,
            base_url: ESA_API_BASE.to_string(),
            request_interval: REQUEST_INTERVAL,
            last_request: Mutex::new(None),
        }
    }

    pub fn with_base_url(token: String, base_url: String) -> Self {
        Self {
            http: Client::new(),
            token,
            base_url,
            request_interval: Duration::ZERO,
            last_request: Mutex::new(None),
        }
    }

    pub fn from_env() -> Result<Self, ClientError> {
        Self::from_env_with(|k| std::env::var(k))
    }

    pub(crate) fn from_env_with(
        get_var: impl Fn(&str) -> Result<String, std::env::VarError>,
    ) -> Result<Self, ClientError> {
        let token = get_var("ESA_ACCESS_TOKEN").map_err(|_| ClientError::TokenNotSet)?;
        if token.is_empty() {
            return Err(ClientError::TokenNotSet);
        }
        Ok(Self::new(token))
    }

    pub async fn list_posts(
        &self,
        team: &str,
        page: u32,
        query: Option<&str>,
    ) -> Result<PostsResponse, ClientError> {
        let mut url = self.build_url(&format!("teams/{team}/posts"))?;
        url.query_pairs_mut()
            .append_pair("page", &page.to_string())
            .append_pair("per_page", &PAGE_SIZE.to_string())
            .append_pair("sort", "updated")
            .append_pair("order", "desc");
        if let Some(q) = query {
            url.query_pairs_mut().append_pair("q", q);
        }
        let url = url.to_string();
        self.send(|http| http.get(&url)).await
    }

    pub async fn get_post(&self, team: &str, post_number: u32) -> Result<EsaPost, ClientError> {
        let url = self
            .build_url(&format!("teams/{team}/posts/{post_number}"))?
            .to_string();
        self.send(|http| http.get(&url)).await
    }

    pub async fn create_post(
        &self,
        team: &str,
        params: &CreatePostParams<'_>,
    ) -> Result<EsaPost, ClientError> {
        let url = self.build_url(&format!("teams/{team}/posts"))?.to_string();
        let body = CreatePostRequest {
            post: CreatePostBody {
                name: params.name.to_string(),
                body_md: params.body_md.map(str::to_string),
                category: params.category.map(str::to_string),
                tags: params.tags.clone(),
                wip: params.wip,
            },
        };
        self.send(|http| http.post(&url).json(&body)).await
    }

    pub async fn update_post(
        &self,
        team: &str,
        post_number: u32,
        params: &UpdatePostParams<'_>,
    ) -> Result<EsaPost, ClientError> {
        let url = self
            .build_url(&format!("teams/{team}/posts/{post_number}"))?
            .to_string();
        let body = UpdatePostRequest {
            post: UpdatePostBody {
                name: params.name.map(str::to_string),
                body_md: params.body_md.map(str::to_string),
                category: params.category.map(str::to_string),
                tags: params.tags.clone(),
                wip: params.wip,
            },
        };
        self.send(|http| http.patch(&url).json(&body)).await
    }

    fn build_url(&self, path: &str) -> Result<reqwest::Url, ClientError> {
        let raw = format!("{}/{path}", self.base_url);
        reqwest::Url::parse(&raw).map_err(|e| ClientError::Api(format!("invalid URL '{raw}': {e}")))
    }

    async fn throttle(&self) {
        if self.request_interval.is_zero() {
            return;
        }
        let sleep_dur = {
            let last = self.last_request.lock().await;
            last.map(|prev| self.request_interval.saturating_sub(prev.elapsed()))
                .filter(|d| !d.is_zero())
        };
        if let Some(dur) = sleep_dur {
            tokio::time::sleep(dur).await;
        }
        *self.last_request.lock().await = Some(Instant::now());
    }

    fn auth_header(&self) -> Result<reqwest::header::HeaderValue, ClientError> {
        let mut v = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.token))
            .map_err(|_| ClientError::Api("invalid token format".into()))?;
        v.set_sensitive(true);
        Ok(v)
    }

    async fn send<T: serde::de::DeserializeOwned>(
        &self,
        build: impl Fn(&Client) -> reqwest::RequestBuilder,
    ) -> Result<T, ClientError> {
        for attempt in 0..=MAX_RETRIES {
            self.throttle().await;
            let auth = self.auth_header()?;

            let resp = build(&self.http)
                .header(reqwest::header::AUTHORIZATION, auth)
                .timeout(REQUEST_TIMEOUT)
                .send()
                .await?;

            let status = resp.status().as_u16();
            if (200..300).contains(&status) {
                return resp.json::<T>().await.map_err(Into::into);
            }

            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            match retry_wait(status, retry_after.as_deref(), attempt) {
                Some(wait) if attempt < MAX_RETRIES => {
                    warn!(
                        attempt,
                        status,
                        wait_ms = wait.as_millis() as u64,
                        "Retrying esa API"
                    );
                    tokio::time::sleep(wait).await;
                }
                Some(_) => return Err(ClientError::MaxRetries(MAX_RETRIES)),
                None => {
                    let text = resp.text().await.unwrap_or_default();
                    let safe = redact_token(&text, &self.token);
                    return Err(ClientError::Api(format!(
                        "HTTP {status}: {}",
                        truncate_str(&safe, 500)
                    )));
                }
            }
        }
        Err(ClientError::MaxRetries(MAX_RETRIES))
    }
}

fn retry_wait(status: u16, retry_after: Option<&str>, attempt: u32) -> Option<Duration> {
    match status {
        429 => {
            let secs = retry_after.and_then(|v| v.parse::<u64>().ok()).unwrap_or(5);
            Some(Duration::from_secs(secs))
        }
        500 | 502 | 503 => Some(Duration::from_millis(500 * 2u64.pow(attempt))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_post_json(number: u32) -> serde_json::Value {
        serde_json::json!({
            "number": number,
            "name": "Getting Started",
            "full_name": "dev/Getting Started",
            "body_md": "# Hello",
            "category": "dev",
            "tags": ["api"],
            "wip": false,
            "kind": "stock",
            "url": "https://example.esa.io/posts/1",
            "created_at": "2025-01-01T00:00:00+09:00",
            "updated_at": "2025-01-02T00:00:00+09:00",
            "created_by": {"screen_name": "alice"},
            "updated_by": {"screen_name": "bob"},
            "revision_number": 3
        })
    }

    async fn test_client(server: &MockServer) -> EsaClient {
        EsaClient::with_base_url("test-token".into(), server.uri())
    }

    #[tokio::test]
    async fn list_posts_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posts": [test_post_json(1)],
                "next_page": null,
                "total_count": 1
            })))
            .mount(&server)
            .await;

        let resp = test_client(&server)
            .await
            .list_posts("myteam", 1, None)
            .await
            .unwrap();
        assert_eq!(resp.posts.len(), 1);
        assert_eq!(resp.posts[0].number, 1);
        assert_eq!(resp.posts[0].name, "Getting Started");
        assert_eq!(resp.total_count, 1);
        assert!(resp.next_page.is_none());
    }

    #[tokio::test]
    async fn list_posts_returns_next_page() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posts": [test_post_json(1)],
                "next_page": 2,
                "total_count": 150
            })))
            .mount(&server)
            .await;

        let resp = test_client(&server)
            .await
            .list_posts("myteam", 1, None)
            .await
            .unwrap();
        assert_eq!(resp.next_page, Some(2));
        assert_eq!(resp.total_count, 150);
    }

    #[tokio::test]
    async fn get_post_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts/42"))
            .respond_with(ResponseTemplate::new(200).set_body_json(test_post_json(42)))
            .mount(&server)
            .await;

        let post = test_client(&server)
            .await
            .get_post("myteam", 42)
            .await
            .unwrap();
        assert_eq!(post.number, 42);
        assert_eq!(post.name, "Getting Started");
    }

    #[tokio::test]
    async fn create_post_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/teams/myteam/posts"))
            .respond_with(ResponseTemplate::new(201).set_body_json(test_post_json(1)))
            .mount(&server)
            .await;

        let post = test_client(&server)
            .await
            .create_post(
                "myteam",
                &CreatePostParams {
                    name: "Test",
                    body_md: Some("# Body"),
                    category: None,
                    tags: vec![],
                    wip: true,
                },
            )
            .await
            .unwrap();
        assert_eq!(post.name, "Getting Started");
    }

    #[tokio::test]
    async fn update_post_success() {
        let server = MockServer::start().await;
        Mock::given(method("PATCH"))
            .and(path("/teams/myteam/posts/1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(test_post_json(1)))
            .mount(&server)
            .await;

        let post = test_client(&server)
            .await
            .update_post(
                "myteam",
                1,
                &UpdatePostParams {
                    name: Some("New Name"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(post.name, "Getting Started");
    }

    #[tokio::test]
    async fn api_error_returns_client_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts/999"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": "not_found",
                "message": "Not Found"
            })))
            .mount(&server)
            .await;

        let err = test_client(&server)
            .await
            .get_post("myteam", 999)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("404"));
    }

    #[tokio::test]
    async fn retry_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts/1"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "1"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts/1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(test_post_json(1)))
            .mount(&server)
            .await;

        let post = test_client(&server)
            .await
            .get_post("myteam", 1)
            .await
            .unwrap();
        assert_eq!(post.name, "Getting Started");
    }

    // T-203: EsaClient::from_env returns TokenNotSet when ESA_ACCESS_TOKEN is absent
    #[test]
    fn from_env_missing_token() {
        let err = EsaClient::from_env_with(|_| Err(std::env::VarError::NotPresent)).unwrap_err();
        assert!(matches!(err, ClientError::TokenNotSet));
    }

    // T-204: EsaClient::from_env returns TokenNotSet when ESA_ACCESS_TOKEN is empty
    #[test]
    fn from_env_empty_token() {
        let err = EsaClient::from_env_with(|key| match key {
            "ESA_ACCESS_TOKEN" => Ok("".into()),
            _ => Err(std::env::VarError::NotPresent),
        })
        .unwrap_err();
        assert!(matches!(err, ClientError::TokenNotSet));
    }

    // T-205: EsaClient Debug output redacts the token value
    #[test]
    fn debug_redacts_token() {
        let client = EsaClient::new("secret-token-123".into());
        let debug = format!("{client:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("secret-token-123"));
    }

    // T-206: retry_wait returns Retry-After header duration on 429
    #[test]
    fn retry_wait_429_uses_header() {
        assert_eq!(retry_wait(429, Some("3"), 0), Some(Duration::from_secs(3)));
    }

    // T-207: retry_wait returns default 5s on 429 when Retry-After header is absent
    #[test]
    fn retry_wait_429_defaults() {
        assert_eq!(retry_wait(429, None, 0), Some(Duration::from_secs(5)));
    }

    // T-208: retry_wait applies exponential backoff for 5xx errors
    #[test]
    fn retry_wait_500_backoff() {
        assert_eq!(retry_wait(500, None, 0), Some(Duration::from_millis(500)));
        assert_eq!(retry_wait(502, None, 1), Some(Duration::from_millis(1000)));
        assert_eq!(retry_wait(503, None, 2), Some(Duration::from_millis(2000)));
    }

    // T-209: retry_wait returns None for non-retryable status codes (4xx)
    #[test]
    fn retry_wait_non_retryable() {
        assert!(retry_wait(400, None, 0).is_none());
        assert!(retry_wait(404, None, 0).is_none());
    }

    // T-210: retry_wait falls back to default 5s when Retry-After header is non-numeric
    #[test]
    fn retry_wait_429_non_numeric_header_defaults() {
        assert_eq!(
            retry_wait(429, Some("Thu, 01 Jan 2026 00:00:00 GMT"), 0),
            Some(Duration::from_secs(5))
        );
    }

    #[tokio::test]
    async fn malformed_json_on_success_returns_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts/1"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let err = test_client(&server)
            .await
            .get_post("myteam", 1)
            .await
            .unwrap_err();
        assert!(matches!(err, ClientError::Network(_)));
    }

    fn test_esa_post(body_md: Option<String>) -> EsaPost {
        EsaPost {
            number: 42,
            name: "Test Post".to_string(),
            full_name: "dev/Test Post".to_string(),
            body_md,
            category: Some("dev".to_string()),
            tags: vec!["rust".to_string()],
            wip: false,
            kind: "stock".to_string(),
            url: "https://example.esa.io/posts/42".to_string(),
            created_at: "2025-01-01T00:00:00+09:00".to_string(),
            updated_at: "2025-01-02T00:00:00+09:00".to_string(),
            created_by: EsaUser {
                screen_name: "alice".to_string(),
            },
            updated_by: EsaUser {
                screen_name: "bob".to_string(),
            },
            revision_number: 3,
        }
    }

    // T-036: get --json → body_md field is null
    #[test]
    fn esa_post_json_body_md_null_when_none() {
        let post = test_esa_post(None);
        let json_str = serde_json::to_string(&post).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(
            v["body_md"].is_null(),
            "[T-036] body_md should be null when None"
        );
        assert_eq!(v["number"], 42);
        assert_eq!(v["name"], "Test Post");
    }

    // T-043: get --json --with-body → body_md is string value
    #[test]
    fn esa_post_json_body_md_present_when_some() {
        let post = test_esa_post(Some("# Hello\nWorld".to_string()));
        let json_str = serde_json::to_string(&post).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(
            v["body_md"], "# Hello\nWorld",
            "[T-043] body_md should be the string value"
        );
    }

    // T-048: EsaPost → JSON serialization contract (name, number, url, wip, tags).
    // Shared by create/update/archive/ship --json code paths.
    #[test]
    fn esa_post_json_has_name_number_url() {
        let post = test_esa_post(Some("body".to_string()));
        let json_str = serde_json::to_string(&post).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["number"], 42, "[T-048] number field");
        assert_eq!(v["name"], "Test Post", "[T-048] name field");
        assert_eq!(
            v["url"], "https://example.esa.io/posts/42",
            "[T-048] url field"
        );
        assert_eq!(v["wip"], false, "[T-048] wip field");
        assert_eq!(v["tags"][0], "rust", "[T-048] tags field");
    }

    // T-108: create_post sends correct request body
    #[tokio::test]
    async fn create_post_sends_correct_payload() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/teams/myteam/posts"))
            .and(body_json(serde_json::json!({
                "post": {
                    "name": "My Title",
                    "body_md": "# Content",
                    "category": "docs",
                    "tags": ["rust", "api"],
                    "wip": true,
                }
            })))
            .respond_with(ResponseTemplate::new(201).set_body_json(test_post_json(1)))
            .mount(&server)
            .await;

        test_client(&server)
            .await
            .create_post(
                "myteam",
                &CreatePostParams {
                    name: "My Title",
                    body_md: Some("# Content"),
                    category: Some("docs"),
                    tags: vec!["rust".into(), "api".into()],
                    wip: true,
                },
            )
            .await
            .unwrap();
    }

    // T-109: create_post omits None fields via skip_serializing_if
    #[tokio::test]
    async fn create_post_omits_none_fields() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/teams/myteam/posts"))
            .and(body_json(serde_json::json!({
                "post": {
                    "name": "Minimal",
                    "wip": false,
                }
            })))
            .respond_with(ResponseTemplate::new(201).set_body_json(test_post_json(1)))
            .mount(&server)
            .await;

        test_client(&server)
            .await
            .create_post(
                "myteam",
                &CreatePostParams {
                    name: "Minimal",
                    body_md: None,
                    category: None,
                    tags: vec![],
                    wip: false,
                },
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn max_retries_exhausted() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts/1"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .mount(&server)
            .await;

        let err = test_client(&server)
            .await
            .get_post("myteam", 1)
            .await
            .unwrap_err();
        assert!(matches!(err, ClientError::MaxRetries(MAX_RETRIES)));
    }

    #[tokio::test]
    async fn list_posts_with_query() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/teams/myteam/posts"))
            .and(query_param("q", "updated:>2025-01-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posts": [test_post_json(1)],
                "next_page": null,
                "total_count": 1
            })))
            .mount(&server)
            .await;

        let resp = test_client(&server)
            .await
            .list_posts("myteam", 1, Some("updated:>2025-01-01"))
            .await
            .unwrap();
        assert_eq!(resp.posts.len(), 1);
    }
}
