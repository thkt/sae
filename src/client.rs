use std::fmt;
use std::time::Duration;

use amici::cli::env_lookup;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderValue};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::time::{Instant, sleep};
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
#[non_exhaustive]
pub enum ClientError {
    #[error("ESA_ACCESS_TOKEN not set")]
    TokenNotSet,

    /// HTTP non-success response from esa API. `status` is preserved so
    /// downstream classification can split client-side (4xx, e.g. 404 →
    /// DATA_ERROR) from server-side (5xx → INTERNAL) per #136.
    #[error("esa API error: HTTP {status}: {body}")]
    Api { status: u16, body: String },

    /// Pre-call request construction failure (URL parse, malformed token
    /// header). Distinct from `Api` because no HTTP exchange occurred —
    /// these are program-detectable bugs and route to INTERNAL (70).
    #[error("invalid request: {0}")]
    InvalidRequest(String),

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

impl fmt::Debug for EsaClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            base_url: ESA_API_BASE.to_owned(),
            request_interval: REQUEST_INTERVAL,
            last_request: Mutex::new(None),
        }
    }

    /// Constructs a client against an arbitrary `base_url`, enforcing the
    /// esa.io SSRF allowlist. Production callers should use [`EsaClient::new`];
    /// tests that need wiremock should use [`EsaClient::with_base_url_unchecked`].
    pub fn with_base_url(token: String, base_url: String) -> Result<Self, ClientError> {
        validate_base_url(&base_url)?;
        Ok(Self::construct(token, base_url, REQUEST_INTERVAL))
    }

    /// Test-only constructor that bypasses the SSRF allowlist so wiremock
    /// servers on `127.0.0.1` / `localhost` are reachable. Not exposed in
    /// production builds (`#[cfg(test)]` only), so the SSRF guard cannot leak
    /// into the binary path.
    #[cfg(test)]
    pub(crate) fn with_base_url_unchecked(token: String, base_url: String) -> Self {
        Self::construct(token, base_url, Duration::ZERO)
    }

    /// Test-only constructor that lets tests pin `request_interval` to a
    /// non-zero value so the throttle path can be exercised under virtual
    /// time. Existing `with_base_url_unchecked` call sites keep `Duration::ZERO`
    /// semantics; this is the dedicated seam for throttle regression tests
    /// (T-002 / T-003) that must observe `request_interval` enforcement.
    #[cfg(test)]
    pub(crate) fn with_test_interval(token: String, base_url: String, interval: Duration) -> Self {
        Self::construct(token, base_url, interval)
    }

    fn construct(token: String, base_url: String, request_interval: Duration) -> Self {
        Self {
            http: Client::new(),
            token,
            base_url,
            request_interval,
            last_request: Mutex::new(None),
        }
    }

    pub fn from_env() -> Result<Self, ClientError> {
        Self::from_env_with(env_lookup())
    }

    pub(crate) fn from_env_with(
        get_var: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ClientError> {
        let token = get_var("ESA_ACCESS_TOKEN").ok_or(ClientError::TokenNotSet)?;
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
                name: params.name.to_owned(),
                body_md: params.body_md.map(str::to_owned),
                category: params.category.map(str::to_owned),
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
                name: params.name.map(str::to_owned),
                body_md: params.body_md.map(str::to_owned),
                category: params.category.map(str::to_owned),
                tags: params.tags.clone(),
                wip: params.wip,
            },
        };
        self.send(|http| http.patch(&url).json(&body)).await
    }

    fn build_url(&self, path: &str) -> Result<reqwest::Url, ClientError> {
        let raw = format!("{}/{path}", self.base_url);
        reqwest::Url::parse(&raw)
            .map_err(|e| ClientError::InvalidRequest(format!("invalid URL '{raw}': {e}")))
    }

    async fn throttle(&self) {
        if self.request_interval.is_zero() {
            return;
        }
        // NOTE: tokio::sync::Mutex の guard を sleep().await 越しに hold して
        // read → sleep → write を single critical section にする。lock を
        // 2 回別々に取ると、解放の隙に並行 caller が stale な last_request を
        // 読んで quota (75 req/15min) を bypass する race window が開く (#155)。
        // std::sync::Mutex で同じ pattern を書くと .await 越し hold が deadlock
        // を招くため refactor 不可。
        let mut last = self.last_request.lock().await;
        if let Some(prev) = *last {
            let wait = self.request_interval.saturating_sub(prev.elapsed());
            if !wait.is_zero() {
                sleep(wait).await;
            }
        }
        *last = Some(Instant::now());
    }

    fn auth_header(&self) -> Result<HeaderValue, ClientError> {
        let mut v = HeaderValue::from_str(&format!("Bearer {}", self.token))
            .map_err(|_| ClientError::InvalidRequest("invalid token format".into()))?;
        v.set_sensitive(true);
        Ok(v)
    }

    async fn send<T: DeserializeOwned>(
        &self,
        build: impl Fn(&Client) -> reqwest::RequestBuilder,
    ) -> Result<T, ClientError> {
        for attempt in 0..=MAX_RETRIES {
            self.throttle().await;
            let auth = self.auth_header()?;

            let resp = build(&self.http)
                .header(AUTHORIZATION, auth)
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
                .map(str::to_owned);
            match retry_wait(status, retry_after.as_deref(), attempt) {
                Some(wait) if attempt < MAX_RETRIES => {
                    warn!(
                        attempt,
                        status,
                        wait_ms = u64::try_from(wait.as_millis()).unwrap_or(u64::MAX),
                        "Retrying esa API"
                    );
                    sleep(wait).await;
                }
                Some(_) => return Err(ClientError::MaxRetries(MAX_RETRIES)),
                None => {
                    let text = resp.text().await.unwrap_or_default();
                    let safe = redact_token(&text, &self.token);
                    return Err(ClientError::Api {
                        status,
                        body: truncate_str(&safe, 500).to_owned(),
                    });
                }
            }
        }
        Err(ClientError::MaxRetries(MAX_RETRIES))
    }
}

/// SSRF guard: accepts only `https://esa.io` or `https://*.esa.io`. Anything
/// else (private IPs, IPv6 loopback, arbitrary hostnames, non-https schemes,
/// missing host) is rejected with [`ClientError::InvalidRequest`]. Tests that
/// need wiremock should construct via [`EsaClient::with_base_url_unchecked`]
/// rather than weakening this guard with a `cfg(test)` carve-out.
fn validate_base_url(url: &str) -> Result<(), ClientError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| ClientError::InvalidRequest(format!("invalid base_url '{url}': {e}")))?;

    let host_str = parsed
        .host_str()
        .ok_or_else(|| ClientError::InvalidRequest(format!("base_url '{url}' has no host")))?;

    if parsed.scheme() != "https" {
        return Err(ClientError::InvalidRequest(format!(
            "base_url '{url}' must use https scheme (got: {})",
            parsed.scheme()
        )));
    }

    if host_str == "esa.io" || host_str.ends_with(".esa.io") {
        Ok(())
    } else {
        Err(ClientError::InvalidRequest(format!(
            "base_url '{url}' host '{host_str}' is not in allowlist (.esa.io required)"
        )))
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
    use std::sync::Arc;

    use super::*;
    use tokio::time::timeout;
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
        EsaClient::with_base_url_unchecked("test-token".into(), server.uri())
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

    // T-345: get_post on 404 returns ClientError::Api with status=404 preserved
    // so downstream routing (SaeError::error_code) can split 404 → DATA_ERROR
    // from non-404 → INTERNAL per #136.
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
        assert!(
            matches!(err, ClientError::Api { status: 404, .. }),
            "expected Api {{ status: 404, .. }}, got: {err:?}"
        );
        assert!(err.to_string().contains("HTTP 404"));
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

    // T-153: EsaClient::from_env returns TokenNotSet when ESA_ACCESS_TOKEN is absent
    #[test]
    fn from_env_missing_token() {
        let err = EsaClient::from_env_with(|_| None).unwrap_err();
        assert!(matches!(err, ClientError::TokenNotSet));
    }

    // T-154: EsaClient::from_env returns TokenNotSet when ESA_ACCESS_TOKEN is empty
    #[test]
    fn from_env_empty_token() {
        let err = EsaClient::from_env_with(|key| match key {
            "ESA_ACCESS_TOKEN" => Some("".into()),
            _ => None,
        })
        .unwrap_err();
        assert!(matches!(err, ClientError::TokenNotSet));
    }

    // T-155: EsaClient Debug output redacts the token value
    #[test]
    fn debug_redacts_token() {
        let client = EsaClient::new("secret-token-123".into());
        let debug = format!("{client:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("secret-token-123"));
    }

    // T-156: retry_wait returns Retry-After header duration on 429
    #[test]
    fn retry_wait_429_uses_header() {
        assert_eq!(retry_wait(429, Some("3"), 0), Some(Duration::from_secs(3)));
    }

    // T-157: retry_wait returns default 5s on 429 when Retry-After header is absent
    #[test]
    fn retry_wait_429_defaults() {
        assert_eq!(retry_wait(429, None, 0), Some(Duration::from_secs(5)));
    }

    // T-158: retry_wait applies exponential backoff for 5xx errors
    #[test]
    fn retry_wait_500_backoff() {
        assert_eq!(retry_wait(500, None, 0), Some(Duration::from_millis(500)));
        assert_eq!(retry_wait(502, None, 1), Some(Duration::from_millis(1000)));
        assert_eq!(retry_wait(503, None, 2), Some(Duration::from_millis(2000)));
    }

    // T-159: retry_wait returns None for non-retryable status codes (4xx)
    #[test]
    fn retry_wait_non_retryable() {
        assert!(retry_wait(400, None, 0).is_none());
        assert!(retry_wait(404, None, 0).is_none());
    }

    // T-160: retry_wait falls back to default 5s when Retry-After header is non-numeric
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
            name: "Test Post".to_owned(),
            full_name: "dev/Test Post".to_owned(),
            body_md,
            category: Some("dev".to_owned()),
            tags: vec!["rust".to_owned()],
            wip: false,
            kind: "stock".to_owned(),
            url: "https://example.esa.io/posts/42".to_owned(),
            created_at: "2025-01-01T00:00:00+09:00".to_owned(),
            updated_at: "2025-01-02T00:00:00+09:00".to_owned(),
            created_by: EsaUser {
                screen_name: "alice".to_owned(),
            },
            updated_by: EsaUser {
                screen_name: "bob".to_owned(),
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
        assert!(v["body_md"].is_null(), "body_md should be null when None");
        assert_eq!(v["number"], 42);
        assert_eq!(v["name"], "Test Post");
    }

    // T-043: get --json --with-body → body_md is string value
    #[test]
    fn esa_post_json_body_md_present_when_some() {
        let post = test_esa_post(Some("# Hello\nWorld".to_owned()));
        let json_str = serde_json::to_string(&post).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(
            v["body_md"], "# Hello\nWorld",
            "body_md should be the string value"
        );
    }

    // T-048: EsaPost → JSON serialization contract (name, number, url, wip, tags).
    // Shared by create/update/archive/ship --json code paths.
    // T-243: esa_post_json_has_name_number_url
    #[test]
    fn esa_post_json_has_name_number_url() {
        let post = test_esa_post(Some("body".to_owned()));
        let json_str = serde_json::to_string(&post).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["number"], 42, "number field");
        assert_eq!(v["name"], "Test Post", "name field");
        assert_eq!(v["url"], "https://example.esa.io/posts/42", "url field");
        assert_eq!(v["wip"], false, "wip field");
        assert_eq!(v["tags"][0], "rust", "tags field");
    }

    // T-058: create_post sends correct request body
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

    // T-059: create_post omits None fields via skip_serializing_if
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

    // T-380: SSRF allowlist accepts https://api.esa.io (canonical production host)
    #[test]
    fn validate_base_url_accepts_api_esa_io() {
        validate_base_url("https://api.esa.io/v1").expect("api.esa.io should be allowed");
    }

    // T-381: SSRF allowlist accepts https://esa.io (root domain)
    #[test]
    fn validate_base_url_accepts_root_esa_io() {
        validate_base_url("https://esa.io").expect("root esa.io should be allowed");
    }

    // T-382: SSRF allowlist rejects arbitrary hosts (defense against config
    // pointing the client at an attacker-controlled host)
    #[test]
    fn validate_base_url_rejects_arbitrary_host() {
        let err = validate_base_url("https://attacker.com")
            .expect_err("non-esa.io host must be rejected");
        assert!(matches!(err, ClientError::InvalidRequest(_)));
    }

    // T-383: SSRF allowlist rejects host suffix attacks (esa.io.attacker.com
    // ends with .com, not .esa.io)
    #[test]
    fn validate_base_url_rejects_host_suffix_attack() {
        let err = validate_base_url("https://esa.io.attacker.com")
            .expect_err("esa.io.attacker.com suffix must be rejected");
        assert!(matches!(err, ClientError::InvalidRequest(_)));
    }

    // T-384: SSRF allowlist rejects private IP literals (defense in depth even
    // though they would not match the .esa.io allowlist anyway)
    #[test]
    fn validate_base_url_rejects_private_ip() {
        let err =
            validate_base_url("https://10.0.0.1").expect_err("private IP literal must be rejected");
        assert!(matches!(err, ClientError::InvalidRequest(_)));
    }

    // T-387: SSRF allowlist rejects IPv6 loopback `::1`. Issue #138 subtask 3
    // explicitly enumerates `::1` as a rejection target alongside IPv4 private
    // ranges. `reqwest::Url::parse` reads it as `host_str = "[::1]"` which
    // cannot satisfy the `.esa.io` suffix.
    #[test]
    fn validate_base_url_rejects_ipv6_loopback() {
        let err = validate_base_url("https://[::1]/").expect_err("IPv6 loopback must be rejected");
        assert!(matches!(err, ClientError::InvalidRequest(_)));
    }

    // T-385: SSRF allowlist rejects non-http(s) schemes (gopher / file / ftp
    // / data are common SSRF amplifiers)
    #[test]
    fn validate_base_url_rejects_ftp_scheme() {
        let err = validate_base_url("ftp://api.esa.io").expect_err("ftp scheme must be rejected");
        assert!(matches!(err, ClientError::InvalidRequest(_)));
    }

    // T-386: SSRF allowlist rejects unparseable URLs
    #[test]
    fn validate_base_url_rejects_invalid_url() {
        let err = validate_base_url("not-a-url").expect_err("garbage must be rejected");
        assert!(matches!(err, ClientError::InvalidRequest(_)));
    }

    // T-388: with_base_url uses production REQUEST_INTERVAL so the esa
    // 75 req / 15 min quota throttle is enforced.
    #[test]
    fn with_base_url_sets_request_interval() {
        let client =
            EsaClient::with_base_url("tok".into(), "https://example.esa.io".into()).unwrap();
        assert_eq!(client.request_interval, REQUEST_INTERVAL);
    }

    // T-389: with_base_url_unchecked keeps Duration::ZERO so wiremock-backed
    // tests are not gated by the production rate limiter.
    #[test]
    fn with_base_url_unchecked_keeps_zero_interval() {
        let client = EsaClient::with_base_url_unchecked("tok".into(), "http://127.0.0.1/".into());
        assert_eq!(client.request_interval, Duration::ZERO);
    }

    // T-002: throttle_serializes_concurrent_callers
    // FR-001 / FR-003: under concurrent `Arc<EsaClient>` callers, `throttle()`
    // must hold the lock across read+sleep+write so the second caller's
    // critical-section exit lags the first by at least `request_interval`.
    // Captures each task's post-throttle `Instant::now()` per-task (rather
    // than re-reading `last_request` after both finish, which would only
    // surface the second writer's timestamp) to satisfy the "diff of each
    // task's critical-section exit" intent of the spec.
    #[tokio::test(start_paused = true)]
    async fn throttle_serializes_concurrent_callers() {
        let client = Arc::new(EsaClient::with_test_interval(
            "tok".into(),
            "https://example.esa.io".into(),
            Duration::from_millis(100),
        ));

        let c1 = Arc::clone(&client);
        let c2 = Arc::clone(&client);

        let h1 = tokio::spawn(async move {
            c1.throttle().await;
            Instant::now()
        });
        let h2 = tokio::spawn(async move {
            c2.throttle().await;
            Instant::now()
        });

        let t1 = h1.await.expect("task 1 panicked");
        let t2 = h2.await.expect("task 2 panicked");

        let gap = if t1 > t2 { t1 - t2 } else { t2 - t1 };
        assert!(
            gap >= Duration::from_millis(100),
            "concurrent throttle gap {gap:?} must be >= 100ms (request_interval)"
        );
    }

    // T-003: throttle_cancellation_preserves_last_request
    // FR-005: dropping a `throttle()` future mid-sleep must leave
    // `last_request` unchanged (no request was issued, so the timestamp
    // must not advance). Verified by capturing `last_request` after the
    // first completed throttle, dropping a second throttle while it sleeps
    // via `tokio::time::timeout`, then asserting the stored timestamp is
    // still the post-first-call value.
    #[tokio::test(start_paused = true)]
    async fn throttle_cancellation_preserves_last_request() {
        let client = EsaClient::with_test_interval(
            "tok".into(),
            "https://example.esa.io".into(),
            Duration::from_millis(500),
        );

        // First throttle completes and writes t1.
        client.throttle().await;
        let t1 = client
            .last_request
            .lock()
            .await
            .expect("last_request must be Some after first throttle");

        // Second throttle gets cancelled mid-sleep via timeout < interval.
        let timed_out = timeout(Duration::from_millis(50), client.throttle()).await;
        assert!(
            timed_out.is_err(),
            "timeout(50ms) must elapse before throttle(500ms) completes"
        );

        // last_request must still be t1 because the second throttle was
        // dropped before its write phase.
        let after = *client.last_request.lock().await;
        assert_eq!(
            after,
            Some(t1),
            "cancelled throttle must not advance last_request"
        );
    }
}
