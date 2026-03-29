#[derive(Debug, Clone)]
pub struct EsaPostRow {
    pub number: u32,
    pub name: String,
    pub full_name: String,
    pub body_md: String,
    pub category: Option<String>,
    pub tags: String,
    pub wip: bool,
    pub kind: String,
    pub url: String,
    pub created_at: String,
    pub updated_at: String,
    pub created_by: String,
    pub updated_by: String,
    pub revision_number: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SyncState {
    pub latest_updated_at: Option<String>,
    pub total_count: u32,
    pub local_count: u32,
    pub last_page: Option<u32>,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncStatus {
    Synced,
    NotSynced,
    Error,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TeamStatus {
    pub team: String,
    pub status: SyncStatus,
    pub posts: u32,
    pub sync_state: Option<SyncState>,
    pub error: Option<String>,
    /// DB path for human-readable output (not included in JSON)
    #[serde(skip)]
    pub db_path: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbedResult {
    pub chunks_embedded: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Failed to open database: {0}")]
    Open(String),

    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-045: status --json → TeamStatus array with team, status, posts fields
    #[test]
    fn team_status_serializes_to_json_with_expected_fields() {
        let status = TeamStatus {
            team: "myteam".to_string(),
            status: SyncStatus::Synced,
            posts: 42,
            sync_state: Some(SyncState {
                latest_updated_at: Some("2025-01-01T00:00:00+09:00".to_string()),
                total_count: 100,
                local_count: 42,
                last_page: None,
                updated_at: "2025-01-01 00:00:00".to_string(),
            }),
            error: None,
            db_path: None,
        };
        let json_str =
            serde_json::to_string(&status).expect("[T-045] TeamStatus should serialize");
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["team"], "myteam");
        assert_eq!(v["status"], "synced");
        assert_eq!(v["posts"], 42);
        assert!(v["sync_state"].is_object(), "[T-045] sync_state should be an object");
    }

    // T-045: TeamStatus array serialization
    #[test]
    fn team_status_array_serializes_to_json_array() {
        let statuses = vec![
            TeamStatus {
                team: "team-a".to_string(),
                status: SyncStatus::Synced,
                posts: 10,
                sync_state: None,
                error: None,
                db_path: None,
            },
            TeamStatus {
                team: "team-b".to_string(),
                status: SyncStatus::NotSynced,
                posts: 0,
                sync_state: None,
                error: None,
                db_path: None,
            },
        ];
        let json_str = serde_json::to_string(&statuses)
            .expect("[T-045] TeamStatus array should serialize");
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(v.is_array(), "[T-045] should be a JSON array");
        assert_eq!(v.as_array().unwrap().len(), 2);
        assert_eq!(v[0]["team"], "team-a");
        assert_eq!(v[1]["status"], "not_synced");
    }

    // TC-010: SyncStatus::Error serializes to "error" with error message
    #[test]
    fn team_status_error_serializes_correctly() {
        let status = TeamStatus {
            team: "broken".to_string(),
            status: SyncStatus::Error,
            posts: 0,
            sync_state: None,
            error: Some("config missing".to_string()),
            db_path: None,
        };
        let json_str = serde_json::to_string(&status).expect("[TC-010] should serialize");
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["status"], "error");
        assert_eq!(v["error"], "config missing");
    }

    // T-045: SyncState serializes (required by TeamStatus.sync_state)
    #[test]
    fn sync_state_serializes_to_json() {
        let state = SyncState {
            latest_updated_at: Some("2025-01-01T00:00:00+09:00".to_string()),
            total_count: 100,
            local_count: 50,
            last_page: Some(3),
            updated_at: "2025-01-01 00:00:00".to_string(),
        };
        let json_str =
            serde_json::to_string(&state).expect("[T-045] SyncState should serialize");
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["total_count"], 100);
        assert_eq!(v["local_count"], 50);
        assert_eq!(v["last_page"], 3);
    }

    // T-053: embed --json → EmbedResult JSON (chunks_embedded field)
    #[test]
    fn embed_result_serializes_to_json_with_chunks_embedded() {
        let result = EmbedResult { chunks_embedded: 150 };
        let json_str =
            serde_json::to_string(&result).expect("[T-053] EmbedResult should serialize");
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["chunks_embedded"], 150);
    }
}
