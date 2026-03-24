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

#[derive(Debug, Clone)]
pub struct SyncState {
    pub latest_updated_at: Option<String>,
    pub total_count: u32,
    pub local_count: u32,
    pub last_page: Option<u32>,
    pub updated_at: String,
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
