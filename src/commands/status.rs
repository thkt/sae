use std::path::Path;

use crate::config::{Config, validate_team_name};
use crate::output;
use crate::storage;
use crate::storage::TeamStatus;
use crate::tools::SaeError;

pub(crate) fn run_status(
    config: &Config,
    team: Option<&str>,
    json: bool,
) -> Result<String, SaeError> {
    let target_teams: Vec<&str> = if let Some(t) = team {
        vec![config.resolve_team(Some(t))?]
    } else {
        config
            .teams
            .iter()
            .filter(|t| match validate_team_name(t) {
                Ok(_) => true,
                Err(_) => {
                    eprintln!("Warning: skipping invalid team name: {t}");
                    false
                }
            })
            .map(String::as_str)
            .collect()
    };
    let statuses = collect_team_statuses(config, &target_teams);
    output::status(&statuses, json)
}

fn collect_team_statuses(config: &Config, teams: &[&str]) -> Vec<TeamStatus> {
    let mut statuses = Vec::new();
    for t in teams {
        let ts = match config.team_db_path(t) {
            Ok(path) if path.exists() => match query_team_status(t, &path) {
                Ok(ts) => ts,
                Err(e) => TeamStatus::error(*t, &e),
            },
            Ok(path) => TeamStatus::not_synced(*t, Some(path.display().to_string())),
            Err(e) => TeamStatus::error(*t, &e),
        };
        statuses.push(ts);
    }
    statuses
}

fn query_team_status(team: &str, path: &Path) -> Result<TeamStatus, SaeError> {
    let db = storage::Db::open(path)?;
    let count = storage::count_posts(db.conn())?;
    let state = storage::get_sync_state(db.conn())?;
    let pending_embed = storage::count_unembedded_chunks(db.conn())?;
    let mut ts = if state.is_some() {
        TeamStatus::synced(team, count, state)
    } else {
        TeamStatus::not_synced(team, None)
    };
    ts.pending_embed = pending_embed;
    Ok(ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_post_row() -> storage::EsaPostRow {
        storage::EsaPostRow {
            category: None,
            ..storage::test_post_row(1)
        }
    }

    // T-074: at least 1 test co-located with status handlers
    #[test]
    fn team_status_error_constructs_correctly() {
        let ts = TeamStatus::error("myteam", &"db error");
        assert_eq!(ts.team, "myteam");
        assert_eq!(ts.status, storage::SyncStatus::Error);
        assert_eq!(ts.error.as_deref(), Some("db error"));
        assert_eq!(ts.posts, 0);
    }

    // T-075: query_team_status with sync state present → TeamStatus::Synced
    #[test]
    fn query_team_status_returns_synced_when_state_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        {
            let db = storage::Db::open(&path).unwrap();
            storage::upsert_post(db.conn(), &make_post_row()).unwrap();
            storage::save_sync_state(
                db.conn(),
                &storage::SyncStateUpdate {
                    latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
                    total_count: 1,
                    local_count: 1,
                    last_page: None,
                },
            )
            .unwrap();
        }
        let ts = query_team_status("myteam", &path).unwrap();
        assert_eq!(ts.team, "myteam");
        assert_eq!(ts.status, storage::SyncStatus::Synced);
        assert_eq!(ts.posts, 1);
        assert!(ts.sync_state.is_some());
    }

    // T-076: query_team_status with no sync state → TeamStatus::NotSynced
    #[test]
    fn query_team_status_returns_not_synced_when_no_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let _db = storage::Db::open(&path).unwrap();
        let ts = query_team_status("myteam", &path).unwrap();
        assert_eq!(ts.team, "myteam");
        assert_eq!(ts.status, storage::SyncStatus::NotSynced);
        assert!(ts.sync_state.is_none());
    }

    // T-073: posts exist but sync_state absent → NotSynced (bug: was reporting Synced)
    #[test]
    fn query_team_status_returns_not_synced_when_posts_exist_but_no_sync_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        {
            let db = storage::Db::open(&path).unwrap();
            storage::upsert_post(db.conn(), &make_post_row()).unwrap();
            // sync_state intentionally NOT saved
        }
        let ts = query_team_status("myteam", &path).unwrap();
        assert_eq!(ts.team, "myteam");
        assert_eq!(ts.status, storage::SyncStatus::NotSynced);
        assert!(ts.sync_state.is_none());
    }

    // T-126: unembedded chunks after interrupted embed → pending_embed > 0
    #[test]
    fn query_team_status_reports_pending_embed_when_chunks_unembedded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        {
            let db = storage::Db::open(&path).unwrap();
            let mut row = make_post_row();
            row.body_md = "# Section\n\nSome content here to generate a chunk.".into();
            storage::upsert_post(db.conn(), &row).unwrap();
            storage::rechunk_post(db.conn(), row.number, &row.body_md).unwrap();
            storage::save_sync_state(
                db.conn(),
                &storage::SyncStateUpdate {
                    latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
                    total_count: 1,
                    local_count: 1,
                    last_page: None,
                },
            )
            .unwrap();
            // embed intentionally NOT run → chunks remain unembedded
        }
        let ts = query_team_status("myteam", &path).unwrap();
        assert_eq!(ts.status, storage::SyncStatus::Synced);
        assert!(ts.pending_embed > 0, "should report unembedded chunks");
    }

    // T-127: no unembedded chunks → pending_embed == 0
    #[test]
    fn query_team_status_reports_zero_pending_embed_when_no_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        {
            let db = storage::Db::open(&path).unwrap();
            storage::save_sync_state(
                db.conn(),
                &storage::SyncStateUpdate {
                    latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
                    total_count: 0,
                    local_count: 0,
                    last_page: None,
                },
            )
            .unwrap();
        }
        let ts = query_team_status("myteam", &path).unwrap();
        assert_eq!(ts.pending_embed, 0);
    }
}
