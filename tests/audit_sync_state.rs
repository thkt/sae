//! T-005: Compile-time signature check for save_sync_state / save_sync_state_at.
//!
//! After FR-006, these functions should accept `&SyncStateUpdate` instead of
//! 4 individual parameters. This test file will fail to compile if the
//! signature does not match the spec.

use sae::storage::{Db, SyncStateUpdate, get_sync_state, save_sync_state};

/// [T-005] save_sync_state accepts (conn, &SyncStateUpdate)
#[test]
fn t_005_save_sync_state_accepts_sync_state_update() {
    let db = Db::open_memory().unwrap();

    let update = SyncStateUpdate {
        latest_updated_at: Some("2025-01-01T00:00:00+09:00"),
        total_count: 100,
        local_count: 50,
        last_page: Some(3),
    };

    save_sync_state(db.conn(), &update).unwrap();

    let state = get_sync_state(db.conn()).unwrap().unwrap();
    assert_eq!(
        state.latest_updated_at.as_deref(),
        Some("2025-01-01T00:00:00+09:00"),
        "[T-005] latest_updated_at should roundtrip"
    );
    assert_eq!(state.total_count, 100, "[T-005] total_count should be 100");
    assert_eq!(state.local_count, 50, "[T-005] local_count should be 50");
    assert_eq!(
        state.last_page,
        Some(3),
        "[T-005] last_page should be Some(3)"
    );
}

/// [T-005] save_sync_state with None optionals
#[test]
fn t_005_save_sync_state_none_optionals() {
    let db = Db::open_memory().unwrap();

    let update = SyncStateUpdate {
        latest_updated_at: None,
        total_count: 0,
        local_count: 0,
        last_page: None,
    };

    save_sync_state(db.conn(), &update).unwrap();

    let state = get_sync_state(db.conn()).unwrap().unwrap();
    assert_eq!(
        state.latest_updated_at, None,
        "[T-005] latest_updated_at should be None"
    );
    assert_eq!(state.last_page, None, "[T-005] last_page should be None");
}

/// [T-005] SyncStateUpdate borrows from caller's Option<String> via .as_deref()
#[test]
fn t_005_sync_state_update_borrows_from_option_string() {
    let db = Db::open_memory().unwrap();

    // Simulate caller holding Option<String> and borrowing via .as_deref()
    let owned: Option<String> = Some("2025-06-01T12:00:00+09:00".to_string());
    let update = SyncStateUpdate {
        latest_updated_at: owned.as_deref(),
        total_count: 200,
        local_count: 150,
        last_page: Some(5),
    };

    save_sync_state(db.conn(), &update).unwrap();

    let state = get_sync_state(db.conn()).unwrap().unwrap();
    assert_eq!(
        state.latest_updated_at.as_deref(),
        Some("2025-06-01T12:00:00+09:00"),
        "[T-005] borrowed value should persist correctly"
    );
    assert_eq!(state.total_count, 200);
    assert_eq!(state.local_count, 150);
    assert_eq!(state.last_page, Some(5));
}
