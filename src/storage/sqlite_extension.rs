//! The only SQLite FFI boundary. Registration precedes opening any connection.
#![allow(unsafe_code)]

use super::StorageError;
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use sqlite_vec::sqlite3_vec_init;
use std::{ffi::c_char, mem::transmute, sync::OnceLock};

pub(super) fn register() -> Result<(), StorageError> {
    static REGISTERED: OnceLock<i32> = OnceLock::new();
    let result = REGISTERED.get_or_init(|| {
        // sqlite-vec exports the SQLite extension ABI. auto_extension installs it
        // for subsequent connections; OnceLock prevents concurrent registration.
        unsafe {
            sqlite3_auto_extension(Some(transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut sqlite3,
                    *mut *mut c_char,
                    *const sqlite3_api_routines,
                ) -> i32,
            >(sqlite3_vec_init as *const ())))
        }
    });
    if *result == 0 {
        Ok(())
    } else {
        Err(StorageError::Open(format!(
            "sqlite-vec registration failed ({result})"
        )))
    }
}
