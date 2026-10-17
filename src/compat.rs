pub(crate) mod filter;
pub(crate) mod fts;
pub(crate) mod query;
mod query_helpers;
mod shorthand;
mod spinner;
pub(crate) mod text;

pub(crate) use filter::{anon_placeholders, as_sql_params, in_placeholders};
pub(crate) use query_helpers::{collect_rows, fetch_by_in_clause};
pub use shorthand::try_expand_shorthand;
pub(crate) use spinner::embed_with_spinners;

use std::{env, io, process::ExitCode};

pub trait CliError {
    fn exit_code(&self) -> ExitCode;
}

pub mod codes {
    pub const SUCCESS: u8 = 0;
    pub const USAGE: u8 = 64;
    pub const DATA_ERROR: u8 = 65;
    pub const SOFTWARE: u8 = 70;
    pub const CANT_CREAT: u8 = 73;
    pub const IO_ERR: u8 = 74;
    pub const TEMP_FAIL: u8 = 75;
    pub const UNKNOWN: u8 = 104;
    pub const INTERNAL: u8 = SOFTWARE;
}

pub(crate) fn env_lookup() -> impl Fn(&str) -> Option<String> {
    |key| env::var(key).ok()
}

pub fn exit_error(message: &str) {
    eprintln!("error: {message}");
}

pub fn hint_arrow<S: AsRef<str>>(items: &[S]) {
    eprintln!(
        "→ {}",
        items
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<_>>()
            .join(" ")
    );
}

pub(crate) fn progress_step<S: AsRef<str>>(items: &[S]) {
    eprintln!(
        "  {}",
        items
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<_>>()
            .join(" — ")
    );
}

pub fn init_subscriber(default_filter: &str) {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .init();
}
