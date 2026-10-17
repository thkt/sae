// Pure helper retained from the previous dependency to preserve behavior.
mod query_normalize;
mod search;
pub(crate) use query_normalize::{QueryNormalizationConfig, normalize_for_fts};
pub(crate) use search::{MatchFtsQuery, prepare_match_query};
