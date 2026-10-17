use rusqlite::Connection;

use super::query_normalize::{QueryNormalizationConfig, normalize_for_fts};

fn strip_near_groups(query: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut paren_depth: usize = 0;
    for w in query.split_whitespace() {
        let upper = w.to_ascii_uppercase();
        if upper.starts_with("NEAR(") || upper.starts_with("NEAR/") {
            paren_depth += w.chars().filter(|&c| c == '(').count();
            paren_depth = paren_depth.saturating_sub(w.chars().filter(|&c| c == ')').count());
            continue;
        }
        if paren_depth > 0 {
            paren_depth = paren_depth.saturating_sub(w.chars().filter(|&c| c == ')').count());
            continue;
        }
        tokens.push(w);
    }
    tokens
}

const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT"];

fn is_fts5_operator(token: &str) -> bool {
    FTS5_OPERATORS
        .iter()
        .any(|op| token.eq_ignore_ascii_case(op))
}

fn drop_dangling_operators(tokens: &[String]) -> Vec<&str> {
    tokens
        .iter()
        .enumerate()
        .filter(|&(i, t)| {
            if !is_fts5_operator(t) {
                return true;
            }
            let has_left = i > 0 && !is_fts5_operator(&tokens[i - 1]);
            let has_right = i + 1 < tokens.len() && !is_fts5_operator(&tokens[i + 1]);
            has_left && has_right
        })
        .map(|(_, t)| t.as_str())
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SanitizedFtsQuery(String);

impl SanitizedFtsQuery {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchFtsQuery(String);

impl MatchFtsQuery {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SanitizeError {
    #[error("search query is empty")]
    EmptyInput,
    #[error("search query contains no searchable terms")]
    NoSearchableTerms,
    #[error("invalid vocab table name: {0:?}")]
    InvalidVocabTable(String),
    #[error("fts vocab lookup failed: {0}")]
    VocabLookupFailed(String),
}

pub(crate) fn sanitize_fts_query(query: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
    if query.trim().is_empty() {
        return Err(SanitizeError::EmptyInput);
    }
    let tokens: Vec<String> = strip_near_groups(query)
        .into_iter()
        .map(|w| {
            let stripped = w.trim_start_matches(['^', '+', '-']);
            let cleaned = stripped.trim_matches(['(', ')']);
            if (cleaned.contains(':') || cleaned.contains('-')) && !cleaned.starts_with('"') {
                let unquoted = cleaned.replace('"', "");
                format!("\"{unquoted}\"")
            } else {
                cleaned.to_owned()
            }
        })
        .filter(|w| !w.is_empty())
        .collect();
    let result = drop_dangling_operators(&tokens).join(" ");
    if result.is_empty() {
        return Err(SanitizeError::NoSearchableTerms);
    }
    let quote_count = result.chars().filter(|&c| c == '"').count();
    if quote_count % 2 != 0 {
        Ok(SanitizedFtsQuery(format!("{result}\"")))
    } else {
        Ok(SanitizedFtsQuery(result))
    }
}

pub fn fts_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

pub fn prepare_match_query(
    conn: &Connection,
    query: &str,
    vocab_table: &str,
    normalization: &QueryNormalizationConfig,
) -> Result<MatchFtsQuery, SanitizeError> {
    let normalized = normalize_for_fts(query, normalization);
    let sanitized = sanitize_fts_query(&normalized)?;
    fts_expand_short_terms(conn, &sanitized, vocab_table)
}

fn expand_token_via_vocab(
    stmt: &mut rusqlite::CachedStatement<'_>,
    token: &str,
) -> Result<Option<String>, SanitizeError> {
    let escaped = token
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    let pattern = format!("{escaped}%");
    match stmt
        .query_map([&pattern], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
    {
        Ok(terms) if !terms.is_empty() => {
            let quoted: Vec<String> = terms.iter().map(|t| fts_quote(t)).collect();
            Ok(Some(format!("({})", quoted.join(" OR "))))
        }
        Ok(_) => Ok(None),
        Err(e) if is_missing_table_error(&e) => Ok(None),
        Err(e) => Err(SanitizeError::VocabLookupFailed(e.to_string())),
    }
}

// SQLite has no distinct missing-table code, so inspect its error message.
fn is_missing_table_error(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(_, Some(msg)) if msg.contains("no such table")
    )
}

fn is_valid_sql_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let head_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let tail_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    head_ok && tail_ok
}

pub(crate) fn fts_expand_short_terms(
    conn: &Connection,
    query: &SanitizedFtsQuery,
    vocab_table: &str,
) -> Result<MatchFtsQuery, SanitizeError> {
    if !is_valid_sql_identifier(vocab_table) {
        return Err(SanitizeError::InvalidVocabTable(vocab_table.to_owned()));
    }
    let sql = format!(
        "SELECT term FROM {vocab_table} \
         WHERE term LIKE ?1 ESCAPE '\\' \
         ORDER BY cnt DESC LIMIT 25"
    );
    let mut stmt = match conn.prepare_cached(&sql) {
        Ok(s) => Some(s),
        Err(e) if is_missing_table_error(&e) => None,
        Err(e) => return Err(SanitizeError::VocabLookupFailed(e.to_string())),
    };

    let mut parts = Vec::new();
    for token in query.as_str().split_whitespace() {
        if token.chars().count() >= 3 || is_fts5_operator(token) {
            parts.push(fts_quote(token));
            continue;
        }
        let expanded = match stmt.as_mut() {
            Some(s) => expand_token_via_vocab(s, token)?,
            None => None,
        };
        parts.push(expanded.unwrap_or_else(|| fts_quote(token)));
    }
    Ok(MatchFtsQuery(parts.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::{SanitizeError, fts_expand_short_terms, sanitize_fts_query};
    use rusqlite::Connection;

    #[test]
    fn sanitization_rejects_empty_and_operator_only_queries_and_preserves_literal_punctuation() {
        assert_eq!(sanitize_fts_query(" \n\t"), Err(SanitizeError::EmptyInput));
        for query in ["AND OR NOT", "NEAR(foo bar)", "NEAR/5(foo bar)", "() ^ + -"] {
            assert_eq!(
                sanitize_fts_query(query),
                Err(SanitizeError::NoSearchableTerms),
                "{query}"
            );
        }
        for (query, expected) in [
            ("AND rust OR auth NOT", "rust OR auth"),
            ("^rust -lang +api", "rust lang api"),
            ("title:auth foo-bar", "\"title:auth\" \"foo-bar\""),
            ("\"rust auth", "\"rust auth\""),
            ("alpha NEAR(foo bar) beta", "alpha beta"),
            ("alpha NEAR/5(foo bar) beta", "alpha beta"),
        ] {
            assert_eq!(
                sanitize_fts_query(query).unwrap().as_str(),
                expected,
                "{query}"
            );
        }
    }

    #[test]
    fn short_terms_expand_by_popularity_and_escape_like_wildcards() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE vocab(term TEXT,cnt INTEGER); INSERT INTO vocab VALUES('rust',1),('ruby',5),('r_underscore',2),('ra',9),('percent%',3);").unwrap();
        let query = sanitize_fts_query("ru xyz OR r_ %").unwrap();
        let expanded = fts_expand_short_terms(&conn, &query, "vocab").unwrap();
        assert_eq!(
            expanded.as_str(),
            "(\"ruby\" OR \"rust\") \"xyz\" \"OR\" (\"r_underscore\") \"%\""
        );
        let missing =
            fts_expand_short_terms(&conn, &sanitize_fts_query("zz").unwrap(), "vocab").unwrap();
        assert_eq!(missing.as_str(), "\"zz\"");
    }

    #[test]
    fn invalid_vocab_names_and_schema_errors_are_not_silently_treated_as_missing_vocab() {
        let conn = Connection::open_in_memory().unwrap();
        let query = sanitize_fts_query("ru").unwrap();
        for name in ["", "1vocab", "vocab;DROP TABLE posts", "vocab.name"] {
            assert_eq!(
                fts_expand_short_terms(&conn, &query, name),
                Err(SanitizeError::InvalidVocabTable(name.into()))
            );
        }
        assert_eq!(
            fts_expand_short_terms(&conn, &query, "missing_vocab")
                .unwrap()
                .as_str(),
            "\"ru\""
        );
        conn.execute_batch("CREATE TABLE broken_vocab(unrelated TEXT);")
            .unwrap();
        assert!(matches!(
            fts_expand_short_terms(&conn, &query, "broken_vocab"),
            Err(SanitizeError::VocabLookupFailed(_))
        ));
        conn.execute_batch(
            "CREATE TABLE bad_values(term,cnt INTEGER); INSERT INTO bad_values VALUES(12,1);",
        )
        .unwrap();
        assert!(matches!(
            fts_expand_short_terms(&conn, &sanitize_fts_query("1").unwrap(), "bad_values"),
            Err(SanitizeError::VocabLookupFailed(_))
        ));
    }
}
