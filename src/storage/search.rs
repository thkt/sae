use std::collections::HashMap;

use rusqlite::Connection;
use tracing::warn;

use super::StorageError;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub post_number: u32,
    pub post_name: String,
    pub post_url: String,
    pub section_title: Option<String>,
    pub snippet: String,
    pub score: f64,
}

/// FTS5 trigram search with fts5vocab short-term expansion.
pub fn fts_search(conn: &Connection, query: &str, limit: u32) -> Result<Vec<FtsHit>, StorageError> {
    let normalized = normalize_punctuation(query);
    let matched = match rurico::storage::prepare_match_query(conn, &normalized) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!(%e, query, "query produced no searchable terms");
            return Ok(Vec::new());
        }
    };

    let match_query = clean_for_trigram(matched.as_str());

    let mut stmt = conn.prepare(
        "SELECT c.id, c.post_number, c.section_title, c.content, f.rank \
         FROM fts_chunks f \
         JOIN chunks c ON c.id = f.rowid \
         WHERE fts_chunks MATCH ?1 \
         ORDER BY f.rank \
         LIMIT ?2",
    )?;

    let rows: Vec<FtsHit> = stmt
        .query_map(rusqlite::params![&match_query, limit], |row| {
            Ok(FtsHit {
                chunk_id: row.get(0)?,
                post_number: row.get(1)?,
                section_title: row.get(2)?,
                content: row.get(3)?,
                rank: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(rows)
}

pub fn vec_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: u32,
) -> Result<Vec<VecHit>, StorageError> {
    let bytes: &[u8] = rurico::storage::f32_as_bytes(query_embedding);

    let mut stmt = conn.prepare(
        "SELECT v.chunk_id, v.distance, c.post_number, c.section_title, c.content \
         FROM vec_chunks v \
         JOIN chunks c ON c.id = v.chunk_id \
         WHERE v.embedding MATCH ?1 AND k = ?2 \
         ORDER BY v.distance",
    )?;

    let rows: Vec<VecHit> = stmt
        .query_map(rusqlite::params![bytes, limit], |row| {
            Ok(VecHit {
                chunk_id: row.get(0)?,
                distance: row.get(1)?,
                post_number: row.get(2)?,
                section_title: row.get(3)?,
                content: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(rows)
}

const RECENCY_HALF_LIFE: f64 = 30.0;
const RECENCY_WEIGHT: f64 = 0.2;
const SECS_PER_DAY: f64 = 86_400.0;

/// Hybrid search: merge FTS5 + vector results via Reciprocal Rank Fusion,
/// then apply recency decay boost based on `updated_at`.
pub fn hybrid_search(
    conn: &Connection,
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: u32,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Vec<SearchResult>, StorageError> {
    let candidate_limit = limit * 3;

    let fts_hits = fts_search(conn, query, candidate_limit)?;

    let vec_hits = match query_embedding {
        Some(emb) if super::has_embeddings(conn) => match vec_search(conn, emb, candidate_limit) {
            Ok(hits) => hits,
            Err(e) => {
                warn!(%e, "vec_search failed, falling back to FTS only");
                Vec::new()
            }
        },
        _ => Vec::new(),
    };

    // RRF scores from rank position only; the f64 value is ignored by rurico::rrf_merge
    let fts_ranked: Vec<(u32, f64)> = fts_hits.iter().map(|h| (h.post_number, 0.0)).collect();
    let vec_ranked: Vec<(u32, f64)> = vec_hits.iter().map(|h| (h.post_number, 0.0)).collect();
    let merged = rurico::storage::rrf_merge(&fts_ranked, &vec_ranked);

    // Fetch metadata for all candidates (not just limit) to apply decay before truncation
    let candidate_numbers: Vec<u32> = merged.iter().map(|(pn, _)| *pn).collect();
    let post_meta = batch_fetch_post_meta(conn, &candidate_numbers)?;

    let fts_map: HashMap<u32, &FtsHit> = fts_hits.iter().map(|h| (h.post_number, h)).collect();
    let vec_map: HashMap<u32, &VecHit> = vec_hits.iter().map(|h| (h.post_number, h)).collect();

    // Apply recency decay to all candidates, then re-sort
    let mut scored: Vec<(u32, f64)> = merged
        .into_iter()
        .map(|(post_number, rrf_score)| {
            let decay = post_meta
                .get(&post_number)
                .and_then(|(_, _, updated_at)| {
                    chrono::DateTime::parse_from_rfc3339(updated_at)
                        .map_err(|e| warn!(%e, %updated_at, "unparseable updated_at, decay=0.0"))
                        .ok()
                })
                .map(|updated| {
                    let age_days = (now - updated.with_timezone(&chrono::Utc)).num_seconds() as f64
                        / SECS_PER_DAY;
                    rurico::storage::recency_decay(age_days, RECENCY_HALF_LIFE)
                })
                .unwrap_or(0.0);
            let boosted = rrf_score * (1.0 + RECENCY_WEIGHT * decay);
            (post_number, boosted)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut results = Vec::new();
    for (post_number, score) in scored.into_iter().take(limit as usize) {
        let (name, url, _) = post_meta
            .get(&post_number)
            .cloned()
            .unwrap_or_else(|| (format!("#{post_number}"), String::new(), String::new()));

        let (section_title, snippet) = fts_map
            .get(&post_number)
            .map(|h| (h.section_title.clone(), h.content.clone()))
            .or_else(|| {
                vec_map
                    .get(&post_number)
                    .map(|h| (h.section_title.clone(), h.content.clone()))
            })
            .unwrap_or_default();

        results.push(SearchResult {
            post_number,
            post_name: name,
            post_url: url,
            section_title,
            snippet: truncate_snippet(&snippet, 200),
            score,
        });
    }

    Ok(results)
}

fn batch_fetch_post_meta(
    conn: &Connection,
    post_numbers: &[u32],
) -> Result<HashMap<u32, (String, String, String)>, StorageError> {
    if post_numbers.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders: Vec<String> = (1..=post_numbers.len()).map(|i| format!("?{i}")).collect();
    let sql = format!(
        "SELECT number, name, url, updated_at FROM posts WHERE number IN ({})",
        placeholders.join(", ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = post_numbers
        .iter()
        .map(|n| n as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt
        .query_map(params.as_slice(), |row| {
            Ok((
                row.get::<_, u32>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows
        .into_iter()
        .map(|(n, name, url, updated_at)| (n, (name, url, updated_at)))
        .collect())
}

#[derive(Debug, Clone)]
pub struct FtsHit {
    pub chunk_id: i64,
    pub post_number: u32,
    pub section_title: Option<String>,
    pub content: String,
    pub rank: f64,
}

#[derive(Debug, Clone)]
pub struct VecHit {
    pub chunk_id: i64,
    pub distance: f32,
    pub post_number: u32,
    pub section_title: Option<String>,
    pub content: String,
}

/// Strip general punctuation while preserving technical term characters.
///
/// Characters like `+`, `#`, `.`, `/`, `_`, `@` appear in technical terms
/// (`C++`, `C#`, `.NET`, `/api/v1`) and must survive so that rurico's
/// `prepare_match_query` can quote or expand them correctly.
///
/// `:` and `-` are intentionally stripped because rurico's
/// `sanitize_fts_query` quotes terms containing them, and
/// `fts_expand_short_terms` then re-quotes, producing double-quoted
/// FTS5 queries that match nothing. Until rurico fixes this, terms like
/// `std::io` and `rate-limit` are split into separate words (`std io`,
/// `rate limit`) for broader-but-working recall.
fn normalize_punctuation(query: &str) -> String {
    let result: String = query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || "+#./_@".contains(c) {
                c
            } else {
                ' '
            }
        })
        .collect();
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Adapt rurico's MatchFtsQuery output for FTS5 trigram tokenizer.
///
/// rurico targets standard FTS5 (Unicode61). The trigram tokenizer does
/// not support parenthesized OR groups combined with other terms via
/// implicit AND (e.g. `("a" OR "b") "c"` fails). This function:
///
/// 1. Strips control characters from vocab-expanded terms (trigram vocab
///    can include newlines, e.g. "認証\n").
/// 2. Drops sub-trigram terms (<3 chars) created by step 1.
/// 3. Distributes OR groups across fixed terms to eliminate parentheses:
///    `(A OR B) C` → `A C OR B C`.
///    When the query is a single group with no other terms, the parens
///    are simply removed (single-group queries work in trigram FTS5).
fn cross_product(groups: &[Vec<String>]) -> Vec<Vec<String>> {
    if groups.is_empty() {
        return vec![vec![]];
    }
    let rest = cross_product(&groups[1..]);
    let mut result = Vec::new();
    for term in &groups[0] {
        for combo in &rest {
            let mut v = vec![term.clone()];
            v.extend(combo.iter().cloned());
            result.push(v);
        }
    }
    result
}

fn clean_for_trigram(query: &str) -> String {
    let cleaned: String = query.chars().filter(|c| !c.is_control()).collect();

    // Parse into segments: quoted terms and OR groups
    let mut fixed: Vec<String> = Vec::new();
    let mut or_groups: Vec<Vec<String>> = Vec::new();
    let mut chars = cleaned.chars();

    while let Some(c) = chars.next() {
        if c == '(' {
            let mut group = String::new();
            for gc in chars.by_ref() {
                if gc == ')' {
                    break;
                }
                group.push(gc);
            }
            let terms: Vec<String> = group
                .split(" OR ")
                .filter(|t| t.trim().trim_matches('"').chars().count() >= 3)
                .map(|t| t.trim().to_string())
                .collect();
            if !terms.is_empty() {
                or_groups.push(terms);
            }
        } else if c == '"' {
            let mut term = String::from('"');
            for tc in chars.by_ref() {
                term.push(tc);
                if tc == '"' {
                    break;
                }
            }
            fixed.push(term);
        }
        // skip whitespace between segments
    }

    // No OR groups → return fixed terms joined
    if or_groups.is_empty() {
        return fixed.join(" ");
    }

    // Cross-product of all OR groups, then append fixed terms to each combo.
    // E.g. (A1 OR A2) (B1 OR B2) C → A1 B1 C OR A1 B2 C OR A2 B1 C OR A2 B2 C
    let combos = cross_product(&or_groups);
    let alternatives: Vec<String> = combos
        .iter()
        .map(|combo| {
            let mut parts = combo.clone();
            parts.extend(fixed.iter().cloned());
            parts.join(" ")
        })
        .collect();
    alternatives.join(" OR ")
}

fn truncate_snippet(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        None => s.to_string(),
        Some((byte_pos, _)) => format!("{}...", &s[..byte_pos]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{self, Db};

    fn test_post(number: u32, name: &str, body_md: &str) -> storage::EsaPostRow {
        storage::EsaPostRow {
            number,
            name: name.to_string(),
            full_name: format!("dev/{name}"),
            body_md: body_md.to_string(),
            category: Some("dev".into()),
            tags: "[]".into(),
            wip: false,
            kind: "stock".into(),
            url: format!("https://example.esa.io/posts/{number}"),
            created_at: "2025-01-01T00:00:00+09:00".into(),
            updated_at: "2025-01-01T00:00:00+09:00".into(),
            created_by: "alice".into(),
            updated_by: "alice".into(),
            revision_number: 1,
        }
    }

    fn setup_db_with_posts(db: &Db) {
        let posts = [
            (
                1,
                "認証ガイド",
                "# 認証フロー\n認証の仕組みを説明します\n# 実装手順\nコードの書き方",
            ),
            (
                2,
                "API設計",
                "# エンドポイント\nREST APIの設計方針\n# 認証\nトークン認証の詳細",
            ),
            (3, "デプロイ", "# 手順\nデプロイの手順ガイド"),
        ];
        for (num, name, body) in &posts {
            storage::upsert_post(
                db.conn(),
                &storage::EsaPostRow {
                    number: *num,
                    name: name.to_string(),
                    full_name: format!("dev/{name}"),
                    body_md: body.to_string(),
                    category: Some("dev".into()),
                    tags: "[]".into(),
                    wip: false,
                    kind: "stock".into(),
                    url: format!("https://example.esa.io/posts/{num}"),
                    created_at: "2025-01-01T00:00:00+09:00".into(),
                    updated_at: "2025-01-01T00:00:00+09:00".into(),
                    created_by: "alice".into(),
                    updated_by: "alice".into(),
                    revision_number: 1,
                },
            )
            .unwrap();
            storage::rechunk_post(db.conn(), *num, body).unwrap();
        }
    }

    #[test]
    fn fts_trigram_match_3chars() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "ガイド", 10).unwrap();
        assert!(!hits.is_empty());
        let post_nums: Vec<u32> = hits.iter().map(|h| h.post_number).collect();
        assert!(post_nums.contains(&3));
    }

    #[test]
    fn fts_vocab_expansion_short_term() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "認証", 10).unwrap();
        assert!(!hits.is_empty());
        let post_nums: Vec<u32> = hits.iter().map(|h| h.post_number).collect();
        assert!(post_nums.contains(&1) || post_nums.contains(&2));
    }

    #[test]
    fn fts_single_char_expansion() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "設", 10).unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn rrf_merge_combines_sources() {
        let fts: Vec<(u32, f64)> = vec![(1, 0.0), (2, 0.0)];
        let vec: Vec<(u32, f64)> = vec![(2, 0.0), (3, 0.0)];
        let merged = rurico::storage::rrf_merge(&fts, &vec);
        assert_eq!(merged[0].0, 2);
        assert!(merged[0].1 > merged[1].1);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn hybrid_search_fts_only() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let results =
            hybrid_search(db.conn(), "認証の仕組み", None, 10, chrono::Utc::now()).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].post_number, 1);
        assert!(!results[0].post_name.is_empty());
        assert!(!results[0].post_url.is_empty());
    }

    #[test]
    fn empty_query_returns_empty() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let results = hybrid_search(db.conn(), "", None, 10, chrono::Utc::now()).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn normalize_punctuation_strips_general_keeps_technical() {
        // General punctuation → space
        assert_eq!(
            normalize_punctuation("hello; DROP TABLE"),
            "hello DROP TABLE"
        );
        assert_eq!(normalize_punctuation("認証、フロー"), "認証 フロー");
        assert_eq!(normalize_punctuation("\"injected\""), "injected");
        assert_eq!(normalize_punctuation("  "), "");

        // Technical characters preserved
        assert_eq!(normalize_punctuation("C++"), "C++");
        assert_eq!(normalize_punctuation("C#"), "C#");
        assert_eq!(normalize_punctuation("/api/v1"), "/api/v1");
        assert_eq!(normalize_punctuation(".NET"), ".NET");
        assert_eq!(normalize_punctuation("user@example"), "user@example");

        // : and - are stripped (rurico double-quoting bug workaround)
        assert_eq!(normalize_punctuation("std::io"), "std io");
        assert_eq!(normalize_punctuation("rate-limit"), "rate limit");
    }

    #[test]
    fn clean_for_trigram_adapts_query() {
        // Control chars removed + sub-trigram dropped + distributed
        assert_eq!(
            clean_for_trigram("(\"認証の\" OR \"認証\n\" OR \"認証フ\") \"フロー\""),
            "\"認証の\" \"フロー\" OR \"認証フ\" \"フロー\""
        );
        // Single-element group + fixed term → distributed
        assert_eq!(clean_for_trigram("\"std\" (\"ioの\")"), "\"ioの\" \"std\"");
        // Multi-element group + fixed term → distributed
        assert_eq!(
            clean_for_trigram("(\"abc\" OR \"def\") \"ghi\""),
            "\"abc\" \"ghi\" OR \"def\" \"ghi\""
        );
        // No parens → unchanged
        assert_eq!(clean_for_trigram("\"hello\""), "\"hello\"");
        // Single group, no fixed terms → just OR
        assert_eq!(
            clean_for_trigram("(\"abc\" OR \"def\")"),
            "\"abc\" OR \"def\""
        );
        // Multiple OR groups → cross-product
        assert_eq!(
            clean_for_trigram("(\"a01\" OR \"a02\") (\"b01\" OR \"b02\")"),
            "\"a01\" \"b01\" OR \"a01\" \"b02\" OR \"a02\" \"b01\" OR \"a02\" \"b02\""
        );
        // Multiple OR groups + fixed term
        assert_eq!(
            clean_for_trigram("(\"a01\" OR \"a02\") \"xyz\" (\"b01\" OR \"b02\")"),
            "\"a01\" \"b01\" \"xyz\" OR \"a01\" \"b02\" \"xyz\" OR \"a02\" \"b01\" \"xyz\" OR \"a02\" \"b02\" \"xyz\""
        );
    }

    #[test]
    fn fts_search_with_punctuation_does_not_error() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "認証、フロー", 10).unwrap();
        let _ = hits;
    }

    #[test]
    fn fts_search_technical_terms_e2e() {
        let db = Db::open_memory().unwrap();
        let posts = [
            (
                10,
                "C++ guide",
                "# C++入門\nC++のテンプレートとstd::ioの使い方",
            ),
            (
                11,
                "rate-limit設計",
                "# rate-limit\nAPIのrate-limitを実装する手順",
            ),
        ];
        for (num, name, body) in &posts {
            storage::upsert_post(
                db.conn(),
                &storage::EsaPostRow {
                    number: *num,
                    name: name.to_string(),
                    full_name: format!("dev/{name}"),
                    body_md: body.to_string(),
                    category: Some("dev".into()),
                    tags: "[]".into(),
                    wip: false,
                    kind: "stock".into(),
                    url: format!("https://example.esa.io/posts/{num}"),
                    created_at: "2025-01-01T00:00:00+09:00".into(),
                    updated_at: "2025-01-01T00:00:00+09:00".into(),
                    created_by: "alice".into(),
                    updated_by: "alice".into(),
                    revision_number: 1,
                },
            )
            .unwrap();
            storage::rechunk_post(db.conn(), *num, body).unwrap();
        }

        // C++ should match post 10 (+ is preserved, no rurico quoting issue)
        let hits = fts_search(db.conn(), "C++", 10).unwrap();
        assert!(!hits.is_empty(), "C++ should match");
        assert!(hits.iter().any(|h| h.post_number == 10));

        // rate-limit → "rate limit" (- stripped). Both ≥3 chars, no vocab
        // expansion needed, so no single-element parenthesization issue.
        let hits = fts_search(db.conn(), "rate-limit", 10).unwrap();
        assert!(!hits.is_empty(), "rate-limit (split) should match");
        assert!(hits.iter().any(|h| h.post_number == 11));

        // std::io → "std io" (: stripped). "io" (2 chars) expands via vocab.
        // clean_for_trigram unwraps single-element parens for trigram compat.
        let hits = fts_search(db.conn(), "std::io", 10).unwrap();
        assert!(!hits.is_empty(), "std::io (split) should match");
        assert!(hits.iter().any(|h| h.post_number == 10));
    }

    #[test]
    fn batch_fetch_post_meta_works() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let meta = batch_fetch_post_meta(db.conn(), &[1, 3]).unwrap();
        assert_eq!(meta.len(), 2);
        assert!(meta.get(&1).unwrap().0.contains("認証")); // name
        assert!(meta.get(&3).unwrap().0.contains("デプロイ")); // name
        assert!(!meta.get(&1).unwrap().2.is_empty()); // updated_at
    }

    #[test]
    fn batch_fetch_empty() {
        let db = Db::open_memory().unwrap();
        let meta = batch_fetch_post_meta(db.conn(), &[]).unwrap();
        assert!(meta.is_empty());
    }

    // T-024: recent post scores higher than old post
    #[test]
    fn hybrid_search_recency_boost_recent_post_scores_higher() {
        let db = Db::open_memory().unwrap();

        // Post A: updated 1 day ago, Post B: updated 30 days ago.
        // Both match the same query so raw RRF scores are equal.
        let shared_body = "# 認証フロー\n認証の仕組みを説明します";
        let posts = [
            (1u32, "PostA", "2025-01-31T00:00:00+09:00"), // 1 day before "now"
            (2u32, "PostB", "2025-01-02T00:00:00+09:00"), // 30 days before "now"
        ];
        for (num, name, updated) in &posts {
            storage::upsert_post(
                db.conn(),
                &storage::EsaPostRow {
                    number: *num,
                    name: name.to_string(),
                    full_name: format!("dev/{name}"),
                    body_md: shared_body.to_string(),
                    category: Some("dev".into()),
                    tags: "[]".into(),
                    wip: false,
                    kind: "stock".into(),
                    url: format!("https://example.esa.io/posts/{num}"),
                    created_at: "2025-01-01T00:00:00+09:00".into(),
                    updated_at: updated.to_string(),
                    created_by: "alice".into(),
                    updated_by: "alice".into(),
                    revision_number: 1,
                },
            )
            .unwrap();
            storage::rechunk_post(db.conn(), *num, shared_body).unwrap();
        }

        let now = chrono::DateTime::parse_from_rfc3339("2025-02-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let results = hybrid_search(db.conn(), "認証の仕組み", None, 10, now).unwrap();
        assert!(results.len() >= 2, "both posts should match");

        let score_a = results.iter().find(|r| r.post_number == 1).unwrap().score;
        let score_b = results.iter().find(|r| r.post_number == 2).unwrap().score;
        assert!(
            score_a > score_b,
            "post A (1 day old) should score higher than post B (30 days old), \
             got A={score_a} B={score_b}"
        );
    }

    // T-025: unparseable updated_at does not panic, decay=0.0 applied
    #[test]
    fn hybrid_search_unparseable_updated_at_no_panic() {
        let db = Db::open_memory().unwrap();

        let body = "# 認証フロー\n認証の仕組みを説明します";
        storage::upsert_post(
            db.conn(),
            &storage::EsaPostRow {
                number: 1,
                name: "BadDate".to_string(),
                full_name: "dev/BadDate".to_string(),
                body_md: body.to_string(),
                category: Some("dev".into()),
                tags: "[]".into(),
                wip: false,
                kind: "stock".into(),
                url: "https://example.esa.io/posts/1".into(),
                created_at: "2025-01-01T00:00:00+09:00".into(),
                updated_at: "not-a-date".into(),
                created_by: "alice".into(),
                updated_by: "alice".into(),
                revision_number: 1,
            },
        )
        .unwrap();
        storage::rechunk_post(db.conn(), 1, body).unwrap();

        let now = chrono::DateTime::parse_from_rfc3339("2025-02-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let results = hybrid_search(db.conn(), "認証の仕組み", None, 10, now).unwrap();

        // Must not panic. With decay=0.0 the boost factor is
        // (1.0 + 0.2 * 0.0) = 1.0, so score equals raw RRF score.
        assert!(!results.is_empty(), "post should still be returned");
        assert!(
            results[0].score > 0.0,
            "score should be positive (raw RRF with 1.0x boost)"
        );
    }

    // T-002: content のみの語で fts_search ヒット — regression (FR-003)
    #[test]
    fn fts_search_matches_content_only_term_regression() {
        let db = Db::open_memory().unwrap();
        let body = "# 認証ガイド\nフローの説明をします";
        let post = test_post(1, "テスト", body);
        storage::upsert_post(db.conn(), &post).unwrap();
        storage::rechunk_post(db.conn(), 1, body).unwrap();

        let hits = fts_search(db.conn(), "フローの説明", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "[T-002] content-only term must still be found (regression)"
        );
        assert_eq!(hits[0].post_number, 1);
    }

    // T-003: heading なし preamble chunk で rechunk + fts_search エラーなし (FR-004)
    #[test]
    fn fts_search_preamble_chunk_no_heading_no_error() {
        let db = Db::open_memory().unwrap();
        let body = "見出しなしの本文テキストです";
        let post = test_post(1, "Preamble", body);
        storage::upsert_post(db.conn(), &post).unwrap();
        let count = storage::rechunk_post(db.conn(), 1, body).unwrap();
        assert_eq!(count, 1, "[T-003] preamble should produce 1 chunk");

        let hits = fts_search(db.conn(), "見出しなし", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "[T-003] preamble chunk should be searchable by content"
        );
        assert!(
            hits[0].section_title.is_none(),
            "[T-003] preamble chunk should have no section_title"
        );
    }

    // T-001: section_title のみの語で fts_search ヒット (FR-003)
    #[test]
    fn fts_search_matches_section_title_only_term() {
        let db = Db::open_memory().unwrap();
        let body = "# 認証ガイド\nフローの説明をします";
        let post = test_post(1, "テスト", body);
        storage::upsert_post(db.conn(), &post).unwrap();
        storage::rechunk_post(db.conn(), 1, body).unwrap();

        // "認証ガイド" は section_title にのみ存在、content には含まれない
        let hits = fts_search(db.conn(), "認証ガイド", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "[T-001] section_title-only term should be found via FTS"
        );
        assert_eq!(hits[0].post_number, 1);
        assert_eq!(hits[0].section_title.as_deref(), Some("認証ガイド"));
    }
}
