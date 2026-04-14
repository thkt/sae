use std::collections::HashMap;

use amici::storage::append_eq_filter;
use rurico::reranker::Rerank;
use rusqlite::Connection;
use tracing::warn;

use super::StorageError;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchSource {
    Semantic,
    Fts,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub post_number: u32,
    pub post_name: String,
    pub post_url: String,
    pub section_title: Option<String>,
    pub snippet: String,
    pub score: f32,
    pub match_source: MatchSource,
}

#[derive(Debug, Clone, Default)]
pub struct SearchFilter<'a> {
    pub tags: Option<&'a [&'a str]>,
    pub category: Option<&'a str>,
    pub created_by: Option<&'a str>,
    pub updated_after: Option<chrono::DateTime<chrono::Utc>>,
    pub updated_before: Option<chrono::DateTime<chrono::Utc>>,
}

pub fn fts_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    filter: &SearchFilter<'_>,
) -> Result<Vec<FtsHit>, StorageError> {
    let normalized = normalize_punctuation(query);
    let matched = match rurico::storage::prepare_match_query(conn, &normalized) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!(%e, query, "query produced no searchable terms");
            return Ok(Vec::new());
        }
    };
    let match_query = clean_for_trigram(matched.as_str());

    let mut sql = String::from(
        "SELECT c.id, c.post_number, c.section_title, c.content, f.rank \
         FROM fts_chunks f \
         JOIN chunks c ON c.id = f.rowid \
         JOIN posts p ON p.number = c.post_number \
         WHERE fts_chunks MATCH ?",
    );
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(match_query)];

    append_tags_filter(&mut sql, &mut params, filter.tags);
    append_eq_filter(&mut sql, &mut params, "p.category", filter.category);
    append_eq_filter(&mut sql, &mut params, "p.created_by", filter.created_by);
    append_date_filter(&mut sql, &mut params, false, filter.updated_after);
    append_date_filter(&mut sql, &mut params, true, filter.updated_before);
    sql.push_str(" ORDER BY f.rank LIMIT ?");
    params.push(Box::new(limit));

    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<FtsHit> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| {
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

const VEC_MAXSIM_OVERSAMPLE: u32 = 10;

pub fn vec_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: u32,
    filter: &SearchFilter<'_>,
) -> Result<Vec<VecHit>, StorageError> {
    let bytes: &[u8] = rurico::storage::f32_as_bytes(query_embedding);
    let oversample = limit.saturating_mul(VEC_MAXSIM_OVERSAMPLE);

    // Step 1: KNN query — fetch only chunk_id + distance to avoid the sqlite-vec
    // restriction that prohibits JOIN conditions on vec0 auxiliary columns (+chunk_id).
    let knn_rows: Vec<(i64, f32)> = {
        let mut stmt = conn.prepare_cached(
            "SELECT chunk_id, distance FROM vec_chunks \
             WHERE embedding MATCH ?1 AND k = ?2 \
             ORDER BY distance",
        )?;
        stmt.query_map(rusqlite::params![bytes, oversample], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?
    };

    if knn_rows.is_empty() {
        return Ok(Vec::new());
    }

    // MaxSim: keep the sub-embedding with the smallest distance per chunk_id
    let mut best: HashMap<i64, f32> = HashMap::new();
    for (chunk_id, distance) in &knn_rows {
        best.entry(*chunk_id)
            .and_modify(|d| {
                if *distance < *d {
                    *d = *distance;
                }
            })
            .or_insert(*distance);
    }

    // Step 2: batch-fetch chunk metadata for the deduplicated chunk_ids
    let chunk_ids: Vec<i64> = best.keys().copied().collect();
    let mut sql = format!(
        "SELECT c.id, c.post_number, c.section_title, c.content \
         FROM chunks c \
         JOIN posts p ON p.number = c.post_number \
         WHERE c.id IN ({})",
        super::anon_placeholders(chunk_ids.len())
    );
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = chunk_ids
        .iter()
        .map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
        .collect();
    append_tags_filter(&mut sql, &mut params, filter.tags);
    append_eq_filter(&mut sql, &mut params, "p.category", filter.category);
    append_eq_filter(&mut sql, &mut params, "p.created_by", filter.created_by);
    append_date_filter(&mut sql, &mut params, false, filter.updated_after);
    append_date_filter(&mut sql, &mut params, true, filter.updated_before);

    let mut stmt2 = conn.prepare(&sql)?;
    let meta: HashMap<i64, (u32, Option<String>, String)> = stmt2
        .query_map(rusqlite::params_from_iter(params.iter()), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                (
                    row.get::<_, u32>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                ),
            ))
        })?
        .collect::<Result<HashMap<_, _>, _>>()?;

    let mut hits: Vec<VecHit> = best
        .into_iter()
        .filter_map(|(chunk_id, distance)| {
            meta.get(&chunk_id)
                .map(|(post_number, section_title, content)| VecHit {
                    chunk_id,
                    distance,
                    post_number: *post_number,
                    section_title: section_title.clone(),
                    content: content.clone(),
                })
        })
        .collect();

    hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    hits.truncate(limit as usize);

    Ok(hits)
}

const RECENCY_HALF_LIFE: f64 = 30.0;
const RECENCY_WEIGHT: f64 = 0.2;
const SECS_PER_DAY: f64 = 86_400.0;

pub fn hybrid_search(
    conn: &Connection,
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: u32,
    now: chrono::DateTime<chrono::Utc>,
    filter: &SearchFilter<'_>,
    reranker: Option<&dyn Rerank>,
) -> Result<Vec<SearchResult>, StorageError> {
    let candidate_limit = if reranker.is_some() {
        limit * 4
    } else {
        limit * 3
    };

    let fts_hits = fts_search(conn, query, candidate_limit, filter)?;

    let vec_hits = match query_embedding {
        Some(emb) if super::has_embeddings(conn) => {
            match vec_search(conn, emb, candidate_limit, filter) {
                Ok(hits) => hits,
                Err(e) => {
                    eprintln!("  warning: vector search failed, falling back to text search only");
                    warn!(%e, %query, candidate_limit, "vec_search failed, falling back to FTS only");
                    Vec::new()
                }
            }
        }
        _ => Vec::new(),
    };

    let fts_rrf_input: Vec<(u32, f64)> = fts_hits.iter().map(|h| (h.post_number, 0.0)).collect();
    let vec_rrf_input: Vec<(u32, f64)> = vec_hits.iter().map(|h| (h.post_number, 0.0)).collect();
    let merged = rurico::storage::rrf_merge(&fts_rrf_input, &vec_rrf_input);

    // Fetch metadata for all candidates (not just limit) to apply decay before truncation
    let candidate_numbers: Vec<u32> = merged.iter().map(|(pn, _)| *pn).collect();
    let post_meta = batch_fetch_post_meta(conn, &candidate_numbers)?;

    // P1: keep the first (highest-ranked) chunk per post so the reranker sees
    // the most relevant content, not whichever chunk happened to be last.
    let mut fts_map: HashMap<u32, &FtsHit> = HashMap::new();
    for h in &fts_hits {
        fts_map.entry(h.post_number).or_insert(h);
    }
    let mut vec_map: HashMap<u32, &VecHit> = HashMap::new();
    for h in &vec_hits {
        vec_map.entry(h.post_number).or_insert(h);
    }

    // P2: apply recency boost in every path so it is never discarded.
    // When the cross-encoder is active it re-scores first, then recency
    // is added on top; the fallback and no-reranker paths behave as before.
    let scored = if let Some(ranker) = reranker {
        let pairs: Vec<(&str, &str)> = merged
            .iter()
            .map(|(pn, _)| {
                let content = fts_map
                    .get(pn)
                    .map(|h| h.content.as_str())
                    .or_else(|| vec_map.get(pn).map(|h| h.content.as_str()))
                    .unwrap_or("");
                (query, content)
            })
            .collect();
        match ranker.score_batch(&pairs) {
            Ok(scores) => {
                let cross_scored: Vec<(u32, f64)> = merged
                    .into_iter()
                    .zip(scores)
                    .map(|((pn, _), s)| (pn, s as f64))
                    .collect();
                apply_recency_boost(cross_scored, &post_meta, now)
            }
            Err(e) => {
                warn!(%e, "cross-encoder reranking failed, keeping heuristic order");
                apply_recency_boost(merged, &post_meta, now)
            }
        }
    } else {
        apply_recency_boost(merged, &post_meta, now)
    };

    let results = scored
        .into_iter()
        .take(limit as usize)
        .map(|(post_number, score)| {
            let meta = post_meta
                .get(&post_number)
                .cloned()
                .unwrap_or_else(|| PostMeta {
                    name: format!("#{post_number}"),
                    url: String::new(),
                    updated_at: String::new(),
                });
            let vec_hit = vec_map.get(&post_number).copied();
            let (section_title, snippet) = fts_map
                .get(&post_number)
                .map(|h| (h.section_title.clone(), h.content.clone()))
                .or_else(|| vec_hit.map(|h| (h.section_title.clone(), h.content.clone())))
                .unwrap_or_default();
            SearchResult {
                post_number,
                post_name: meta.name,
                post_url: meta.url,
                section_title,
                snippet: truncate_snippet(&snippet, 200),
                score: score as f32,
                // Semantic takes priority when a post matched both sources.
                match_source: if vec_hit.is_some() {
                    MatchSource::Semantic
                } else {
                    MatchSource::Fts
                },
            }
        })
        .collect();
    Ok(results)
}

fn append_tags_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn rusqlite::types::ToSql>>,
    tags: Option<&[&str]>,
) {
    if let Some(tags) = tags
        && !tags.is_empty()
    {
        sql.push_str(&format!(
            " AND EXISTS (SELECT 1 FROM json_each(p.tags) jt WHERE jt.value IN ({}))",
            super::anon_placeholders(tags.len())
        ));
        for &tag in tags {
            params.push(Box::new(tag.to_string()));
        }
    }
}

fn append_date_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn rusqlite::types::ToSql>>,
    before: bool,
    value: Option<chrono::DateTime<chrono::Utc>>,
) {
    let Some(dt) = value else { return };
    // For `before`, advance one day and use `<` so the boundary day is fully
    // included ("2025-01-15T23:59:59+09:00" < "2025-01-16" is TRUE).
    let (op, date_str) = if before {
        let next = dt
            .date_naive()
            .succ_opt()
            .unwrap_or(dt.date_naive())
            .format("%Y-%m-%d")
            .to_string();
        ("<", next)
    } else {
        (">=", dt.format("%Y-%m-%d").to_string())
    };
    sql.push_str(&format!(" AND p.updated_at {op} ?"));
    params.push(Box::new(date_str));
}

fn apply_recency_boost(
    merged: Vec<(u32, f64)>,
    post_meta: &HashMap<u32, PostMeta>,
    now: chrono::DateTime<chrono::Utc>,
) -> Vec<(u32, f64)> {
    let mut scored: Vec<(u32, f64)> = merged
        .into_iter()
        .map(|(post_number, rrf_score)| {
            let decay = post_meta
                .get(&post_number)
                .and_then(|meta| {
                    let updated_at = &meta.updated_at;
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
    scored
}

#[derive(Debug, Clone)]
struct PostMeta {
    name: String,
    url: String,
    updated_at: String,
}

fn batch_fetch_post_meta(
    conn: &Connection,
    post_numbers: &[u32],
) -> Result<HashMap<u32, PostMeta>, StorageError> {
    if post_numbers.is_empty() {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT number, name, url, updated_at FROM posts WHERE number IN ({})",
        super::in_placeholders(post_numbers.len())
    );
    let mut stmt = conn.prepare(&sql)?;
    let params = super::as_sql_params(post_numbers);
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
        .map(|(n, name, url, updated_at)| {
            (
                n,
                PostMeta {
                    name,
                    url,
                    updated_at,
                },
            )
        })
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

/// Strip general punctuation while preserving technical term characters
/// (`+`, `#`, `.`, `/`, `_`, `@`).
///
/// `:` and `-` are stripped to work around rurico double-quoting in
/// `sanitize_fts_query` + `fts_expand_short_terms` (produces FTS5
/// queries that match nothing).
fn normalize_punctuation(query: &str) -> String {
    query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || "+#./_@".contains(c) {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

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

fn parse_fts_segments(cleaned: &str) -> (Vec<String>, Vec<Vec<String>>) {
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
    }

    (fixed, or_groups)
}

/// Adapt rurico's MatchFtsQuery output for FTS5 trigram tokenizer.
///
/// Trigram FTS5 does not support `("a" OR "b") "c"` (parenthesized
/// OR + implicit AND). This distributes OR groups into flat alternatives:
/// `(A OR B) C` → `A C OR B C`. Also strips control chars and drops
/// sub-trigram terms (<3 chars).
fn clean_for_trigram(query: &str) -> String {
    let cleaned: String = query.chars().filter(|c| !c.is_control()).collect();
    let (fixed, or_groups) = parse_fts_segments(&cleaned);

    if or_groups.is_empty() {
        return fixed.join(" ");
    }

    // (A1 OR A2) (B1 OR B2) C → A1 B1 C OR A1 B2 C OR A2 B1 C OR A2 B2 C
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
        let mut row = storage::test_post_row(number);
        row.name = name.to_string();
        row.full_name = format!("dev/{name}");
        row.body_md = body_md.to_string();
        row
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
            let post = test_post(*num, name, body);
            storage::upsert_post(db.conn(), &post).unwrap();
            storage::rechunk_post(db.conn(), *num, body).unwrap();
        }
    }

    // T-228: fts_search matches a 3-character trigram term
    #[test]
    fn fts_trigram_match_3chars() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "ガイド", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty());
        let post_nums: Vec<u32> = hits.iter().map(|h| h.post_number).collect();
        assert!(post_nums.contains(&3));
    }

    // T-229: fts_search expands a short (< 3 char) query via vocab table
    #[test]
    fn fts_vocab_expansion_short_term() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "認証", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty());
        let post_nums: Vec<u32> = hits.iter().map(|h| h.post_number).collect();
        assert!(post_nums.contains(&1) || post_nums.contains(&2));
    }

    // T-230: fts_search expands a single-character query via vocab table
    #[test]
    fn fts_single_char_expansion() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "設", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty());
    }

    // T-231: rrf_merge ranks the result appearing in both sources first
    #[test]
    fn rrf_merge_combines_sources() {
        let fts: Vec<(u32, f64)> = vec![(1, 0.0), (2, 0.0)];
        let vec: Vec<(u32, f64)> = vec![(2, 0.0), (3, 0.0)];
        let merged = rurico::storage::rrf_merge(&fts, &vec);
        assert_eq!(merged[0].0, 2);
        assert!(merged[0].1 > merged[1].1);
        assert_eq!(merged.len(), 3);
    }

    // T-232: hybrid_search returns results via FTS when no embeddings are present
    #[test]
    fn hybrid_search_fts_only() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let results = hybrid_search(
            db.conn(),
            "認証の仕組み",
            None,
            10,
            chrono::Utc::now(),
            &SearchFilter::default(),
            None,
        )
        .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].post_number, 1);
        assert!(!results[0].post_name.is_empty());
        assert!(!results[0].post_url.is_empty());
    }

    // T-233: hybrid_search returns empty results for an empty query string
    #[test]
    fn empty_query_returns_empty() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let results = hybrid_search(
            db.conn(),
            "",
            None,
            10,
            chrono::Utc::now(),
            &SearchFilter::default(),
            None,
        )
        .unwrap();
        assert!(results.is_empty());
    }

    // T-234: normalize_punctuation strips general punct but preserves technical symbols
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

    // T-235: clean_for_trigram distributes OR groups and removes sub-trigram terms
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

    // T-236: fts_search does not error when query contains punctuation characters
    #[test]
    fn fts_search_with_punctuation_does_not_error() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        fts_search(db.conn(), "認証、フロー", 10, &SearchFilter::default()).unwrap();
    }

    // T-237: fts_search finds C++, rate-limit, and std::io after normalization
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
            let post = test_post(*num, name, body);
            storage::upsert_post(db.conn(), &post).unwrap();
            storage::rechunk_post(db.conn(), *num, body).unwrap();
        }

        // C++ should match post 10 (+ is preserved, no rurico quoting issue)
        let hits = fts_search(db.conn(), "C++", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty(), "C++ should match");
        assert!(hits.iter().any(|h| h.post_number == 10));

        // rate-limit → "rate limit" (- stripped). Both ≥3 chars, no vocab
        // expansion needed, so no single-element parenthesization issue.
        let hits = fts_search(db.conn(), "rate-limit", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty(), "rate-limit (split) should match");
        assert!(hits.iter().any(|h| h.post_number == 11));

        // std::io → "std io" (: stripped). "io" (2 chars) expands via vocab.
        // clean_for_trigram unwraps single-element parens for trigram compat.
        let hits = fts_search(db.conn(), "std::io", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty(), "std::io (split) should match");
        assert!(hits.iter().any(|h| h.post_number == 10));
    }

    // T-238: batch_fetch_post_meta returns metadata for each requested post number
    #[test]
    fn batch_fetch_post_meta_works() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let meta = batch_fetch_post_meta(db.conn(), &[1, 3]).unwrap();
        assert_eq!(meta.len(), 2);
        assert!(meta.get(&1).unwrap().name.contains("認証"));
        assert!(meta.get(&3).unwrap().name.contains("デプロイ"));
        assert!(!meta.get(&1).unwrap().updated_at.is_empty());
    }

    // T-239: batch_fetch_post_meta returns empty map for an empty input slice
    #[test]
    fn batch_fetch_empty() {
        let db = Db::open_memory().unwrap();
        let meta = batch_fetch_post_meta(db.conn(), &[]).unwrap();
        assert!(meta.is_empty());
    }

    // T-153: recent post scores higher than old post
    #[test]
    fn hybrid_search_recency_boost_recent_post_scores_higher() {
        let db = Db::open_memory().unwrap();

        let shared_body = "# 認証フロー\n認証の仕組みを説明します";
        let posts = [
            (1u32, "PostA", "2025-01-31T00:00:00+09:00"),
            (2u32, "PostB", "2025-01-02T00:00:00+09:00"),
        ];
        for (num, name, updated) in &posts {
            let mut post = test_post(*num, name, shared_body);
            post.updated_at = updated.to_string();
            storage::upsert_post(db.conn(), &post).unwrap();
            storage::rechunk_post(db.conn(), *num, shared_body).unwrap();
        }

        let now = chrono::DateTime::parse_from_rfc3339("2025-02-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let results = hybrid_search(
            db.conn(),
            "認証の仕組み",
            None,
            10,
            now,
            &SearchFilter::default(),
            None,
        )
        .unwrap();
        assert!(results.len() >= 2, "both posts should match");

        let score_a = results.iter().find(|r| r.post_number == 1).unwrap().score;
        let score_b = results.iter().find(|r| r.post_number == 2).unwrap().score;
        assert!(
            score_a > score_b,
            "post A (1 day old) should score higher than post B (30 days old), \
             got A={score_a} B={score_b}"
        );
    }

    // T-154: unparseable updated_at does not panic, decay=0.0 applied
    #[test]
    fn hybrid_search_unparseable_updated_at_no_panic() {
        let db = Db::open_memory().unwrap();

        let body = "# 認証フロー\n認証の仕組みを説明します";
        let mut post = test_post(1, "BadDate", body);
        post.updated_at = "not-a-date".into();
        storage::upsert_post(db.conn(), &post).unwrap();
        storage::rechunk_post(db.conn(), 1, body).unwrap();

        let now = chrono::DateTime::parse_from_rfc3339("2025-02-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let results = hybrid_search(
            db.conn(),
            "認証の仕組み",
            None,
            10,
            now,
            &SearchFilter::default(),
            None,
        )
        .unwrap();

        assert!(!results.is_empty(), "post should still be returned");
        assert!(
            results[0].score > 0.0,
            "score should be positive (raw RRF with 1.0x boost)"
        );
    }

    // T-151: content のみの語で fts_search ヒット — regression (FR-003)
    #[test]
    fn fts_search_matches_content_only_term_regression() {
        let db = Db::open_memory().unwrap();
        let body = "# 認証ガイド\nフローの説明をします";
        let post = test_post(1, "テスト", body);
        storage::upsert_post(db.conn(), &post).unwrap();
        storage::rechunk_post(db.conn(), 1, body).unwrap();

        let hits = fts_search(db.conn(), "フローの説明", 10, &SearchFilter::default()).unwrap();
        assert!(
            !hits.is_empty(),
            "content-only term must still be found (regression)"
        );
        assert_eq!(hits[0].post_number, 1);
    }

    // T-152: heading なし preamble chunk で rechunk + fts_search エラーなし (FR-004)
    #[test]
    fn fts_search_preamble_chunk_no_heading_no_error() {
        let db = Db::open_memory().unwrap();
        let body = "見出しなしの本文テキストです";
        let post = test_post(1, "Preamble", body);
        storage::upsert_post(db.conn(), &post).unwrap();
        let count = storage::rechunk_post(db.conn(), 1, body).unwrap();
        assert_eq!(count, 1, "preamble should produce 1 chunk");

        let hits = fts_search(db.conn(), "見出しなし", 10, &SearchFilter::default()).unwrap();
        assert!(
            !hits.is_empty(),
            "preamble chunk should be searchable by content"
        );
        assert!(
            hits[0].section_title.is_none(),
            "preamble chunk should have no section_title"
        );
    }

    // T-150: section_title のみの語で fts_search ヒット (FR-003)
    #[test]
    fn fts_search_matches_section_title_only_term() {
        let db = Db::open_memory().unwrap();
        let body = "# 認証ガイド\nフローの説明をします";
        let post = test_post(1, "テスト", body);
        storage::upsert_post(db.conn(), &post).unwrap();
        storage::rechunk_post(db.conn(), 1, body).unwrap();

        let hits = fts_search(db.conn(), "認証ガイド", 10, &SearchFilter::default()).unwrap();
        assert!(
            !hits.is_empty(),
            "section_title-only term should be found via FTS"
        );
        assert_eq!(hits[0].post_number, 1);
        assert_eq!(hits[0].section_title.as_deref(), Some("認証ガイド"));
    }

    // T-240: fts_search finds posts by name, author, category, and tag metadata
    #[test]
    fn fts_search_matches_enriched_metadata() {
        let db = Db::open_memory().unwrap();
        let mut row = storage::test_post_row(1);
        row.name = "Daily振り返り".into();
        row.body_md = "# 作業内容\n実装した".into();
        row.category = Some("チーム/日報".into());
        row.tags = vec!["日報".into(), "振り返り".into()];
        row.created_by = "thkt".into();
        storage::upsert_post(db.conn(), &row).unwrap();

        let enriched = storage::enrich_body(&row);
        storage::rechunk_post(db.conn(), 1, &enriched).unwrap();

        // Post name
        let hits = fts_search(db.conn(), "Daily振り返り", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty(), "post name should be searchable");

        // Author
        let hits = fts_search(db.conn(), "thkt", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty(), "author should be searchable");

        // Category
        let hits = fts_search(db.conn(), "日報", 10, &SearchFilter::default()).unwrap();
        assert!(!hits.is_empty(), "category/tag should be searchable");

        // Combined query (matching esa web search behavior)
        let hits = fts_search(
            db.conn(),
            "Daily 振り返り thkt",
            10,
            &SearchFilter::default(),
        )
        .unwrap();
        assert!(
            !hits.is_empty(),
            "combined name + author query should match"
        );
    }

    // T-035: search --json → JSON array with post_number, post_name, score
    #[test]
    fn search_result_serializes_to_json_with_expected_fields() {
        let result = SearchResult {
            post_number: 42,
            post_name: "Test Post".to_string(),
            post_url: "https://example.esa.io/posts/42".to_string(),
            section_title: Some("Section".to_string()),
            snippet: "snippet text".to_string(),
            score: 0.75,
            match_source: MatchSource::Fts,
        };
        let json_str = serde_json::to_string(&result).expect("SearchResult should serialize");
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["post_number"], 42);
        assert_eq!(v["post_name"], "Test Post");
        assert_eq!(v["score"], 0.75);
        assert_eq!(v["match_source"], "fts");
        assert_eq!(v["post_url"], "https://example.esa.io/posts/42");
        assert_eq!(v["section_title"], "Section");
        assert_eq!(v["snippet"], "snippet text");
    }

    // T-156: truncate_snippet
    #[test]
    fn truncate_snippet_short_unchanged() {
        assert_eq!(truncate_snippet("abc", 5), "abc");
    }

    // T-241: truncate_snippet appends "..." when snippet exceeds char limit
    #[test]
    fn truncate_snippet_over_limit_truncated() {
        assert_eq!(truncate_snippet("abcdef", 3), "abc...");
    }

    // T-242: truncate_snippet truncates at character boundary for multi-byte chars
    #[test]
    fn truncate_snippet_multibyte_boundary() {
        let s = "あいうえお";
        let result = truncate_snippet(s, 3);
        assert_eq!(result, "あいう...");
    }

    // T-243: hybrid_search uses vector search when embeddings are present
    #[test]
    fn hybrid_search_with_embeddings() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let unembedded = storage::get_unembedded_chunks(db.conn(), 100).unwrap();
        assert!(!unembedded.is_empty());
        let embeddings: Vec<(i64, ChunkedEmbedding)> = unembedded
            .iter()
            .map(|(id, _)| {
                (
                    *id,
                    ChunkedEmbedding {
                        chunks: vec![vec![0.1; EMBEDDING_DIMS]],
                    },
                )
            })
            .collect();
        storage::add_chunked_embeddings(db.conn(), &embeddings).unwrap();

        let query_emb = vec![0.1; EMBEDDING_DIMS];
        let results = hybrid_search(
            db.conn(),
            "認証",
            Some(&query_emb),
            10,
            chrono::Utc::now(),
            &SearchFilter::default(),
            None,
        )
        .unwrap();
        assert!(!results.is_empty());
        assert!(!results[0].post_name.is_empty());
        assert!(!results[0].post_url.is_empty());
    }

    /// Distinct embeddings produce different vec_search distances and ranking.
    // T-244: vec_search ranks the chunk with the closer embedding first
    #[test]
    fn vec_search_ranks_closer_embedding_first() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        // Distinct embeddings per chunk: vary dim 0 weight to control distance
        let chunks = storage::get_unembedded_chunks(db.conn(), 100).unwrap();
        assert!(chunks.len() >= 2, "need at least 2 chunks");

        let embeddings: Vec<(i64, ChunkedEmbedding)> = chunks
            .iter()
            .enumerate()
            .map(|(i, (id, _))| {
                let mut emb = vec![0.01_f32; EMBEDDING_DIMS];
                // chunk 0: strong dim-0 signal (close to query)
                // chunk 1+: weaker dim-0 signal (farther from query)
                emb[0] = if i == 0 { 1.0 } else { 0.01 };
                emb[1] = if i == 0 { 0.01 } else { 1.0 };
                (*id, ChunkedEmbedding { chunks: vec![emb] })
            })
            .collect();
        storage::add_chunked_embeddings(db.conn(), &embeddings).unwrap();

        // Query: strong dim-0 signal
        let mut query_emb = vec![0.01_f32; EMBEDDING_DIMS];
        query_emb[0] = 1.0;

        let results = hybrid_search(
            db.conn(),
            "認証",
            Some(&query_emb),
            10,
            chrono::Utc::now(),
            &SearchFilter::default(),
            None,
        )
        .unwrap();
        assert!(results.len() >= 2, "expected at least 2 results");

        // First result should come from the chunk with the closest embedding
        // (chunk 0 with strong dim-0 matches query's strong dim-0)
        assert!(
            results[0].score >= results[1].score,
            "closer embedding should score higher: s0={} s1={}",
            results[0].score,
            results[1].score,
        );
    }

    /// MaxSim dedup: multiple sub-embeddings per chunk, best distance wins.
    // T-245: vec_search deduplicates chunks and keeps only the best sub-embedding
    #[test]
    fn maxsim_dedup_selects_best_sub_embedding() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let chunks = storage::get_unembedded_chunks(db.conn(), 100).unwrap();
        assert!(chunks.len() >= 2, "need at least 2 chunks");

        // Each chunk gets 2 sub-embeddings with different dim-0 weights
        let embeddings: Vec<(i64, ChunkedEmbedding)> = chunks
            .iter()
            .enumerate()
            .map(|(i, (id, _))| {
                // sub-emb A: closer to query for chunk 0, farther for others
                let mut sub_a = vec![0.01_f32; EMBEDDING_DIMS];
                sub_a[0] = if i == 0 { 0.9 } else { 0.01 };
                sub_a[1] = if i == 0 { 0.01 } else { 0.9 };

                // sub-emb B: always far from query
                let mut sub_b = vec![0.01_f32; EMBEDDING_DIMS];
                sub_b[2] = 1.0;

                (
                    *id,
                    ChunkedEmbedding {
                        chunks: vec![sub_a, sub_b],
                    },
                )
            })
            .collect();
        storage::add_chunked_embeddings(db.conn(), &embeddings).unwrap();

        // Query: strong dim-0
        let mut query_emb = vec![0.01_f32; EMBEDDING_DIMS];
        query_emb[0] = 1.0;

        let results = hybrid_search(
            db.conn(),
            "認証",
            Some(&query_emb),
            10,
            chrono::Utc::now(),
            &SearchFilter::default(),
            None,
        )
        .unwrap();
        assert!(
            !results.is_empty(),
            "should have results with multi-sub embeddings"
        );

        // Dedup: each chunk_id produces at most one result (2 sub-embs per chunk → not 2× results)
        assert!(
            results.len() <= chunks.len(),
            "dedup should keep at most one result per chunk: got {} results for {} chunks",
            results.len(),
            chunks.len()
        );
        // No duplicate (post_number, section_title) pairs in results
        let mut seen = std::collections::HashSet::new();
        for r in &results {
            let key = (r.post_number, r.section_title.clone());
            assert!(
                seen.insert(key.clone()),
                "duplicate chunk in results: post {} section {:?}",
                key.0,
                key.1
            );
        }
    }

    // T-155: MaxSim selects the sub-embedding with the lower distance (closer to query)
    #[test]
    fn vec_search_maxsim_picks_closer_sub_embedding() {
        use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS};

        let db = Db::open_memory().unwrap();
        let body = "# Rust\nOwnership and borrowing";
        let post = test_post(1, "Rust", body);
        storage::upsert_post(db.conn(), &post).unwrap();
        storage::rechunk_post(db.conn(), 1, body).unwrap();

        let chunks = storage::get_unembedded_chunks(db.conn(), 10).unwrap();
        assert_eq!(chunks.len(), 1);
        let chunk_id = chunks[0].0;

        // sub_idx 0: orthogonal to query (far)
        let mut sub_far = vec![0.0f32; EMBEDDING_DIMS];
        sub_far[1] = 1.0;
        // sub_idx 1: aligned with query (close, distance ≈ 0)
        let mut sub_close = vec![0.0f32; EMBEDDING_DIMS];
        sub_close[0] = 1.0;

        storage::add_chunked_embeddings(
            db.conn(),
            &[(
                chunk_id,
                ChunkedEmbedding {
                    chunks: vec![sub_far, sub_close],
                },
            )],
        )
        .unwrap();

        let mut query = vec![0.0f32; EMBEDDING_DIMS];
        query[0] = 1.0;

        let hits = vec_search(db.conn(), &query, 1, &SearchFilter::default()).unwrap();
        assert_eq!(hits.len(), 1, "should return exactly one hit");
        assert!(
            hits[0].distance < 0.1,
            "MaxSim must select sub_idx=1 (closer sub-embedding), got distance={}",
            hits[0].distance
        );
    }

    // T-246: fts_search with category filter returns only posts in that category
    #[test]
    fn fts_filter_category_narrows_results() {
        let db = Db::open_memory().unwrap();

        let mut post1 = storage::test_post_row(1);
        post1.body_md = "# 認証フロー\n認証の仕組みを説明します".into();
        post1.category = Some("backend".into());
        storage::upsert_post(db.conn(), &post1).unwrap();
        storage::rechunk_post(db.conn(), 1, &post1.body_md).unwrap();

        let mut post2 = storage::test_post_row(2);
        post2.body_md = "# 認証フロー\n認証の実装".into();
        post2.category = Some("frontend".into());
        storage::upsert_post(db.conn(), &post2).unwrap();
        storage::rechunk_post(db.conn(), 2, &post2.body_md).unwrap();

        let hits = fts_search(
            db.conn(),
            "認証",
            10,
            &SearchFilter {
                category: Some("backend"),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!hits.is_empty(), "category filter should return results");
        assert!(
            hits.iter().all(|h| h.post_number == 1),
            "category=backend should only match post 1"
        );
    }

    // T-247: fts_search with created_by filter returns only posts by that author
    #[test]
    fn fts_filter_created_by_narrows_results() {
        let db = Db::open_memory().unwrap();

        let mut post1 = storage::test_post_row(1);
        post1.body_md = "# 認証フロー\n認証の仕組みを説明します".into();
        post1.created_by = "alice".into();
        storage::upsert_post(db.conn(), &post1).unwrap();
        storage::rechunk_post(db.conn(), 1, &post1.body_md).unwrap();

        let mut post2 = storage::test_post_row(2);
        post2.body_md = "# 認証フロー\n認証の実装".into();
        post2.created_by = "bob".into();
        storage::upsert_post(db.conn(), &post2).unwrap();
        storage::rechunk_post(db.conn(), 2, &post2.body_md).unwrap();

        let hits = fts_search(
            db.conn(),
            "認証",
            10,
            &SearchFilter {
                created_by: Some("alice"),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!hits.is_empty(), "created_by filter should return results");
        assert!(
            hits.iter().all(|h| h.post_number == 1),
            "created_by=alice should only match post 1"
        );
    }

    // T-248: fts_search with tags filter returns only posts containing that tag
    #[test]
    fn fts_filter_tags_narrows_results() {
        let db = Db::open_memory().unwrap();

        let mut post1 = storage::test_post_row(1);
        post1.body_md = "# 認証フロー\n認証の仕組みを説明します".into();
        post1.tags = vec!["security".into(), "auth".into()];
        storage::upsert_post(db.conn(), &post1).unwrap();
        storage::rechunk_post(db.conn(), 1, &post1.body_md).unwrap();

        let mut post2 = storage::test_post_row(2);
        post2.body_md = "# 認証フロー\n認証の実装".into();
        post2.tags = vec!["frontend".into()];
        storage::upsert_post(db.conn(), &post2).unwrap();
        storage::rechunk_post(db.conn(), 2, &post2.body_md).unwrap();

        let hits = fts_search(
            db.conn(),
            "認証",
            10,
            &SearchFilter {
                tags: Some(&["security"]),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!hits.is_empty(), "tags filter should return results");
        assert!(
            hits.iter().all(|h| h.post_number == 1),
            "tags=[security] should only match post 1"
        );
    }

    // T-249: fts_search with no filters returns all matching posts
    #[test]
    fn fts_filter_all_none_returns_all_matching() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(db.conn(), "認証", 10, &SearchFilter::default()).unwrap();
        assert!(
            !hits.is_empty(),
            "no filters should return all matching posts"
        );
    }

    // T-250: fts_search with empty tags slice does not filter results
    #[test]
    fn fts_filter_empty_tags_returns_all_matching() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let hits = fts_search(
            db.conn(),
            "認証",
            10,
            &SearchFilter {
                tags: Some(&[]),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            !hits.is_empty(),
            "empty tags slice should not filter anything"
        );
    }

    // T-251: hybrid_search with category filter returns only posts in that category
    #[test]
    fn hybrid_search_filter_category_narrows_results() {
        let db = Db::open_memory().unwrap();

        let mut post1 = storage::test_post_row(1);
        post1.body_md = "# 認証フロー\n認証の仕組みを説明します".into();
        post1.category = Some("backend".into());
        storage::upsert_post(db.conn(), &post1).unwrap();
        storage::rechunk_post(db.conn(), 1, &post1.body_md).unwrap();

        let mut post2 = storage::test_post_row(2);
        post2.body_md = "# 認証フロー\n認証の実装".into();
        post2.category = Some("frontend".into());
        storage::upsert_post(db.conn(), &post2).unwrap();
        storage::rechunk_post(db.conn(), 2, &post2.body_md).unwrap();

        let results = hybrid_search(
            db.conn(),
            "認証の仕組み",
            None,
            10,
            chrono::Utc::now(),
            &SearchFilter {
                category: Some("backend"),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert!(!results.is_empty(), "category filter should return results");
        assert!(
            results.iter().all(|r| r.post_number == 1),
            "category=backend should only match post 1"
        );
    }

    fn test_post_dated(number: u32, updated_at: &str) -> storage::EsaPostRow {
        let mut row = storage::test_post_row(number);
        row.name = format!("ガイド {number}");
        row.full_name = format!("dev/ガイド{number}");
        row.body_md = "# ガイド\n設定方法を解説".to_string();
        row.updated_at = updated_at.to_string();
        row
    }

    fn setup_db_with_dated_posts(db: &Db) {
        // post 1: 2025-01-10 (old)
        // post 2: 2025-06-15 (new)
        for (num, date) in [
            (1u32, "2025-01-10T00:00:00+00:00"),
            (2, "2025-06-15T00:00:00+00:00"),
        ] {
            let post = test_post_dated(num, date);
            storage::upsert_post(db.conn(), &post).unwrap();
            storage::rechunk_post(db.conn(), num, &post.body_md).unwrap();
        }
    }

    fn date_utc(s: &str) -> chrono::DateTime<chrono::Utc> {
        chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
    }

    // T-158: updated_after filters out posts with updated_at before threshold
    #[test]
    fn hybrid_search_updated_after_excludes_old_posts() {
        let db = Db::open_memory().unwrap();
        setup_db_with_dated_posts(&db);

        let filter = SearchFilter {
            updated_after: Some(date_utc("2025-03-01")),
            ..Default::default()
        };
        let results = hybrid_search(
            db.conn(),
            "ガイド",
            None,
            10,
            chrono::Utc::now(),
            &filter,
            None,
        )
        .unwrap();

        assert_eq!(results.len(), 1, "only the newer post should be returned");
        assert_eq!(results[0].post_number, 2);
    }

    // T-159: updated_before filters out posts with updated_at after threshold
    #[test]
    fn hybrid_search_updated_before_excludes_new_posts() {
        let db = Db::open_memory().unwrap();
        setup_db_with_dated_posts(&db);

        let filter = SearchFilter {
            updated_before: Some(date_utc("2025-03-01")),
            ..Default::default()
        };
        let results = hybrid_search(
            db.conn(),
            "ガイド",
            None,
            10,
            chrono::Utc::now(),
            &filter,
            None,
        )
        .unwrap();

        assert_eq!(results.len(), 1, "only the older post should be returned");
        assert_eq!(results[0].post_number, 1);
    }

    // T-160: updated_after and updated_before combined apply AND condition
    #[test]
    fn hybrid_search_date_range_returns_matching_posts_only() {
        let db = Db::open_memory().unwrap();
        setup_db_with_dated_posts(&db);

        // Range that covers only post 1 (2025-01-10)
        let filter = SearchFilter {
            updated_after: Some(date_utc("2025-01-01")),
            updated_before: Some(date_utc("2025-03-01")),
            ..Default::default()
        };
        let results = hybrid_search(
            db.conn(),
            "ガイド",
            None,
            10,
            chrono::Utc::now(),
            &filter,
            None,
        )
        .unwrap();

        assert_eq!(
            results.len(),
            1,
            "range 2025-01-01..2025-03-01 should include only post 1"
        );
        assert_eq!(results[0].post_number, 1);
    }

    // T-157: hybrid_search with MockReranker applies cross-encoder scores to results
    #[test]
    fn hybrid_search_reranker_applies_cross_encoder_scores() {
        use rurico::reranker::MockReranker;

        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let reranker = MockReranker::with_score(0.99);
        let results = hybrid_search(
            db.conn(),
            "認証の仕組み",
            None,
            10,
            chrono::Utc::now(),
            &SearchFilter::default(),
            Some(&reranker),
        )
        .unwrap();
        assert!(!results.is_empty(), "reranker search should return results");
        for r in &results {
            assert!(
                r.score >= 0.99,
                "score should be cross-encoder (0.99) × recency boost (≥1.0); got {}",
                r.score
            );
        }
    }
}
