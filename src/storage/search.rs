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
pub fn fts_search(
    conn: &Connection,
    query: &str,
    limit: u32,
) -> Result<Vec<FtsHit>, StorageError> {
    let sanitized = sanitize_fts(query);
    let expanded = expand_short_terms(conn, &sanitized);

    if expanded.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut stmt = conn.prepare(
        "SELECT c.id, c.post_number, c.section_title, c.content, f.rank \
         FROM fts_chunks f \
         JOIN chunks c ON c.id = f.rowid \
         WHERE fts_chunks MATCH ?1 \
         ORDER BY f.rank \
         LIMIT ?2",
    )?;

    let rows: Vec<FtsHit> = stmt
        .query_map(rusqlite::params![expanded, limit], |row| {
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
    let bytes: &[u8] = bytemuck::cast_slice(query_embedding);

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

/// Hybrid search: merge FTS5 + vector results via Reciprocal Rank Fusion.
pub fn hybrid_search(
    conn: &Connection,
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: u32,
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

    let merged = rrf_merge(&fts_hits, &vec_hits);

    let post_numbers: Vec<u32> = merged
        .iter()
        .take(limit as usize)
        .map(|(pn, _)| *pn)
        .collect();
    let post_meta = batch_fetch_post_meta(conn, &post_numbers)?;

    let fts_map: HashMap<u32, &FtsHit> = fts_hits.iter().map(|h| (h.post_number, h)).collect();
    let vec_map: HashMap<u32, &VecHit> = vec_hits.iter().map(|h| (h.post_number, h)).collect();

    let mut results = Vec::new();
    for (post_number, score) in merged.into_iter().take(limit as usize) {
        let (name, url) = post_meta
            .get(&post_number)
            .cloned()
            .unwrap_or_else(|| (format!("#{post_number}"), String::new()));

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
) -> Result<HashMap<u32, (String, String)>, StorageError> {
    if post_numbers.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders: Vec<String> = (1..=post_numbers.len()).map(|i| format!("?{i}")).collect();
    let sql = format!(
        "SELECT number, name, url FROM posts WHERE number IN ({})",
        placeholders.join(", ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = post_numbers
        .iter()
        .map(|n| n as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt
        .query_map(params.as_slice(), |row| {
            Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows.into_iter().map(|(n, name, url)| (n, (name, url))).collect())
}

const RRF_K: f64 = 60.0;

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

fn rrf_merge(fts_hits: &[FtsHit], vec_hits: &[VecHit]) -> Vec<(u32, f64)> {
    let mut scores: HashMap<u32, f64> = HashMap::new();

    for (rank, hit) in fts_hits.iter().enumerate() {
        *scores.entry(hit.post_number).or_default() += 1.0 / (RRF_K + rank as f64);
    }
    for (rank, hit) in vec_hits.iter().enumerate() {
        *scores.entry(hit.post_number).or_default() += 1.0 / (RRF_K + rank as f64);
    }

    let mut results: Vec<(u32, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

fn fts_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

fn sanitize_fts(query: &str) -> String {
    query
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}

fn expand_short_terms(conn: &Connection, sanitized: &str) -> String {
    let mut parts = Vec::new();
    for token in sanitized.split_whitespace() {
        let upper = token.to_ascii_uppercase();
        if matches!(upper.as_str(), "AND" | "OR" | "NOT") {
            parts.push(token.to_string());
            continue;
        }
        if token.is_empty() {
            continue;
        }
        if token.chars().count() >= 3 {
            parts.push(fts_quote(token));
            continue;
        }
        let escaped = token
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("{escaped}%");
        let expanded: Vec<String> = conn
            .prepare(
                "SELECT term FROM fts_chunks_vocab \
                 WHERE term LIKE ?1 ESCAPE '\\' \
                 ORDER BY cnt DESC LIMIT 25",
            )
            .and_then(|mut stmt| {
                stmt.query_map([&pattern], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap_or_default();
        if expanded.is_empty() {
            parts.push(fts_quote(token));
        } else {
            let quoted: Vec<String> = expanded.iter().map(|t| fts_quote(t)).collect();
            parts.push(format!("({})", quoted.join(" OR ")));
        }
    }
    parts.join(" ")
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

    fn setup_db_with_posts(db: &Db) {
        let posts = [
            (1, "認証ガイド", "# 認証フロー\n認証の仕組みを説明します\n# 実装手順\nコードの書き方"),
            (2, "API設計", "# エンドポイント\nREST APIの設計方針\n# 認証\nトークン認証の詳細"),
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
        let fts = vec![
            FtsHit { chunk_id: 1, post_number: 1, section_title: None, content: String::new(), rank: -1.0 },
            FtsHit { chunk_id: 2, post_number: 2, section_title: None, content: String::new(), rank: -0.5 },
        ];
        let vec = vec![
            VecHit { chunk_id: 3, post_number: 2, distance: 0.1, section_title: None, content: String::new() },
            VecHit { chunk_id: 4, post_number: 3, distance: 0.2, section_title: None, content: String::new() },
        ];

        let merged = rrf_merge(&fts, &vec);
        assert_eq!(merged[0].0, 2);
        assert!(merged[0].1 > merged[1].1);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn hybrid_search_fts_only() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let results = hybrid_search(db.conn(), "認証フロー", None, 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].post_number, 1);
        assert!(!results[0].post_name.is_empty());
        assert!(!results[0].post_url.is_empty());
    }

    #[test]
    fn empty_query_returns_empty() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let results = hybrid_search(db.conn(), "", None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn sanitize_removes_special_chars_and_quotes() {
        assert_eq!(sanitize_fts("hello; DROP TABLE"), "hello DROP TABLE");
        assert_eq!(sanitize_fts("認証 AND フロー"), "認証 AND フロー");
        assert_eq!(sanitize_fts("\"injected\""), "injected");
    }

    #[test]
    fn batch_fetch_post_meta_works() {
        let db = Db::open_memory().unwrap();
        setup_db_with_posts(&db);

        let meta = batch_fetch_post_meta(db.conn(), &[1, 3]).unwrap();
        assert_eq!(meta.len(), 2);
        assert!(meta.get(&1).unwrap().0.contains("認証"));
        assert!(meta.get(&3).unwrap().0.contains("デプロイ"));
    }

    #[test]
    fn batch_fetch_empty() {
        let db = Db::open_memory().unwrap();
        let meta = batch_fetch_post_meta(db.conn(), &[]).unwrap();
        assert!(meta.is_empty());
    }
}
