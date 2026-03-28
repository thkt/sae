#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Section,
    Full,
}

impl ChunkType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Section => "section",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub section_title: Option<String>,
    pub content: String,
    pub chunk_type: ChunkType,
}

const MAX_CHUNK_BYTES: usize = 16_000;

pub fn chunk_markdown(body_md: &str) -> Vec<Chunk> {
    if body_md.trim().is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = body_md.lines().collect();
    let headings = find_headings(&lines);

    if headings.is_empty() {
        return split_oversized(Chunk {
            section_title: None,
            content: body_md.to_string(),
            chunk_type: ChunkType::Full,
        });
    }

    let mut chunks = Vec::new();

    if headings[0].0 > 0 {
        let content: String = lines[..headings[0].0].join("\n");
        if !content.trim().is_empty() {
            chunks.push(Chunk {
                section_title: None,
                content,
                chunk_type: ChunkType::Section,
            });
        }
    }

    for (idx, (line_idx, title)) in headings.iter().enumerate() {
        let end = headings.get(idx + 1).map_or(lines.len(), |(next, _)| *next);
        let content: String = lines[(*line_idx + 1)..end].join("\n");
        if !content.trim().is_empty() {
            chunks.push(Chunk {
                section_title: if title.is_empty() {
                    None
                } else {
                    Some(title.clone())
                },
                content,
                chunk_type: ChunkType::Section,
            });
        }
    }

    chunks.into_iter().flat_map(split_oversized).collect()
}

fn split_oversized(chunk: Chunk) -> Vec<Chunk> {
    if chunk.content.len() <= MAX_CHUNK_BYTES {
        return vec![chunk];
    }
    // Heading is already excluded from content at chunk creation (line 56).
    // FTS indexing of section_title is tracked in #8.
    rurico::text::split_text(&chunk.content, MAX_CHUNK_BYTES)
        .into_iter()
        .map(|part| Chunk {
            section_title: chunk.section_title.clone(),
            content: part.to_string(),
            chunk_type: chunk.chunk_type.clone(),
        })
        .collect()
}

fn find_headings(lines: &[&str]) -> Vec<(usize, String)> {
    let mut headings = Vec::new();
    let mut in_fence = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(title) = parse_heading(line) {
            headings.push((i, title));
        }
    }

    headings
}

fn parse_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    // Markdown spec: up to 3 spaces of indentation allowed
    if line.len() - trimmed.len() > 3 {
        return None;
    }
    let level = trimmed.bytes().take_while(|&b| b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    Some(title.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_body_returns_empty() {
        assert!(chunk_markdown("").is_empty());
        assert!(chunk_markdown("   ").is_empty());
        assert!(chunk_markdown("\n\n").is_empty());
    }

    #[test]
    fn no_headings_returns_full_chunk() {
        let chunks = chunk_markdown("Just some text\nwithout headings");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, ChunkType::Full);
        assert!(chunks[0].section_title.is_none());
    }

    #[test]
    fn three_headings_three_chunks() {
        let md = "# Introduction\nHello world\n# Details\nSome details\n# Conclusion\nGoodbye";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].section_title.as_deref(), Some("Introduction"));
        assert_eq!(chunks[1].section_title.as_deref(), Some("Details"));
        assert_eq!(chunks[2].section_title.as_deref(), Some("Conclusion"));
        assert!(chunks.iter().all(|c| c.chunk_type == ChunkType::Section));
    }

    #[test]
    fn preamble_before_first_heading() {
        let md = "Some preamble\n\n# First Section\nContent";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].section_title.is_none());
        assert!(chunks[0].content.contains("preamble"));
        assert_eq!(chunks[1].section_title.as_deref(), Some("First Section"));
    }

    #[test]
    fn respects_code_fences() {
        let md = "# Real Heading\nContent\n```\n# Not A Heading\n```\n# Another Heading\nMore";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_title.as_deref(), Some("Real Heading"));
        assert!(chunks[0].content.contains("# Not A Heading"));
        assert_eq!(chunks[1].section_title.as_deref(), Some("Another Heading"));
    }

    #[test]
    fn nested_headings() {
        let md = "# H1\nText1\n## H2\nText2\n### H3\nText3";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 3);
        assert!(!chunks[0].content.contains("# H1"));
        assert!(chunks[0].content.contains("Text1"));
    }

    #[test]
    fn heading_only_returns_empty() {
        let chunks = chunk_markdown("# Title Only");
        assert!(chunks.is_empty());
    }

    #[test]
    fn headings_only_no_body_returns_empty() {
        let chunks = chunk_markdown("# H1\n## H2\n### H3");
        assert!(chunks.is_empty());
    }

    #[test]
    fn heading_with_trailing_hashes() {
        let md = "# Title ##\nContent";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks[0].section_title.as_deref(), Some("Title"));
    }

    #[test]
    fn rejects_invalid_headings() {
        assert!(parse_heading("#NoSpace").is_none());
        assert!(parse_heading("####### Seven").is_none());
        assert!(parse_heading("    # Indented").is_none());
    }

    // T-026: oversized section is split
    #[test]
    fn oversized_chunk_is_split() {
        // 20KB section with paragraph breaks
        let paragraph = "あ".repeat(500); // ~1500 bytes in UTF-8
        let mut body = "# Big Section\n".to_string();
        for i in 0..15 {
            body.push_str(&format!("Paragraph {i}: {paragraph}\n\n"));
        }
        let chunks = chunk_markdown(&body);
        assert!(chunks.len() > 1, "should be split into multiple chunks");
        for chunk in &chunks {
            assert!(
                chunk.content.len() <= MAX_CHUNK_BYTES,
                "chunk {} bytes exceeds MAX_CHUNK_BYTES",
                chunk.content.len()
            );
        }
    }

    // T-027: normal chunk is not split, heading excluded from content
    #[test]
    fn normal_chunk_not_split() {
        let md = "# Normal\nSmall content here";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section_title.as_deref(), Some("Normal"));
        assert!(!chunks[0].content.contains("# Normal"));
        assert!(chunks[0].content.contains("Small content here"));
    }

    // T-028: split chunks preserve section_title; heading excluded from content
    #[test]
    fn split_chunks_preserve_section_title() {
        let paragraph = "あ".repeat(500);
        let mut body = "# My Title\n".to_string();
        for i in 0..15 {
            body.push_str(&format!("Paragraph {i}: {paragraph}\n\n"));
        }
        let chunks = chunk_markdown(&body);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert_eq!(chunk.section_title.as_deref(), Some("My Title"));
            assert_eq!(chunk.chunk_type, ChunkType::Section);
            assert!(
                !chunk.content.starts_with("# "),
                "content should not contain heading line"
            );
        }
    }

    // T-032: exactly MAX_CHUNK_BYTES is NOT split (threshold is >, not >=)
    #[test]
    fn exact_threshold_not_split() {
        // Content excludes the heading line, so build body content of exactly
        // MAX_CHUNK_BYTES after the heading.
        let padding = "a".repeat(MAX_CHUNK_BYTES);
        let body = format!("# E\n{padding}");
        let chunks = chunk_markdown(&body);
        assert_eq!(
            chunks[0].content.len(),
            MAX_CHUNK_BYTES,
            "chunk content should be exactly MAX_CHUNK_BYTES"
        );
        assert_eq!(chunks.len(), 1, "exactly at threshold should not split");
    }

    #[test]
    fn tilde_fences() {
        let md = "# Before\nText\n~~~\n# Inside Fence\n~~~\n# After\nMore";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_title.as_deref(), Some("Before"));
        assert_eq!(chunks[1].section_title.as_deref(), Some("After"));
    }
}
