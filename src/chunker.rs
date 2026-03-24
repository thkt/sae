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

pub fn chunk_markdown(body_md: &str) -> Vec<Chunk> {
    if body_md.trim().is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = body_md.lines().collect();
    let headings = find_headings(&lines);

    if headings.is_empty() {
        return vec![Chunk {
            section_title: None,
            content: body_md.to_string(),
            chunk_type: ChunkType::Full,
        }];
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
        let end = headings
            .get(idx + 1)
            .map_or(lines.len(), |(next, _)| *next);
        let content: String = lines[*line_idx..end].join("\n");
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

    if chunks.is_empty() {
        return vec![Chunk {
            section_title: None,
            content: body_md.to_string(),
            chunk_type: ChunkType::Full,
        }];
    }

    chunks
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
        let md = "# H1\n## H2\n### H3";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 3);
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

    #[test]
    fn tilde_fences() {
        let md = "# Before\nText\n~~~\n# Inside Fence\n~~~\n# After\nMore";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_title.as_deref(), Some("Before"));
        assert_eq!(chunks[1].section_title.as_deref(), Some("After"));
    }
}
