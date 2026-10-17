// Pure helper retained from the previous dependency to preserve behavior.
pub(crate) fn split_text(text: &str, max_bytes: usize) -> Vec<&str> {
    if max_bytes < 4 || text.len() <= max_bytes {
        return vec![text];
    }
    let mut fragments = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        if remaining.len() <= max_bytes {
            fragments.push(remaining);
            break;
        }
        let boundary = remaining.floor_char_boundary(max_bytes);
        let split_at = remaining[..boundary]
            .rfind("\n\n")
            .map(|p| p + 2)
            .or_else(|| remaining[..boundary].rfind('\n').map(|p| p + 1))
            .unwrap_or(boundary);
        fragments.push(&remaining[..split_at]);
        remaining = &remaining[split_at..];
    }
    fragments
}

#[cfg(test)]
mod tests {
    #[test]
    fn tiny_byte_budget_does_not_split_inside_unicode_or_loop() {
        for budget in 0..4 {
            assert_eq!(super::split_text("日本語🙂", budget), ["日本語🙂"]);
        }
        assert_eq!(super::split_text("", 100), [""]);
    }
}
