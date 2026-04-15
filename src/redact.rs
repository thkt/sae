pub(crate) fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

pub(crate) fn redact_token(message: &str, token: &str) -> String {
    if token.is_empty() {
        return message.to_owned();
    }
    message.replace(token, "[REDACTED]")
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-132: redact_token replaces matching token with [REDACTED]
    #[test]
    fn redacts_known_token() {
        assert_eq!(
            redact_token("Error: token abc123secret in request", "abc123secret"),
            "Error: token [REDACTED] in request"
        );
    }

    // T-133: redact_token returns input unchanged when token is empty
    #[test]
    fn handles_empty_token() {
        assert_eq!(redact_token("Normal error", ""), "Normal error");
    }

    // T-134: redact_token returns input unchanged when token is not found
    #[test]
    fn handles_no_match() {
        assert_eq!(redact_token("Normal error", "secret"), "Normal error");
    }

    // T-135: redact_token replaces all occurrences when token appears multiple times
    #[test]
    fn redacts_multiple_occurrences() {
        assert_eq!(
            redact_token("tok=abc tok=abc", "abc"),
            "tok=[REDACTED] tok=[REDACTED]"
        );
    }

    // T-136: truncate_str returns full string when len is within limit
    #[test]
    fn truncate_within_limit() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    // T-137: truncate_str returns full string when len equals limit exactly
    #[test]
    fn truncate_at_limit() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    // T-138: truncate_str truncates ASCII string to byte limit
    #[test]
    fn truncate_ascii() {
        assert_eq!(truncate_str("hello world", 5), "hello");
    }

    // T-139: truncate_str does not split a multi-byte character
    #[test]
    fn truncate_respects_char_boundary() {
        assert_eq!(truncate_str("日本語テスト", 7), "日本");
    }
}
