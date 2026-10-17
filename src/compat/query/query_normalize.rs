// Pure helper retained from the previous dependency to preserve behavior.

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryNormalizationConfig {
    pub nfkc: bool,
    pub ascii_lowercase: bool,
    pub collapse_whitespace: bool,
}

impl Default for QueryNormalizationConfig {
    fn default() -> Self {
        Self {
            nfkc: true,
            ascii_lowercase: true,
            collapse_whitespace: true,
        }
    }
}

#[must_use]
pub fn normalize_for_fts(text: &str, config: &QueryNormalizationConfig) -> String {
    if !config.nfkc && !config.ascii_lowercase && !config.collapse_whitespace {
        return text.to_owned();
    }
    let mut buf = if config.nfkc {
        text.nfkc().collect::<String>()
    } else {
        text.to_owned()
    };
    if config.ascii_lowercase {
        buf.make_ascii_lowercase();
    }
    if config.collapse_whitespace {
        buf = collapse_whitespace(&buf);
    }
    buf
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::{QueryNormalizationConfig, normalize_for_fts};

    #[test]
    fn normalization_flags_can_preserve_or_selectively_transform_input() {
        let input = " ＡBC\tRust\n  ";
        for (nfkc, ascii_lowercase, collapse_whitespace, expected) in [
            (false, false, false, " ＡBC\tRust\n  "),
            (false, true, false, " Ａbc\trust\n  "),
            (true, false, false, " ABC\tRust\n  "),
            (false, false, true, "ＡBC Rust"),
            (true, true, true, "abc rust"),
        ] {
            let config = QueryNormalizationConfig {
                nfkc,
                ascii_lowercase,
                collapse_whitespace,
            };
            assert_eq!(normalize_for_fts(input, &config), expected);
        }
    }
}
