//! Structural audit tests for the sae codebase.
//!
//! These tests validate source-level properties (line counts, grep matches,
//! module structure) and read source files at test time to assert on their
//! textual content. This approach lets `cargo test` catch structural
//! regressions without requiring external CI scripts.

use std::fs;
use std::path::{Path, PathBuf};

/// Locate the project src/ directory relative to the cargo manifest.
fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Count non-test, non-blank lines in a Rust source file.
/// Excludes lines inside `#[cfg(test)]` modules (from the marker to EOF or
/// matching closing brace).
fn count_non_test_lines(path: &Path) -> usize {
    let content = fs::read_to_string(path).expect("failed to read source file");
    let mut in_test_module = false;
    let mut count = 0;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.contains("#[cfg(test)]") {
            in_test_module = true;
            continue;
        }

        // Treat everything from #[cfg(test)] to EOF as test code
        if in_test_module {
            continue;
        }

        if !trimmed.is_empty() {
            count += 1;
        }
    }

    count
}

/// Count occurrences of `needle` in the given text, restricted to lines
/// between `fn {fn_name}` and the next `fn ` at the same or lesser indent.
fn count_in_function(source: &str, fn_name: &str, needle: &str) -> usize {
    let fn_start = format!("fn {fn_name}");
    let mut in_fn = false;
    let mut brace_depth: i32 = 0;
    let mut entered_body = false;
    let mut count = 0;

    for line in source.lines() {
        if !in_fn {
            if line.contains(&fn_start) {
                in_fn = true;
                brace_depth = 0;
                for ch in line.chars() {
                    match ch {
                        '{' => brace_depth += 1,
                        '}' => brace_depth -= 1,
                        _ => {}
                    }
                }
                if brace_depth > 0 {
                    entered_body = true;
                }
                if line.contains(needle) {
                    count += 1;
                }
            }
            continue;
        }

        // Inside the function (signature or body)
        for ch in line.chars() {
            match ch {
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                _ => {}
            }
        }
        if brace_depth > 0 {
            entered_body = true;
        }

        if line.contains(needle) {
            count += 1;
        }

        if entered_body && brace_depth <= 0 {
            break;
        }
    }

    count
}

// T-165: main.rs stays under 400 non-test lines after extraction
#[test]
fn main_rs_under_400_non_test_lines() {
    let path = src_dir().join("main.rs");
    let lines = count_non_test_lines(&path);
    assert!(
        lines < 400,
        "[T-165] main.rs has {lines} non-test lines, expected < 400"
    );
}

// T-166: each command file stays under 200 non-test lines
#[test]
fn each_command_file_under_200_non_test_lines() {
    let commands_dir = src_dir().join("commands");
    assert!(
        commands_dir.is_dir(),
        "[T-166] src/commands/ directory does not exist"
    );

    let expected_files = ["search.rs", "post.rs", "archive.rs", "data.rs", "status.rs"];
    for filename in &expected_files {
        let path = commands_dir.join(filename);
        assert!(
            path.exists(),
            "[T-166] src/commands/{filename} does not exist"
        );
        let lines = count_non_test_lines(&path);
        assert!(
            lines < 200,
            "[T-166] commands/{filename} has {lines} non-test lines, expected < 200"
        );
    }
}

// T-167: run_embed uses tracing instead of eprintln! (zero eprintln! calls)
#[test]
fn run_embed_no_eprintln() {
    let path = src_dir().join("commands").join("data.rs");
    assert!(
        path.exists(),
        "[T-167] src/commands/data.rs does not exist (run_embed should live there)"
    );
    let source = fs::read_to_string(&path).unwrap();
    let count = count_in_function(&source, "run_embed", "eprintln!");
    assert_eq!(
        count, 0,
        "[T-167] run_embed should contain 0 eprintln! calls, found {count}"
    );
}

// T-168: run_embed emits exactly one tracing::info! call
#[test]
fn run_embed_has_1_tracing_info() {
    let path = src_dir().join("commands").join("data.rs");
    assert!(path.exists(), "[T-168] src/commands/data.rs does not exist");
    let source = fs::read_to_string(&path).unwrap();
    let count = count_in_function(&source, "run_embed", "tracing::info!");
    assert_eq!(
        count, 1,
        "[T-168] run_embed should contain 1 tracing::info! call, found {count}"
    );
}

// T-169: embed_query warn! includes %query structured field
#[test]
fn embed_query_warn_includes_query_field() {
    let path = src_dir().join("commands").join("search.rs");
    assert!(
        path.exists(),
        "[T-169] src/commands/search.rs does not exist"
    );
    let source = fs::read_to_string(&path).unwrap();

    // Find warn! calls near embed_query that contain %query
    let mut found = false;
    for line in source.lines() {
        if line.contains("embed_query") || line.contains("warn!") {
            if line.contains("%query") {
                found = true;
                break;
            }
        }
    }
    // Broader search: look for %query in the embed_query error-handling block
    if !found {
        // Look for %query anywhere in the run_search function
        let count = count_in_function(&source, "run_search", "%query");
        assert!(
            count >= 1,
            "[T-169] run_search should contain at least 1 occurrence of '%query' \
             in a warn! call for embed_query failure, found {count}"
        );
    }
}

// T-170: each commands/*.rs has #[cfg(test)] module with at least 1 test
#[test]
fn each_command_file_has_test_module() {
    let commands_dir = src_dir().join("commands");
    assert!(
        commands_dir.is_dir(),
        "[T-170] src/commands/ directory does not exist"
    );

    let expected_files = ["search.rs", "post.rs", "archive.rs", "data.rs", "status.rs"];
    for filename in &expected_files {
        let path = commands_dir.join(filename);
        assert!(
            path.exists(),
            "[T-170] src/commands/{filename} does not exist"
        );
        let source = fs::read_to_string(&path).unwrap();
        assert!(
            source.contains("#[cfg(test)]"),
            "[T-170] commands/{filename} should contain #[cfg(test)] module"
        );
        assert!(
            source.contains("#[test]"),
            "[T-170] commands/{filename} should contain at least one #[test] function"
        );
    }
}

// T-171: vec_search warn! includes %query structured field
#[test]
fn vec_search_warn_includes_query_field() {
    let path = src_dir().join("storage").join("search.rs");
    let source = fs::read_to_string(&path).unwrap();

    // Look for %query in the hybrid_search function (where vec_search error is handled)
    let count = count_in_function(&source, "hybrid_search", "%query");
    assert!(
        count >= 1,
        "[T-171] hybrid_search should contain '%query' in vec_search warn!, found {count}"
    );
}

// T-172: vec_search warn! includes candidate_limit structured field
#[test]
fn vec_search_warn_includes_candidate_limit() {
    let path = src_dir().join("storage").join("search.rs");
    let source = fs::read_to_string(&path).unwrap();

    let count = count_in_function(&source, "hybrid_search", "candidate_limit");
    assert!(
        count >= 1,
        "[T-172] hybrid_search should contain 'candidate_limit' in vec_search warn!, found {count}"
    );
}
