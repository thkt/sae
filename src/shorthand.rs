use std::ffi::OsString;

/// Expands shorthand `sae "query"` → `sae [global_flags] search "query" [rest_flags]`.
///
/// Returns `Some(expanded_args)` when `args` match the shorthand pattern,
/// `None` otherwise. Caller provides known subcommand names and global flag
/// strings (e.g. `&["--json"]`); this fn has no dependency on `Cli`.
pub(crate) fn try_expand_shorthand(
    args: &[OsString],
    known_subcommands: &[&str],
    global_flags: &[&str],
) -> Option<Vec<OsString>> {
    let positional_count = args
        .iter()
        .filter(|a| !a.to_str().is_some_and(|s| s.starts_with('-')))
        .count();

    if positional_count < 2 {
        return None;
    }

    let (flags, rest): (Vec<_>, Vec<_>) = args
        .iter()
        .enumerate()
        .partition(|(i, a)| *i > 0 && a.to_str().is_some_and(|s| global_flags.contains(&s)));
    let rest: Vec<&OsString> = rest.into_iter().map(|(_, a)| a).collect();

    if rest.len() >= 2
        && let Some(first_arg) = rest[1].to_str()
        && !first_arg.starts_with('-')
        && first_arg != "help"
        && !known_subcommands.contains(&first_arg)
        && !known_subcommands
            .iter()
            .any(|k| strsim::osa_distance(first_arg, k) <= 1)
    {
        let mut expanded: Vec<OsString> = vec![rest[0].clone()];
        for (_, f) in &flags {
            expanded.push((*f).clone());
        }
        expanded.push("search".into());
        for arg in &rest[1..] {
            expanded.push((*arg).clone());
        }
        Some(expanded)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNOWN: &[&str] = &[
        "harvest", "search", "get", "update", "ship", "archive", "embed", "status", "create",
        "model",
    ];
    const GLOBAL: &[&str] = &["--json"];

    fn os(s: &[&str]) -> Vec<OsString> {
        s.iter().map(|&a| a.into()).collect()
    }

    // T-029: bare query expands with "search" inserted as subcommand
    #[test]
    fn single_query_expands_to_search() {
        let exp = try_expand_shorthand(&os(&["sae", "認証"]), KNOWN, GLOBAL).unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "search", "認証"]);
    }

    // T-031: known subcommand as first positional → not expanded
    #[test]
    fn known_subcommand_not_expanded() {
        assert!(try_expand_shorthand(&os(&["sae", "harvest", "myteam"]), KNOWN, GLOBAL).is_none());
    }

    // T-022: trailing options pass through after the inserted "search"
    #[test]
    fn query_with_trailing_option_expanded() {
        let exp =
            try_expand_shorthand(&os(&["sae", "query", "--limit", "2"]), KNOWN, GLOBAL).unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "search", "query", "--limit", "2"]);
    }

    // T-023: global flag (--json) is hoisted to before the inserted "search"
    #[test]
    fn global_flag_hoisted_before_search() {
        let exp = try_expand_shorthand(
            &os(&["sae", "--json", "query", "--limit", "2"]),
            KNOWN,
            GLOBAL,
        )
        .unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "--json", "search", "query", "--limit", "2"]);
    }

    // T-024: non-global option (--team) stays after the inserted "search"
    #[test]
    fn non_global_option_stays_after_search() {
        let exp = try_expand_shorthand(&os(&["sae", "query", "--team", "myteam"]), KNOWN, GLOBAL)
            .unwrap();
        let s: Vec<&str> = exp.iter().filter_map(|a| a.to_str()).collect();
        assert_eq!(s, ["sae", "search", "query", "--team", "myteam"]);
    }

    // T-025: typo within OSA distance 1 → not expanded (typo guard)
    #[test]
    fn typo_within_distance_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "serach"]), KNOWN, GLOBAL).is_none(),
            "typo 'serach' (osa=1 from 'search') should not expand"
        );
    }

    // TC-013: bare dash counts as flag prefix → positional_count < 2 → not expanded
    #[test]
    fn bare_dash_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "-"]), KNOWN, GLOBAL).is_none(),
            "`sae -` should not expand"
        );
    }

    // TC-011: flag-like arg (--) → positional_count < 2 → not expanded
    #[test]
    fn flag_only_not_expanded() {
        assert!(
            try_expand_shorthand(&os(&["sae", "--unknown"]), KNOWN, GLOBAL).is_none(),
            "--unknown should not expand"
        );
    }
}
