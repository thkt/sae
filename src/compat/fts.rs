// Pure helper retained from the previous dependency to preserve behavior.

use crate::compat::query::MatchFtsQuery;

const MAX_COMBOS: usize = 100;

pub fn clean_for_trigram(query: &MatchFtsQuery) -> Option<String> {
    clean_impl(query.as_str())
}

fn clean_impl(match_query: &str) -> Option<String> {
    let cleaned: String = match_query.chars().filter(|c| !c.is_control()).collect();
    let (fixed, or_groups) = parse_fts_segments(&cleaned);
    let fixed_only = || (!fixed.is_empty()).then(|| fixed.join(" "));

    if or_groups.is_empty() {
        return fixed_only();
    }

    // Estimate Π m_i without materializing cross_product. Saturates at
    // usize::MAX on overflow so the threshold check still fires.
    let estimated_combos = or_groups
        .iter()
        .map(Vec::len)
        .try_fold(1usize, usize::checked_mul)
        .unwrap_or(usize::MAX);
    if estimated_combos > MAX_COMBOS {
        return fixed_only();
    }

    let combos = cross_product(&or_groups);
    Some(
        combos
            .iter()
            .map(|combo| {
                let mut parts = combo.clone();
                parts.extend(fixed.iter().cloned());
                parts.join(" ")
            })
            .collect::<Vec<_>>()
            .join(" OR "),
    )
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
                .map(|t| t.trim().to_owned())
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
            // Trigram tokenizer cannot index <3 char terms; apply the same
            // filter used inside OR-groups so the Option<String> contract
            // stays honest across all input shapes.
            if term.trim_matches('"').chars().count() >= 3 {
                fixed.push(term);
            }
        }
    }

    (fixed, or_groups)
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

#[cfg(test)]
mod tests {
    #[test]
    fn excessive_or_expansion_retains_fixed_search_terms_without_cartesian_explosion() {
        let group = format!(
            "({})",
            (0..11)
                .map(|i| format!("\"term{i}\""))
                .collect::<Vec<_>>()
                .join(" OR ")
        );
        let query = format!("{group} {group} \"retained\"");
        assert_eq!(super::clean_impl(&query).as_deref(), Some("\"retained\""));
        assert_eq!(super::clean_impl(&format!("{group} {group}")), None);
    }
}
