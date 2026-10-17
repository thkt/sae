// Pure helper retained from the previous dependency to preserve behavior.
use std::ffi::OsString;

pub fn try_expand_shorthand(
    args: &[OsString],
    known_subcommands: &[&str],
    global_flags: &[&str],
) -> Option<Vec<OsString>> {
    let (flag_pairs, indexed_rest): (Vec<_>, Vec<_>) = args
        .iter()
        .enumerate()
        .partition(|(i, a)| *i > 0 && a.to_str().is_some_and(|s| global_flags.contains(&s)));
    let rest: Vec<&OsString> = indexed_rest.into_iter().map(|(_, a)| a).collect();
    let flags: Vec<&OsString> = flag_pairs.into_iter().map(|(_, a)| a).collect();

    if rest.len() >= 2
        && let Some(first_arg) = rest[1].to_str()
        && !first_arg.starts_with('-')
        && first_arg != "help"
        && !known_subcommands.contains(&first_arg)
        && !known_subcommands
            .iter()
            .any(|k| osa_distance(first_arg, k) <= 1)
    {
        let mut expanded: Vec<OsString> = vec![rest[0].clone()];
        expanded.extend(flags.into_iter().cloned());
        expanded.push("search".into());
        expanded.extend(rest[1..].iter().copied().cloned());
        Some(expanded)
    } else {
        None
    }
}

fn osa_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (na, nb) = (a.len(), b.len());
    let mut d = vec![vec![0usize; nb + 1]; na + 1];
    for (i, row) in d.iter_mut().enumerate().take(na + 1) {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate().take(nb + 1) {
        *cell = j;
    }
    for i in 1..=na {
        for j in 1..=nb {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + cost);
            }
        }
    }
    d[na][nb]
}
