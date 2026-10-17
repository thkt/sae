use rusqlite::types::ToSql;

pub(crate) fn in_placeholders(len: usize) -> String {
    (1..=len)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn anon_placeholders(n: usize) -> String {
    vec!["?"; n].join(", ")
}

pub(crate) fn as_sql_params<T: ToSql>(values: &[T]) -> Vec<&dyn ToSql> {
    values.iter().map(|v| v as &dyn ToSql).collect()
}

pub(crate) fn append_eq_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    value: Option<&str>,
) {
    if let Some(v) = value {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(" = ?");
        params.push(Box::new(v.to_owned()));
    }
}

pub(crate) fn append_in_filter<T>(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    values: Option<&[T]>,
) where
    T: ToSql + Clone + 'static,
{
    let Some(values) = values else {
        return;
    };
    if values.is_empty() {
        sql.push_str(" AND 1 = 0");
        return;
    }
    sql.push_str(&format!(
        " AND {column} IN ({})",
        anon_placeholders(values.len())
    ));
    params.extend(
        values
            .iter()
            .cloned()
            .map(|value| Box::new(value) as Box<dyn ToSql>),
    );
}

pub(crate) fn append_timestamp_day_cutoff_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    before: bool,
    date_iso: Option<&str>,
) {
    if let Some(date) = date_iso {
        sql.push_str(" AND ");
        sql.push_str(column);
        if before {
            sql.push_str(" < date(?, '+1 day')");
        } else {
            sql.push_str(" >= ?");
        }
        params.push(Box::new(date.to_owned()));
    }
}

#[cfg(test)]
mod tests {
    use super::append_in_filter;
    use rusqlite::{Connection, params_from_iter, types::ToSql};

    #[test]
    fn absent_in_filter_preserves_query_but_empty_selection_matches_nothing() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE items(id INTEGER); INSERT INTO items VALUES(1),(2);")
            .unwrap();
        for (selection, expected) in [(None, 2), (Some(&[][..]), 0), (Some(&[2][..]), 1)] {
            let mut sql = "SELECT count(*) FROM items WHERE 1=1".to_owned();
            let mut values: Vec<Box<dyn ToSql>> = Vec::new();
            append_in_filter(&mut sql, &mut values, "id", selection);
            let found: i64 = conn
                .query_row(&sql, params_from_iter(values.iter()), |row| row.get(0))
                .unwrap();
            assert_eq!(found, expected);
        }
    }
}
