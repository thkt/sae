use std::iter;

use rusqlite::{Connection, Row, params_from_iter, types::ToSql};

use crate::compat::filter::anon_placeholders;

pub fn collect_rows<I, T, C, E>(rows: I) -> Result<C, E>
where
    I: Iterator<Item = Result<T, rusqlite::Error>>,
    C: FromIterator<T>,
    E: From<rusqlite::Error>,
{
    rows.collect::<Result<C, _>>().map_err(E::from)
}

pub fn fetch_by_in_clause<P, T, C, E, F>(
    conn: &Connection,
    keys: &[P],
    sql_template: &str,
    map_row: F,
) -> Result<C, E>
where
    P: ToSql,
    F: FnMut(&Row<'_>) -> Result<T, rusqlite::Error>,
    C: FromIterator<T>,
    E: From<rusqlite::Error>,
{
    if keys.is_empty() {
        return Ok(iter::empty::<T>().collect());
    }
    let placeholders = anon_placeholders(keys.len());
    let sql = sql_template.replace("{placeholders}", &placeholders);
    let mut stmt = conn.prepare(&sql).map_err(E::from)?;
    let rows = stmt
        .query_map(params_from_iter(keys.iter()), map_row)
        .map_err(E::from)?;
    collect_rows(rows)
}
