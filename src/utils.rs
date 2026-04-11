use polars::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;

/// Groups a DataFrame by "unique_id" and aggregates the "y" column.
/// (Casting "unique_id" as String and "y" as Float64.)
pub fn get_groups(df: &DataFrame) -> Result<LazyFrame, PolarsError> {
    Ok(df.clone().lazy()
        .select([
            col("unique_id").cast(DataType::String),
            col("y").cast(DataType::Float64)
        ])
        .group_by([col("unique_id")])
        .agg([col("y")])
    )
}

/// Optimized conversion of a grouped DataFrame into a HashMap mapping id -> Vec<f64>.
///
/// This version first collects the "unique_id" column and the list-of-f64
/// from the "y" column into two vectors. Then, using a parallel index loop,
/// it zips them together into a HashMap.
pub fn df_to_hashmap(df: &DataFrame) -> HashMap<String, Vec<f64>> {
    // Retrieve the columns.
    let unique_id_col = df.column("unique_id").expect("expected column unique_id");
    let y_col = df.column("y").expect("expected column y");

    // Collect unique IDs into a Vec<String>.
    let unique_ids: Vec<String> = unique_id_col
        .str()
        .expect("expected utf8 column for unique_id")
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    // Collect each list element into a Vec<f64>.
    let y_lists: Vec<Vec<f64>> = y_col
        .list()
        .expect("expected a List type for y")
        .into_iter()
        .map(|opt_series| {
            let series = opt_series.expect("null entry in 'y' list column");
            series
                .f64()
                .expect("expected a f64 Series inside the list")
                .into_no_null_iter()
                .collect::<Vec<f64>>()
        })
        .collect();

    // Sanity-check that we have the same number of ids and y vectors.
    assert_eq!(unique_ids.len(), y_lists.len(), "Mismatched lengths in unique_ids and y_lists");

    // Build the HashMap in parallel.
    let hashmap: HashMap<String, Vec<f64>> = (0..unique_ids.len())
        .into_par_iter()
        .map(|i| (unique_ids[i].clone(), y_lists[i].clone()))
        .collect();
    hashmap
}

/// Groups a DataFrame by "unique_id" and aggregates all dimension columns.
/// Assumes that besides the "unique_id" column, every other column is a numeric dimension.
pub fn get_groups_multivariate(df: &DataFrame) -> Result<LazyFrame, PolarsError> {
    // Identify dimension columns: all columns except "unique_id"
    let dims: Vec<_> = df.get_column_names()
        .iter()
        .filter(|&&name| name != "unique_id")
        .map(|s| s.to_string())
        .collect();

    // Build aggregation expressions for each dimension column.
    let agg_exprs: Vec<Expr> = dims.iter()
        .map(|col_name| col(col_name).cast(DataType::Float64))
        .collect();

    // Group by unique_id and aggregate each dimension column into a list.
    Ok(df.clone().lazy()
        .select([col("unique_id").cast(DataType::String)]
            .into_iter()
            .chain(agg_exprs.iter().cloned())
            .collect::<Vec<_>>())
        .group_by([col("unique_id")])
        .agg(agg_exprs)
    )
}

/// Converts a grouped DataFrame into a HashMap mapping unique_id -> multivariate time series.
/// Each row in the DataFrame must have a "unique_id" column and one list column per dimension.
/// For each unique_id the time series is represented as Vec<Vec<f64>> where each inner Vec is a data point
/// (with one entry per dimension).
pub fn df_to_hashmap_multivariate(df: &DataFrame) -> HashMap<String, Vec<Vec<f64>>> {
    // Get the unique IDs.
    let unique_id_col = df.column("unique_id").expect("expected column unique_id");
    let unique_ids: Vec<String> = unique_id_col
        .str()
        .expect("expected Utf8 for unique_id")
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    // Identify dimension columns: all columns except "unique_id".
    let dims: Vec<&str> = df.get_column_names()
        .iter()
        .filter(|&&name| name != "unique_id")
        .map(|s| s.as_str())
        .collect();

    // For each dimension, extract the list-of-f64 values.
    // dims_data[d][i] gives the full list for unique_id[i] in dimension d.
    let mut dims_data: Vec<Vec<Vec<f64>>> = Vec::with_capacity(dims.len());
    for d in dims.iter() {
        let col_series = df.column(d).expect("expected dimension column");
        let lists: Vec<Vec<f64>> = col_series
            .list()
            .expect("expected list type in dimension column")
            .into_iter()
            .map(|opt_series| {
                let series = opt_series.expect("null entry in dimension list");
                series.f64()
                    .expect("expected f64 Series inside the list")
                    .into_no_null_iter()
                    .collect::<Vec<f64>>()
            })
            .collect();
        dims_data.push(lists);
    }

    // Build the multivariate time series for each unique_id.
    let mut hashmap = HashMap::new();
    let num_series = unique_ids.len();
    for i in 0..num_series {
        // For each unique_id, assume all dimensions have the same series length.
        let series_len = dims_data[0][i].len();
        let mut series: Vec<Vec<f64>> = Vec::with_capacity(series_len);
        for t in 0..series_len {
            let mut point: Vec<f64> = Vec::with_capacity(dims.len());
            for d in 0..dims.len() {
                point.push(dims_data[d][i][t]);
            }
            series.push(point);
        }
        hashmap.insert(unique_ids[i].clone(), series);
    }
    hashmap
}
