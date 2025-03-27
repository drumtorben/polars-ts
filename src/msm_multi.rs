use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

/// Compute Manhattan distance between two vectors.
fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Compute squared Euclidean distance between two vectors.
/// This function mimics the _univariate_squared_distance in the Python code.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| {
        let diff = x - y;
        diff * diff
    }).sum()
}

/// Compute the dependent cost given three vectors, following the Python logic.
/// Let x be the current observation, y and z be consecutive observations:
///
///   let diameter = squared_distance(y, z)
///   let mid = (y + z) / 2  (computed element-wise)
///   let distance_to_mid = squared_distance(mid, x)
///
///   if distance_to_mid <= (diameter / 2) { cost = c }
///   else { cost = c + min(squared_distance(y, x), squared_distance(z, x)) }
fn cost_dependent(x: &[f64], y: &[f64], z: &[f64], c: f64) -> f64 {
    let diameter = squared_distance(y, z);
    let mid: Vec<f64> = y.iter().zip(z).map(|(a, b)| (a + b) / 2.0).collect();
    let distance_to_mid = squared_distance(&mid, x);
    if distance_to_mid <= (diameter / 2.0) {
        c
    } else {
        let dist_to_y = squared_distance(y, x);
        let dist_to_z = squared_distance(z, x);
        c + dist_to_y.min(dist_to_z)
    }
}

/// Optimized MSM distance for multivariate time series using two rows of memory.
/// Each time series is represented as a slice of observations (Vec<Vec<f64>>).
///
/// Recurrence:
///   dp[0,0] = manhattan_distance(a[0], b[0])
///   dp[i,0] = dp[i-1,0] + cost_dependent(a[i], a[i-1], b[0], c)
///   dp[0,j] = dp[0,j-1] + cost_dependent(b[j], b[j-1], a[0], c)
/// and for i,j ≥ 1:
///   dp[i,j] = min{
///      dp[i-1,j-1] + manhattan_distance(a[i], b[j]),      // match
///      dp[i-1,j]   + cost_dependent(a[i], a[i-1], b[j], c),  // delete
///      dp[i,j-1]   + cost_dependent(b[j], a[i], b[j-1], c)   // insert
///   }
fn msm_distance(a: &[Vec<f64>], b: &[Vec<f64>], c: f64) -> f64 {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut prev = vec![f64::MAX; m];
    let mut curr = vec![f64::MAX; m];

    // dp[0,0]
    prev[0] = manhattan_distance(&a[0], &b[0]);

    // First row: compare b elements against the first observation of a.
    for j in 1..m {
        let cost = cost_dependent(&b[j], &b[j - 1], &a[0], c);
        prev[j] = prev[j - 1] + cost;
    }

    // Main dynamic programming loop.
    for i in 1..n {
        let cost = cost_dependent(&a[i], &a[i - 1], &b[0], c);
        curr[0] = prev[0] + cost;

        for j in 1..m {
            let d1 = prev[j - 1] + manhattan_distance(&a[i], &b[j]); // match
            let d2 = prev[j] + cost_dependent(&a[i], &a[i - 1], &b[j], c); // delete
            let d3 = curr[j - 1] + cost_dependent(&b[j], &a[i], &b[j - 1], c); // insert
            curr[j] = d1.min(d2).min(d3);
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m - 1]
}

/// Groups a DataFrame by "unique_id" and aggregates all dimension columns.
/// Assumes that besides the "unique_id" column, every other column is a numeric dimension.
fn get_groups(df: &DataFrame) -> Result<LazyFrame, PolarsError> {
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
fn df_to_hashmap(df: &DataFrame) -> HashMap<String, Vec<Vec<f64>>> {
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

/// Compute pairwise MSM distances between multivariate time series in two DataFrames,
/// using parallelism.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and multiple y-columns (e.g., y1, y2, y3).
/// * `input2` - Second PyDataFrame with columns "unique_id" and multiple y-columns.
/// * `c` - Cost parameter for MSM distance (default 1.0).
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "msm".
#[pyfunction]
pub fn compute_pairwise_msm_multi(
    input1: PyDataFrame,
    input2: PyDataFrame,
    c: Option<f64>
) -> PyResult<PyDataFrame> {
    let c_value = c.unwrap_or(1.0);

    // Convert PyDataFrames to Polars DataFrames.
    let df_1: DataFrame = input1.into();
    let df_2: DataFrame = input2.into();

    let uid_a_dtype = df_1.column("unique_id")
        .expect("df_1 must have unique_id")
        .dtype().clone();
    let uid_b_dtype = df_2.column("unique_id")
        .expect("df_2 must have unique_id")
        .dtype().clone();

    // Cast unique_id to String.
    let df_a = df_1.lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect().unwrap();
    let df_b = df_2.lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect().unwrap();

    // Group each DataFrame by "unique_id" and aggregate the y-columns.
    let grouped_a = get_groups(&df_a).unwrap().collect().unwrap();
    let grouped_b = get_groups(&df_b).unwrap().collect().unwrap();

    // Convert grouped DataFrames into HashMaps mapping unique_id -> multivariate time series.
    let raw_map_a = df_to_hashmap(&grouped_a);
    let raw_map_b = df_to_hashmap(&grouped_b);

    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    let left_series_by_key: Vec<(&String, &Vec<Vec<f64>>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<Vec<f64>>)> = map_b.iter().collect();

    // Compute pairwise MSM distances.
    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            let map_a = Arc::clone(&map_a);
            let map_b = Arc::clone(&map_b);

            right_series_by_key
                .par_iter()
                .filter_map(move |&(right_key, right_series)| {
                    // Skip self-comparisons.
                    if left_key == right_key {
                        return None;
                    }
                    // Avoid duplicate comparisons.
                    if map_b.contains_key(left_key) && map_a.contains_key(right_key) {
                        if left_key >= right_key {
                            return None;
                        }
                    }
                    let distance = msm_distance(left_series, right_series, c_value);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    // Build output DataFrame.
    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let msm_vals: Vec<f64> = results.iter().map(|(_, _, msm)| *msm).collect();

    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("msm_multi".into(), msm_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ])
        .collect()
        .unwrap();
    Ok(PyDataFrame(casted_out_df))
}
