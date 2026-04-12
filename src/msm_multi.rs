use polars::prelude::*;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use crate::utils::{get_groups_multivariate, df_to_hashmap_multivariate, cast_column};

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
/// and for i,j >= 1:
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
#[pyo3(signature = (input1, input2, c=None))]
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
    let df_a = cast_column(&df_1, "unique_id", DataType::String).unwrap();
    let df_b = cast_column(&df_2, "unique_id", DataType::String).unwrap();

    // Group each DataFrame by "unique_id" and aggregate the y-columns.
    let grouped_a = get_groups_multivariate(&df_a).unwrap();
    let grouped_b = get_groups_multivariate(&df_b).unwrap();

    // Convert grouped DataFrames into HashMaps mapping unique_id -> multivariate time series.
    let raw_map_a = df_to_hashmap_multivariate(&grouped_a);
    let raw_map_b = df_to_hashmap_multivariate(&grouped_b);

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
    let mut casted_out_df = out_df;
    let id1_casted = casted_out_df.column("id_1").unwrap().cast(&uid_a_dtype).unwrap().take_materialized_series();
    let _ = casted_out_df.replace("id_1", id1_casted).unwrap();
    let id2_casted = casted_out_df.column("id_2").unwrap().cast(&uid_b_dtype).unwrap().take_materialized_series();
    let _ = casted_out_df.replace("id_2", id2_casted).unwrap();
    Ok(PyDataFrame(casted_out_df))
}
