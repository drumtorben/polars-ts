use polars::prelude::*;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use rayon::prelude::*;

use crate::utils::{get_groups, df_to_hashmap};

/// LCSS (Longest Common Subsequence) distance.
/// Two points match if their absolute difference is within `epsilon`.
/// Returns 1 - (LCSS_length / min(n, m)), so 0 = identical, 1 = no matches.
/// O(m) memory.
fn lcss_distance(a: &[f64], b: &[f64], epsilon: f64) -> f64 {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return 1.0;
    }

    let mut prev = vec![0_usize; m + 1];
    let mut curr = vec![0_usize; m + 1];

    for i in 1..=n {
        for j in 1..=m {
            if (a[i - 1] - b[j - 1]).abs() <= epsilon {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        // Reset curr for next row
        for j in 0..=m {
            curr[j] = 0;
        }
    }

    let lcss_len = prev[m] as f64;
    let min_len = n.min(m) as f64;
    1.0 - (lcss_len / min_len)
}

/// Compute pairwise LCSS distances between time series in two DataFrames.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
/// * `epsilon` - Matching threshold (default 1.0). Two points match if |a[i] - b[j]| <= epsilon.
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "lcss".
#[pyfunction]
#[pyo3(signature = (input1, input2, epsilon=None))]
pub fn compute_pairwise_lcss(input1: PyDataFrame, input2: PyDataFrame, epsilon: Option<f64>) -> PyResult<PyDataFrame> {
    let eps = epsilon.unwrap_or(1.0);

    let df_1: DataFrame = input1.into();
    let df_2: DataFrame = input2.into();

    let uid_a_dtype = df_1.column("unique_id")
        .expect("df_a must have unique_id")
        .dtype().clone();

    let uid_b_dtype = df_2.column("unique_id")
        .expect("df_b must have unique_id")
        .dtype().clone();

    let df_a = df_1
        .lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect().unwrap();

    let df_b = df_2
        .lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect().unwrap();

    let grouped_a = get_groups(&df_a).unwrap().collect().unwrap();
    let grouped_b = get_groups(&df_b).unwrap().collect().unwrap();

    let raw_map_a = df_to_hashmap(&grouped_a);
    let raw_map_b = df_to_hashmap(&grouped_b);

    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    let left_series_by_key: Vec<(&String, &Vec<f64>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<f64>)> = map_b.iter().collect();

    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            let map_a = Arc::clone(&map_a);
            let map_b = Arc::clone(&map_b);
            let epsilon = eps;

            right_series_by_key
                .par_iter()
                .filter_map(move |&(right_key, right_series)| {
                    if left_key == right_key {
                        return None;
                    }
                    if map_b.contains_key(left_key) && map_a.contains_key(right_key) {
                        if left_key >= right_key {
                            return None;
                        }
                    }
                    let distance = lcss_distance(left_series, right_series, epsilon);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let lcss_vals: Vec<f64> = results.iter().map(|(_, _, v)| *v).collect();

    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("lcss".into(), lcss_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ]).collect().unwrap();
    Ok(PyDataFrame(casted_out_df))
}
