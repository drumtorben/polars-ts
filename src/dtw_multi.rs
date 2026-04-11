use polars::prelude::*;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use crate::utils::{get_groups_multivariate, df_to_hashmap_multivariate};

/// Distance metric for DTW cost calculations.
#[derive(Clone, Copy)]
enum DistanceMetric {
    Manhattan,
    Euclidean,
}

/// Computes the DTW distance between two multivariate time series using the specified metric.
/// Each time series is represented as a slice of points (Vec<f64>).
/// For each pair of points, the cost is computed as follows:
/// - Manhattan: sum of absolute differences.
/// - Euclidean: Euclidean distance.
fn dtw_distance_multivariate(a: &[Vec<f64>], b: &[Vec<f64>], metric: DistanceMetric) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::MAX;
        for j in 1..=m {
            // Compute the cost between points using the selected metric.
            let cost: f64 = match metric {
                DistanceMetric::Manhattan => {
                    a[i - 1].iter().zip(b[j - 1].iter())
                        .map(|(x, y)| (x - y).abs())
                        .sum()
                },
                DistanceMetric::Euclidean => {
                    let sum_sq: f64 = a[i - 1].iter().zip(b[j - 1].iter())
                        .map(|(x, y)| (x - y).powi(2))
                        .sum();
                    sum_sq.sqrt()
                },
            };
            let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            curr[j] = cost + min_prev;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Compute pairwise multivariate DTW distances between time series in two DataFrames,
/// using extensive parallelism.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and one or more dimension columns (e.g. y1, y2, y3, ...).
/// * `input2` - Second PyDataFrame with columns "unique_id" and one or more dimension columns.
/// * `metric` - Optional string to choose the distance metric: "manhattan" or "euclidean".
///              Defaults to "manhattan" if not provided or unrecognized.
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "dtw".
#[pyfunction]
#[pyo3(signature = (input1, input2, metric=None))]
pub fn compute_pairwise_dtw_multi(input1: PyDataFrame, input2: PyDataFrame, metric: Option<String>) -> PyResult<PyDataFrame> {
    // Determine the distance metric.
    let distance_metric = match metric.as_deref() {
        Some("euclidean") => DistanceMetric::Euclidean,
        _ => DistanceMetric::Manhattan,
    };

    // Convert PyDataFrames to Polars DataFrames.
    let df_1: DataFrame = input1.into();
    let df_2: DataFrame = input2.into();

    let uid_a_dtype = df_1.column("unique_id")
        .expect("df_a must have unique_id")
        .dtype().clone();

    let uid_b_dtype = df_2.column("unique_id")
        .expect("df_b must have unique_id")
        .dtype().clone();

    // Cast unique_id columns to string.
    let df_a = df_1
        .lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect()
        .unwrap();
    let df_b = df_2
        .lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect()
        .unwrap();

    // Group each DataFrame by "unique_id" and aggregate all dimension columns.
    let grouped_a = get_groups_multivariate(&df_a).unwrap().collect().unwrap();
    let grouped_b = get_groups_multivariate(&df_b).unwrap().collect().unwrap();

    // Build HashMaps mapping unique_id -> multivariate time series for each input.
    let raw_map_a = df_to_hashmap_multivariate(&grouped_a);
    let raw_map_b = df_to_hashmap_multivariate(&grouped_b);

    // Wrap the maps in an Arc so they can be shared safely across threads.
    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    // Create vectors of references for keys and series.
    let left_series_by_key: Vec<(&String, &Vec<Vec<f64>>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<Vec<f64>>)> = map_b.iter().collect();

    // Compute pairwise DTW distances: id_1 always comes from left, id_2 from right.
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
                    // If both keys are common (i.e. appear in both maps), enforce an ordering to avoid duplicate pairs.
                    if map_b.contains_key(left_key) && map_a.contains_key(right_key) {
                        if left_key >= right_key {
                            return None;
                        }
                    }
                    // Compute the multivariate DTW distance with the selected metric.
                    let distance = dtw_distance_multivariate(left_series, right_series, distance_metric);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    // Build output columns.
    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let dtw_vals: Vec<f64> = results.iter().map(|(_, _, dtw)| *dtw).collect();

    // Create a new Polars DataFrame.
    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("dtw_multi".into(), dtw_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();

    // Cast id columns back to the original dtypes.
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ])
        .collect()
        .unwrap();
    Ok(PyDataFrame(casted_out_df))
}
