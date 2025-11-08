use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

/// Distance metric for DTW cost calculations.
#[derive(Clone, Copy)]
enum DistanceMetric {
    Manhattan,
    Euclidean,
}

/// Groups a DataFrame by "unique_id" and aggregates all dimension columns.
/// Assumes that besides the "unique_id" column, every other column is a numeric dimension.
fn get_groups_multivariate(df: &DataFrame) -> Result<LazyFrame, PolarsError> {
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

/// Converts a grouped DataFrame into a HashMap mapping unique_id -> multivariate time series.
/// Each row in the DataFrame must have a "unique_id" column and one list column per dimension.
/// For each unique_id the time series is represented as Vec<Vec<f64>> where each inner Vec is a data point
/// (with one entry per dimension).
fn df_to_hashmap_multivariate(df: &DataFrame) -> HashMap<String, Vec<Vec<f64>>> {
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
