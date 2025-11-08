use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use rayon::prelude::*;

/// Groups a DataFrame by "unique_id" and aggregates the "y" column.
/// (Casting "unique_id" as Utf8 and "y" as Float64.)
fn get_groups(df: &DataFrame) -> Result<LazyFrame, PolarsError> {
    Ok(df.clone().lazy()
        .select([
            col("unique_id").cast(DataType::String),
            col("y").cast(DataType::Float64)
        ])
        .group_by([col("unique_id")])
        .agg([col("y")])
    )
}

/// Compute the derivative of a time series using the method from Keogh & Pazzani (2001).
/// 
/// The derivative is calculated as:
/// q'(i) = (q(i) - q(i-1) + ((q(i+1) - q(i-1))/2)) / 2
/// 
/// This implementation handles the boundary conditions by returning a vector with length len(q) - 2.
fn compute_derivative(q: &[f64]) -> Vec<f64> {
    if q.len() < 3 {
        return Vec::new(); // Not enough points to compute derivative
    }
    
    // Preallocate output vector
    let mut derivative = Vec::with_capacity(q.len() - 2);
    
    for i in 1..q.len() - 1 {
        // Calculate the derivative at point i
        // q'(i) = (q(i) - q(i-1) + ((q(i+1) - q(i-1))/2)) / 2
        let term1 = q[i] - q[i-1];
        let term2 = (q[i+1] - q[i-1]) / 2.0;
        let derivative_i = (term1 + term2) / 2.0;
        
        // Alternatively, simplified as in the Python code:
        // let derivative_i = 0.25 * q[i+1] + 0.5 * q[i] - 0.75 * q[i-1];
        
        derivative.push(derivative_i);
    }
    
    derivative
}

/// Optimized DTW distance implementation using two rows.
/// This version uses O(m) memory instead of allocating the full (n+1)×(m+1) matrix.
fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    
    // Handle edge cases
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }
    
    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::MAX;
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            // Choose the best previous cell.
            let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            curr[j] = cost + min_prev;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Computes the DDTW (Derivative DTW) distance between two time series.
/// First calculates the derivative of each series, then computes DTW on those derivatives.
fn ddtw_distance(a: &[f64], b: &[f64]) -> f64 {
    // Calculate derivatives
    let a_derivative = compute_derivative(a);
    let b_derivative = compute_derivative(b);
    
    // Handle edge cases
    if a_derivative.is_empty() || b_derivative.is_empty() {
        return f64::INFINITY;
    }
    
    // Compute DTW on derivatives
    dtw_distance(&a_derivative, &b_derivative)
}

/// Optimized conversion of a grouped DataFrame into a HashMap mapping id -> Vec<f64>.
fn df_to_hashmap(df: &DataFrame) -> HashMap<String, Vec<f64>> {
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

/// Compute pairwise DDTW distances between time series in two DataFrames,
/// using extensive parallelism.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "ddtw".
#[pyfunction]
pub fn compute_pairwise_ddtw(input1: PyDataFrame, input2: PyDataFrame) -> PyResult<PyDataFrame> {
    // Convert PyDataFrames to Polars DataFrames.
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

    // Group each DataFrame by "unique_id" and aggregate the "y" column.
    let grouped_a = get_groups(&df_a).unwrap().collect().unwrap();
    let grouped_b = get_groups(&df_b).unwrap().collect().unwrap();

    // Build HashMaps mapping unique_id -> time series (Vec<f64>) for each input.
    let raw_map_a = df_to_hashmap(&grouped_a);
    let raw_map_b = df_to_hashmap(&grouped_b);

    // Wrap the maps in an Arc so that they can be shared safely across threads.
    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    // Create vectors of references for the keys and series.
    let left_series_by_key: Vec<(&String, &Vec<f64>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<f64>)> = map_b.iter().collect();

    // Compute pairwise DDTW distances: id_1 always comes from left, id_2 from right.
    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            // Clone the Arc pointers for use in the inner closure.
            let map_a = Arc::clone(&map_a);
            let map_b = Arc::clone(&map_b);
            right_series_by_key
                .par_iter()
                .filter_map(move |&(right_key, right_series)| {
                    // Skip self-comparisons.
                    if left_key == right_key {
                        return None;
                    }
                    // If both keys are common (i.e. appear in both maps), enforce an ordering to avoid duplicates.
                    if map_b.contains_key(left_key) && map_a.contains_key(right_key) {
                        if left_key >= right_key {
                            return None;
                        }
                    }
                    // Compute the DDTW distance.
                    let distance = ddtw_distance(left_series, right_series);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    // Build output columns.
    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let ddtw_vals: Vec<f64> = results.iter().map(|(_, _, ddtw)| *ddtw).collect();

    // Create a new Polars DataFrame.
    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("ddtw".into(), ddtw_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ]).collect().unwrap();
    Ok(PyDataFrame(casted_out_df))
}
