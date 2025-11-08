use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use rayon::prelude::*;

/// Helper function to calculate the MSM cost
fn msm_cost(x: f64, y: f64, z: f64, c: f64) -> f64 {
    if (y <= x && x <= z) || (y >= x && x >= z) {
        return c;
    }
    c + (x - y).abs().min((x - z).abs())
}

/// Optimized MSM distance implementation using two rows.
/// This version uses O(m) memory instead of allocating the full n×m matrix.
fn msm_distance(a: &[f64], b: &[f64], c: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    
    // Early return for empty sequences
    if n == 0 || m == 0 {
        return 0.0;
    }
    
    let mut prev = vec![f64::MAX; m];
    let mut curr = vec![f64::MAX; m];
    
    // Initialize first cell
    prev[0] = (a[0] - b[0]).abs();
    
    // Initialize first row
    for j in 1..m {
        let cost = msm_cost(b[j], a[0], b[j-1], c);
        prev[j] = prev[j-1] + cost;
    }
    
    // Main dynamic programming loop
    for i in 1..n {
        // Initialize first column of current row
        let cost = msm_cost(a[i], a[i-1], b[0], c);
        curr[0] = prev[0] + cost;
        
        for j in 1..m {
            // Calculate the three possible transitions
            let d1 = prev[j-1] + (a[i] - b[j]).abs();  // Match
            let d2 = prev[j] + msm_cost(a[i], a[i-1], b[j], c);  // Delete
            let d3 = curr[j-1] + msm_cost(b[j], a[i], b[j-1], c);  // Insert
            
            // Take the minimum cost
            curr[j] = d1.min(d2).min(d3);
        }
        
        // Swap rows for next iteration
        std::mem::swap(&mut prev, &mut curr);
    }
    
    // Final MSM distance is in the bottom-right corner
    prev[m-1]
}

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

/// Compute pairwise MSM distances between time series in two DataFrames,
/// using extensive parallelism.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
/// * `c` - Cost parameter for MSM distance (default 1.0).
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "msm".
#[pyfunction]
pub fn compute_pairwise_msm(input1: PyDataFrame, input2: PyDataFrame, c: Option<f64>) -> PyResult<PyDataFrame> {
    // Set default value for c parameter
    let c_value = c.unwrap_or(1.0);

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

    // Create vectors of references for the keys and series. These are now references into the
    // data held by the Arc-ed maps.
    let left_series_by_key: Vec<(&String, &Vec<f64>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<f64>)> = map_b.iter().collect();

    // Compute pairwise MSM distances: id_1 always comes from left, id_2 from right.
    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            // Clone the Arc pointers for use in the inner closure.
            let map_a = Arc::clone(&map_a);
            let map_b = Arc::clone(&map_b);
            // Capture g_value for the inner closure
            let c = c_value;

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
                    // Compute the MSM distance.
                    let distance = msm_distance(left_series, right_series, c);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    // Build output columns.
    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let msm_vals: Vec<f64> = results.iter().map(|(_, _, msm)| *msm).collect();

    // Create a new Polars DataFrame.
    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("msm".into(), msm_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ]).collect().unwrap();
    Ok(PyDataFrame(casted_out_df))
}
