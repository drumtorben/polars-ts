use polars::prelude::*;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use rayon::prelude::*;

/// Groups a DataFrame by "unique_id" and aggregates the "y" column.
/// (Casting "unique_id" as Utf8 and "y" as Float32.)
fn get_groups(df: &DataFrame) -> Result<LazyFrame, PolarsError> {
    Ok(df.clone().lazy()
        .select([
            col("unique_id").cast(DataType::String),
            col("y").cast(DataType::Float32)
        ])
        .group_by([col("unique_id")])
        .agg([col("y")])
    )
}

/// Optimized DTW distance implementation using two rows.
/// This version uses O(m) memory instead of allocating the full (n+1)×(m+1) matrix.
fn dtw_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![f32::MAX; m + 1];
    let mut curr = vec![f32::MAX; m + 1];
    prev[0] = 0.0;
    
    for i in 1..=n {
        curr[0] = f32::MAX;
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

/// Optimized conversion of a grouped DataFrame into a HashMap mapping id -> Vec<f32>.
///
/// This version first collects the "unique_id" column and the list-of-f32
/// from the "y" column into two vectors. Then, using a parallel index loop,
/// it zips them together into a HashMap.
fn df_to_hashmap(df: &DataFrame) -> HashMap<String, Vec<f32>> {
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
    
    // Collect each list element into a Vec<f32>.
    let y_lists: Vec<Vec<f32>> = y_col
        .list()
        .expect("expected a List type for y")
        .into_iter()
        .map(|opt_series| {
            let series = opt_series.expect("null entry in 'y' list column");
            series
                .f32()
                .expect("expected a f32 Series inside the list")
                .into_no_null_iter()
                .collect::<Vec<f32>>()
        })
        .collect();
    
    // Sanity-check that we have the same number of ids and y vectors.
    assert_eq!(unique_ids.len(), y_lists.len(), "Mismatched lengths in unique_ids and y_lists");
    
    // Build the HashMap in parallel.
    let hashmap: HashMap<String, Vec<f32>> = (0..unique_ids.len())
        .into_par_iter()
        .map(|i| (unique_ids[i].clone(), y_lists[i].clone()))
        .collect();
    hashmap
}

/// Compute pairwise DTW distances between time series in two DataFrames,
/// using extensive parallelism.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "dtw".
#[pyfunction]
pub fn compute_pairwise_dtw(input1: PyDataFrame, input2: PyDataFrame) -> PyResult<PyDataFrame> {
    // Convert PyDataFrames to Polars DataFrames.
    let df_a: DataFrame = input1.into();
    let df_b: DataFrame = input2.into();

    // Group each DataFrame by "unique_id" and aggregate the "y" column.
    let grouped_a = get_groups(&df_a).unwrap().collect().unwrap();
    let grouped_b = get_groups(&df_b).unwrap().collect().unwrap();

    // Build HashMaps mapping unique_id -> time series (Vec<f32>).
    let map_a = df_to_hashmap(&grouped_a);
    let map_b = df_to_hashmap(&grouped_b);

    // Compute all pairwise DTW distances.
    // The outer loop (over map_a) is done in parallel.
    let results: Vec<(String, String, f32)> = map_a.par_iter().flat_map(|(id1, series1)| {
        map_b.iter().map(move |(id2, series2)| {
            let distance = dtw_distance(series1, series2);
            (id1.clone(), id2.clone(), distance)
        }).collect::<Vec<_>>()
    }).collect();

    // Build output columns.
    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let dtw_vals: Vec<f32> = results.iter().map(|(_, _, dtw)| *dtw).collect();

    // Create a new Polars DataFrame.
    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("dtw".into(), dtw_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    Ok(PyDataFrame(out_df))
}
