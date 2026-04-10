use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use rayon::prelude::*;

/// ERP (Edit Distance with Real Penalty) distance.
/// Uses a gap value `g` as the reference point for insertions/deletions.
/// O(m) memory.
fn erp_distance(a: &[f64], b: &[f64], g: f64) -> f64 {
    let n = a.len();
    let m = b.len();

    let mut prev = vec![0.0_f64; m + 1];
    let mut curr = vec![0.0_f64; m + 1];

    // Initialize first row: cost of inserting all of b[0..j]
    for j in 1..=m {
        prev[j] = prev[j - 1] + (b[j - 1] - g).abs();
    }

    let mut first_col_cost = 0.0_f64;
    for i in 1..=n {
        first_col_cost += (a[i - 1] - g).abs();
        curr[0] = first_col_cost;

        for j in 1..=m {
            let d_match = prev[j - 1] + (a[i - 1] - b[j - 1]).abs();
            let d_delete = prev[j] + (a[i - 1] - g).abs();
            let d_insert = curr[j - 1] + (b[j - 1] - g).abs();
            curr[j] = d_match.min(d_delete).min(d_insert);
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Groups a DataFrame by "unique_id" and aggregates the "y" column.
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
    let unique_id_col = df.column("unique_id").expect("expected column unique_id");
    let y_col = df.column("y").expect("expected column y");

    let unique_ids: Vec<String> = unique_id_col
        .str()
        .expect("expected utf8 column for unique_id")
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

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

    assert_eq!(unique_ids.len(), y_lists.len(), "Mismatched lengths in unique_ids and y_lists");

    let hashmap: HashMap<String, Vec<f64>> = (0..unique_ids.len())
        .into_par_iter()
        .map(|i| (unique_ids[i].clone(), y_lists[i].clone()))
        .collect();
    hashmap
}

/// Compute pairwise ERP distances between time series in two DataFrames.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
/// * `g` - Gap value (default 0.0). Points are compared against this reference when inserted/deleted.
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "erp".
#[pyfunction]
#[pyo3(signature = (input1, input2, g=None))]
pub fn compute_pairwise_erp(input1: PyDataFrame, input2: PyDataFrame, g: Option<f64>) -> PyResult<PyDataFrame> {
    let g_value = g.unwrap_or(0.0);

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
            let g = g_value;

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
                    let distance = erp_distance(left_series, right_series, g);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let erp_vals: Vec<f64> = results.iter().map(|(_, _, v)| *v).collect();

    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("erp".into(), erp_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ]).collect().unwrap();
    Ok(PyDataFrame(casted_out_df))
}
