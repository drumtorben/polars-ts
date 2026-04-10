use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use rayon::prelude::*;

/// TWE (Time Warp Edit) distance.
/// Combines edit distance with a stiffness penalty for time warping.
/// Parameters:
/// - `nu`: stiffness parameter (default 0.001). Higher = less warping allowed.
/// - `lambda`: penalty for delete/insert operations (default 1.0).
/// O(m) memory.
fn twe_distance(a: &[f64], b: &[f64], nu: f64, lambda: f64) -> f64 {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];

    prev[0] = 0.0;
    for j in 1..=m {
        prev[j] = prev[j - 1] + (b[j - 1] - if j > 1 { b[j - 2] } else { 0.0 }).abs() + nu + lambda;
    }

    for i in 1..=n {
        let a_i = a[i - 1];
        let a_prev = if i > 1 { a[i - 2] } else { 0.0 };

        curr[0] = prev[0] + (a_i - a_prev).abs() + nu + lambda;

        for j in 1..=m {
            let b_j = b[j - 1];
            let b_prev = if j > 1 { b[j - 2] } else { 0.0 };

            // Match: align a[i] with b[j]
            let d_match = prev[j - 1]
                + (a_i - b_j).abs()
                + (a_prev - b_prev).abs()
                + nu * ((i as f64 - j as f64).abs()).min(2.0 * nu);

            // Delete: a[i] is unmatched
            let d_delete = prev[j]
                + (a_i - a_prev).abs()
                + nu + lambda;

            // Insert: b[j] is unmatched
            let d_insert = curr[j - 1]
                + (b_j - b_prev).abs()
                + nu + lambda;

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

/// Compute pairwise TWE distances between time series in two DataFrames.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
/// * `nu` - Stiffness parameter (default 0.001). Controls how much time warping is penalized.
/// * `lambda` - Penalty for delete/insert operations (default 1.0).
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "twe".
#[pyfunction]
#[pyo3(signature = (input1, input2, nu=None, lambda=None))]
pub fn compute_pairwise_twe(input1: PyDataFrame, input2: PyDataFrame, nu: Option<f64>, lambda: Option<f64>) -> PyResult<PyDataFrame> {
    let nu_value = nu.unwrap_or(0.001);
    let lambda_value = lambda.unwrap_or(1.0);

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
            let nu = nu_value;
            let lambda = lambda_value;

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
                    let distance = twe_distance(left_series, right_series, nu, lambda);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let twe_vals: Vec<f64> = results.iter().map(|(_, _, v)| *v).collect();

    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("twe".into(), twe_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ]).collect().unwrap();
    Ok(PyDataFrame(casted_out_df))
}
