use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::sync::Arc;
use rayon::prelude::*;

/// Cast a column in-place to the given DataType, returning a new DataFrame.
pub fn cast_column(df: &DataFrame, col_name: &str, dtype: DataType) -> Result<DataFrame, PolarsError> {
    let mut df = df.clone();
    let casted = df.column(col_name)?.cast(&dtype)?.take_materialized_series();
    let _ = df.replace(col_name, casted)?;
    Ok(df)
}

/// Validate that a required column exists in the DataFrame.
fn validate_column_exists(df: &DataFrame, col_name: &str) -> PyResult<()> {
    if df.column(col_name).is_err() {
        return Err(PyKeyError::new_err(format!(
            "Missing required column '{col_name}' in DataFrame"
        )));
    }
    Ok(())
}

/// Groups a DataFrame by "unique_id" and aggregates the "y" column into lists.
/// Returns a DataFrame with columns: unique_id (String), y (List<f64>).
pub fn get_groups(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    // Cast columns first
    let df = cast_column(df, "unique_id", DataType::String)?;
    let df = cast_column(&df, "y", DataType::Float64)?;

    // Select only the columns we need
    let df = df.select(["unique_id", "y"])?;

    // Group by unique_id and aggregate all columns into lists
    #[allow(deprecated)]
    let mut result = df.group_by(["unique_id"])?.agg_list()?;
    // agg_list renames columns to "{name}_agg_list", rename back
    result.rename("y_agg_list", "y".into())?;
    Ok(result)
}

/// Optimized conversion of a grouped DataFrame into a HashMap mapping id -> Vec<f64>.
pub fn df_to_hashmap(df: &DataFrame) -> PyResult<HashMap<String, Vec<f64>>> {
    let unique_id_col = df.column("unique_id")
        .map_err(|e| PyKeyError::new_err(format!("Missing column 'unique_id': {e}")))?;
    let y_col = df.column("y")
        .map_err(|e| PyKeyError::new_err(format!("Missing column 'y': {e}")))?;

    let unique_ids: Vec<String> = unique_id_col
        .str()
        .map_err(|e| PyValueError::new_err(format!("Column 'unique_id' must be string type: {e}")))?
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    let y_lists: Vec<Vec<f64>> = y_col
        .list()
        .map_err(|e| PyValueError::new_err(format!("Column 'y' must be list type: {e}")))?
        .into_iter()
        .map(|opt_series| {
            let series = opt_series.ok_or_else(|| {
                PyValueError::new_err("Null entry found in 'y' list column. Ensure no null values in 'y'.")
            })?;
            let chunked = series.f64().map_err(|e| {
                PyValueError::new_err(format!("Values in 'y' column must be f64: {e}"))
            })?;
            Ok(chunked.into_no_null_iter().collect::<Vec<f64>>())
        })
        .collect::<PyResult<Vec<Vec<f64>>>>()?;

    if unique_ids.len() != y_lists.len() {
        return Err(PyValueError::new_err(format!(
            "Mismatched lengths: {} unique_ids vs {} y_lists",
            unique_ids.len(),
            y_lists.len()
        )));
    }

    let hashmap: HashMap<String, Vec<f64>> = unique_ids
        .into_iter()
        .zip(y_lists)
        .collect();
    Ok(hashmap)
}

/// Groups a DataFrame by "unique_id" and aggregates all dimension columns into lists.
pub fn get_groups_multivariate(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    let dims: Vec<String> = df.get_column_names()
        .iter()
        .filter(|&&name| name != "unique_id")
        .map(|s| s.to_string())
        .collect();

    // Cast unique_id to String and dimension columns to Float64
    let mut df = cast_column(df, "unique_id", DataType::String)?;
    for dim in &dims {
        df = cast_column(&df, dim.as_str(), DataType::Float64)?;
    }

    // Group by unique_id and aggregate all columns into lists
    #[allow(deprecated)]
    let mut result = df.group_by(["unique_id"])?.agg_list()?;
    // agg_list renames columns to "{name}_agg_list", rename back
    for dim in &dims {
        let agg_name = format!("{}_agg_list", dim);
        result.rename(agg_name.as_str(), dim.clone().into())?;
    }
    Ok(result)
}

/// Converts a grouped DataFrame into a HashMap mapping unique_id -> multivariate time series.
pub fn df_to_hashmap_multivariate(df: &DataFrame) -> PyResult<HashMap<String, Vec<Vec<f64>>>> {
    let unique_id_col = df.column("unique_id")
        .map_err(|e| PyKeyError::new_err(format!("Missing column 'unique_id': {e}")))?;
    let unique_ids: Vec<String> = unique_id_col
        .str()
        .map_err(|e| PyValueError::new_err(format!("Column 'unique_id' must be string type: {e}")))?
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    let dims: Vec<&str> = df.get_column_names()
        .iter()
        .filter(|&&name| name != "unique_id")
        .map(|s| s.as_str())
        .collect();

    let mut dims_data: Vec<Vec<Vec<f64>>> = Vec::with_capacity(dims.len());
    for d in dims.iter() {
        let col_series = df.column(d)
            .map_err(|e| PyKeyError::new_err(format!("Missing dimension column '{d}': {e}")))?;
        let lists: Vec<Vec<f64>> = col_series
            .list()
            .map_err(|e| PyValueError::new_err(format!("Column '{d}' must be list type: {e}")))?
            .into_iter()
            .map(|opt_series| {
                let series = opt_series.ok_or_else(|| {
                    PyValueError::new_err(format!("Null entry in dimension column '{d}'"))
                })?;
                let chunked = series.f64().map_err(|e| {
                    PyValueError::new_err(format!("Values in column '{d}' must be f64: {e}"))
                })?;
                Ok(chunked.into_no_null_iter().collect::<Vec<f64>>())
            })
            .collect::<PyResult<Vec<Vec<f64>>>>()?;
        dims_data.push(lists);
    }

    let mut hashmap = HashMap::new();
    let num_series = unique_ids.len();
    for i in 0..num_series {
        let series_len = dims_data[0][i].len();
        let mut series: Vec<Vec<f64>> = Vec::with_capacity(series_len);
        for t in 0..series_len {
            let point: Vec<f64> = dims_data.iter().map(|dim| dim[i][t]).collect();
            series.push(point);
        }
        hashmap.insert(unique_ids[i].clone(), series);
    }
    Ok(hashmap)
}

/// Generic pairwise computation for univariate distance functions.
///
/// Handles: column validation, casting, grouping, HashMap construction,
/// Arc wrapping, parallel pairwise iteration with deduplication, output assembly.
pub fn compute_pairwise<F>(
    input1: PyDataFrame,
    input2: PyDataFrame,
    distance_col: &str,
    distance_fn: F,
) -> PyResult<PyDataFrame>
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync,
{
    let df_1: DataFrame = input1.into();
    let df_2: DataFrame = input2.into();

    validate_column_exists(&df_1, "unique_id")?;
    validate_column_exists(&df_1, "y")?;
    validate_column_exists(&df_2, "unique_id")?;
    validate_column_exists(&df_2, "y")?;

    let uid_a_dtype = df_1.column("unique_id")
        .map_err(|e| PyKeyError::new_err(e.to_string()))?
        .dtype().clone();
    let uid_b_dtype = df_2.column("unique_id")
        .map_err(|e| PyKeyError::new_err(e.to_string()))?
        .dtype().clone();

    let df_a = cast_column(&df_1, "unique_id", DataType::String)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let df_b = cast_column(&df_2, "unique_id", DataType::String)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let grouped_a = get_groups(&df_a)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let grouped_b = get_groups(&df_b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let raw_map_a = df_to_hashmap(&grouped_a)?;
    let raw_map_b = df_to_hashmap(&grouped_b)?;

    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    let left_series_by_key: Vec<(&String, &Vec<f64>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<f64>)> = map_b.iter().collect();

    let distance_fn = Arc::new(distance_fn);
    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            let map_a = Arc::clone(&map_a);
            let map_b = Arc::clone(&map_b);
            let distance_fn = Arc::clone(&distance_fn);
            right_series_by_key
                .par_iter()
                .filter_map(move |&(right_key, right_series)| {
                    if left_key == right_key {
                        return None;
                    }
                    if map_b.contains_key(left_key) && map_a.contains_key(right_key) && left_key >= right_key {
                        return None;
                    }
                    let distance = distance_fn(left_series, right_series);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    build_output_df(&results, distance_col, &uid_a_dtype, &uid_b_dtype)
}

/// Generic pairwise computation for multivariate distance functions.
pub fn compute_pairwise_multivariate<F>(
    input1: PyDataFrame,
    input2: PyDataFrame,
    distance_col: &str,
    distance_fn: F,
) -> PyResult<PyDataFrame>
where
    F: Fn(&[Vec<f64>], &[Vec<f64>]) -> f64 + Send + Sync,
{
    let df_1: DataFrame = input1.into();
    let df_2: DataFrame = input2.into();

    validate_column_exists(&df_1, "unique_id")?;
    validate_column_exists(&df_2, "unique_id")?;

    let uid_a_dtype = df_1.column("unique_id")
        .map_err(|e| PyKeyError::new_err(e.to_string()))?
        .dtype().clone();
    let uid_b_dtype = df_2.column("unique_id")
        .map_err(|e| PyKeyError::new_err(e.to_string()))?
        .dtype().clone();

    let df_a = cast_column(&df_1, "unique_id", DataType::String)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let df_b = cast_column(&df_2, "unique_id", DataType::String)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let grouped_a = get_groups_multivariate(&df_a)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let grouped_b = get_groups_multivariate(&df_b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let raw_map_a = df_to_hashmap_multivariate(&grouped_a)?;
    let raw_map_b = df_to_hashmap_multivariate(&grouped_b)?;

    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    let left_series_by_key: Vec<(&String, &Vec<Vec<f64>>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<Vec<f64>>)> = map_b.iter().collect();

    let distance_fn = Arc::new(distance_fn);
    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            let map_a = Arc::clone(&map_a);
            let map_b = Arc::clone(&map_b);
            let distance_fn = Arc::clone(&distance_fn);
            right_series_by_key
                .par_iter()
                .filter_map(move |&(right_key, right_series)| {
                    if left_key == right_key {
                        return None;
                    }
                    if map_b.contains_key(left_key) && map_a.contains_key(right_key) && left_key >= right_key {
                        return None;
                    }
                    let distance = distance_fn(left_series, right_series);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    build_output_df(&results, distance_col, &uid_a_dtype, &uid_b_dtype)
}

/// Build the output DataFrame from pairwise results, casting IDs back to original dtypes.
fn build_output_df(
    results: &[(String, String, f64)],
    distance_col: &str,
    uid_a_dtype: &DataType,
    uid_b_dtype: &DataType,
) -> PyResult<PyDataFrame> {
    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let vals: Vec<f64> = results.iter().map(|(_, _, v)| *v).collect();

    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new(distance_col.into(), vals),
    ];
    let mut out_df = DataFrame::new(columns)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let id1_casted = out_df.column("id_1")
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .cast(uid_a_dtype)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .take_materialized_series();
    let _ = out_df.replace("id_1", id1_casted)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let id2_casted = out_df.column("id_2")
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .cast(uid_b_dtype)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .take_materialized_series();
    let _ = out_df.replace("id_2", id2_casted)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyDataFrame(out_df))
}
