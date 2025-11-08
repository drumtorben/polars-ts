use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

// ... (all the necessary imports and your existing data-processing code)

// For demonstration, let's say we match on a string to pick the method.
#[pyfunction]
#[pyo3(text_signature = "(input1, input2, method)")]
pub fn compute_pairwise_dtw(input1: PyDataFrame,
                            input2: PyDataFrame,
                            method: &str) -> PyResult<PyDataFrame> {

    // Convert PyDataFrames -> Polars DataFrames
    let df_1: DataFrame = input1.into();
    let df_2: DataFrame = input2.into();

    // (Same type-casting code for unique_id as in your snippet) ...
    let df_a = df_1
        .lazy()
        .with_column(col("unique_id").cast(DataType::Utf8))
        .collect()
        .unwrap();
    let df_b = df_2
        .lazy()
        .with_column(col("unique_id").cast(DataType::Utf8))
        .collect()
        .unwrap();

    // Group by "unique_id" => gather into HashMaps
    let grouped_a = get_groups(&df_a).unwrap().collect().unwrap();
    let grouped_b = get_groups(&df_b).unwrap().collect().unwrap();
    let raw_map_a = df_to_hashmap(&grouped_a);
    let raw_map_b = df_to_hashmap(&grouped_b);

    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    // Prepare a (key, series) list for each map
    let left_series_by_key: Vec<(&String, &Vec<f64>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<f64>)> = map_b.iter().collect();

    // Convert the "method" string into our DtwMethod enum
    let chosen_method = match method {
        "standard" => DtwMethod::Standard,
        "fast" => DtwMethod::Fast { radius: 5 },
        "sakoe" => DtwMethod::SakoeChiba { window_size: 10 },
        "itakura" => DtwMethod::Itakura,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown DTW method: {}", method)
            ));
        }
    };

    let calculator = DtwCalculator::new(chosen_method);

    // Now do the pairwise comparison
    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            right_series_by_key
                .par_iter()
                .filter_map(|&(right_key, right_series)| {
                    // skip self-comparisons or duplicates if needed
                    if left_key == right_key {
                        return None;
                    }
                    // compute the distance
                    let distance = calculator.compute(left_series, right_series);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    // Build the output DataFrame
    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let dtw_vals: Vec<f64> = results.iter().map(|(_, _, dtw)| *dtw).collect();

    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("dtw".into(), dtw_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();

    // Convert back to the original dtype if needed, or just keep as is.
    Ok(PyDataFrame(out_df))
}
