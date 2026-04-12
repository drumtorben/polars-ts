use polars::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;

/// Cast a column in-place to the given DataType, returning a new DataFrame.
pub fn cast_column(df: &DataFrame, col_name: &str, dtype: DataType) -> Result<DataFrame, PolarsError> {
    let mut df = df.clone();
    let casted = df.column(col_name)?.cast(&dtype)?.take_materialized_series();
    let _ = df.replace(col_name, casted)?;
    Ok(df)
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
    let mut result = df.group_by(["unique_id"])?.agg_list()?;
    // agg_list renames columns to "{name}_agg_list", rename back
    result.rename("y_agg_list", "y".into())?;
    Ok(result)
}

/// Optimized conversion of a grouped DataFrame into a HashMap mapping id -> Vec<f64>.
pub fn df_to_hashmap(df: &DataFrame) -> HashMap<String, Vec<f64>> {
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
    let mut result = df.group_by(["unique_id"])?.agg_list()?;
    // agg_list renames columns to "{name}_agg_list", rename back
    for dim in &dims {
        let agg_name = format!("{}_agg_list", dim);
        result.rename(agg_name.as_str(), dim.clone().into())?;
    }
    Ok(result)
}

/// Converts a grouped DataFrame into a HashMap mapping unique_id -> multivariate time series.
pub fn df_to_hashmap_multivariate(df: &DataFrame) -> HashMap<String, Vec<Vec<f64>>> {
    let unique_id_col = df.column("unique_id").expect("expected column unique_id");
    let unique_ids: Vec<String> = unique_id_col
        .str()
        .expect("expected Utf8 for unique_id")
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

    let mut hashmap = HashMap::new();
    let num_series = unique_ids.len();
    for i in 0..num_series {
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
