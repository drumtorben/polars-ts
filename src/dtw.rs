use polars::prelude::*;
use itertools::iproduct;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;


/// Computes pairwise Dynamic Time Warping (DTW) between time series.
///
/// # Arguments
/// * `df1` - First dataframe containing time series data with columns "unique_id" and "y".
/// * `df2` - Second dataframe containing time series data with columns "unique_id" and "y".

fn get_groups(df: &DataFrame) -> Result<LazyFrame, PolarsError>  {
    Ok(df.clone().lazy()
        .select([
            col("unique_id").cast(DataType::String),
            col("y").cast(DataType::Float32)
        ])
        .group_by([col("unique_id")])
        .agg([col("y")])
    )
}

/// Classic O(n*m) DTW using a 2D DP matrix.
///
/// - `dp[i][j]` = minimal cumulative distance aligning
///   `a[..i]` and `b[..j]`.
/// - `a` has length n, `b` has length m.
fn dtw_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let m = b.len();

    // Allocate a (n+1) x (m+1) matrix, initialize to MAX
    let mut dp = vec![vec![f32::MAX; m + 1]; n + 1];

    // DP boundary initialization
    dp[0][0] = 0.0;

    // Fill DP
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let min_prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + min_prev;
        }
    }
    // The bottom-right corner holds the final DTW distance
    dp[n][m]
}

#[pyfunction]
pub fn compute_pairwise_dtw(input1: PyDataFrame, input2: PyDataFrame) -> PyResult<PyDataFrame> {
    let a = input1.into();
    let b = input2.into();
    let df1 = get_groups(&a).unwrap().collect().unwrap();
    let df2 = get_groups(&b).unwrap().collect().unwrap();

    let ids1 = df_to_hashmap(&df1);
    let ids2 = df_to_hashmap(&df2);

    let mut results = Vec::new();

    for ((id1, series1), (id2, series2)) in iproduct!(ids1.iter(), ids2.iter()) {
        let distance = dtw_distance(series1, series2);
        //println!("DTW distance between '{}' and '{}' = {}", id1, id2, distance);
        results.push((id1.to_string(), id2.to_string(), distance))
    }

    // Step 4: Convert results to DataFrame
    let columns = vec![
        Column::new("id_1".into(), results.iter().map(|row| row.0.clone()).collect::<Vec<_>>()),
        Column::new("id_2".into(), results.iter().map(|row| row.1.clone()).collect::<Vec<_>>()),
        Column::new("dtw".into(), results.iter().map(|row| row.2).collect::<Vec<_>>()),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    Ok(PyDataFrame(out_df))
}

fn df_to_hashmap(df: &DataFrame) -> HashMap<String, Vec<f32>> {
    let unique_id_col = df.column("unique_id").unwrap();
    let y_col = df.column("y").unwrap();

    // 1) Get the string IDs out of the UTF8 column.
    let unique_id_iter = unique_id_col
        .str()
        .expect("expected a utf8 column for unique_id")
        .into_no_null_iter()
        .map(|s| s.to_string());

    // 2) Grab the list column as a ListChunked.
    let y_list = y_col
        .list()
        .expect("expected a List type for y");

    // 3) For each element in the list column, we get back a Series.
    //    Then we call .f32() on that Series (assuming it’s f32),
    //    collect the values, and return a Vec<f32>.
    // 3) Instead of `amortized_iter()`, use `into_iter()` to get `Option<Series>`.
    let y_iter = y_list
        .into_iter()
        .map(|opt_series| {
            let series = opt_series.expect("null entry in 'y' list column");
            // Now you have a real Series, so you can call .f32().
            series
                .f32()
                .expect("expected a f32 Series inside the list")
                // Convert to a Vec<f32> (assuming no nulls).
                .into_no_null_iter()
                .collect::<Vec<f32>>()
        });

    // 4) Finally, zip the two iterators and build the HashMap.
    let mut map = HashMap::new();
    for (id, y_vals) in unique_id_iter.zip(y_iter) {
        map.insert(id, y_vals);
    }
    map
}
