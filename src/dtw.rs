use polars::prelude::*;
use std::collections::HashMap;
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

// ---------------------------------------------------------------------------
// DTW distance functions
// ---------------------------------------------------------------------------

/// Standard unconstrained DTW using O(m) memory (two-row approach).
fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::MAX;
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            curr[j] = cost + min_prev;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// DTW with Sakoe-Chiba band constraint.
/// Only cells within `window` of the diagonal are computed.
fn dtw_sakoe_chiba(a: &[f64], b: &[f64], window: usize) -> f64 {
    let n = a.len();
    let m = b.len();
    // Adjust window to accommodate length differences.
    let w = window.max(if n > m { n - m } else { m - n });
    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::MAX;
        let j_start = if i > w { i - w } else { 1 };
        let j_end = (i + w).min(m);
        // Reset cells just outside the band to MAX so they don't bleed in.
        if j_start > 1 {
            curr[j_start - 1] = f64::MAX;
        }
        for j in j_start..=j_end {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            curr[j] = cost + min_prev;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// DTW with Itakura parallelogram constraint.
/// `max_slope` controls the steepness of the parallelogram (typical: 2.0).
fn dtw_itakura(a: &[f64], b: &[f64], max_slope: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];
    prev[0] = 0.0;

    let nf = n as f64;
    let mf = m as f64;

    for i in 1..=n {
        curr[0] = f64::MAX;
        for j in 1..=m {
            // Check Itakura parallelogram bounds.
            // The parallelogram constrains j/i and (m-j)/(n-i) by max_slope
            // and 1/max_slope. Equivalently, for position (i, j):
            //   j >= i / max_slope  AND  j <= i * max_slope
            //   j >= m - (n - i) * max_slope  AND  j <= m - (n - i) / max_slope
            let fi = i as f64;
            let fj = j as f64;
            let lower1 = fi / max_slope;
            let upper1 = fi * max_slope;
            let lower2 = mf - (nf - fi) * max_slope;
            let upper2 = mf - (nf - fi) / max_slope;
            let lower = lower1.max(lower2);
            let upper = upper1.min(upper2);
            if fj < lower || fj > upper {
                curr[j] = f64::MAX;
            } else {
                let cost = (a[i - 1] - b[j - 1]).abs();
                let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
                curr[j] = cost + min_prev;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Reduce a time series by averaging consecutive pairs.
fn reduce_by_half(x: &[f64]) -> Vec<f64> {
    let mut reduced = Vec::with_capacity((x.len() + 1) / 2);
    let mut i = 0;
    while i + 1 < x.len() {
        reduced.push((x[i] + x[i + 1]) / 2.0);
        i += 2;
    }
    if i < x.len() {
        reduced.push(x[i]);
    }
    reduced
}

/// FastDTW: approximate DTW in O(N) time using multi-resolution coarsening.
/// `radius` controls the size of the neighborhood around the projected path.
fn fast_dtw(a: &[f64], b: &[f64], radius: usize) -> f64 {
    let min_size = radius + 2;
    if a.len() <= min_size || b.len() <= min_size {
        return dtw_distance(a, b);
    }

    // 1. Coarsen
    let a_shrunk = reduce_by_half(a);
    let b_shrunk = reduce_by_half(b);

    // 2. Recurse on coarsened series to get the warping path
    let path = fast_dtw_path(&a_shrunk, &b_shrunk, radius);

    // 3. Project path back to original resolution and expand by radius
    let n = a.len();
    let m = b.len();
    let mut window = vec![vec![false; m]; n];
    for &(pi, pj) in &path {
        // Each coarse cell (pi, pj) maps to up to 4 cells in original resolution
        for di in 0..2 {
            for dj in 0..2 {
                let oi = pi * 2 + di;
                let oj = pj * 2 + dj;
                if oi < n && oj < m {
                    // Expand by radius
                    let r_start_i = if oi >= radius { oi - radius } else { 0 };
                    let r_end_i = (oi + radius).min(n - 1);
                    let r_start_j = if oj >= radius { oj - radius } else { 0 };
                    let r_end_j = (oj + radius).min(m - 1);
                    for ri in r_start_i..=r_end_i {
                        for rj in r_start_j..=r_end_j {
                            window[ri][rj] = true;
                        }
                    }
                }
            }
        }
    }

    // 4. Compute DTW only on the projected window
    dtw_with_window(a, b, &window)
}

/// Compute DTW restricted to a boolean window mask.
fn dtw_with_window(a: &[f64], b: &[f64], window: &[Vec<bool>]) -> f64 {
    let n = a.len();
    let m = b.len();
    // Use full matrix since window is sparse and irregular
    let mut cost_matrix = vec![vec![f64::MAX; m + 1]; n + 1];
    cost_matrix[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            if !window[i - 1][j - 1] {
                continue;
            }
            let cost = (a[i - 1] - b[j - 1]).abs();
            let min_prev = cost_matrix[i - 1][j]
                .min(cost_matrix[i][j - 1])
                .min(cost_matrix[i - 1][j - 1]);
            cost_matrix[i][j] = cost + min_prev;
        }
    }
    cost_matrix[n][m]
}

/// FastDTW helper that returns the warping path (used for recursive projection).
fn fast_dtw_path(a: &[f64], b: &[f64], radius: usize) -> Vec<(usize, usize)> {
    let min_size = radius + 2;
    if a.len() <= min_size || b.len() <= min_size {
        return dtw_full_path(a, b);
    }

    let a_shrunk = reduce_by_half(a);
    let b_shrunk = reduce_by_half(b);
    let path = fast_dtw_path(&a_shrunk, &b_shrunk, radius);

    let n = a.len();
    let m = b.len();
    let mut window = vec![vec![false; m]; n];
    for &(pi, pj) in &path {
        for di in 0..2 {
            for dj in 0..2 {
                let oi = pi * 2 + di;
                let oj = pj * 2 + dj;
                if oi < n && oj < m {
                    let r_start_i = if oi >= radius { oi - radius } else { 0 };
                    let r_end_i = (oi + radius).min(n - 1);
                    let r_start_j = if oj >= radius { oj - radius } else { 0 };
                    let r_end_j = (oj + radius).min(m - 1);
                    for ri in r_start_i..=r_end_i {
                        for rj in r_start_j..=r_end_j {
                            window[ri][rj] = true;
                        }
                    }
                }
            }
        }
    }

    dtw_path_with_window(a, b, &window)
}

/// Compute the full DTW cost matrix and extract the optimal warping path.
fn dtw_full_path(a: &[f64], b: &[f64]) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    let mut cost_matrix = vec![vec![f64::MAX; m + 1]; n + 1];
    cost_matrix[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let min_prev = cost_matrix[i - 1][j]
                .min(cost_matrix[i][j - 1])
                .min(cost_matrix[i - 1][j - 1]);
            cost_matrix[i][j] = cost + min_prev;
        }
    }

    // Traceback
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let diag = cost_matrix[i - 1][j - 1];
        let left = cost_matrix[i][j - 1];
        let up = cost_matrix[i - 1][j];
        if diag <= left && diag <= up {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    path
}

/// Compute DTW path restricted to a boolean window mask.
fn dtw_path_with_window(a: &[f64], b: &[f64], window: &[Vec<bool>]) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    let mut cost_matrix = vec![vec![f64::MAX; m + 1]; n + 1];
    cost_matrix[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            if !window[i - 1][j - 1] {
                continue;
            }
            let cost = (a[i - 1] - b[j - 1]).abs();
            let min_prev = cost_matrix[i - 1][j]
                .min(cost_matrix[i][j - 1])
                .min(cost_matrix[i - 1][j - 1]);
            cost_matrix[i][j] = cost + min_prev;
        }
    }

    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let diag = cost_matrix[i - 1][j - 1];
        let left = cost_matrix[i][j - 1];
        let up = cost_matrix[i - 1][j];
        if diag <= left && diag <= up {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    path
}

/// Dispatch to the appropriate DTW variant.
fn compute_dtw(a: &[f64], b: &[f64], method: &str, param: f64) -> f64 {
    match method {
        "standard" => dtw_distance(a, b),
        "sakoe_chiba" => dtw_sakoe_chiba(a, b, param as usize),
        "itakura" => dtw_itakura(a, b, param),
        "fast" => fast_dtw(a, b, param as usize),
        _ => dtw_distance(a, b),
    }
}

/// Optimized conversion of a grouped DataFrame into a HashMap mapping id -> Vec<f64>.
///
/// This version first collects the "unique_id" column and the list-of-f64
/// from the "y" column into two vectors. Then, using a parallel index loop,
/// it zips them together into a HashMap.
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

/// Compute pairwise DTW distances between time series in two DataFrames,
/// using extensive parallelism.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
/// * `method` - DTW variant: "standard", "sakoe_chiba", "itakura", or "fast".
/// * `param` - Method-specific parameter:
///   - sakoe_chiba: window size (default 10)
///   - itakura: max slope (default 2.0)
///   - fast: radius (default 5)
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "dtw".
#[pyfunction]
#[pyo3(signature = (input1, input2, method=None, param=None))]
pub fn compute_pairwise_dtw(
    input1: PyDataFrame,
    input2: PyDataFrame,
    method: Option<&str>,
    param: Option<f64>,
) -> PyResult<PyDataFrame> {
    let method = method.unwrap_or("standard");
    // Validate method
    match method {
        "standard" | "sakoe_chiba" | "itakura" | "fast" => {}
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown DTW method: '{}'. Expected one of: standard, sakoe_chiba, itakura, fast", method)
            ));
        }
    }
    let param = param.unwrap_or(match method {
        "sakoe_chiba" => 10.0,
        "itakura" => 2.0,
        "fast" => 5.0,
        _ => 0.0,
    });
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

    // Compute pairwise DTW distances: id_1 always comes from left, id_2 from right.
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
                    // Compute the DTW distance.
                    let distance = compute_dtw(left_series, right_series, method, param);
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
        Column::new("dtw".into(), dtw_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ]).collect().unwrap();
    Ok(PyDataFrame(casted_out_df))
}
