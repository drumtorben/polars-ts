use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashSet;

use crate::utils::compute_pairwise;

// ---------------------------------------------------------------------------
// DTW distance kernels
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
fn dtw_sakoe_chiba(a: &[f64], b: &[f64], window: usize) -> f64 {
    let n = a.len();
    let m = b.len();
    let w = window.max(n.abs_diff(m));
    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::MAX;
        let j_start = if i > w { i - w } else { 1 };
        let j_end = (i + w).min(m);
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
            let fi = i as f64;
            let fj = j as f64;
            let lower = (fi / max_slope).max(mf - (nf - fi) * max_slope);
            let upper = (fi * max_slope).min(mf - (nf - fi) / max_slope);
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
    let mut reduced = Vec::with_capacity(x.len().div_ceil(2));
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
/// Uses HashSet for O(path_len * radius^2) memory instead of O(n*m).
fn fast_dtw(a: &[f64], b: &[f64], radius: usize) -> f64 {
    let min_size = radius + 2;
    if a.len() <= min_size || b.len() <= min_size {
        return dtw_distance(a, b);
    }

    let a_shrunk = reduce_by_half(a);
    let b_shrunk = reduce_by_half(b);
    let path = fast_dtw_path(&a_shrunk, &b_shrunk, radius);

    let n = a.len();
    let m = b.len();
    let mut window: HashSet<(usize, usize)> = HashSet::new();
    for &(pi, pj) in &path {
        for di in 0..2 {
            for dj in 0..2 {
                let oi = pi * 2 + di;
                let oj = pj * 2 + dj;
                if oi < n && oj < m {
                    let r_start_i = oi.saturating_sub(radius);
                    let r_end_i = (oi + radius).min(n - 1);
                    let r_start_j = oj.saturating_sub(radius);
                    let r_end_j = (oj + radius).min(m - 1);
                    for ri in r_start_i..=r_end_i {
                        for rj in r_start_j..=r_end_j {
                            window.insert((ri, rj));
                        }
                    }
                }
            }
        }
    }

    dtw_with_window(a, b, &window)
}

/// Compute DTW restricted to a HashSet window mask.
fn dtw_with_window(a: &[f64], b: &[f64], window: &HashSet<(usize, usize)>) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut cost_matrix = vec![vec![f64::MAX; m + 1]; n + 1];
    cost_matrix[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            if !window.contains(&(i - 1, j - 1)) {
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
    let mut window: HashSet<(usize, usize)> = HashSet::new();
    for &(pi, pj) in &path {
        for di in 0..2 {
            for dj in 0..2 {
                let oi = pi * 2 + di;
                let oj = pj * 2 + dj;
                if oi < n && oj < m {
                    let r_start_i = oi.saturating_sub(radius);
                    let r_end_i = (oi + radius).min(n - 1);
                    let r_start_j = oj.saturating_sub(radius);
                    let r_end_j = (oj + radius).min(m - 1);
                    for ri in r_start_i..=r_end_i {
                        for rj in r_start_j..=r_end_j {
                            window.insert((ri, rj));
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

/// Compute DTW path restricted to a HashSet window mask.
fn dtw_path_with_window(a: &[f64], b: &[f64], window: &HashSet<(usize, usize)>) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    let mut cost_matrix = vec![vec![f64::MAX; m + 1]; n + 1];
    cost_matrix[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            if !window.contains(&(i - 1, j - 1)) {
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

// ---------------------------------------------------------------------------
// Pairwise wrapper
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (input1, input2, method=None, param=None))]
pub fn compute_pairwise_dtw(
    input1: PyDataFrame,
    input2: PyDataFrame,
    method: Option<&str>,
    param: Option<f64>,
) -> PyResult<PyDataFrame> {
    let method = method.unwrap_or("standard");
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

    let method_owned = method.to_string();
    compute_pairwise(input1, input2, "dtw", move |a, b| {
        compute_dtw(a, b, &method_owned, param)
    })
}
