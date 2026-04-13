use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// Compute the derivative of a time series using the method from Keogh & Pazzani (2001).
fn compute_derivative(q: &[f64]) -> Vec<f64> {
    if q.len() < 3 {
        return Vec::new();
    }
    let mut derivative = Vec::with_capacity(q.len() - 2);
    for i in 1..q.len() - 1 {
        let term1 = q[i] - q[i - 1];
        let term2 = (q[i + 1] - q[i - 1]) / 2.0;
        derivative.push((term1 + term2) / 2.0);
    }
    derivative
}

/// DTW distance using O(m) memory (two-row approach).
fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }
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

/// DDTW distance: derivative of each series, then DTW on derivatives.
fn ddtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let a_d = compute_derivative(a);
    let b_d = compute_derivative(b);
    if a_d.is_empty() || b_d.is_empty() {
        return f64::INFINITY;
    }
    dtw_distance(&a_d, &b_d)
}

#[pyfunction]
pub fn compute_pairwise_ddtw(input1: PyDataFrame, input2: PyDataFrame) -> PyResult<PyDataFrame> {
    compute_pairwise(input1, input2, "ddtw", ddtw_distance)
}
