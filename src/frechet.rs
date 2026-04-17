use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// Discrete Frechet distance between two time series. Standard O(nm) DP.
fn frechet_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }

    // O(m) memory using two-row approach
    let mut prev = vec![f64::NEG_INFINITY; m];
    let mut curr = vec![f64::NEG_INFINITY; m];

    prev[0] = (a[0] - b[0]).abs();
    for j in 1..m {
        prev[j] = prev[j - 1].max((a[0] - b[j]).abs());
    }

    for &ai in &a[1..] {
        curr[0] = prev[0].max((ai - b[0]).abs());
        for j in 1..m {
            let min_prev = prev[j - 1].min(prev[j]).min(curr[j - 1]);
            curr[j] = min_prev.max((ai - b[j]).abs());
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m - 1]
}

#[pyfunction]
pub fn compute_pairwise_frechet(
    input1: PyDataFrame,
    input2: PyDataFrame,
) -> PyResult<PyDataFrame> {
    compute_pairwise(input1, input2, "frechet", frechet_distance)
}
