use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// Precompute weight vector for WDTW calculation.
fn compute_weight_vector(len: usize, g: f64) -> Vec<f64> {
    let half_len = len as f64 / 2.0;
    (0..len)
        .map(|i| 1.0 / (1.0 + (-g * (i as f64 - half_len)).exp()))
        .collect()
}

/// Memory-optimized WDTW distance using O(m) memory.
fn wdtw_distance_optimized(a: &[f64], b: &[f64], g: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }

    let max_len = n.max(m);
    let weight_vector = compute_weight_vector(max_len, g);

    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::INFINITY;
        for j in 1..=m {
            let weight = weight_vector[((i - 1) as isize - (j - 1) as isize).unsigned_abs()];
            let diff = (a[i - 1] - b[j - 1]).powi(2);
            let prev_min = prev[j - 1].min(prev[j]).min(curr[j - 1]);
            curr[j] = prev_min + weight * diff;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

#[pyfunction]
#[pyo3(signature = (input1, input2, g=None))]
pub fn compute_pairwise_wdtw(
    input1: PyDataFrame,
    input2: PyDataFrame,
    g: Option<f64>,
) -> PyResult<PyDataFrame> {
    let g_value = g.unwrap_or(0.05);
    compute_pairwise(input1, input2, "wdtw", move |a, b| {
        wdtw_distance_optimized(a, b, g_value)
    })
}
