use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// Compute the normalized cross-correlation (NCC) sequence between two series.
/// Returns a vector of length n+m-1 containing the cross-correlation at each lag.
fn normalized_cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let m = b.len();
    let len = n + m - 1;

    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;

    if denom == 0.0 {
        return vec![0.0; len];
    }

    // Direct cross-correlation computation
    let mut ncc = vec![0.0; len];
    for (k, ncc_k) in ncc.iter_mut().enumerate() {
        let mut sum = 0.0;
        let shift = k as isize - (m as isize - 1);
        for (i, &ai) in a.iter().enumerate() {
            let j = i as isize - shift;
            if j >= 0 && (j as usize) < m {
                sum += ai * b[j as usize];
            }
        }
        *ncc_k = sum / denom;
    }
    ncc
}

/// Shape-Based Distance (SBD) between two time series.
/// SBD = 1 - max(NCC(a, b)), where NCC is the normalized cross-correlation.
/// Range: [0, 2].
fn sbd_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 2.0;
    }

    let ncc = normalized_cross_correlation(a, b);
    let max_ncc = ncc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    1.0 - max_ncc
}

#[pyfunction]
pub fn compute_pairwise_sbd(input1: PyDataFrame, input2: PyDataFrame) -> PyResult<PyDataFrame> {
    compute_pairwise(input1, input2, "sbd", sbd_distance)
}
