use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// ERP (Edit Distance with Real Penalty) distance. O(m) memory.
fn erp_distance(a: &[f64], b: &[f64], g: f64) -> f64 {
    let n = a.len();
    let m = b.len();

    let mut prev = vec![0.0_f64; m + 1];
    let mut curr = vec![0.0_f64; m + 1];

    for j in 1..=m {
        prev[j] = prev[j - 1] + (b[j - 1] - g).abs();
    }

    let mut first_col_cost = 0.0_f64;
    for i in 1..=n {
        first_col_cost += (a[i - 1] - g).abs();
        curr[0] = first_col_cost;
        for j in 1..=m {
            let d_match = prev[j - 1] + (a[i - 1] - b[j - 1]).abs();
            let d_delete = prev[j] + (a[i - 1] - g).abs();
            let d_insert = curr[j - 1] + (b[j - 1] - g).abs();
            curr[j] = d_match.min(d_delete).min(d_insert);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

#[pyfunction]
#[pyo3(signature = (input1, input2, g=None))]
pub fn compute_pairwise_erp(
    input1: PyDataFrame,
    input2: PyDataFrame,
    g: Option<f64>,
) -> PyResult<PyDataFrame> {
    let g_value = g.unwrap_or(0.0);
    compute_pairwise(input1, input2, "erp", move |a, b| {
        erp_distance(a, b, g_value)
    })
}
