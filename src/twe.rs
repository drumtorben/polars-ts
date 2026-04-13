use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// TWE (Time Warp Edit) distance. O(m) memory.
fn twe_distance(a: &[f64], b: &[f64], nu: f64, lambda: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];

    prev[0] = 0.0;
    for j in 1..=m {
        prev[j] = prev[j - 1] + (b[j - 1] - if j > 1 { b[j - 2] } else { 0.0 }).abs() + nu + lambda;
    }

    for i in 1..=n {
        let a_i = a[i - 1];
        let a_prev = if i > 1 { a[i - 2] } else { 0.0 };
        curr[0] = prev[0] + (a_i - a_prev).abs() + nu + lambda;

        for j in 1..=m {
            let b_j = b[j - 1];
            let b_prev = if j > 1 { b[j - 2] } else { 0.0 };

            let d_match = prev[j - 1]
                + (a_i - b_j).abs()
                + (a_prev - b_prev).abs()
                + nu * ((i as f64 - j as f64).abs()).min(2.0 * nu);
            let d_delete = prev[j] + (a_i - a_prev).abs() + nu + lambda;
            let d_insert = curr[j - 1] + (b_j - b_prev).abs() + nu + lambda;

            curr[j] = d_match.min(d_delete).min(d_insert);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

#[pyfunction]
#[pyo3(signature = (input1, input2, nu=None, lambda=None))]
pub fn compute_pairwise_twe(
    input1: PyDataFrame,
    input2: PyDataFrame,
    nu: Option<f64>,
    lambda: Option<f64>,
) -> PyResult<PyDataFrame> {
    let nu_value = nu.unwrap_or(0.001);
    let lambda_value = lambda.unwrap_or(1.0);
    compute_pairwise(input1, input2, "twe", move |a, b| {
        twe_distance(a, b, nu_value, lambda_value)
    })
}
