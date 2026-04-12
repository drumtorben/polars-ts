use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// Helper function to calculate the MSM cost.
fn msm_cost(x: f64, y: f64, z: f64, c: f64) -> f64 {
    if (y <= x && x <= z) || (y >= x && x >= z) {
        return c;
    }
    c + (x - y).abs().min((x - z).abs())
}

/// Optimized MSM distance using O(m) memory.
fn msm_distance(a: &[f64], b: &[f64], c: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut prev = vec![f64::MAX; m];
    let mut curr = vec![f64::MAX; m];
    prev[0] = (a[0] - b[0]).abs();

    for j in 1..m {
        prev[j] = prev[j - 1] + msm_cost(b[j], a[0], b[j - 1], c);
    }

    for i in 1..n {
        curr[0] = prev[0] + msm_cost(a[i], a[i - 1], b[0], c);
        for j in 1..m {
            let d1 = prev[j - 1] + (a[i] - b[j]).abs();
            let d2 = prev[j] + msm_cost(a[i], a[i - 1], b[j], c);
            let d3 = curr[j - 1] + msm_cost(b[j], a[i], b[j - 1], c);
            curr[j] = d1.min(d2).min(d3);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m - 1]
}

#[pyfunction]
#[pyo3(signature = (input1, input2, c=None))]
pub fn compute_pairwise_msm(
    input1: PyDataFrame,
    input2: PyDataFrame,
    c: Option<f64>,
) -> PyResult<PyDataFrame> {
    let c_value = c.unwrap_or(1.0);
    compute_pairwise(input1, input2, "msm", move |a, b| {
        msm_distance(a, b, c_value)
    })
}
