use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// LCSS distance. Returns 1 - (LCSS_length / min(n, m)). O(m) memory.
fn lcss_distance(a: &[f64], b: &[f64], epsilon: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 1.0;
    }

    let mut prev = vec![0_usize; m + 1];
    let mut curr = vec![0_usize; m + 1];

    for i in 1..=n {
        for j in 1..=m {
            if (a[i - 1] - b[j - 1]).abs() <= epsilon {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        for item in curr.iter_mut().take(m + 1) {
            *item = 0;
        }
    }

    let lcss_len = prev[m] as f64;
    let min_len = n.min(m) as f64;
    1.0 - (lcss_len / min_len)
}

#[pyfunction]
#[pyo3(signature = (input1, input2, epsilon=None))]
pub fn compute_pairwise_lcss(
    input1: PyDataFrame,
    input2: PyDataFrame,
    epsilon: Option<f64>,
) -> PyResult<PyDataFrame> {
    let eps = epsilon.unwrap_or(1.0);
    compute_pairwise(input1, input2, "lcss", move |a, b| {
        lcss_distance(a, b, eps)
    })
}
