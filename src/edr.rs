use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise;

/// Edit Distance on Real sequences (EDR) with epsilon threshold. O(m) memory.
/// Returns the normalized EDR distance: edr_count / max(n, m).
fn edr_distance(a: &[f64], b: &[f64], epsilon: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return 1.0;
    }
    if m == 0 {
        return 1.0;
    }

    let mut prev = vec![0usize; m + 1];
    let mut curr = vec![0usize; m + 1];

    for (j, prev_j) in prev.iter_mut().enumerate() {
        *prev_j = j;
    }

    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let subcost = if (a[i - 1] - b[j - 1]).abs() <= epsilon {
                0
            } else {
                1
            };
            curr[j] = (prev[j - 1] + subcost)
                .min(prev[j] + 1)
                .min(curr[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m] as f64 / n.max(m) as f64
}

#[pyfunction]
#[pyo3(signature = (input1, input2, epsilon=None))]
pub fn compute_pairwise_edr(
    input1: PyDataFrame,
    input2: PyDataFrame,
    epsilon: Option<f64>,
) -> PyResult<PyDataFrame> {
    let eps = epsilon.unwrap_or(0.1);
    compute_pairwise(input1, input2, "edr", move |a, b| {
        edr_distance(a, b, eps)
    })
}
