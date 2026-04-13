use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise_multivariate;

/// Compute Manhattan distance between two vectors.
fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Compute squared Euclidean distance between two vectors.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| {
        let diff = x - y;
        diff * diff
    }).sum()
}

/// Compute the dependent cost given three vectors.
fn cost_dependent(x: &[f64], y: &[f64], z: &[f64], c: f64) -> f64 {
    let diameter = squared_distance(y, z);
    let mid: Vec<f64> = y.iter().zip(z).map(|(a, b)| (a + b) / 2.0).collect();
    let distance_to_mid = squared_distance(&mid, x);
    if distance_to_mid <= (diameter / 2.0) {
        c
    } else {
        c + squared_distance(y, x).min(squared_distance(z, x))
    }
}

/// Multivariate MSM distance using O(m) memory.
fn msm_distance(a: &[Vec<f64>], b: &[Vec<f64>], c: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut prev = vec![f64::MAX; m];
    let mut curr = vec![f64::MAX; m];
    prev[0] = manhattan_distance(&a[0], &b[0]);

    for j in 1..m {
        prev[j] = prev[j - 1] + cost_dependent(&b[j], &b[j - 1], &a[0], c);
    }

    for i in 1..n {
        curr[0] = prev[0] + cost_dependent(&a[i], &a[i - 1], &b[0], c);
        for j in 1..m {
            let d1 = prev[j - 1] + manhattan_distance(&a[i], &b[j]);
            let d2 = prev[j] + cost_dependent(&a[i], &a[i - 1], &b[j], c);
            let d3 = curr[j - 1] + cost_dependent(&b[j], &a[i], &b[j - 1], c);
            curr[j] = d1.min(d2).min(d3);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m - 1]
}

#[pyfunction]
#[pyo3(signature = (input1, input2, c=None))]
pub fn compute_pairwise_msm_multi(
    input1: PyDataFrame,
    input2: PyDataFrame,
    c: Option<f64>,
) -> PyResult<PyDataFrame> {
    let c_value = c.unwrap_or(1.0);
    compute_pairwise_multivariate(input1, input2, "msm_multi", move |a, b| {
        msm_distance(a, b, c_value)
    })
}
