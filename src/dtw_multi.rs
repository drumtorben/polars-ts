use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::utils::compute_pairwise_multivariate;

/// Distance metric for DTW cost calculations.
#[derive(Clone, Copy)]
enum DistanceMetric {
    Manhattan,
    Euclidean,
}

/// Multivariate DTW distance using the specified metric. O(m) memory.
fn dtw_distance_multivariate(a: &[Vec<f64>], b: &[Vec<f64>], metric: DistanceMetric) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![f64::MAX; m + 1];
    let mut curr = vec![f64::MAX; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::MAX;
        for j in 1..=m {
            let cost: f64 = match metric {
                DistanceMetric::Manhattan => {
                    a[i - 1].iter().zip(b[j - 1].iter())
                        .map(|(x, y)| (x - y).abs())
                        .sum()
                }
                DistanceMetric::Euclidean => {
                    let sum_sq: f64 = a[i - 1].iter().zip(b[j - 1].iter())
                        .map(|(x, y)| (x - y).powi(2))
                        .sum();
                    sum_sq.sqrt()
                }
            };
            let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            curr[j] = cost + min_prev;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

#[pyfunction]
#[pyo3(signature = (input1, input2, metric=None))]
pub fn compute_pairwise_dtw_multi(
    input1: PyDataFrame,
    input2: PyDataFrame,
    metric: Option<String>,
) -> PyResult<PyDataFrame> {
    let distance_metric = match metric.as_deref() {
        Some("euclidean") => DistanceMetric::Euclidean,
        _ => DistanceMetric::Manhattan,
    };

    compute_pairwise_multivariate(input1, input2, "dtw_multi", move |a, b| {
        dtw_distance_multivariate(a, b, distance_metric)
    })
}
