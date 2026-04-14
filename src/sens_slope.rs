//! Sen's slope estimator for Polars Series.
//!
//! The Theil-Sen estimator computes the median of all pairwise slopes
//! `(x[j] - x[i]) / (j - i)` for `i < j`, providing a robust measure
//! of trend magnitude that is resistant to outliers.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// Compute Sen's slope (median of pairwise slopes) for the first Series in `inputs`.
///
/// # Parameters
/// - `inputs`: a slice of Series, with the first entry expected to be a float Series (f64).
///
/// # Returns
/// A single-valued Float64 Series containing the Sen's slope estimate.
///
/// # Errors
/// Returns an error if the first Series is not of f64 type or if it contains nulls.
#[polars_expr(output_type=Float64)]
pub fn sens_slope(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    // Filter out null values instead of panicking
    let vals: Vec<f64> = s.f64()?.into_iter().flatten().collect();
    let n = vals.len();

    if n < 2 {
        return Ok(Series::new(s.name().clone(), [0.0f64]));
    }

    // Collect all pairwise slopes: (x[j] - x[i]) / (j - i) for i < j
    let mut slopes: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            slopes.push((vals[j] - vals[i]) / ((j - i) as f64));
        }
    }

    // Sort and take median
    slopes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if slopes.len().is_multiple_of(2) {
        let mid = slopes.len() / 2;
        (slopes[mid - 1] + slopes[mid]) / 2.0
    } else {
        slopes[slopes.len() / 2]
    };

    Ok(Series::new(s.name().clone(), [median]))
}
