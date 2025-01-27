//! This module provides an implementation of the Mann-Kendall trend test for Polars Series.
//!
//! The Mann-Kendall test checks for a trend in a time series by comparing each data point
//! to subsequent data points. It computes a statistic `S` indicating how many pairs are
//! in increasing vs. decreasing order and then normalizes it via
//! \[ S / (0.5 * n * (n - 1)) \], matching standard Mann-Kendall definitions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use ordered_float::OrderedFloat;


/// Apply the Mann-Kendall trend test to the first Series in `inputs`.
///
/// # Parameters
/// - `inputs`: a slice of Series, with the first entry expected to be a float Series (f64).
///
/// # Returns
/// A single-valued Float64 Series containing the normalized Mann-Kendall statistic.
/// This matches the Python version `mk_stat = S / (0.5 * n * (n - 1))`.
///
/// # Errors
/// Returns an error if the first Series is not of f64 type or if it contains nulls.
#[polars_expr(output_type=Float64)]
pub fn mann_kendall(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];

    // Collect non-null f64 values
    let vals = s.f64()?.into_no_null_iter().collect::<Vec<_>>();
    let n = vals.len() as f64;
    if n < 2.0 {
        return Ok(Series::new(s.name().clone(), [0.0f64]));
    }

    // Sort and deduplicate distinct floats (wrapped for total ordering)
    let mut unique: Vec<OrderedFloat<f64>> = vals.iter().copied().map(OrderedFloat).collect();
    unique.sort();
    unique.dedup();

    // Fenwick (BIT) for counting
    let mut fen = vec![0i64; unique.len() + 1];
    fn update(bit: &mut [i64], mut i: usize) {
        while i < bit.len() {
            bit[i] += 1;
            i += i & (!i + 1);
        }
    }
    /// Query how many increments occur up to and including index `i` in the Fenwick tree.
    fn query(bit: &[i64], mut i: usize) -> i64 {
        let mut s = 0;
        while i > 0 {
            s += bit[i];
            i -= i & (!i + 1);
        }
        s
    }

    // Accumulate S in reverse
    let mut s_stat = 0i64; 
    for (i, &val) in vals.iter().enumerate().rev() {
        // Binary search to find the 1-based rank
        let c = unique.binary_search(&OrderedFloat(val)).unwrap() + 1;
        let less = query(&fen, c - 1);
        let equal = query(&fen, c) - less;
        s_stat += (n as i64 - 1 - i as i64) - 2 * less - equal;
        update(&mut fen, c);
    }

    // Normalize to match the Mann-Kendall metric = S / (0.5 * n * (n - 1))
    let metric = s_stat as f64 / (0.5 * n * (n - 1.0));
    Ok(Series::new(s.name().clone(), [metric]))
}