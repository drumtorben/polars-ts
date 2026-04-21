//! PELT (Pruned Exact Linear Time) changepoint detection in Rust.
//!
//! Accelerates the O(n²) dynamic programming algorithm by moving
//! inner loops to Rust and using precomputed cumulative sums for
//! O(1) segment cost evaluation.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

/// Precomputed cumulative sums for O(1) segment cost evaluation.
struct CumStats {
    /// cumsum[i] = sum(data[0..i])
    cumsum: Vec<f64>,
    /// cumsum_sq[i] = sum(data[0..i]^2)
    cumsum_sq: Vec<f64>,
    _n: usize,
}

impl CumStats {
    fn new(data: &[f64]) -> Self {
        let n = data.len();
        let mut cumsum = vec![0.0; n + 1];
        let mut cumsum_sq = vec![0.0; n + 1];
        for i in 0..n {
            cumsum[i + 1] = cumsum[i] + data[i];
            cumsum_sq[i + 1] = cumsum_sq[i] + data[i] * data[i];
        }
        Self {
            cumsum,
            cumsum_sq,
            _n: n,
        }
    }

    /// Sum of data[start..end]
    #[inline]
    fn sum(&self, start: usize, end: usize) -> f64 {
        self.cumsum[end] - self.cumsum[start]
    }

    /// Sum of data[start..end]^2
    #[inline]
    fn sum_sq(&self, start: usize, end: usize) -> f64 {
        self.cumsum_sq[end] - self.cumsum_sq[start]
    }

    /// Mean of data[start..end]
    #[inline]
    fn mean(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        if n == 0.0 {
            return 0.0;
        }
        self.sum(start, end) / n
    }
}

/// Cost of segment [start, end) under a change-in-mean model.
/// cost = sum((x - mean)^2) = sum(x^2) - n * mean^2
#[inline]
fn cost_mean(stats: &CumStats, start: usize, end: usize) -> f64 {
    let n = (end - start) as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = stats.mean(start, end);
    stats.sum_sq(start, end) - n * mean * mean
}

/// Cost of segment [start, end) under a change-in-variance model.
/// cost = n * log(var) where var = sample variance (ddof=1)
#[inline]
fn cost_var(stats: &CumStats, start: usize, end: usize) -> f64 {
    let n = end - start;
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = stats.mean(start, end);
    let var = (stats.sum_sq(start, end) - nf * mean * mean) / (nf - 1.0);
    if var <= 0.0 {
        return 0.0;
    }
    nf * var.ln()
}

/// Cost of segment under change-in-mean-and-variance model.
#[inline]
fn cost_meanvar(stats: &CumStats, start: usize, end: usize) -> f64 {
    cost_mean(stats, start, end) + cost_var(stats, start, end)
}

type CostFn = fn(&CumStats, usize, usize) -> f64;

fn get_cost_fn(cost: &str) -> PyResult<CostFn> {
    match cost {
        "mean" => Ok(cost_mean),
        "var" => Ok(cost_var),
        "meanvar" => Ok(cost_meanvar),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown cost {cost:?}. Choose from [\"mean\", \"meanvar\", \"var\"]"
        ))),
    }
}

/// Run PELT on a single group's data, returning changepoint indices.
fn pelt_single(data: &[f64], cost_fn: CostFn, penalty: f64, min_size: usize) -> Vec<i64> {
    let n = data.len();
    if n < 2 * min_size {
        return vec![];
    }

    let stats = CumStats::new(data);

    // F[t] = minimum cost of optimally segmenting data[0..t]
    let mut f = vec![f64::INFINITY; n + 1];
    f[0] = -penalty;

    // cp[t] stores the last changepoint for the optimal segmentation of data[0..t]
    let mut last_cp = vec![0usize; n + 1];

    let mut candidates = vec![0usize];

    for t in min_size..=n {
        let mut best_cost = f64::INFINITY;
        let mut best_s = 0usize;

        for &s in &candidates {
            if t - s >= min_size {
                let c = f[s] + cost_fn(&stats, s, t) + penalty;
                if c < best_cost {
                    best_cost = c;
                    best_s = s;
                }
            }
        }

        f[t] = best_cost;
        last_cp[t] = best_s;

        // Pruning step
        candidates.retain(|&s| f[s] + cost_fn(&stats, s, t) <= f[t]);
        candidates.push(t);
    }

    // Backtrack to find all changepoints
    let mut changepoints = Vec::new();
    let mut idx = n;
    while idx > 0 {
        let prev = last_cp[idx];
        if prev > 0 {
            changepoints.push(prev as i64);
        }
        idx = prev;
    }
    changepoints.sort();
    changepoints
}

#[pyfunction]
#[pyo3(signature = (input, cost="mean", pen=None, min_size=2, id_col="unique_id", target_col="y"))]
pub fn pelt(
    input: PyDataFrame,
    cost: &str,
    pen: Option<f64>,
    min_size: usize,
    id_col: &str,
    target_col: &str,
) -> PyResult<PyDataFrame> {
    let cost_fn = get_cost_fn(cost)?;
    let df = input.0;

    let id_col_series = df
        .column(id_col)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let target_ca = df
        .column(target_col)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .f64()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Convert id column to string representations and collect groups
    let id_strs: Vec<String> = (0..id_col_series.len())
        .map(|i| {
            match id_col_series.get(i) {
                Ok(polars::prelude::AnyValue::String(s)) => s.to_string(),
                Ok(polars::prelude::AnyValue::StringOwned(s)) => s.to_string(),
                Ok(v) => format!("{v}"),
                Err(_) => String::new(),
            }
        })
        .collect();

    // Build groups: map group_id -> Vec<f64>
    let mut group_map: std::collections::BTreeMap<String, Vec<f64>> = std::collections::BTreeMap::new();
    for (i, gid) in id_strs.iter().enumerate() {
        let val = target_ca.get(i).unwrap_or(f64::NAN);
        group_map.entry(gid.clone()).or_default().push(val);
    }
    let groups: Vec<(String, Vec<f64>)> = group_map.into_iter().collect();

    // Process each group in parallel with rayon
    let results: Vec<(String, Vec<i64>)> = groups
        .into_par_iter()
        .map(|(gid, data)| {
            let penalty = pen.unwrap_or_else(|| 2.0 * (data.len() as f64).ln());
            let cps = pelt_single(&data, cost_fn, penalty, min_size);
            (gid, cps)
        })
        .collect();

    // Build output DataFrame
    let mut id_vals: Vec<String> = Vec::new();
    let mut cp_vals: Vec<i64> = Vec::new();

    for (gid, cps) in results {
        for cp in cps {
            id_vals.push(gid.clone());
            cp_vals.push(cp);
        }
    }

    let out_df = DataFrame::new(vec![
        Column::new(id_col.into(), &id_vals),
        Column::new("changepoint_idx".into(), &cp_vals),
    ])
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyDataFrame(out_df))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_mean_constant() {
        let data = vec![5.0; 10];
        let stats = CumStats::new(&data);
        let c = cost_mean(&stats, 0, 10);
        assert!(c.abs() < 1e-10);
    }

    #[test]
    fn test_cost_mean_shift() {
        let mut data = vec![0.0; 5];
        data.extend(vec![10.0; 5]);
        let stats = CumStats::new(&data);
        // Full segment cost should be high (mixed means)
        let full = cost_mean(&stats, 0, 10);
        // Two segments should have low cost
        let seg1 = cost_mean(&stats, 0, 5);
        let seg2 = cost_mean(&stats, 5, 10);
        assert!(full > seg1 + seg2);
    }

    #[test]
    fn test_pelt_single_no_change() {
        let data: Vec<f64> = (0..50).map(|_| 1.0).collect();
        let cps = pelt_single(&data, cost_mean, 50.0, 2);
        assert!(cps.is_empty());
    }

    #[test]
    fn test_pelt_single_one_change() {
        let mut data = vec![0.0; 50];
        data.extend(vec![10.0; 50]);
        let cps = pelt_single(&data, cost_mean, 10.0, 2);
        assert!(!cps.is_empty());
        // Changepoint should be near index 50
        assert!((cps[0] - 50).abs() <= 5);
    }
}
