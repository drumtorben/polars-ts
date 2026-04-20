//! Exponential smoothing forecasters: SES, Holt, Holt-Winters.
//!
//! Pure numerical core — accepts a slice of f64 values and parameters,
//! returns a Vec of h forecast values. The Python wrapper handles
//! grouping, time columns, and DataFrame construction.

use pyo3::prelude::*;

/// Simple Exponential Smoothing.
///
/// Smooths with level parameter `alpha` and projects a flat forecast of length `h`.
fn ses_core(values: &[f64], alpha: f64, h: usize) -> Vec<f64> {
    let mut level = values[0];
    for &v in &values[1..] {
        level = alpha * v + (1.0 - alpha) * level;
    }
    vec![level; h]
}

/// Holt's linear trend method.
///
/// Smooths level and trend, then extrapolates linearly for `h` steps.
fn holt_core(values: &[f64], alpha: f64, beta: f64, h: usize) -> Vec<f64> {
    let mut level = values[0];
    let mut trend = values[1] - values[0];

    for &v in &values[1..] {
        let prev_level = level;
        level = alpha * v + (1.0 - alpha) * (level + trend);
        trend = beta * (level - prev_level) + (1.0 - beta) * trend;
    }

    (1..=h).map(|step| level + step as f64 * trend).collect()
}

/// Holt-Winters seasonal method (additive or multiplicative).
///
/// Smooths level, trend, and seasonal components, then forecasts `h` steps.
fn holt_winters_core(
    values: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    m: usize,
    additive: bool,
    h: usize,
) -> Vec<f64> {
    let n = values.len();

    // Initialize
    let first_season_avg: f64 = values[..m].iter().sum::<f64>() / m as f64;
    let mut level = first_season_avg;
    let second_season_avg: f64 = values[m..2 * m].iter().sum::<f64>() / m as f64;
    let mut trend = (second_season_avg - first_season_avg) / m as f64;

    let mut seasons: Vec<f64> = if additive {
        values[..m].iter().map(|v| v - first_season_avg).collect()
    } else {
        values[..m]
            .iter()
            .map(|v| {
                if first_season_avg != 0.0 {
                    v / first_season_avg
                } else {
                    1.0
                }
            })
            .collect()
    };

    // Smooth through observations starting from index m
    // We need `t` for both values[t] and seasonal index t % m
    #[allow(clippy::needless_range_loop)]
    for t in m..n {
        let v = values[t];
        let s_idx = t % m;
        let prev_level = level;

        if additive {
            level = alpha * (v - seasons[s_idx]) + (1.0 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
            seasons[s_idx] = gamma * (v - level) + (1.0 - gamma) * seasons[s_idx];
        } else {
            level = if seasons[s_idx] != 0.0 {
                alpha * (v / seasons[s_idx])
            } else {
                alpha * v
            } + (1.0 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
            seasons[s_idx] = gamma
                * (if level != 0.0 {
                    v / level
                } else {
                    1.0
                })
                + (1.0 - gamma) * seasons[s_idx];
        }
    }

    // Forecast
    (1..=h)
        .map(|step| {
            let s_idx = (n - 1 + step) % m;
            if additive {
                level + step as f64 * trend + seasons[s_idx]
            } else {
                (level + step as f64 * trend) * seasons[s_idx]
            }
        })
        .collect()
}

// --- PyO3 exports ---

/// Rust-accelerated Simple Exponential Smoothing.
///
/// Takes values for a single group and returns h forecast values.
#[pyfunction]
#[pyo3(signature = (values, alpha, h))]
pub fn ets_ses(values: Vec<f64>, alpha: f64, h: usize) -> PyResult<Vec<f64>> {
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values must not be empty",
        ));
    }
    Ok(ses_core(&values, alpha, h))
}

/// Rust-accelerated Holt's linear trend method.
#[pyfunction]
#[pyo3(signature = (values, alpha, beta, h))]
pub fn ets_holt(values: Vec<f64>, alpha: f64, beta: f64, h: usize) -> PyResult<Vec<f64>> {
    if values.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values must have at least 2 elements",
        ));
    }
    Ok(holt_core(&values, alpha, beta, h))
}

/// Rust-accelerated Holt-Winters seasonal method.
#[pyfunction]
#[pyo3(signature = (values, alpha, beta, gamma, season_length, additive, h))]
pub fn ets_holt_winters(
    values: Vec<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    season_length: usize,
    additive: bool,
    h: usize,
) -> PyResult<Vec<f64>> {
    if values.len() < 2 * season_length {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "values must have at least 2*season_length={} elements, got {}",
            2 * season_length,
            values.len()
        )));
    }
    Ok(holt_winters_core(
        &values,
        alpha,
        beta,
        gamma,
        season_length,
        additive,
        h,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ses_constant() {
        let values = vec![5.0; 20];
        let fc = ses_core(&values, 0.3, 3);
        assert_eq!(fc.len(), 3);
        for v in &fc {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ses_flat_forecast() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let fc = ses_core(&values, 0.3, 3);
        // SES produces flat forecast
        assert!((fc[0] - fc[1]).abs() < 1e-10);
        assert!((fc[1] - fc[2]).abs() < 1e-10);
    }

    #[test]
    fn test_holt_trend() {
        let values: Vec<f64> = (0..20).map(|i| i as f64 + 10.0).collect();
        let fc = holt_core(&values, 0.3, 0.1, 3);
        // Upward trend
        assert!(fc[1] > fc[0]);
        assert!(fc[2] > fc[1]);
    }

    #[test]
    fn test_holt_winters_additive() {
        // Simple seasonal pattern
        let m = 4;
        let values: Vec<f64> = (0..24)
            .map(|i| 10.0 + 0.5 * i as f64 + 3.0 * ((i % m) as f64 - 1.5))
            .collect();
        let fc = holt_winters_core(&values, 0.3, 0.1, 0.1, m, true, 4);
        assert_eq!(fc.len(), 4);
    }

    #[test]
    fn test_holt_winters_multiplicative() {
        let m = 4;
        let values: Vec<f64> = (0..24)
            .map(|i| (20.0 + i as f64) * (1.0 + 0.3 * ((i % m) as f64 / m as f64)))
            .collect();
        let fc = holt_winters_core(&values, 0.3, 0.1, 0.1, m, false, 4);
        assert_eq!(fc.len(), 4);
        for v in &fc {
            assert!(v.is_finite());
        }
    }
}
