//! PyO3 bindings for KASBA clustering (issue #192).
//!
//! Exposes `kasba_fit` and `kasba_predict` as Python-callable functions
//! that accept 3D numpy arrays of shape (n_cases, n_channels, n_timepoints).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::kasba::clustering;
use crate::kasba::pairwise::{DistanceMethod, DistanceParams};

/// Fit KASBA clustering on multivariate time series data.
///
/// # Arguments
/// * `data` - 3D array of shape (n_cases, n_channels, n_timepoints), f64
/// * `n_clusters` - Number of clusters
/// * `c` - MSM cost parameter
/// * `independent` - MSM mode (true=per-channel, false=multivariate dependent)
/// * `ba_subset_size` - Fraction of cluster members for barycenter averaging
/// * `initial_step_size` - Starting SGD step size
/// * `decay_rate` - Exponential decay rate for step size
/// * `n_iters` - Maximum clustering iterations
/// * `random_seed` - Random seed for determinism
///
/// # Returns
/// Tuple of (labels, centers, inertia, n_iter)
#[pyfunction]
#[pyo3(signature = (
    data,
    n_clusters = 8,
    c = 1.0,
    independent = true,
    ba_subset_size = 0.5,
    initial_step_size = 0.05,
    decay_rate = 0.1,
    n_iters = 10,
    random_seed = 42
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn kasba_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    n_clusters: usize,
    c: f64,
    independent: bool,
    ba_subset_size: f64,
    initial_step_size: f64,
    decay_rate: f64,
    n_iters: usize,
    random_seed: u64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>, f64, usize)> {
    let shape = data.shape();
    let n_cases = shape[0];
    let n_channels = shape[1];
    let ts_len = shape[2];

    if n_cases == 0 || n_channels == 0 || ts_len == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "data dimensions must all be > 0",
        ));
    }
    if n_clusters < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_clusters must be >= 1",
        ));
    }
    if n_clusters > n_cases {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_clusters ({n_clusters}) must be <= n_cases ({n_cases})"
        )));
    }

    // Flatten to channel-major layout: for each series, channels are contiguous
    // Input numpy shape: (n_cases, n_channels, n_timepoints) — already channel-major per series
    let array = data.as_array();
    let mut flat = Vec::with_capacity(n_cases * n_channels * ts_len);
    for case in 0..n_cases {
        for ch in 0..n_channels {
            for t in 0..ts_len {
                flat.push(array[[case, ch, t]]);
            }
        }
    }

    let (labels, centers, inertia, n_iter) = clustering::kasba_fit(
        &flat,
        n_cases,
        n_channels,
        ts_len,
        n_clusters,
        independent,
        c,
        n_iters,
        1e-6,             // tol
        ba_subset_size,
        initial_step_size,
        decay_rate,
        random_seed,
        50,    // max_ba_iters
        false, // verbose
    );

    // Convert labels to i64 numpy array
    let labels_i64: Vec<i64> = labels.into_iter().map(|l| l as i64).collect();
    let labels_array = labels_i64.into_pyarray(py);

    // Reshape centers to (n_clusters, n_channels * ts_len)
    let stride = n_channels * ts_len;
    let centers_array = Array2::from_shape_vec((n_clusters, stride), centers)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Center reshape failed: {e}")))?;
    let centers_py = centers_array.into_pyarray(py);

    Ok((labels_array, centers_py, inertia, n_iter))
}

/// Predict cluster assignments for new data using pre-computed centers.
///
/// # Arguments
/// * `centers` - 2D array of shape (n_clusters, n_channels * n_timepoints)
/// * `data` - 3D array of shape (n_cases, n_channels, n_timepoints)
/// * `c` - MSM cost parameter
/// * `independent` - MSM mode
///
/// # Returns
/// 1D array of cluster labels
#[pyfunction]
#[pyo3(signature = (centers, data, c = 1.0, independent = true))]
pub fn kasba_predict<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray2<'py, f64>,
    data: PyReadonlyArray3<'py, f64>,
    c: f64,
    independent: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let data_shape = data.shape();
    let n_cases = data_shape[0];
    let n_channels = data_shape[1];
    let ts_len = data_shape[2];

    let centers_shape = centers.shape();
    let n_clusters = centers_shape[0];
    let expected_stride = n_channels * ts_len;

    if centers_shape[1] != expected_stride {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "centers shape[1] ({}) must equal n_channels * n_timepoints ({})",
            centers_shape[1], expected_stride
        )));
    }

    let params = DistanceParams {
        method: DistanceMethod::Msm,
        n_channels,
        independent,
        c,
    };

    // Flatten data (channel-major per series)
    let data_arr = data.as_array();
    let mut flat_data = Vec::with_capacity(n_cases * n_channels * ts_len);
    for case in 0..n_cases {
        for ch in 0..n_channels {
            for t in 0..ts_len {
                flat_data.push(data_arr[[case, ch, t]]);
            }
        }
    }

    // Flatten centers — already stored as (n_clusters, n_channels * ts_len) row-major
    let centers_arr = centers.as_array();
    let flat_centers: Vec<f64> = centers_arr.iter().copied().collect();

    let labels = clustering::kasba_predict(
        &flat_data,
        &flat_centers,
        n_cases,
        n_clusters,
        n_channels,
        ts_len,
        &params,
    );

    let labels_i64: Vec<i64> = labels.into_iter().map(|l| l as i64).collect();
    Ok(labels_i64.into_pyarray(py))
}
