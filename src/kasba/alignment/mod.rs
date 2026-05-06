/// MSM cost matrix computation and alignment path extraction.
pub mod traceback;

use traceback::compute_min_return_path;
use crate::kasba::msm::{cost_independent, squared_distance_cols, cost_dependent_cols};

// ---------------------------------------------------------------------------
// Independent mode: full cost matrix (single channel)
// ---------------------------------------------------------------------------

fn msm_cost_matrix_single_channel_into(a: &[f64], b: &[f64], c: f64, cm: &mut [f64]) {
    let n = a.len();
    let m = b.len();

    cm[0] = (a[0] - b[0]).abs();

    for i in 1..n {
        let cost = cost_independent(a[i], a[i - 1], b[0], c);
        unsafe { *cm.get_unchecked_mut(i * m) = *cm.get_unchecked((i - 1) * m) + cost };
    }

    for j in 1..m {
        let cost = cost_independent(b[j], a[0], unsafe { *b.get_unchecked(j - 1) }, c);
        unsafe { *cm.get_unchecked_mut(j) = *cm.get_unchecked(j - 1) + cost };
    }

    for i in 1..n {
        let ai = unsafe { *a.get_unchecked(i) };
        let ai1 = unsafe { *a.get_unchecked(i - 1) };
        for j in 1..m {
            unsafe {
                let bj = *b.get_unchecked(j);
                let bj1 = *b.get_unchecked(j - 1);
                let d1 = *cm.get_unchecked((i - 1) * m + (j - 1)) + (ai - bj).abs();
                let d2 = *cm.get_unchecked((i - 1) * m + j) + cost_independent(ai, ai1, bj, c);
                let d3 = *cm.get_unchecked(i * m + (j - 1)) + cost_independent(bj, ai, bj1, c);
                *cm.get_unchecked_mut(i * m + j) = d1.min(d2).min(d3);
            }
        }
    }
}

pub fn msm_cost_matrix_independent(
    x: &[f64], y: &[f64], n_channels: usize, x_len: usize, y_len: usize, c: f64,
) -> Vec<f64> {
    let total_cells = x_len * y_len;
    let mut total_cm = vec![0.0; total_cells];
    let mut ch_cm = vec![0.0; total_cells];

    for ch in 0..n_channels {
        let x_ch = &x[ch * x_len..(ch + 1) * x_len];
        let y_ch = &y[ch * y_len..(ch + 1) * y_len];
        ch_cm.fill(f64::INFINITY);
        msm_cost_matrix_single_channel_into(x_ch, y_ch, c, &mut ch_cm);
        for k in 0..total_cells {
            unsafe { *total_cm.get_unchecked_mut(k) += *ch_cm.get_unchecked(k) };
        }
    }

    total_cm
}

// ---------------------------------------------------------------------------
// Dependent mode: full cost matrix
// ---------------------------------------------------------------------------

pub fn msm_cost_matrix_dependent(
    x: &[f64], y: &[f64], n_channels: usize, x_len: usize, y_len: usize, c: f64,
) -> Vec<f64> {
    let mut cm = vec![f64::INFINITY; x_len * y_len];

    cm[0] = squared_distance_cols(x, 0, x_len, y, 0, y_len, n_channels);

    for i in 1..x_len {
        let cost = cost_dependent_cols(x, i, x_len, x, i - 1, x_len, y, 0, y_len, n_channels, c);
        cm[i * y_len] = cm[(i - 1) * y_len] + cost;
    }

    for j in 1..y_len {
        let cost = cost_dependent_cols(y, j, y_len, y, j - 1, y_len, x, 0, x_len, n_channels, c);
        cm[j] = cm[j - 1] + cost;
    }

    for i in 1..x_len {
        for j in 1..y_len {
            unsafe {
                let d1 = *cm.get_unchecked((i - 1) * y_len + (j - 1))
                    + squared_distance_cols(x, i, x_len, y, j, y_len, n_channels);
                let d2 = *cm.get_unchecked((i - 1) * y_len + j)
                    + cost_dependent_cols(x, i, x_len, x, i - 1, x_len, y, j, y_len, n_channels, c);
                let d3 = *cm.get_unchecked(i * y_len + (j - 1))
                    + cost_dependent_cols(y, j, y_len, x, i, x_len, y, j - 1, y_len, n_channels, c);
                *cm.get_unchecked_mut(i * y_len + j) = d1.min(d2).min(d3);
            }
        }
    }

    cm
}

// ---------------------------------------------------------------------------
// Alignment path
// ---------------------------------------------------------------------------

pub fn msm_alignment_path(
    x: &[f64], y: &[f64], n_channels: usize, x_len: usize, y_len: usize,
    independent: bool, c: f64,
) -> (Vec<(usize, usize)>, f64) {
    let cm = if independent {
        msm_cost_matrix_independent(x, y, n_channels, x_len, y_len, c)
    } else {
        msm_cost_matrix_dependent(x, y, n_channels, x_len, y_len, c)
    };

    let dist = cm[x_len * y_len - 1];
    let path = compute_min_return_path(&cm, x_len, y_len);
    (path, dist)
}
