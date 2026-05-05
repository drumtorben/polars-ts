/// Pairwise distance matrix computation with adaptive parallelism.
use rayon::prelude::*;

use super::msm::msm_distance;

const PARALLEL_THRESHOLD: usize = 16;

#[derive(Clone, Copy)]
pub enum DistanceMethod {
    Msm,
}

#[derive(Clone, Copy)]
pub struct DistanceParams {
    pub method: DistanceMethod,
    pub n_channels: usize,
    pub independent: bool,
    pub c: f64,
}

#[inline]
pub fn compute_distance(
    x: &[f64], y: &[f64], x_len: usize, y_len: usize, params: &DistanceParams,
) -> f64 {
    match params.method {
        DistanceMethod::Msm => msm_distance(
            x, y, params.n_channels, x_len, y_len, params.independent, params.c,
        ),
    }
}

pub fn pairwise_distance_xy(
    x_data: &[f64], y_data: &[f64], n_x: usize, n_y: usize,
    x_len: usize, y_len: usize, params: &DistanceParams,
) -> Vec<f64> {
    let stride_x = params.n_channels * x_len;
    let stride_y = params.n_channels * y_len;
    let mut result = vec![0.0; n_x * n_y];

    if n_x >= PARALLEL_THRESHOLD {
        result.par_chunks_mut(n_y).enumerate().for_each(|(i, row)| {
            let x_row = &x_data[i * stride_x..(i + 1) * stride_x];
            for j in 0..n_y {
                let y_row = &y_data[j * stride_y..(j + 1) * stride_y];
                row[j] = compute_distance(x_row, y_row, x_len, y_len, params);
            }
        });
    } else {
        for i in 0..n_x {
            let x_row = &x_data[i * stride_x..(i + 1) * stride_x];
            let row = &mut result[i * n_y..(i + 1) * n_y];
            for j in 0..n_y {
                let y_row = &y_data[j * stride_y..(j + 1) * stride_y];
                row[j] = compute_distance(x_row, y_row, x_len, y_len, params);
            }
        }
    }

    result
}

pub fn pairwise_distance_self(
    x_data: &[f64], n_cases: usize, ts_len: usize, params: &DistanceParams,
) -> Vec<f64> {
    let stride = params.n_channels * ts_len;
    let mut matrix = vec![0.0; n_cases * n_cases];

    for i in 0..n_cases {
        let x_row = &x_data[i * stride..(i + 1) * stride];
        for j in (i + 1)..n_cases {
            let y_row = &x_data[j * stride..(j + 1) * stride];
            let d = compute_distance(x_row, y_row, ts_len, ts_len, params);
            matrix[i * n_cases + j] = d;
            matrix[j * n_cases + i] = d;
        }
    }
    matrix
}
