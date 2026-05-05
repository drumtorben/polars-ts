/// Fast assignment step exploiting the triangle inequality.
use rayon::prelude::*;

use crate::kasba::pairwise::{compute_distance, pairwise_distance_self, DistanceParams};

const ASSIGN_PARALLEL_THRESHOLD: usize = 16;

pub fn fast_assign(
    x_data: &[f64], centers: &[f64], n_cases: usize, n_clusters: usize,
    n_channels: usize, ts_len: usize,
    distances_to_centers: &mut Vec<f64>, labels: &mut Vec<usize>,
    is_first_iteration: bool, params: &DistanceParams,
) -> f64 {
    let stride = n_channels * ts_len;
    let center_distances = pairwise_distance_self(centers, n_clusters, ts_len, params);

    if n_cases >= ASSIGN_PARALLEL_THRESHOLD {
        let results: Vec<(usize, f64)> = (0..n_cases)
            .into_par_iter()
            .map(|i| {
                let mut min_dist = distances_to_centers[i];
                let mut closest = labels[i];
                let x_i = &x_data[i * stride..(i + 1) * stride];

                for j in 0..n_clusters {
                    if !is_first_iteration && j == closest {
                        continue;
                    }
                    let bound = center_distances[j * n_clusters + closest] / 2.0;
                    if min_dist < bound {
                        continue;
                    }
                    let center_j = &centers[j * stride..(j + 1) * stride];
                    let dist = compute_distance(x_i, center_j, ts_len, ts_len, params);
                    if dist < min_dist {
                        min_dist = dist;
                        closest = j;
                    }
                }
                (closest, min_dist)
            })
            .collect();

        let mut inertia = 0.0;
        for (i, (label, dist)) in results.into_iter().enumerate() {
            labels[i] = label;
            distances_to_centers[i] = dist;
            inertia += dist * dist;
        }
        inertia
    } else {
        for i in 0..n_cases {
            let mut min_dist = distances_to_centers[i];
            let mut closest = labels[i];
            let x_i = &x_data[i * stride..(i + 1) * stride];

            for j in 0..n_clusters {
                if !is_first_iteration && j == closest {
                    continue;
                }
                let bound = center_distances[j * n_clusters + closest] / 2.0;
                if min_dist < bound {
                    continue;
                }
                let center_j = &centers[j * stride..(j + 1) * stride];
                let dist = compute_distance(x_i, center_j, ts_len, ts_len, params);
                if dist < min_dist {
                    min_dist = dist;
                    closest = j;
                }
            }
            labels[i] = closest;
            distances_to_centers[i] = min_dist;
        }

        distances_to_centers.iter().map(|d| d * d).sum()
    }
}
