/// Elastic k-means++ initialization for KASBA.
use crate::kasba::pairwise::{pairwise_distance_xy, DistanceParams};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

pub fn elastic_kmeans_plus_plus(
    x_data: &[f64], n_cases: usize, n_channels: usize, ts_len: usize,
    n_clusters: usize, params: &DistanceParams, rng: &mut ChaCha8Rng,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let stride = n_channels * ts_len;
    let mut centers = Vec::with_capacity(n_clusters * stride);
    let mut indexes = Vec::with_capacity(n_clusters);

    let init_center_idx = rng.gen_range(0..n_cases);
    indexes.push(init_center_idx);
    centers.extend_from_slice(&x_data[init_center_idx * stride..(init_center_idx + 1) * stride]);

    let first_center = &x_data[init_center_idx * stride..(init_center_idx + 1) * stride];
    let pw = pairwise_distance_xy(x_data, first_center, n_cases, 1, ts_len, ts_len, params);
    let mut min_distances = pw;
    let mut labels = vec![0usize; n_cases];

    for i in 1..n_clusters {
        let sum: f64 = min_distances.iter().sum();
        let random_val: f64 = rng.gen();

        let next_center_idx = if sum > 0.0 {
            let threshold = random_val * sum;
            let mut cumsum = 0.0;
            let mut selected = n_cases - 1;
            for j in 0..n_cases {
                cumsum += min_distances[j];
                if cumsum > threshold {
                    selected = j;
                    break;
                }
            }
            selected
        } else {
            rng.gen_range(0..n_cases)
        };

        indexes.push(next_center_idx);
        centers.extend_from_slice(&x_data[next_center_idx * stride..(next_center_idx + 1) * stride]);

        let new_center = &x_data[next_center_idx * stride..(next_center_idx + 1) * stride];
        let new_distances =
            pairwise_distance_xy(x_data, new_center, n_cases, 1, ts_len, ts_len, params);

        for j in 0..n_cases {
            if new_distances[j] < min_distances[j] {
                min_distances[j] = new_distances[j];
                labels[j] = i;
            }
        }
    }

    (centers, min_distances, labels)
}
