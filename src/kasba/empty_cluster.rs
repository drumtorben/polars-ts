/// Empty cluster recovery.
use crate::kasba::pairwise::{pairwise_distance_xy, DistanceParams};

pub fn handle_empty_clusters(
    x_data: &[f64], centers: &mut Vec<f64>, distances_to_centers: &mut Vec<f64>,
    labels: &mut Vec<usize>, n_cases: usize, n_clusters: usize,
    n_channels: usize, ts_len: usize, params: &DistanceParams,
) -> bool {
    let stride = n_channels * ts_len;
    let mut recovery_count = 0;

    loop {
        let mut cluster_occupied = vec![false; n_clusters];
        for &l in labels.iter() {
            cluster_occupied[l] = true;
        }

        let empty_clusters: Vec<usize> =
            (0..n_clusters).filter(|&k| !cluster_occupied[k]).collect();

        if empty_clusters.is_empty() {
            break;
        }

        let empty_cluster_idx = empty_clusters[0];

        let furthest_idx = distances_to_centers
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let src_start = furthest_idx * stride;
        let dst_start = empty_cluster_idx * stride;
        centers[dst_start..dst_start + stride]
            .copy_from_slice(&x_data[src_start..src_start + stride]);

        let pw = pairwise_distance_xy(x_data, centers, n_cases, n_clusters, ts_len, ts_len, params);

        for i in 0..n_cases {
            let mut min_dist = f64::INFINITY;
            let mut min_j = 0;
            for j in 0..n_clusters {
                let d = pw[i * n_clusters + j];
                if d < min_dist {
                    min_dist = d;
                    min_j = j;
                }
            }
            labels[i] = min_j;
            distances_to_centers[i] = min_dist;
        }

        recovery_count += 1;
        if recovery_count > n_clusters {
            return false;
        }
    }

    true
}
