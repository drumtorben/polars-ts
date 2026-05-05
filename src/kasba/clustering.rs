/// Main KASBA clustering orchestration.
use crate::kasba::assign::fast_assign;
use crate::kasba::average::kasba_average;
use crate::kasba::empty_cluster::handle_empty_clusters;
use crate::kasba::init::elastic_kmeans_plus_plus;
use crate::kasba::pairwise::{pairwise_distance_xy, DistanceMethod, DistanceParams};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[allow(clippy::too_many_arguments)]
pub fn kasba_fit(
    x_data: &[f64], n_cases: usize, n_channels: usize, ts_len: usize,
    n_clusters: usize, independent: bool, c: f64, max_iter: usize, tol: f64,
    ba_subset_size: f64, initial_step_size: f64, decay_rate: f64,
    random_seed: u64, max_ba_iters: usize, verbose: bool,
) -> (Vec<usize>, Vec<f64>, f64, usize) {
    let params = DistanceParams {
        method: DistanceMethod::Msm,
        n_channels,
        independent,
        c,
    };

    let mut rng = ChaCha8Rng::seed_from_u64(random_seed);

    let (mut centers, mut distances_to_centers, mut labels) = elastic_kmeans_plus_plus(
        x_data, n_cases, n_channels, ts_len, n_clusters, &params, &mut rng,
    );

    let stride = n_channels * ts_len;
    let mut inertia = f64::INFINITY;
    let mut prev_inertia = f64::INFINITY;
    let mut prev_labels = vec![0usize; n_cases];
    let mut prev_centers = vec![0.0; n_clusters * stride];
    let mut has_prev = false;
    let mut n_iter = 0;

    for i in 0..max_iter {
        recalculate_centroids(
            x_data, &mut centers, &labels, &mut distances_to_centers,
            n_cases, n_clusters, n_channels, ts_len, tol, ba_subset_size,
            initial_step_size, decay_rate, independent, c, &mut rng,
            max_ba_iters, verbose,
        );

        inertia = fast_assign(
            x_data, &centers, n_cases, n_clusters, n_channels, ts_len,
            &mut distances_to_centers, &mut labels, i == 0, &params,
        );

        let ok = handle_empty_clusters(
            x_data, &mut centers, &mut distances_to_centers, &mut labels,
            n_cases, n_clusters, n_channels, ts_len, &params,
        );
        if !ok {
            eprintln!("Warning: Empty cluster recovery exceeded max iterations");
        }

        if has_prev && prev_labels == labels {
            if verbose {
                eprintln!("Converged at iteration {}, inertia {:.5}.", i, inertia);
            }
            n_iter = i + 1;
            break;
        }

        prev_inertia = inertia;
        prev_labels.copy_from_slice(&labels);
        prev_centers.copy_from_slice(&centers);
        has_prev = true;
        n_iter = i + 1;

        if verbose {
            eprintln!("Iteration {}, inertia {}.", i, prev_inertia);
        }
    }

    if has_prev && inertia < prev_inertia {
        return (prev_labels, prev_centers, prev_inertia, n_iter);
    }

    (labels, centers, inertia, n_iter)
}

#[allow(clippy::too_many_arguments)]
fn recalculate_centroids(
    x_data: &[f64], centers: &mut Vec<f64>, labels: &[usize],
    distances_to_centers: &mut Vec<f64>, n_cases: usize, n_clusters: usize,
    n_channels: usize, ts_len: usize, tol: f64, ba_subset_size: f64,
    initial_step_size: f64, decay_rate: f64, independent: bool, c: f64,
    rng: &mut ChaCha8Rng, max_ba_iters: usize, verbose: bool,
) {
    let stride = n_channels * ts_len;

    for j in 0..n_clusters {
        let cluster_indices: Vec<usize> = (0..n_cases).filter(|&i| labels[i] == j).collect();
        let cluster_size = cluster_indices.len();

        if cluster_size == 0 {
            continue;
        }

        let mut cluster_data = vec![0.0; cluster_size * stride];
        let mut cluster_distances = vec![0.0; cluster_size];
        for (local_idx, &global_idx) in cluster_indices.iter().enumerate() {
            cluster_data[local_idx * stride..(local_idx + 1) * stride]
                .copy_from_slice(&x_data[global_idx * stride..(global_idx + 1) * stride]);
            cluster_distances[local_idx] = distances_to_centers[global_idx];
        }

        let previous_cost: f64 = cluster_distances.iter().sum();
        let init_barycenter = centers[j * stride..(j + 1) * stride].to_vec();

        let (new_center, dist_to_center) = kasba_average(
            &cluster_data, cluster_size, n_channels, ts_len, &init_barycenter,
            previous_cost, &cluster_distances, max_ba_iters, tol, ba_subset_size,
            initial_step_size, decay_rate, DistanceMethod::Msm, independent, c,
            rng, verbose,
        );

        centers[j * stride..(j + 1) * stride].copy_from_slice(&new_center);
        for (local_idx, &global_idx) in cluster_indices.iter().enumerate() {
            distances_to_centers[global_idx] = dist_to_center[local_idx];
        }
    }
}

pub fn kasba_predict(
    x_data: &[f64], centers: &[f64], n_cases: usize, n_clusters: usize,
    _n_channels: usize, ts_len: usize, params: &DistanceParams,
) -> Vec<usize> {
    let pw = pairwise_distance_xy(x_data, centers, n_cases, n_clusters, ts_len, ts_len, params);

    let mut labels = vec![0usize; n_cases];
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
    }
    labels
}
