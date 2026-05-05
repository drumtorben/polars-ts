/// KASBA barycenter averaging using stochastic subgradient descent.
use crate::kasba::alignment::msm_alignment_path;
use crate::kasba::pairwise::{pairwise_distance_xy, DistanceMethod, DistanceParams};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

fn get_alignment_path(
    center: &[f64], ts: &[f64], n_channels: usize, ts_len: usize,
    independent: bool, c: f64,
) -> (Vec<(usize, usize)>, f64) {
    msm_alignment_path(ts, center, n_channels, ts_len, ts_len, independent, c)
}

fn kasba_refine_one_iter(
    barycenter: &[f64], x_data: &[f64], n_channels: usize, ts_len: usize,
    shuffled_indices: &[usize], current_step_size: f64, independent: bool, c: f64,
) -> Vec<f64> {
    let stride = n_channels * ts_len;
    let mut bc = barycenter.to_vec();
    let mut new_ba = vec![0.0; stride];

    for &idx in shuffled_indices {
        let curr_ts = &x_data[idx * stride..(idx + 1) * stride];
        let (alignment, _) = get_alignment_path(&bc, curr_ts, n_channels, ts_len, independent, c);

        new_ba.fill(0.0);
        for &(j, k) in &alignment {
            for ch in 0..n_channels {
                new_ba[ch * ts_len + k] += bc[ch * ts_len + k] - curr_ts[ch * ts_len + j];
            }
        }

        let factor = 2.0 * current_step_size;
        for i in 0..stride {
            bc[i] -= factor * new_ba[i];
        }
    }

    bc
}

#[allow(clippy::too_many_arguments)]
pub fn kasba_average(
    x_data: &[f64], n_cases: usize, n_channels: usize, ts_len: usize,
    init_barycenter: &[f64], previous_cost: f64, previous_distances: &[f64],
    max_iters: usize, tol: f64, ba_subset_size: f64,
    initial_step_size: f64, decay_rate: f64,
    method: DistanceMethod, independent: bool, c: f64,
    rng: &mut ChaCha8Rng, verbose: bool,
) -> (Vec<f64>, Vec<f64>) {
    let _stride = n_channels * ts_len;

    if n_cases <= 1 {
        return (init_barycenter.to_vec(), vec![0.0; n_cases]);
    }

    let num_ts_to_use = n_cases.min(10_usize.max((ba_subset_size * n_cases as f64) as usize));

    let mut barycenter = init_barycenter.to_vec();
    let mut prev_barycenter = init_barycenter.to_vec();
    let mut prev_cost = previous_cost;
    let mut prev_distances = previous_distances.to_vec();
    let mut distances_to_center = prev_distances.clone();

    let params = DistanceParams {
        method,
        n_channels,
        independent,
        c,
    };

    let mut perm_buf: Vec<usize> = (0..n_cases).collect();

    for i in 0..max_iters {
        perm_buf.shuffle(rng);

        let indices: &[usize] = if i > 0 {
            &perm_buf[..num_ts_to_use]
        } else {
            &perm_buf
        };

        let current_step_size = initial_step_size * (-decay_rate * i as f64).exp();

        barycenter = kasba_refine_one_iter(
            &barycenter, x_data, n_channels, ts_len, indices,
            current_step_size, independent, c,
        );

        let pw_dist =
            pairwise_distance_xy(x_data, &barycenter, n_cases, 1, ts_len, ts_len, &params);
        let cost: f64 = pw_dist.iter().sum();
        distances_to_center = pw_dist;

        if (prev_cost - cost).abs() < tol {
            if prev_cost < cost {
                barycenter = prev_barycenter;
                distances_to_center = prev_distances;
            }
            if verbose {
                eprintln!(
                    "[KASBA-BA] epoch {}, early convergence change in cost between epochs {} - {} < tol: {}",
                    i, cost, prev_cost, tol
                );
            }
            break;
        } else if prev_cost < cost {
            barycenter = prev_barycenter;
            distances_to_center = prev_distances;
            if verbose {
                eprintln!(
                    "[KASBA-BA] epoch {}, early convergence cost increasing: {} > previous cost: {}",
                    i, cost, prev_cost
                );
            }
            break;
        } else {
            prev_barycenter = barycenter.clone();
            prev_distances = distances_to_center.clone();
            prev_cost = cost;
        }

        if verbose {
            eprintln!("[KASBA-BA] epoch {}, cost {}", i, cost);
        }
    }

    (barycenter, distances_to_center)
}
