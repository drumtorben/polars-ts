//! K-Medoids (PAM) clustering with precomputed distance matrix.
//!
//! Accepts a flat distance matrix and runs the Partitioning Around Medoids
//! swap algorithm in Rust, returning cluster assignments.

use pyo3::prelude::*;
use rayon::prelude::*;

/// Run the PAM swap algorithm on a precomputed distance matrix.
///
/// # Arguments
/// * `dist_flat` - Flattened n×n distance matrix (row-major)
/// * `n` - Number of series
/// * `k` - Number of clusters
/// * `max_iter` - Maximum swap iterations
/// * `seed` - Random seed for initial medoid selection
///
/// Returns (medoid_indices, cluster_assignments) as Vec<usize>.
fn pam_swap(dist_flat: &[f64], n: usize, k: usize, max_iter: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    // Simple LCG for deterministic initial selection (no external dep needed)
    let mut rng_state = seed;
    let mut rand_next = || -> usize {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as usize
    };

    // Initialize medoids via random selection
    let mut medoids: Vec<usize> = Vec::with_capacity(k);
    let mut available: Vec<usize> = (0..n).collect();
    for _ in 0..k {
        let idx = rand_next() % available.len();
        medoids.push(available[idx]);
        available.swap_remove(idx);
    }
    medoids.sort();

    #[inline]
    fn dist_at(dist_flat: &[f64], n: usize, i: usize, j: usize) -> f64 {
        dist_flat[i * n + j]
    }

    // Assign each point to nearest medoid
    let mut assignments = vec![0usize; n];
    let assign = |medoids: &[usize], assignments: &mut [usize]| {
        for (i, assignment) in assignments.iter_mut().enumerate() {
            let mut best_med = 0;
            let mut best_dist = f64::INFINITY;
            for (mi, &m) in medoids.iter().enumerate() {
                let d = dist_at(dist_flat, n, i, m);
                if d < best_dist {
                    best_dist = d;
                    best_med = mi;
                }
            }
            *assignment = best_med;
        }
    };

    let total_cost = |medoids: &[usize]| -> f64 {
        (0..n)
            .map(|i| {
                medoids
                    .iter()
                    .map(|&m| dist_at(dist_flat, n, i, m))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum()
    };

    assign(&medoids, &mut assignments);
    let mut current_cost = total_cost(&medoids);

    // PAM swap loop
    for _ in 0..max_iter {
        let mut improved = false;

        // Build all (medoid_index, candidate) swap pairs
        let non_medoids: Vec<usize> = (0..n).filter(|i| !medoids.contains(i)).collect();
        let swap_candidates: Vec<(usize, usize)> = (0..k)
            .flat_map(|mi| non_medoids.iter().map(move |&c| (mi, c)))
            .collect();

        // Evaluate all swaps in parallel
        let best_swap: Option<(usize, usize, f64)> = swap_candidates
            .par_iter()
            .filter_map(|&(mi, candidate)| {
                let mut new_medoids = medoids.clone();
                new_medoids[mi] = candidate;
                let new_cost = total_cost(&new_medoids);
                if new_cost < current_cost - 1e-12 {
                    Some((mi, candidate, new_cost))
                } else {
                    None
                }
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((mi, candidate, new_cost)) = best_swap {
            medoids[mi] = candidate;
            medoids.sort();
            current_cost = new_cost;
            assign(&medoids, &mut assignments);
            improved = true;
        }

        if !improved {
            break;
        }
    }

    (medoids, assignments)
}

/// Rust-accelerated k-medoids PAM algorithm.
///
/// Accepts a flat distance matrix (list of n*n floats, row-major),
/// the number of series n, clusters k, max iterations, and seed.
/// Returns a list of cluster assignments (0-indexed) for each series.
#[pyfunction]
#[pyo3(signature = (dist_flat, n, k, max_iter=100, seed=42))]
pub fn kmedoids_pam(
    dist_flat: Vec<f64>,
    n: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    if k < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err("k must be >= 1"));
    }
    if k > n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "k ({k}) must be <= n ({n})"
        )));
    }
    if dist_flat.len() != n * n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dist_flat length ({}) must be n*n ({})",
            dist_flat.len(),
            n * n
        )));
    }

    let (medoids, assignments) = pam_swap(&dist_flat, n, k, max_iter, seed);
    Ok((medoids, assignments))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dist_matrix(n: usize) -> Vec<f64> {
        // Simple distance: |i - j| as float
        let mut dist = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                dist[i * n + j] = (i as f64 - j as f64).abs();
            }
        }
        dist
    }

    #[test]
    fn test_single_cluster() {
        let dist = make_dist_matrix(5);
        let (medoids, assignments) = pam_swap(&dist, 5, 1, 100, 42);
        assert_eq!(medoids.len(), 1);
        assert!(assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_two_clusters_separated() {
        // Two groups: 0,1,2 close together, 100,101,102 close together
        let n = 6;
        let points = [0.0, 1.0, 2.0, 100.0, 101.0, 102.0];
        let mut dist = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                dist[i * n + j] = (points[i] - points[j]).abs();
            }
        }
        let (_medoids, assignments) = pam_swap(&dist, n, 2, 100, 42);
        // First 3 should be in same cluster, last 3 in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }

    #[test]
    fn test_k_equals_n() {
        let dist = make_dist_matrix(3);
        let (medoids, assignments) = pam_swap(&dist, 3, 3, 100, 42);
        assert_eq!(medoids.len(), 3);
        // Each point is its own medoid
        let mut sorted_assignments: Vec<usize> = assignments.clone();
        sorted_assignments.sort();
        assert_eq!(sorted_assignments, vec![0, 1, 2]);
    }
}
