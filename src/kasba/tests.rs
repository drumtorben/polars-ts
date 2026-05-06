#[cfg(test)]
mod tests {
    use crate::kasba::msm::{cost_independent, msm_distance, msm_distance_univariate};
    use crate::kasba::pairwise::{
        pairwise_distance_self, pairwise_distance_xy, DistanceMethod,
        DistanceParams,
    };
    use crate::kasba::alignment::{msm_alignment_path, traceback::compute_min_return_path};
    use crate::kasba::average::kasba_average;
    use crate::kasba::init::elastic_kmeans_plus_plus;
    use crate::kasba::assign::fast_assign;
    use crate::kasba::empty_cluster::handle_empty_clusters;
    use crate::kasba::clustering::{kasba_fit, kasba_predict};

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // -----------------------------------------------------------------------
    // 1. MSM distance: known pair -> expected distance value
    // -----------------------------------------------------------------------

    #[test]
    fn test_msm_distance_univariate_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = msm_distance_univariate(&a, &a, 1.0);
        assert!((d - 0.0).abs() < 1e-10, "Identical series should have distance 0, got {d}");
    }

    #[test]
    fn test_msm_distance_univariate_known_pair() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0];
        let d = msm_distance_univariate(&a, &b, 1.0);
        assert!(d > 0.0, "Different series should have distance > 0, got {d}");
        // MSM(a, b, c=1) for a=[1,2,3], b=[1,2,4]: diagonal path gives |3-4|=1.0
        assert!((d - 1.0).abs() < 1e-10, "Expected MSM distance 1.0, got {d}");
    }

    #[test]
    fn test_msm_distance_univariate_symmetry() {
        let a = vec![1.0, 3.0, 5.0, 2.0];
        let b = vec![2.0, 4.0, 1.0];
        let d_ab = msm_distance_univariate(&a, &b, 1.0);
        let d_ba = msm_distance_univariate(&b, &a, 1.0);
        assert!((d_ab - d_ba).abs() < 1e-10, "MSM should be symmetric: {d_ab} vs {d_ba}");
    }

    #[test]
    fn test_msm_distance_empty_series() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0, 2.0];
        assert_eq!(msm_distance_univariate(&a, &b, 1.0), 0.0);
        assert_eq!(msm_distance_univariate(&b, &a, 1.0), 0.0);
    }

    #[test]
    fn test_cost_independent_between() {
        // x between y and z: cost = c
        assert_eq!(cost_independent(2.0, 1.0, 3.0, 1.0), 1.0);
        assert_eq!(cost_independent(2.0, 3.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn test_cost_independent_outside() {
        // x not between y and z: cost = c + min(|x-y|, |x-z|)
        let cost = cost_independent(5.0, 1.0, 3.0, 1.0);
        // min(|5-1|, |5-3|) = min(4, 2) = 2; total = 1+2 = 3
        assert!((cost - 3.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 2. MSM distance multivariate (independent): sum of per-channel distances
    // -----------------------------------------------------------------------

    #[test]
    fn test_msm_distance_independent_multivariate() {
        // 2 channels, 3 timepoints each
        // Layout: [ch0_t0, ch0_t1, ch0_t2, ch1_t0, ch1_t1, ch1_t2]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![1.0, 2.0, 4.0, 4.0, 5.0, 7.0];
        let d = msm_distance(&x, &y, 2, 3, 3, true, 1.0);
        // Independent: sum of per-channel MSM distances
        let d_ch0 = msm_distance_univariate(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0], 1.0);
        let d_ch1 = msm_distance_univariate(&[4.0, 5.0, 6.0], &[4.0, 5.0, 7.0], 1.0);
        assert!((d - (d_ch0 + d_ch1)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 3. MSM distance multivariate (dependent): cross-channel DP
    // -----------------------------------------------------------------------

    #[test]
    fn test_msm_distance_dependent_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = msm_distance(&x, &x, 2, 3, 3, false, 1.0);
        assert!((d - 0.0).abs() < 1e-10, "Identical multivariate series should be 0, got {d}");
    }

    #[test]
    fn test_msm_distance_dependent_vs_independent() {
        let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let y = vec![1.5, 2.5, 3.5, 10.5, 20.5, 30.5];
        let d_ind = msm_distance(&x, &y, 2, 3, 3, true, 1.0);
        let d_dep = msm_distance(&x, &y, 2, 3, 3, false, 1.0);
        // Both should be > 0; they use different cost functions
        assert!(d_ind > 0.0);
        assert!(d_dep > 0.0);
    }

    // -----------------------------------------------------------------------
    // 4. Alignment path: valid path from (0,0) to (n-1, m-1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_alignment_path_endpoints() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.5, 3.0];
        let (path, dist) = msm_alignment_path(&x, &y, 1, 3, 3, true, 1.0);
        assert!(dist >= 0.0);
        assert!(path.contains(&(0, 0)), "Path must contain (0,0)");
        assert!(path.contains(&(2, 2)), "Path must contain (n-1, m-1)");
    }

    #[test]
    fn test_alignment_path_adjacent_steps() {
        let x = vec![1.0, 3.0, 2.0, 4.0];
        let y = vec![1.0, 2.0, 3.0];
        let (path, _) = msm_alignment_path(&x, &y, 1, 4, 3, true, 1.0);
        // Each step should move at most 1 in each dimension
        for w in path.windows(2) {
            let (i0, j0) = w[0];
            let (i1, j1) = w[1];
            let di = (i0 as isize - i1 as isize).unsigned_abs();
            let dj = (j0 as isize - j1 as isize).unsigned_abs();
            assert!(di <= 1 && dj <= 1, "Steps should be adjacent: ({i0},{j0})->({i1},{j1})");
        }
    }

    #[test]
    fn test_traceback_simple() {
        // 2x2 cost matrix where diagonal is cheapest
        let cm = vec![0.0, 1.0, 1.0, 0.5];
        let path = compute_min_return_path(&cm, 2, 2);
        assert!(path.contains(&(0, 0)));
        assert!(path.contains(&(1, 1)));
    }

    // -----------------------------------------------------------------------
    // 5. Stochastic barycenter: output length matches input, cost decreases
    // -----------------------------------------------------------------------

    #[test]
    fn test_barycenter_output_length() {
        let n_cases = 5;
        let n_channels = 1;
        let ts_len = 4;
        let stride = n_channels * ts_len;
        // 5 simple univariate series of length 4
        let x_data: Vec<f64> = (0..n_cases)
            .flat_map(|i| (0..ts_len).map(move |t| (i as f64) + (t as f64) * 0.1))
            .collect();

        let init_bc = x_data[0..stride].to_vec();
        let params = DistanceParams {
            method: DistanceMethod::Msm,
            n_channels,
            independent: true,
            c: 1.0,
        };
        let init_distances =
            pairwise_distance_xy(&x_data, &init_bc, n_cases, 1, ts_len, ts_len, &params);
        let init_cost: f64 = init_distances.iter().sum();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (bc, dists) = kasba_average(
            &x_data, n_cases, n_channels, ts_len, &init_bc, init_cost, &init_distances,
            10, 1e-5, 1.0, 0.05, 0.5, DistanceMethod::Msm, true, 1.0, &mut rng, false,
        );
        assert_eq!(bc.len(), stride, "Barycenter should have length n_channels*ts_len");
        assert_eq!(dists.len(), n_cases, "Should return one distance per case");
    }

    // -----------------------------------------------------------------------
    // 6. k-means++ init: returns k distinct centers
    // -----------------------------------------------------------------------

    #[test]
    fn test_kmeans_pp_init_returns_k_centers() {
        let n_cases = 10;
        let n_channels = 1;
        let ts_len = 3;
        let n_clusters = 3;
        let stride = n_channels * ts_len;
        // 10 univariate series of length 3
        let x_data: Vec<f64> = (0..n_cases)
            .flat_map(|i| vec![i as f64, i as f64 + 0.5, i as f64 + 1.0])
            .collect();

        let params = DistanceParams {
            method: DistanceMethod::Msm,
            n_channels,
            independent: true,
            c: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (centers, min_dists, labels) =
            elastic_kmeans_plus_plus(&x_data, n_cases, n_channels, ts_len, n_clusters, &params, &mut rng);

        assert_eq!(centers.len(), n_clusters * stride);
        assert_eq!(min_dists.len(), n_cases);
        assert_eq!(labels.len(), n_cases);
        // All labels should be in [0, n_clusters)
        for &l in &labels {
            assert!(l < n_clusters, "Label {l} out of range");
        }
    }

    // -----------------------------------------------------------------------
    // 7. Triangle inequality assignment: same labels as brute-force
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_assign_matches_brute_force() {
        let n_cases = 6;
        let n_channels = 1;
        let ts_len = 3;
        let n_clusters = 2;
        let stride = n_channels * ts_len;

        // Two distinct groups
        let x_data: Vec<f64> = vec![
            0.0, 0.1, 0.2,  // close to center 0
            0.1, 0.0, 0.3,
            0.0, 0.2, 0.1,
            10.0, 10.1, 10.2, // close to center 1
            10.1, 10.0, 10.3,
            10.0, 10.2, 10.1,
        ];
        let centers: Vec<f64> = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let params = DistanceParams {
            method: DistanceMethod::Msm,
            n_channels,
            independent: true,
            c: 1.0,
        };

        // Initialize with all-zero distances and labels
        let mut distances = vec![f64::INFINITY; n_cases];
        let mut labels = vec![0usize; n_cases];

        let _inertia = fast_assign(
            &x_data, &centers, n_cases, n_clusters, n_channels, ts_len,
            &mut distances, &mut labels, true, &params,
        );

        // First 3 should be cluster 0, last 3 cluster 1
        for i in 0..3 {
            assert_eq!(labels[i], 0, "Point {i} should be cluster 0, got {}", labels[i]);
        }
        for i in 3..6 {
            assert_eq!(labels[i], 1, "Point {i} should be cluster 1, got {}", labels[i]);
        }
    }

    // -----------------------------------------------------------------------
    // 8. Empty cluster recovery: no empty clusters in output
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_cluster_recovery() {
        let n_cases = 6;
        let n_channels = 1;
        let ts_len = 3;
        let n_clusters = 2;
        let stride = n_channels * ts_len;

        let x_data: Vec<f64> = vec![
            0.0, 0.1, 0.2,
            0.1, 0.0, 0.3,
            0.0, 0.2, 0.1,
            10.0, 10.1, 10.2,
            10.1, 10.0, 10.3,
            10.0, 10.2, 10.1,
        ];
        // All points assigned to cluster 0 — cluster 1 is empty
        let mut centers: Vec<f64> = vec![0.05, 0.1, 0.2, 99.0, 99.0, 99.0];
        let mut distances = vec![0.1; n_cases];
        let mut labels = vec![0usize; n_cases];

        let params = DistanceParams {
            method: DistanceMethod::Msm,
            n_channels,
            independent: true,
            c: 1.0,
        };

        let ok = handle_empty_clusters(
            &x_data, &mut centers, &mut distances, &mut labels,
            n_cases, n_clusters, n_channels, ts_len, &params,
        );

        assert!(ok, "Recovery should succeed");
        // Both clusters should be occupied
        let mut occupied = vec![false; n_clusters];
        for &l in &labels {
            occupied[l] = true;
        }
        assert!(occupied.iter().all(|&o| o), "All clusters should be occupied");
    }

    // -----------------------------------------------------------------------
    // 9. Full kasba_fit: deterministic with fixed seed
    // -----------------------------------------------------------------------

    #[test]
    fn test_kasba_fit_deterministic() {
        let n_cases = 10;
        let n_channels = 1;
        let ts_len = 4;
        let n_clusters = 2;

        // Two clear groups
        let mut x_data = Vec::new();
        for i in 0..5 {
            for t in 0..ts_len {
                x_data.push(i as f64 * 0.1 + t as f64 * 0.01);
            }
        }
        for i in 0..5 {
            for t in 0..ts_len {
                x_data.push(100.0 + i as f64 * 0.1 + t as f64 * 0.01);
            }
        }

        let (labels1, centers1, inertia1, _) = kasba_fit(
            &x_data, n_cases, n_channels, ts_len, n_clusters,
            true, 1.0, 10, 1e-5, 1.0, 0.05, 0.5, 42, 5, false,
        );
        let (labels2, centers2, inertia2, _) = kasba_fit(
            &x_data, n_cases, n_channels, ts_len, n_clusters,
            true, 1.0, 10, 1e-5, 1.0, 0.05, 0.5, 42, 5, false,
        );

        assert_eq!(labels1, labels2, "Same seed should produce same labels");
        assert!((inertia1 - inertia2).abs() < 1e-10, "Same seed should produce same inertia");
        assert_eq!(centers1, centers2, "Same seed should produce same centers");
    }

    #[test]
    fn test_kasba_fit_separates_clusters() {
        let n_cases = 10;
        let n_channels = 1;
        let ts_len = 4;
        let n_clusters = 2;

        let mut x_data = Vec::new();
        for i in 0..5 {
            for t in 0..ts_len {
                x_data.push(i as f64 * 0.1 + t as f64 * 0.01);
            }
        }
        for i in 0..5 {
            for t in 0..ts_len {
                x_data.push(100.0 + i as f64 * 0.1 + t as f64 * 0.01);
            }
        }

        let (labels, _, _, _) = kasba_fit(
            &x_data, n_cases, n_channels, ts_len, n_clusters,
            true, 1.0, 10, 1e-5, 1.0, 0.05, 0.5, 42, 5, false,
        );

        // First 5 should all have same label, last 5 a different label
        let group_a: std::collections::HashSet<usize> = labels[0..5].iter().cloned().collect();
        let group_b: std::collections::HashSet<usize> = labels[5..10].iter().cloned().collect();
        assert_eq!(group_a.len(), 1, "First group should have one label");
        assert_eq!(group_b.len(), 1, "Second group should have one label");
        assert_ne!(
            group_a.iter().next(),
            group_b.iter().next(),
            "Groups should have different labels"
        );
    }

    // -----------------------------------------------------------------------
    // 10. kasba_predict: assign new data to known centers
    // -----------------------------------------------------------------------

    #[test]
    fn test_kasba_predict_assigns_correctly() {
        let n_channels = 1;
        let ts_len = 3;
        let n_clusters = 2;

        let centers = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let x_new = vec![0.05, 0.15, 0.25, 9.9, 10.0, 10.1];
        let n_new = 2;

        let params = DistanceParams {
            method: DistanceMethod::Msm,
            n_channels,
            independent: true,
            c: 1.0,
        };
        let labels = kasba_predict(&x_new, &centers, n_new, n_clusters, n_channels, ts_len, &params);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
    }

    // -----------------------------------------------------------------------
    // Pairwise distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pairwise_distance_self_symmetric() {
        let n_cases = 3;
        let n_channels = 1;
        let ts_len = 3;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 4.0];
        let params = DistanceParams {
            method: DistanceMethod::Msm,
            n_channels,
            independent: true,
            c: 1.0,
        };
        let dm = pairwise_distance_self(&x, n_cases, ts_len, &params);
        assert_eq!(dm.len(), n_cases * n_cases);
        // Diagonal should be 0
        for i in 0..n_cases {
            assert!((dm[i * n_cases + i]).abs() < 1e-10);
        }
        // Symmetric
        for i in 0..n_cases {
            for j in 0..n_cases {
                assert!((dm[i * n_cases + j] - dm[j * n_cases + i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pairwise_distance_xy_shape() {
        let n_x = 3;
        let n_y = 2;
        let ts_len = 4;
        let n_channels = 1;
        let x: Vec<f64> = (0..n_x * n_channels * ts_len).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n_y * n_channels * ts_len).map(|i| i as f64 * 0.5).collect();
        let params = DistanceParams {
            method: DistanceMethod::Msm,
            n_channels,
            independent: true,
            c: 1.0,
        };
        let dm = pairwise_distance_xy(&x, &y, n_x, n_y, ts_len, ts_len, &params);
        assert_eq!(dm.len(), n_x * n_y);
    }
}
