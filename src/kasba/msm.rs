/// MSM (Move-Split-Merge) distance for KASBA clustering.
///
/// Supports independent (channel-by-channel) and dependent (multivariate) modes.
/// Uses thread-local DP buffers to avoid per-call allocation.
use std::cell::Cell;

// ---------------------------------------------------------------------------
// Thread-local reusable DP row buffers
// ---------------------------------------------------------------------------

thread_local! {
    static KASBA_DP_PREV: Cell<Vec<f64>> = Cell::new(Vec::new());
    static KASBA_DP_CURR: Cell<Vec<f64>> = Cell::new(Vec::new());
}

#[inline]
fn take_dp_bufs(size: usize, init_val: f64) -> (Vec<f64>, Vec<f64>) {
    let mut prev = KASBA_DP_PREV.with(|c| c.take());
    let mut curr = KASBA_DP_CURR.with(|c| c.take());
    prev.resize(size, init_val);
    curr.resize(size, init_val);
    for v in prev.iter_mut() {
        *v = init_val;
    }
    for v in curr.iter_mut() {
        *v = init_val;
    }
    (prev, curr)
}

#[inline]
fn return_dp_bufs(prev: Vec<f64>, curr: Vec<f64>) {
    KASBA_DP_PREV.with(|c| c.set(prev));
    KASBA_DP_CURR.with(|c| c.set(curr));
}

// ---------------------------------------------------------------------------
// Independent mode cost function (scalar)
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn cost_independent(x: f64, y: f64, z: f64, c: f64) -> f64 {
    if (y <= x && x <= z) || (y >= x && x >= z) {
        c
    } else {
        c + (x - y).abs().min((x - z).abs())
    }
}

// ---------------------------------------------------------------------------
// Dependent mode helpers (multivariate) - allocation-free
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn squared_distance_cols(
    a: &[f64], a_t: usize, a_tp: usize,
    b: &[f64], b_t: usize, b_tp: usize,
    n_channels: usize,
) -> f64 {
    let mut sum = 0.0;
    for ch in 0..n_channels {
        // SAFETY: a has n_channels*a_tp elements, ch < n_channels, a_t < a_tp
        let d = unsafe { *a.get_unchecked(ch * a_tp + a_t) - *b.get_unchecked(ch * b_tp + b_t) };
        sum += d * d;
    }
    sum
}

#[inline(always)]
pub fn cost_dependent_cols(
    x_data: &[f64], x_t: usize, x_tp: usize,
    y_data: &[f64], y_t: usize, y_tp: usize,
    z_data: &[f64], z_t: usize, z_tp: usize,
    n_channels: usize, c: f64,
) -> f64 {
    let mut diameter = 0.0;
    let mut distance_to_mid = 0.0;
    let mut dist_to_y = 0.0;
    let mut dist_to_z = 0.0;
    for ch in 0..n_channels {
        unsafe {
            let xv = *x_data.get_unchecked(ch * x_tp + x_t);
            let yv = *y_data.get_unchecked(ch * y_tp + y_t);
            let zv = *z_data.get_unchecked(ch * z_tp + z_t);
            let d_yz = yv - zv;
            diameter += d_yz * d_yz;
            let mid = (yv + zv) * 0.5;
            let d_mid = xv - mid;
            distance_to_mid += d_mid * d_mid;
            let dy = xv - yv;
            let dz = xv - zv;
            dist_to_y += dy * dy;
            dist_to_z += dz * dz;
        }
    }
    if distance_to_mid <= diameter * 0.25 {
        c
    } else {
        c + dist_to_y.min(dist_to_z)
    }
}

// ---------------------------------------------------------------------------
// Univariate MSM distance (two-row DP)
// ---------------------------------------------------------------------------

pub fn msm_distance_univariate(a: &[f64], b: &[f64], c: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }

    let (mut prev, mut curr) = take_dp_bufs(m, f64::MAX);
    prev[0] = (a[0] - b[0]).abs();

    for j in 1..m {
        let cost = cost_independent(b[j], a[0], unsafe { *b.get_unchecked(j - 1) }, c);
        prev[j] = prev[j - 1] + cost;
    }

    for i in 1..n {
        let ai = unsafe { *a.get_unchecked(i) };
        let ai1 = unsafe { *a.get_unchecked(i - 1) };
        curr[0] = prev[0] + cost_independent(ai, ai1, b[0], c);
        for j in 1..m {
            unsafe {
                let bj = *b.get_unchecked(j);
                let bj1 = *b.get_unchecked(j - 1);
                let d1 = *prev.get_unchecked(j - 1) + (ai - bj).abs();
                let d2 = *prev.get_unchecked(j) + cost_independent(ai, ai1, bj, c);
                let d3 = *curr.get_unchecked(j - 1) + cost_independent(bj, ai, bj1, c);
                *curr.get_unchecked_mut(j) = d1.min(d2).min(d3);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let result = prev[m - 1];
    return_dp_bufs(prev, curr);
    result
}

// ---------------------------------------------------------------------------
// Dependent MSM distance (multivariate, two-row DP)
// ---------------------------------------------------------------------------

pub fn msm_distance_dependent(
    x: &[f64], y: &[f64], n_channels: usize, x_len: usize, y_len: usize, c: f64,
) -> f64 {
    if x_len == 0 || y_len == 0 {
        return 0.0;
    }

    let (mut prev, mut curr) = take_dp_bufs(y_len, f64::MAX);
    prev[0] = squared_distance_cols(x, 0, x_len, y, 0, y_len, n_channels);

    for j in 1..y_len {
        let cost = cost_dependent_cols(y, j, y_len, y, j - 1, y_len, x, 0, x_len, n_channels, c);
        prev[j] = prev[j - 1] + cost;
    }

    for i in 1..x_len {
        let cost = cost_dependent_cols(x, i, x_len, x, i - 1, x_len, y, 0, y_len, n_channels, c);
        curr[0] = prev[0] + cost;
        for j in 1..y_len {
            unsafe {
                let d1 = *prev.get_unchecked(j - 1)
                    + squared_distance_cols(x, i, x_len, y, j, y_len, n_channels);
                let d2 = *prev.get_unchecked(j)
                    + cost_dependent_cols(x, i, x_len, x, i - 1, x_len, y, j, y_len, n_channels, c);
                let d3 = *curr.get_unchecked(j - 1)
                    + cost_dependent_cols(y, j, y_len, x, i, x_len, y, j - 1, y_len, n_channels, c);
                *curr.get_unchecked_mut(j) = d1.min(d2).min(d3);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let result = prev[y_len - 1];
    return_dp_bufs(prev, curr);
    result
}

// ---------------------------------------------------------------------------
// Independent MSM distance (multivariate, sums per-channel distances)
// ---------------------------------------------------------------------------

pub fn msm_distance_independent(
    x: &[f64], y: &[f64], n_channels: usize, x_len: usize, y_len: usize, c: f64,
) -> f64 {
    let mut total = 0.0;
    for ch in 0..n_channels {
        let x_ch = &x[ch * x_len..(ch + 1) * x_len];
        let y_ch = &y[ch * y_len..(ch + 1) * y_len];
        total += msm_distance_univariate(x_ch, y_ch, c);
    }
    total
}

// ---------------------------------------------------------------------------
// Top-level dispatcher
// ---------------------------------------------------------------------------

pub fn msm_distance(
    x: &[f64], y: &[f64], n_channels: usize, x_len: usize, y_len: usize,
    independent: bool, c: f64,
) -> f64 {
    if independent {
        msm_distance_independent(x, y, n_channels, x_len, y_len, c)
    } else {
        msm_distance_dependent(x, y, n_channels, x_len, y_len, c)
    }
}
