/// Generic minimum-cost path traceback through a cost matrix.
pub fn compute_min_return_path(cost_matrix: &[f64], n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut i = n - 1;
    let mut j = m - 1;
    let mut alignment = Vec::with_capacity(n + m);

    while i > 0 || j > 0 {
        alignment.push((i, j));

        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let diag = cost_matrix[(i - 1) * m + (j - 1)];
            let up = cost_matrix[(i - 1) * m + j];
            let left = cost_matrix[i * m + (j - 1)];

            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }

    alignment.push((0, 0));
    alignment
}
