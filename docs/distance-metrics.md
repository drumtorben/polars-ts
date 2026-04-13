# Distance Metric Comparison Guide

`polars_ts` provides 10 time-series distance metrics implemented in Rust for high performance. This guide covers each metric in detail and offers domain-specific recommendations.

All functions accept Polars DataFrames with columns `"unique_id"` (series identifier) and `"y"` (numeric values). They return a pairwise distance matrix as a DataFrame.

```python
import polars as pl
from polars_ts import compute_pairwise_dtw

df = pl.DataFrame({
    "unique_id": ["A"] * 4 + ["B"] * 4,
    "y": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
})
result = compute_pairwise_dtw(df, df)
```

---

## Quick Comparison Table

| Metric | True Metric? | Complexity | Handles Shifts? | Noise Robust? | Key Param |
|---|---|---|---|---|---|
| DTW | No | O(n*m) | Yes | Moderate | -- |
| Sakoe-Chiba DTW | No | O(n*w) | Partial (within band) | Moderate | `param` (band width) |
| Itakura DTW | No | O(n*m) constrained | Partial (within parallelogram) | Moderate | `param` (slope factor) |
| FastDTW | No | O(n) approx | Yes | Moderate | `param` (radius) |
| DDTW | No | O(n*m) | Yes | High | -- |
| WDTW | No | O(n*m) | Yes | High | `g` (weight decay) |
| MSM | Yes | O(n*m) | Yes | High | `c` (split/merge cost) |
| ERP | Yes | O(n*m) | Yes | High | `g` (gap penalty ref) |
| LCSS | No | O(n*m) | Yes | Very High | `epsilon` (match threshold) |
| TWE | Yes | O(n*m) | Yes | High | `nu` (stiffness), `lambda_` (penalty) |

---

## Metric Details

### 1. DTW (Dynamic Time Warping)

**Description:** The classic elastic distance measure for time series. DTW finds the optimal alignment between two sequences by warping the time axis, minimizing the total Euclidean distance between aligned points.

**How it works:** Builds a cost matrix where each cell (i, j) represents the cumulative cost of aligning the first i points of one series with the first j points of the other. The optimal warping path is found via dynamic programming.

**Pros:**

- Handles time shifts and variable-speed patterns
- Intuitive and well-studied
- No parameters to tune

**Cons:**

- O(n*m) complexity can be slow for long series
- Not a true metric (violates triangle inequality)
- Susceptible to pathological warping paths

**When to use:** General-purpose time-series comparison where series may be shifted or locally stretched.

```python
from polars_ts import compute_pairwise_dtw

result = compute_pairwise_dtw(df, df)
```

---

### 2. Sakoe-Chiba DTW

**Description:** A constrained variant of DTW that restricts the warping path to a fixed-width band around the diagonal. This prevents excessive warping and reduces computation time.

**How it works:** Same dynamic programming approach as DTW, but cells outside a symmetric band of width `param` around the diagonal are set to infinity, forcing the alignment to stay close to the diagonal.

**Pros:**

- Faster than unconstrained DTW
- Prevents pathological alignments
- Simple single parameter

**Cons:**

- Cannot handle large time shifts exceeding the band width
- Band width must be chosen a priori

**When to use:** When you know the maximum expected temporal offset between series, or when you need faster DTW with controlled warping.

```python
from polars_ts import compute_pairwise_dtw

result = compute_pairwise_dtw(df, df, method="sakoe_chiba", param=10.0)
```

---

### 3. Itakura DTW

**Description:** A constrained DTW variant that restricts the warping path to lie within a parallelogram (Itakura parallelogram). This limits the allowed slope of the warping path, preventing both excessive compression and expansion.

**How it works:** The parallelogram constraint enforces a maximum and minimum slope on the warping path. The `param` value controls the slope factor -- higher values allow more warping flexibility.

**Pros:**

- Prevents both excessive compression and stretching
- More nuanced constraint than a fixed band
- Good for speech and audio where tempo varies smoothly

**Cons:**

- Less intuitive parameter than Sakoe-Chiba
- Can be too restrictive for highly variable series

**When to use:** When temporal distortion is smooth and gradual, especially in speech and audio domains.

```python
from polars_ts import compute_pairwise_dtw

result = compute_pairwise_dtw(df, df, method="itakura", param=2.0)
```

---

### 4. FastDTW

**Description:** An approximate DTW algorithm that operates in linear time and space. It uses a multi-resolution approach: coarsen the series, compute DTW at low resolution, then refine the path at progressively finer resolutions.

**How it works:** The series are recursively downsampled by half. DTW is computed at the coarsest level, and the resulting path is projected upward and expanded by `param` (radius) cells at each finer level to define the search neighborhood.

**Pros:**

- O(n) time and space complexity
- Practical for very long time series
- Accuracy close to exact DTW for reasonable radius values

**Cons:**

- Approximate -- can miss the true optimal alignment
- Accuracy degrades if radius is too small
- Not a true metric

**When to use:** Large-scale exploratory analysis, very long time series, or when exact DTW is too slow.

```python
from polars_ts import compute_pairwise_dtw

result = compute_pairwise_dtw(df, df, method="fast", param=5.0)
```

---

### 5. DDTW (Derivative Dynamic Time Warping)

**Description:** Applies DTW to the first derivatives of the time series rather than the raw values. This makes the comparison shape-based rather than value-based.

**How it works:** Estimates the discrete derivative of each series (using the average of left and right finite differences), then runs standard DTW on the derivative sequences.

**Pros:**

- Invariant to vertical offset (baseline shifts)
- Focuses on shape rather than amplitude
- No extra parameters beyond standard DTW

**Cons:**

- Derivative estimation amplifies noise
- Loses absolute magnitude information
- Slightly more expensive due to derivative computation

**When to use:** When the shape of the series matters more than its absolute values, e.g., comparing trends regardless of baseline level.

```python
from polars_ts import compute_pairwise_ddtw

result = compute_pairwise_ddtw(df, df)
```

---

### 6. WDTW (Weighted Dynamic Time Warping)

**Description:** A variant of DTW that applies a multiplicative weight penalty based on the phase difference between aligned points. Points aligned far from the diagonal receive higher penalties.

**How it works:** A logistic weight function controlled by parameter `g` assigns weights to each cell based on |i - j|. Small `g` values produce nearly uniform weights (behaves like DTW); large `g` values heavily penalize off-diagonal alignments.

**Pros:**

- Smooth penalty discourages excessive warping without hard cutoffs
- Single intuitive parameter
- Retains DTW flexibility while adding regularization

**Cons:**

- Not a true metric
- Choosing `g` requires experimentation
- Same O(n*m) complexity as standard DTW

**When to use:** When you want DTW behavior but with soft penalization of excessive warping, particularly for noisy series.

```python
from polars_ts import compute_pairwise_wdtw

result = compute_pairwise_wdtw(df, df, g=0.05)
```

---

### 7. MSM (Move-Split-Merge)

**Description:** An edit-distance-style metric that uses three operations -- Move (change a value), Split (duplicate a point), and Merge (combine two points) -- each with cost `c`. It is a true metric.

**How it works:** Dynamic programming computes the minimum cost of transforming one series into the other using move, split, and merge operations. The parameter `c` controls the cost of split and merge relative to moves.

**Pros:**

- True metric (satisfies triangle inequality)
- Invariant to certain transformations depending on `c`
- Good theoretical properties for indexing and clustering

**Cons:**

- Less intuitive than DTW
- Cost parameter `c` requires domain knowledge to set
- O(n*m) complexity

**When to use:** When metric properties are needed (e.g., for metric-tree indexing, k-medoids clustering), or when you need principled handling of insertions and deletions.

```python
from polars_ts import compute_pairwise_msm

result = compute_pairwise_msm(df, df, c=1.0)
```

---

### 8. ERP (Edit Distance with Real Penalty)

**Description:** A true metric that combines edit distance with a real-valued gap penalty. Unmatched points are compared against a fixed reference value `g` (often 0), making it robust to noise and partial matches.

**How it works:** Dynamic programming with three operations: match (Euclidean cost), insert-gap (cost is |point - g|), and delete-gap (cost is |point - g|). The parameter `g` acts as a reference level for gap penalties.

**Pros:**

- True metric
- Handles series of different lengths gracefully
- Gap reference `g` provides noise robustness

**Cons:**

- Sensitive to the choice of `g`
- O(n*m) complexity
- Less commonly used than DTW, so less community tooling

**When to use:** When metric properties are required and series may have missing segments or different lengths. Works well for zero-centered or normalized data with `g=0.0`.

```python
from polars_ts import compute_pairwise_erp

result = compute_pairwise_erp(df, df, g=0.0)
```

---

### 9. LCSS (Longest Common Subsequence)

**Description:** Measures similarity by finding the longest subsequence of points that match within a threshold `epsilon`. The distance is derived from the length of this subsequence.

**How it works:** Two points match if their absolute difference is at most `epsilon`. Dynamic programming finds the longest sequence of such matches (not necessarily contiguous). The distance is typically `1 - LCSS_length / min(n, m)`.

**Pros:**

- Extremely robust to noise and outliers (mismatched points are simply skipped)
- Intuitive threshold parameter
- Good for partial matching

**Cons:**

- Not a true metric
- Only considers match/no-match, losing fine-grained distance information
- `epsilon` must be tuned relative to data scale

**When to use:** Noisy environments where outliers are common, or when you care about structural similarity rather than exact amplitude matching.

```python
from polars_ts import compute_pairwise_lcss

result = compute_pairwise_lcss(df, df, epsilon=1.0)
```

---

### 10. TWE (Time Warp Edit Distance)

**Description:** A true metric that combines DTW-style elastic matching with edit-distance-style insert/delete operations. It uses a stiffness parameter `nu` and a gap penalty `lambda_`.

**How it works:** Dynamic programming considers three operations: match (with temporal stiffness `nu` controlling how much consecutive point spacing matters), delete, and insert (both penalized by `lambda_`). The stiffness parameter makes TWE sensitive to the temporal regularity of the alignment.

**Pros:**

- True metric
- Two parameters allow fine-grained control over warping vs. gap behavior
- Good balance between DTW and edit distance

**Cons:**

- Two parameters to tune
- Less widely known than DTW
- O(n*m) complexity

**When to use:** When you need metric properties and want control over both elastic warping and gap penalties. Particularly useful for trajectory and motion data.

```python
from polars_ts import compute_pairwise_twe

result = compute_pairwise_twe(df, df, nu=0.001, lambda_=1.0)
```

---

## Domain-Specific Guidance

### Finance (Stock Prices, Returns, Macro Indicators)

Financial time series often have regime shifts, trends, and varying volatility. Shape-based comparison is usually more meaningful than raw value comparison.

**Recommended metrics:**

- **DDTW** -- Compares return profiles (derivatives) rather than price levels, which is more meaningful since absolute price is often arbitrary.
- **WDTW** (g=0.05 to 0.1) -- Allows flexible alignment with soft warping penalty, useful for comparing securities that move similarly but with slight lead/lag.
- **MSM** -- True metric, good for clustering financial instruments into groups for portfolio construction.

```python
from polars_ts import compute_pairwise_ddtw, compute_pairwise_wdtw

# Compare return shapes (ignoring price levels)
shape_distances = compute_pairwise_ddtw(stocks_df, stocks_df)

# Flexible alignment with controlled warping
aligned_distances = compute_pairwise_wdtw(stocks_df, stocks_df, g=0.05)
```

---

### IoT / Sensor Data

Sensor data is often noisy, may contain dropouts, and can have varying sampling rates. Robustness to noise and missing data is critical.

**Recommended metrics:**

- **LCSS** (epsilon tuned to sensor noise floor) -- Excellent noise robustness; outlier readings are simply ignored.
- **ERP** (g=0.0 for zero-centered data) -- Handles gaps and dropouts gracefully as a true metric.
- **Sakoe-Chiba DTW** -- Faster than full DTW, suitable for real-time or near-real-time processing with bounded sensor lag.

```python
from polars_ts import compute_pairwise_lcss, compute_pairwise_erp

# Robust to noisy sensor readings
noisy_distances = compute_pairwise_lcss(sensor_df, sensor_df, epsilon=0.5)

# Handle sensor dropouts with gap penalty
gap_distances = compute_pairwise_erp(sensor_df, sensor_df, g=0.0)
```

---

### Speech / Audio

Speech signals exhibit smooth tempo variation and local frequency shifts. Constraints that enforce gradual warping are well-suited here.

**Recommended metrics:**

- **Itakura DTW** (param=1.5 to 3.0) -- The parallelogram constraint naturally models gradual tempo changes in speech.
- **WDTW** (g=0.1 to 0.5) -- Soft warping penalty accommodates pronunciation variation.
- **DTW** -- The unconstrained baseline remains strong for general speech comparison.

```python
from polars_ts import compute_pairwise_dtw

# Smooth tempo variation constraint
speech_distances = compute_pairwise_dtw(mfcc_df, mfcc_df, method="itakura", param=2.0)
```

---

### Gesture / Motion Capture

Gesture data involves 3D trajectories performed at varying speeds. Metrics need to handle speed variation while preserving spatial structure.

**Recommended metrics:**

- **TWE** -- True metric with stiffness control; low `nu` allows flexible speed variation, `lambda_` penalizes skipped frames.
- **MSM** -- True metric that handles split/merge of motion segments naturally.
- **DTW** -- Reliable baseline for gesture recognition.

```python
from polars_ts import compute_pairwise_twe, compute_pairwise_msm

# Flexible speed, penalize skipped frames
gesture_distances = compute_pairwise_twe(motion_df, motion_df, nu=0.001, lambda_=1.0)

# Edit-based comparison
edit_distances = compute_pairwise_msm(motion_df, motion_df, c=1.0)
```

---

### Large-Scale Exploration

When working with thousands of long time series, computational cost dominates. Approximate or constrained methods are essential.

**Recommended metrics:**

- **FastDTW** (param=5 to 20) -- Linear-time approximation, the go-to choice for large datasets.
- **Sakoe-Chiba DTW** (narrow band) -- Reduces quadratic cost significantly with a tight band.
- **LCSS** -- Can be pruned efficiently and provides coarse but fast similarity.

```python
from polars_ts import compute_pairwise_dtw

# Fast approximate DTW for large datasets
approx_distances = compute_pairwise_dtw(large_df, large_df, method="fast", param=10.0)

# Constrained DTW with narrow band
band_distances = compute_pairwise_dtw(large_df, large_df, method="sakoe_chiba", param=5.0)
```

---

### Trajectory Analysis (GPS, Vehicle, Pedestrian)

Trajectories involve spatial paths with temporal components. Metrics that handle gaps (stops, signal loss) and provide true metric properties for spatial indexing are preferred.

**Recommended metrics:**

- **ERP** (g=0.0) -- True metric, handles GPS signal dropouts as gaps against a reference point.
- **TWE** -- True metric with temporal stiffness; good for distinguishing paths taken at different speeds.
- **LCSS** (epsilon tuned to spatial tolerance) -- Ignores GPS jitter and noise, focuses on structural path similarity.

```python
from polars_ts import compute_pairwise_erp, compute_pairwise_twe

# Handle GPS dropouts
traj_distances = compute_pairwise_erp(gps_df, gps_df, g=0.0)

# Speed-sensitive trajectory comparison
speed_distances = compute_pairwise_twe(gps_df, gps_df, nu=0.01, lambda_=0.5)
```

---

## Choosing a Metric: Decision Flowchart

1. **Do you need true metric properties?** (e.g., for metric trees, k-medoids, triangle inequality pruning)
    - Yes: Choose from **MSM**, **ERP**, or **TWE**
    - No: Continue below

2. **Is noise/outlier robustness the top priority?**
    - Yes: Use **LCSS** (set `epsilon` to your noise tolerance)
    - No: Continue below

3. **Do you care about shape rather than absolute values?**
    - Yes: Use **DDTW**
    - No: Continue below

4. **Are your series very long (>10,000 points) or do you have many series?**
    - Yes: Use **FastDTW** or **Sakoe-Chiba DTW**
    - No: Continue below

5. **Do you want controlled warping without hard boundaries?**
    - Yes: Use **WDTW**
    - No: Use standard **DTW**
