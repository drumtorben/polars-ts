import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Polars Time Series Exercises

        Hands-on exercises using `polars-ts` — a Polars extension for time series
        analysis with Rust-powered distance metrics, trend detection, and decomposition.

        **What you'll learn:**

        1. Comparing time series with DTW and its variants
        2. Exploring distance metrics (ERP, LCSS, TWE, MSM)
        3. Detecting trends with Mann-Kendall
        4. Decomposing seasonal patterns
        5. Fourier decomposition for multi-seasonal data
        6. Multivariate time series distances
        """
    )
    return (mo,)


# ---------------------------------------------------------------------------
# Exercise 1: Time Series Similarity with DTW
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ## Exercise 1: Time Series Similarity with DTW

        Dynamic Time Warping (DTW) measures similarity between time series that may
        vary in speed or phase. Unlike Euclidean distance, DTW can align shifted or
        stretched patterns.

        We'll create three synthetic series and compare them using different DTW methods:

        - **Standard**: unconstrained warping
        - **Sakoe-Chiba**: band constraint (limits how far the alignment can deviate)
        - **Itakura**: parallelogram constraint (limits warping slope)
        - **FastDTW**: approximate algorithm for speed
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import polars as pl

    from polars_ts import (
        compute_pairwise_ddtw,
        compute_pairwise_dtw,
        compute_pairwise_dtw_multi,
        compute_pairwise_erp,
        compute_pairwise_lcss,
        compute_pairwise_msm,
        compute_pairwise_msm_multi,
        compute_pairwise_twe,
        compute_pairwise_wdtw,
        fourier_decomposition,
        mann_kendall,
        seasonal_decomposition,
    )

    return (
        compute_pairwise_ddtw,
        compute_pairwise_dtw,
        compute_pairwise_dtw_multi,
        compute_pairwise_erp,
        compute_pairwise_lcss,
        compute_pairwise_msm,
        compute_pairwise_msm_multi,
        compute_pairwise_twe,
        compute_pairwise_wdtw,
        fourier_decomposition,
        mann_kendall,
        np,
        pl,
        seasonal_decomposition,
    )


@app.cell
def _(np, pl):
    # Create three synthetic time series
    np.random.seed(42)
    _n = 100

    _random_walk = np.cumsum(np.random.randn(_n)) + 50
    _shifted_walk = np.cumsum(np.random.randn(_n)) + 55  # similar shape, shifted up
    _sinusoidal = np.sin(np.linspace(0, 4 * np.pi, _n)) * 10 + 50  # very different shape

    df_ex1 = pl.DataFrame(
        {
            "unique_id": ["stock_A"] * _n + ["stock_B"] * _n + ["stock_C"] * _n,
            "y": np.concatenate([_random_walk, _shifted_walk, _sinusoidal]).tolist(),
        }
    )

    df_ex1.head(5)
    return (df_ex1,)


@app.cell
def _(compute_pairwise_dtw, df_ex1, mo):
    # Standard DTW — unconstrained
    dtw_standard = compute_pairwise_dtw(df_ex1, df_ex1)

    # Sakoe-Chiba — band constraint with window=10
    dtw_sakoe = compute_pairwise_dtw(df_ex1, df_ex1, method="sakoe_chiba", param=10.0)

    # Itakura — parallelogram constraint
    dtw_itakura = compute_pairwise_dtw(df_ex1, df_ex1, method="itakura", param=2.0)

    # FastDTW — approximate
    dtw_fast = compute_pairwise_dtw(df_ex1, df_ex1, method="fast", param=5.0)

    mo.md(
        f"""
        ### DTW Results Comparison

        | Method | stock_A vs stock_B | stock_A vs stock_C | stock_B vs stock_C |
        |--------|-------------------:|-------------------:|-------------------:|
        | Standard | {dtw_standard["dtw"][0]:.2f} | {dtw_standard["dtw"][1]:.2f} | {dtw_standard["dtw"][2]:.2f} |
        | Sakoe-Chiba (w=10) | {dtw_sakoe["dtw"][0]:.2f} | {dtw_sakoe["dtw"][1]:.2f} | {dtw_sakoe["dtw"][2]:.2f} |
        | Itakura (s=2) | {dtw_itakura["dtw"][0]:.2f} | {dtw_itakura["dtw"][1]:.2f} | {dtw_itakura["dtw"][2]:.2f} |
        | FastDTW (r=5) | {dtw_fast["dtw"][0]:.2f} | {dtw_fast["dtw"][1]:.2f} | {dtw_fast["dtw"][2]:.2f} |

        **Observations:**
        - Constrained methods produce distances >= standard DTW (less flexibility)
        - FastDTW approximates the standard result
        - The sinusoidal series (stock_C) is the most different from the random walks
        """
    )
    return dtw_fast, dtw_itakura, dtw_sakoe, dtw_standard


@app.cell
def _(mo):
    mo.md(
        """
        ### Try it yourself

        **Task:** Change the Sakoe-Chiba window from 10 to 1 and observe how the
        distances increase. What happens with a very large window (e.g., 100)?
        """
    )
    return


# ---------------------------------------------------------------------------
# Exercise 1b: DDTW and WDTW variants
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ### DTW Variants: Shape vs. Value Similarity

        - **DDTW** (Derivative DTW): compares the *slopes* of the series instead of
          raw values — captures shape similarity regardless of vertical offset
        - **WDTW** (Weighted DTW): applies time-dependent weights that penalize
          alignments with large temporal gaps
        """
    )
    return


@app.cell
def _(compute_pairwise_ddtw, compute_pairwise_wdtw, df_ex1):  # noqa: ARG001
    # DDTW — shape-based similarity
    ddtw_result = compute_pairwise_ddtw(df_ex1, df_ex1)
    print("DDTW (shape similarity):")
    print(ddtw_result)
    return (ddtw_result,)


@app.cell
def _(compute_pairwise_wdtw, df_ex1):
    # WDTW — penalizes large time gaps
    wdtw_result = compute_pairwise_wdtw(df_ex1, df_ex1, g=0.05)
    print("WDTW (g=0.05):")
    print(wdtw_result)
    return (wdtw_result,)


# ---------------------------------------------------------------------------
# Exercise 2: Exploring Distance Metrics
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ## Exercise 2: Distance Metrics — Handling Noise & Outliers

        Different metrics respond differently to noise and outliers. We'll create
        three variants of a sine wave and compare:

        - **ERP** (Edit distance with Real Penalty): robust to gaps via a penalty value
        - **LCSS** (Longest Common Subsequence): threshold-based matching, ignores outliers
        - **TWE** (Time Warping Edit): combines DTW with edit distance
        - **MSM** (Move-Split-Merge): edit-based with move/split/merge operations
        """
    )
    return


@app.cell
def _(np, pl):
    np.random.seed(0)
    _n = 50

    _base = np.sin(np.linspace(0, 2 * np.pi, _n))
    _noisy = _base + np.random.normal(0, 0.3, _n)
    _outlier = _base.copy()
    _outlier[25] = 5.0  # single large outlier

    df_ex2 = pl.DataFrame(
        {
            "unique_id": ["clean"] * _n + ["noisy"] * _n + ["outlier"] * _n,
            "y": np.concatenate([_base, _noisy, _outlier]).tolist(),
        }
    )

    df_ex2.head(5)
    return (df_ex2,)


@app.cell
def _(
    compute_pairwise_erp,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_twe,
    df_ex2,
    mo,
):
    erp_result = compute_pairwise_erp(df_ex2, df_ex2, g=0.0)
    lcss_result = compute_pairwise_lcss(df_ex2, df_ex2, epsilon=0.5)
    twe_result = compute_pairwise_twe(df_ex2, df_ex2, 0.01, 1.0)
    msm_result = compute_pairwise_msm(df_ex2, df_ex2, c=1.0)

    mo.md(
        f"""
        ### Metric Comparison

        | Metric | clean vs noisy | clean vs outlier | noisy vs outlier |
        |--------|---------------:|-----------------:|-----------------:|
        | ERP    | {erp_result["erp"][0]:.4f} | {erp_result["erp"][1]:.4f} | {erp_result["erp"][2]:.4f} |
        | LCSS   | {lcss_result["lcss"][0]:.4f} | {lcss_result["lcss"][1]:.4f} | {lcss_result["lcss"][2]:.4f} |
        | TWE    | {twe_result["twe"][0]:.4f} | {twe_result["twe"][1]:.4f} | {twe_result["twe"][2]:.4f} |
        | MSM    | {msm_result["msm"][0]:.4f} | {msm_result["msm"][1]:.4f} | {msm_result["msm"][2]:.4f} |

        **Questions to consider:**
        - Which metric is most robust to the single outlier?
        - Which is most sensitive to random noise?
        - How does the LCSS `epsilon` parameter change the results?
        """
    )
    return erp_result, lcss_result, msm_result, twe_result


@app.cell
def _(mo):
    mo.md(
        """
        ### Try it yourself

        **Task:** Increase `epsilon` in LCSS to 1.0 and then to 2.0. What happens
        to the clean-vs-outlier distance? Why?
        """
    )
    return


# ---------------------------------------------------------------------------
# Exercise 3: Trend Detection with Mann-Kendall
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ## Exercise 3: Trend Detection with Mann-Kendall

        The Mann-Kendall test is a non-parametric test for monotonic trends.
        It returns a normalized statistic in [-1, 1]:

        - **+1** = perfect upward trend
        - **0** = no trend
        - **-1** = perfect downward trend

        Great for detecting trends that are hidden under noise or seasonality.
        """
    )
    return


@app.cell
def _(mann_kendall, np, pl):
    np.random.seed(42)
    _n = 200

    df_mk = pl.DataFrame(
        {
            "unique_id": (["rising"] * _n + ["falling"] * _n + ["flat"] * _n + ["seasonal"] * _n),
            "ds": list(range(_n)) * 4,
            "y": np.concatenate(
                [
                    np.linspace(10, 50, _n) + np.random.randn(_n) * 2,
                    np.linspace(50, 10, _n) + np.random.randn(_n) * 2,
                    np.ones(_n) * 30 + np.random.randn(_n) * 5,
                    np.sin(np.linspace(0, 8 * np.pi, _n)) * 10 + 30,
                ]
            ).tolist(),
        }
    )

    mk_result = df_mk.group_by("unique_id").agg(mann_kendall(pl.col("y")).alias("mk_stat")).sort("unique_id")

    print("Mann-Kendall trend statistics:")
    print(mk_result)
    return df_mk, mk_result


@app.cell
def _(mo):
    mo.md(
        """
        ### Try it yourself

        **Task:** Add a slight upward trend to the seasonal series (e.g.,
        `+ np.linspace(0, 5, n)`) and re-run. Does Mann-Kendall detect it?
        At what trend magnitude does it become clearly visible?
        """
    )
    return


# ---------------------------------------------------------------------------
# Exercise 4: Seasonal Decomposition
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ## Exercise 4: Seasonal Decomposition

        Decompose a time series into three components:

        - **Trend**: the long-term direction
        - **Seasonal**: repeating periodic pattern
        - **Residual**: what's left (noise, anomalies)

        Two models:
        - **Additive**: `Y(t) = T(t) + S(t) + R(t)` — constant seasonal amplitude
        - **Multiplicative**: `Y(t) = T(t) * S(t) * R(t)` — seasonal amplitude grows with trend
        """
    )
    return


@app.cell
def _(np, pl, seasonal_decomposition):
    np.random.seed(42)
    _n = 365

    _t = np.arange(_n)
    _trend = 0.05 * _t + 50
    _seasonal = 10 * np.sin(2 * np.pi * _t / 30)  # monthly cycle
    _noise = np.random.randn(_n) * 2
    _y = _trend + _seasonal + _noise

    df_seasonal = pl.DataFrame(
        {
            "unique_id": ["sensor"] * _n,
            "ds": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True),
            "y": _y.tolist(),
        }
    )

    # Additive decomposition with period=30 (monthly cycle)
    decomposed = seasonal_decomposition(df_seasonal, freq=30, method="additive")
    print("Decomposed columns:", decomposed.columns)
    print(decomposed.select("ds", "y", "trend", "seasonal", "resid").head(10))
    return decomposed, df_seasonal


@app.cell
def _(mo):
    mo.md(
        """
        ### Inspect the components

        The decomposition adds `trend`, `seasonal`, and `resid` columns. Check that:
        - `trend` is a smoothed version of `y`
        - `seasonal` repeats with period 30
        - `resid` should look like white noise if the model is well-specified
        """
    )
    return


@app.cell
def _(decomposed, pl):
    # Verify: trend + seasonal + resid ≈ y (additive model)
    check = decomposed.with_columns(
        (pl.col("trend") + pl.col("seasonal") + pl.col("resid")).alias("reconstructed")
    ).select("y", "reconstructed")

    # Drop nulls from trend edges and check reconstruction error
    check_valid = check.drop_nulls()
    max_error = check_valid.select((pl.col("y") - pl.col("reconstructed")).abs().max()).item()
    print(f"Max reconstruction error: {max_error:.10f}")
    return check, check_valid, max_error


@app.cell
def _(mo):
    mo.md(
        """
        ### Try it yourself

        **Task:** Create multiplicative data where the seasonal amplitude grows
        with the trend. Use `method="multiplicative"` and compare.

        ```python
        y_mult = trend * (1 + 0.2 * np.sin(2 * np.pi * t / 30))
        ```
        """
    )
    return


# ---------------------------------------------------------------------------
# Exercise 5: Fourier Decomposition
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ## Exercise 5: Fourier Decomposition

        When your data has **multiple seasonal patterns** (e.g., weekly + monthly),
        Fourier decomposition captures them using harmonic terms.

        The `n_fourier_terms` parameter controls how many harmonics to use —
        more terms = more flexible seasonal fit, but risk of overfitting.
        """
    )
    return


@app.cell
def _(fourier_decomposition, np, pl):
    np.random.seed(42)
    _n = 365

    _t = np.arange(_n)
    _weekly = 5 * np.sin(2 * np.pi * _t / 7)
    _monthly = 8 * np.sin(2 * np.pi * _t / 30)
    _trend = 0.02 * _t
    _noise = np.random.randn(_n) * 1.5

    df_fourier = pl.DataFrame(
        {
            "unique_id": ["multi_seasonal"] * _n,
            "ds": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True),
            "y": (_trend + _weekly + _monthly + _noise).tolist(),
        }
    )

    # Fourier decomposition capturing weekly pattern
    fourier_result = fourier_decomposition(
        df_fourier,
        ts_freq=365,
        freqs=("week",),
        n_fourier_terms=3,
    )
    print("Fourier decomposition columns:", fourier_result.columns)
    print(fourier_result.head(10))
    return df_fourier, fourier_result


@app.cell
def _(mo):
    mo.md(
        """
        ### Try it yourself

        **Task:** Experiment with different `n_fourier_terms` (1, 3, 5, 10).
        Look at the residual variance — at what point does adding more terms
        stop improving the fit?

        ```python
        for k in [1, 3, 5, 10]:
            result = fourier_decomposition(df_fourier, ts_freq=365, freqs=("week",), n_fourier_terms=k)
            resid_var = result.select(pl.col("resid").var()).item()
            print(f"n_fourier_terms={k}: residual variance = {resid_var:.4f}")
        ```
        """
    )
    return


# ---------------------------------------------------------------------------
# Exercise 6: Multivariate Time Series Distance
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ## Exercise 6: Multivariate Time Series Distance

        When each time step has **multiple measurements** (e.g., temperature +
        humidity), you need multivariate distance metrics.

        `polars-ts` supports:
        - **Multivariate DTW**: with Manhattan or Euclidean point-wise distance
        - **Multivariate MSM**: edit-based metric for multi-dimensional series
        """
    )
    return


@app.cell
def _(
    compute_pairwise_dtw_multi,
    compute_pairwise_msm_multi,  # noqa: ARG001
    np,
    pl,
):
    np.random.seed(42)
    _n = 50
    _t = np.linspace(0, 2 * np.pi, _n)

    df_multi = pl.DataFrame(
        {
            "unique_id": ["sensor_A"] * _n + ["sensor_B"] * _n + ["sensor_C"] * _n,
            "temp": np.concatenate(
                [
                    np.sin(_t),
                    np.sin(_t + 0.5),  # phase shifted
                    np.cos(_t),  # very different
                ]
            ).tolist(),
            "humidity": np.concatenate(
                [
                    np.cos(_t),
                    np.cos(_t + 0.3),  # slightly shifted
                    np.sin(_t),  # swapped axes
                ]
            ).tolist(),
        }
    )

    # Multivariate DTW — Manhattan distance (default)
    mdtw_manhattan = compute_pairwise_dtw_multi(df_multi, df_multi)
    print("Multivariate DTW (Manhattan):")
    print(mdtw_manhattan)
    return (df_multi, mdtw_manhattan)


@app.cell
def _(compute_pairwise_dtw_multi, df_multi):
    # Multivariate DTW — Euclidean distance
    mdtw_euclidean = compute_pairwise_dtw_multi(df_multi, df_multi, metric="euclidean")
    print("Multivariate DTW (Euclidean):")
    print(mdtw_euclidean)
    return (mdtw_euclidean,)


@app.cell
def _(compute_pairwise_msm_multi, df_multi):
    # Multivariate MSM
    mmsm_result = compute_pairwise_msm_multi(df_multi, df_multi, c=1.0)
    print("Multivariate MSM:")
    print(mmsm_result)
    return (mmsm_result,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Try it yourself

        **Task:** Add a third dimension (e.g., `pressure`) to the DataFrame and
        re-run the multivariate metrics. How does the additional dimension
        change which sensors are most similar?

        **Task:** Compare the ranking of pairs between Manhattan and Euclidean.
        Do they always agree on which pair is closest?
        """
    )
    return


# ---------------------------------------------------------------------------
# Exercise 7: Putting It All Together
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md(
        """
        ## Exercise 7: Putting It All Together

        Combine what you've learned: generate grouped time series data, detect
        trends, decompose seasonality, and cluster by distance.
        """
    )
    return


@app.cell
def _(
    compute_pairwise_dtw,
    mann_kendall,
    np,
    pl,
    seasonal_decomposition,
):
    np.random.seed(123)
    _n = 120  # 10 "months" of daily data

    # Build 4 series with different characteristics
    _t = np.arange(_n)
    series_data = {
        "trending_seasonal": 0.1 * _t + 5 * np.sin(2 * np.pi * _t / 30) + np.random.randn(_n),
        "flat_seasonal": 50.0 + 5 * np.sin(2 * np.pi * _t / 30) + np.random.randn(_n),
        "trending_noisy": 0.2 * _t + 30 + np.random.randn(_n) * 5,
        "stationary": np.random.randn(_n) * 3 + 40,
    }

    rows = []
    for name, values in series_data.items():
        for i, v in enumerate(values):
            rows.append({"unique_id": name, "ds": i, "y": float(v)})
    df_combined = pl.DataFrame(rows)

    # Step 1: Trend detection
    trends = df_combined.group_by("unique_id").agg(mann_kendall(pl.col("y")).alias("trend_strength")).sort("unique_id")
    print("=== Trend Detection ===")
    print(trends)
    print()

    # Step 2: DTW distance matrix
    distances = compute_pairwise_dtw(df_combined, df_combined)
    print("=== DTW Distance Matrix ===")
    print(distances)
    print()

    # Step 3: Decompose one series
    _one_series = df_combined.filter(pl.col("unique_id") == "trending_seasonal")
    decomp = seasonal_decomposition(_one_series, freq=30, method="additive")
    print("=== Decomposition of 'trending_seasonal' ===")
    print(decomp.select("ds", "y", "trend", "seasonal", "resid").head(10))
    return decomp, df_combined, rows, series_data, trends


@app.cell
def _(mo):
    mo.md(
        """
        ### Summary

        | Feature | Function | Key Parameters |
        |---------|----------|----------------|
        | DTW variants | `compute_pairwise_dtw` | `method`, `param` |
        | Derivative DTW | `compute_pairwise_ddtw` | — |
        | Weighted DTW | `compute_pairwise_wdtw` | `g` |
        | MSM | `compute_pairwise_msm` | `c` |
        | ERP | `compute_pairwise_erp` | `g` |
        | LCSS | `compute_pairwise_lcss` | `epsilon` |
        | TWE | `compute_pairwise_twe` | `nu`, `lambda` |
        | Multivariate DTW | `compute_pairwise_dtw_multi` | `metric` |
        | Multivariate MSM | `compute_pairwise_msm_multi` | `c` |
        | Mann-Kendall | `mann_kendall` | — |
        | Seasonal decomp | `seasonal_decomposition` | `freq`, `method` |
        | Fourier decomp | `fourier_decomposition` | `ts_freq`, `freqs`, `n_fourier_terms` |
        """
    )
    return


if __name__ == "__main__":
    app.run()
