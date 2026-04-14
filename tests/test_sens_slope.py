import polars as pl
import pytest

from polars_ts import sens_slope


class TestSensSlopeBasic:
    def test_perfect_upward_trend(self):
        """Monotonically increasing by 1 should give slope 1.0."""
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(1.0)

    def test_perfect_downward_trend(self):
        """Monotonically decreasing by 1 should give slope -1.0."""
        df = pl.DataFrame({"y": [5.0, 4.0, 3.0, 2.0, 1.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(-1.0)

    def test_constant_series(self):
        """A constant series should give slope 0.0."""
        df = pl.DataFrame({"y": [3.0, 3.0, 3.0, 3.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(0.0)

    def test_single_element(self):
        """A single element should return 0.0 (n < 2)."""
        df = pl.DataFrame({"y": [42.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(0.0)

    def test_two_elements(self):
        """Two elements: slope = (4 - 2) / (1 - 0) = 2.0."""
        df = pl.DataFrame({"y": [2.0, 4.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(2.0)

    def test_slope_of_two(self):
        """Series increasing by 2 each step should give slope 2.0."""
        df = pl.DataFrame({"y": [0.0, 2.0, 4.0, 6.0, 8.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(2.0)


class TestSensSlopeRobustness:
    def test_robust_to_outlier(self):
        """Sen's slope should be robust to a single outlier."""
        df = pl.DataFrame({"y": [1.0, 2.0, 100.0, 4.0, 5.0]})
        result = df.select(sens_slope("y").alias("ss"))
        # Median of pairwise slopes should be close to 1.0, not pulled by the outlier
        ss = result["ss"][0]
        assert 0.5 < ss < 3.0

    def test_noisy_upward(self):
        """A noisy but mostly increasing series should have a positive slope."""
        df = pl.DataFrame({"y": [1.0, 3.0, 2.0, 4.0, 6.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] > 0


class TestSensSlopeEdgeCases:
    def test_with_nulls_skipped(self):
        """Null values should be filtered out, not cause a panic."""
        df = pl.DataFrame({"y": [1.0, None, 3.0, None, 5.0]})
        result = df.select(sens_slope("y").alias("ss"))
        # After filtering nulls: [1.0, 3.0, 5.0] at positions [0, 1, 2] → slope 2.0
        assert result["ss"][0] == pytest.approx(2.0)

    def test_all_nulls(self):
        """All-null series should return 0.0 (n < 2 after filtering)."""
        df = pl.DataFrame({"y": pl.Series([None, None, None], dtype=pl.Float64)})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(0.0)


class TestSensSlopeWithGroupBy:
    def test_group_by_usage(self):
        """Sen's slope should work inside a group_by context."""
        df = pl.DataFrame(
            {
                "group": ["A"] * 5 + ["B"] * 5,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 8.0, 6.0, 4.0, 2.0],
            }
        )
        result = df.group_by("group").agg(sens_slope("y").alias("ss")).sort("group")
        result = result.explode("ss")
        ss_a = result.filter(pl.col("group") == "A")["ss"].to_list()[0]
        ss_b = result.filter(pl.col("group") == "B")["ss"].to_list()[0]
        assert ss_a == pytest.approx(1.0)
        assert ss_b == pytest.approx(-2.0)
