import polars as pl
import pytest

from polars_ts import mann_kendall


class TestMannKendallBasic:
    def test_perfect_upward_trend(self):
        """Monotonically increasing series should give +1.0."""
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(1.0)

    def test_perfect_downward_trend(self):
        """Monotonically decreasing series should give -1.0."""
        df = pl.DataFrame({"y": [5.0, 4.0, 3.0, 2.0, 1.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(-1.0)

    def test_no_trend_symmetric(self):
        """A symmetric series should give 0.0."""
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 2.0, 1.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(0.0)

    def test_constant_series(self):
        """A constant series has no trend — should give 0.0."""
        df = pl.DataFrame({"y": [3.0, 3.0, 3.0, 3.0, 3.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(0.0)

    def test_single_element(self):
        """A single element should return 0.0 (n < 2)."""
        df = pl.DataFrame({"y": [42.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(0.0)

    def test_two_elements_increasing(self):
        """Two increasing elements: S=1, n=2, metric = 1/(0.5*2*1) = 1.0."""
        df = pl.DataFrame({"y": [1.0, 2.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(1.0)

    def test_two_elements_decreasing(self):
        df = pl.DataFrame({"y": [2.0, 1.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(-1.0)

    def test_two_elements_equal(self):
        df = pl.DataFrame({"y": [1.0, 1.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] == pytest.approx(0.0)


class TestMannKendallRange:
    def test_result_between_minus_one_and_one(self):
        """The normalized statistic should always be in [-1, 1]."""
        import random

        random.seed(42)
        values = [random.gauss(0, 1) for _ in range(50)]
        df = pl.DataFrame({"y": values})
        result = df.select(mann_kendall("y").alias("mk"))
        mk = result["mk"][0]
        assert -1.0 <= mk <= 1.0

    def test_mostly_increasing_positive(self):
        """A mostly increasing series should have a positive statistic."""
        df = pl.DataFrame({"y": [1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 7.0, 9.0, 8.0, 10.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] > 0

    def test_mostly_decreasing_negative(self):
        """A mostly decreasing series should have a negative statistic."""
        df = pl.DataFrame({"y": [10.0, 8.0, 9.0, 7.0, 5.0, 6.0, 4.0, 2.0, 3.0, 1.0]})
        result = df.select(mann_kendall("y").alias("mk"))
        assert result["mk"][0] < 0


class TestMannKendallWithGroupBy:
    def test_group_by_usage(self):
        """Mann-Kendall should work inside a group_by context."""
        df = pl.DataFrame(
            {
                "group": ["A"] * 5 + ["B"] * 5,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = df.group_by("group").agg(mann_kendall("y").alias("mk")).sort("group")

        # Explode the aggregated list column to get scalar values
        result = result.explode("mk")
        # Group A: perfect upward → +1.0
        mk_a = result.filter(pl.col("group") == "A")["mk"].to_list()[0]
        assert mk_a == pytest.approx(1.0)
        # Group B: perfect downward → -1.0
        mk_b = result.filter(pl.col("group") == "B")["mk"].to_list()[0]
        assert mk_b == pytest.approx(-1.0)
