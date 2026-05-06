"""Tests for forecast reconciliation (#55)."""

from datetime import date

import polars as pl
import pytest

from polars_ts.reconciliation import reconcile


def _make_hierarchy() -> dict[str, str]:
    return {"A": "X", "B": "X", "C": "Y", "D": "Y", "X": "Total", "Y": "Total"}


def _make_forecasts() -> pl.DataFrame:
    """Incoherent forecasts at all levels."""
    return pl.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D", "X", "Y", "Total"] * 2,
            "ds": [date(2024, 1, 1)] * 7 + [date(2024, 1, 2)] * 7,
            "y_hat": [10.0, 20.0, 30.0, 40.0, 35.0, 65.0, 90.0] * 2,
        }
    )


class TestReconcile:
    def test_bottom_up(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="bottom_up")
        # Bottom-level values preserved, mid/top levels re-aggregated
        a = result.filter((pl.col("unique_id") == "A") & (pl.col("ds") == date(2024, 1, 1)))
        assert a["y_hat"][0] == pytest.approx(10.0)

    def test_bottom_up_mid_level(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="bottom_up")
        x = result.filter((pl.col("unique_id") == "X") & (pl.col("ds") == date(2024, 1, 1)))
        assert x["y_hat"][0] == pytest.approx(30.0)  # A + B

    def test_top_down(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="top_down")
        # Should produce forecasts for all levels
        ids = result["unique_id"].unique().to_list()
        assert "Total" in ids
        assert "A" in ids

    def test_ols(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="ols")
        # OLS should produce coherent forecasts
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a = t1.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b = t1.filter(pl.col("unique_id") == "B")["y_hat"][0]
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        assert x == pytest.approx(a + b, abs=0.1)

    def test_ols_coherent_total(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="ols")
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        y = t1.filter(pl.col("unique_id") == "Y")["y_hat"][0]
        total = t1.filter(pl.col("unique_id") == "Total")["y_hat"][0]
        assert total == pytest.approx(x + y, abs=0.1)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            reconcile(_make_forecasts(), _make_hierarchy(), method="invalid")

    def test_multiple_timestamps(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="bottom_up")
        assert len(result["ds"].unique()) == 2


class TestMiddleOut:
    """Middle-out reconciliation: start from an intermediate level."""

    def test_middle_out_preserves_middle_level(self):
        """Middle-level forecasts should be preserved as the anchor."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="middle_out",
            middle_level=["X", "Y"],
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        assert x == pytest.approx(35.0)

    def test_middle_out_disaggregates_bottom(self):
        """Bottom-level nodes should be disaggregated from middle using proportions."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="middle_out",
            middle_level=["X", "Y"],
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a = t1.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b = t1.filter(pl.col("unique_id") == "B")["y_hat"][0]
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        # A + B should equal X (coherent)
        assert x == pytest.approx(a + b, abs=0.1)

    def test_middle_out_aggregates_top(self):
        """Top-level should be aggregated from middle level."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="middle_out",
            middle_level=["X", "Y"],
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        y = t1.filter(pl.col("unique_id") == "Y")["y_hat"][0]
        total = t1.filter(pl.col("unique_id") == "Total")["y_hat"][0]
        assert total == pytest.approx(x + y, abs=0.1)

    def test_middle_out_requires_middle_level(self):
        """middle_out should raise if middle_level is not provided."""
        with pytest.raises(ValueError, match="middle_level"):
            reconcile(_make_forecasts(), _make_hierarchy(), method="middle_out")


class TestPERMBU:
    """PERMBU: projection-based empirical residual bottom-up."""

    def _make_residuals(self) -> pl.DataFrame:
        """Residuals (actuals - forecast) for computing empirical covariance."""
        return pl.DataFrame(
            {
                "unique_id": ["A", "B", "C", "D", "X", "Y", "Total"] * 3,
                "ds": [date(2024, 1, d) for d in range(1, 4) for _ in range(7)],
                "residual": [
                    1.0,
                    -1.0,
                    2.0,
                    -2.0,
                    0.5,
                    -0.5,
                    0.2,
                    -0.5,
                    0.5,
                    -1.0,
                    1.0,
                    0.3,
                    -0.3,
                    0.1,
                    0.8,
                    -0.8,
                    1.5,
                    -1.5,
                    0.4,
                    -0.4,
                    0.15,
                ],
            }
        )

    def test_permbu_coherent(self):
        """PERMBU should produce coherent forecasts."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="permbu",
            residuals=self._make_residuals(),
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a = t1.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b = t1.filter(pl.col("unique_id") == "B")["y_hat"][0]
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        assert x == pytest.approx(a + b, abs=0.1)

    def test_permbu_coherent_total(self):
        """PERMBU total should equal sum of mid-levels."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="permbu",
            residuals=self._make_residuals(),
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        y = t1.filter(pl.col("unique_id") == "Y")["y_hat"][0]
        total = t1.filter(pl.col("unique_id") == "Total")["y_hat"][0]
        assert total == pytest.approx(x + y, abs=0.1)

    def test_permbu_requires_residuals(self):
        """PERMBU should raise if residuals not provided."""
        with pytest.raises(ValueError, match="residuals"):
            reconcile(_make_forecasts(), _make_hierarchy(), method="permbu")


class TestMinTraceCV:
    """MinTrace with cross-validation for weight optimization."""

    def _make_train_data(self) -> pl.DataFrame:
        """Historical data for cross-validation."""
        rows = []
        for d in range(1, 11):
            for uid, val in [
                ("A", 10.0),
                ("B", 20.0),
                ("C", 30.0),
                ("D", 40.0),
                ("X", 30.0),
                ("Y", 70.0),
                ("Total", 100.0),
            ]:
                rows.append({"unique_id": uid, "ds": date(2023, 12, d), "y_hat": val + d * 0.5})
        return pl.DataFrame(rows)

    def test_mintrace_cv_coherent(self):
        """MinTrace-CV should produce coherent forecasts."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="mint_cv",
            train_data=self._make_train_data(),
            n_folds=3,
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a = t1.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b = t1.filter(pl.col("unique_id") == "B")["y_hat"][0]
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        assert x == pytest.approx(a + b, abs=0.1)

    def test_mintrace_cv_coherent_total(self):
        """Total should sum coherently."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="mint_cv",
            train_data=self._make_train_data(),
            n_folds=3,
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        y = t1.filter(pl.col("unique_id") == "Y")["y_hat"][0]
        total = t1.filter(pl.col("unique_id") == "Total")["y_hat"][0]
        assert total == pytest.approx(x + y, abs=0.1)

    def test_mintrace_cv_default_folds(self):
        """Should work with default n_folds."""
        result = reconcile(
            _make_forecasts(),
            _make_hierarchy(),
            method="mint_cv",
            train_data=self._make_train_data(),
        )
        assert len(result) > 0


class TestProbabilisticReconciliation:
    """Reconcile prediction intervals, not just point forecasts."""

    def _make_interval_forecasts(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "unique_id": ["A", "B", "C", "D", "X", "Y", "Total"] * 2,
                "ds": [date(2024, 1, 1)] * 7 + [date(2024, 1, 2)] * 7,
                "y_hat": [10.0, 20.0, 30.0, 40.0, 35.0, 65.0, 90.0] * 2,
                "y_lower": [8.0, 17.0, 27.0, 36.0, 30.0, 58.0, 80.0] * 2,
                "y_upper": [12.0, 23.0, 33.0, 44.0, 40.0, 72.0, 100.0] * 2,
            }
        )

    def test_probabilistic_reconciles_point(self):
        """Point forecasts should be coherent."""
        result = reconcile(
            self._make_interval_forecasts(),
            _make_hierarchy(),
            method="ols",
            interval_cols=["y_lower", "y_upper"],
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a = t1.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b = t1.filter(pl.col("unique_id") == "B")["y_hat"][0]
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        assert x == pytest.approx(a + b, abs=0.1)

    def test_probabilistic_reconciles_lower(self):
        """Lower bounds should be coherent."""
        result = reconcile(
            self._make_interval_forecasts(),
            _make_hierarchy(),
            method="ols",
            interval_cols=["y_lower", "y_upper"],
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a_lo = t1.filter(pl.col("unique_id") == "A")["y_lower"][0]
        b_lo = t1.filter(pl.col("unique_id") == "B")["y_lower"][0]
        x_lo = t1.filter(pl.col("unique_id") == "X")["y_lower"][0]
        assert x_lo == pytest.approx(a_lo + b_lo, abs=0.1)

    def test_probabilistic_reconciles_upper(self):
        """Upper bounds should be coherent."""
        result = reconcile(
            self._make_interval_forecasts(),
            _make_hierarchy(),
            method="ols",
            interval_cols=["y_lower", "y_upper"],
        )
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a_hi = t1.filter(pl.col("unique_id") == "A")["y_upper"][0]
        b_hi = t1.filter(pl.col("unique_id") == "B")["y_upper"][0]
        x_hi = t1.filter(pl.col("unique_id") == "X")["y_upper"][0]
        assert x_hi == pytest.approx(a_hi + b_hi, abs=0.1)

    def test_probabilistic_preserves_interval_columns(self):
        """Output should contain interval columns."""
        result = reconcile(
            self._make_interval_forecasts(),
            _make_hierarchy(),
            method="ols",
            interval_cols=["y_lower", "y_upper"],
        )
        assert "y_lower" in result.columns
        assert "y_upper" in result.columns

    def test_probabilistic_lower_leq_upper(self):
        """Reconciled lower should be <= upper for all nodes."""
        result = reconcile(
            self._make_interval_forecasts(),
            _make_hierarchy(),
            method="ols",
            interval_cols=["y_lower", "y_upper"],
        )
        assert (result["y_lower"] <= result["y_upper"]).all()


def test_top_level_import():
    import polars_ts

    assert polars_ts.reconcile is reconcile
