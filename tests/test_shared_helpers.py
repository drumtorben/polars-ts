"""Tests for extracted shared helpers (Phase 2, #216).

Tests that _time_utils.py and _inverse.py exist, are importable,
and produce correct results. Also verifies that pipeline.py and
global_model.py no longer contain their own copies.
"""

import ast
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_ts

POLARS_TS_ROOT = Path(polars_ts.__file__).parent


# ---------------------------------------------------------------------------
# T2.3: _time_utils.py exists and is correct
# ---------------------------------------------------------------------------


class TestTimeUtils:
    """Verify polars_ts.models._time_utils provides _infer_freq and _make_future_dates."""

    def test_module_importable(self):
        from polars_ts.models._time_utils import _infer_freq, _make_future_dates

        assert callable(_infer_freq)
        assert callable(_make_future_dates)

    def test_infer_freq_daily_datetime(self):
        from polars_ts.models._time_utils import _infer_freq

        dates = pl.Series([datetime(2024, 1, i) for i in range(1, 11)])
        freq = _infer_freq(dates)
        assert freq == timedelta(days=1)

    def test_infer_freq_hourly(self):
        from polars_ts.models._time_utils import _infer_freq

        dates = pl.Series([datetime(2024, 1, 1, h) for h in range(10)])
        freq = _infer_freq(dates)
        assert freq == timedelta(hours=1)

    def test_infer_freq_date_column(self):
        from datetime import date

        from polars_ts.models._time_utils import _infer_freq

        dates = pl.Series([date(2024, 1, i) for i in range(1, 11)])
        freq = _infer_freq(dates)
        assert freq == timedelta(days=1)

    def test_infer_freq_too_short_raises(self):
        from polars_ts.models._time_utils import _infer_freq

        dates = pl.Series([datetime(2024, 1, 1)])
        with pytest.raises(ValueError, match="at least 2"):
            _infer_freq(dates)

    def test_make_future_dates_generates_correct_count(self):
        from polars_ts.models._time_utils import _make_future_dates

        last = datetime(2024, 1, 10)
        freq = timedelta(days=1)
        result = _make_future_dates(last, freq, 5)
        assert len(result) == 5
        assert result[0] == datetime(2024, 1, 11)
        assert result[-1] == datetime(2024, 1, 15)

    def test_make_future_dates_hourly(self):
        from polars_ts.models._time_utils import _make_future_dates

        last = datetime(2024, 1, 1, 12)
        freq = timedelta(hours=1)
        result = _make_future_dates(last, freq, 3)
        assert result == [
            datetime(2024, 1, 1, 13),
            datetime(2024, 1, 1, 14),
            datetime(2024, 1, 1, 15),
        ]


# ---------------------------------------------------------------------------
# T2.1: _inverse.py exists and is correct
# ---------------------------------------------------------------------------


class TestInverseHelpers:
    """Verify polars_ts.transforms._inverse provides inverse_single and transform_buffer."""

    def test_module_importable(self):
        from polars_ts.transforms._inverse import inverse_single, transform_buffer

        assert callable(inverse_single)
        assert callable(transform_buffer)

    # -- inverse_single tests --

    def test_inverse_single_no_transform(self):
        from polars_ts.transforms._inverse import inverse_single

        assert inverse_single(5.0, None, {}, []) == 5.0

    def test_inverse_single_log(self):
        from polars_ts.transforms._inverse import inverse_single

        pred = float(np.log1p(10.0))
        result = inverse_single(pred, "log", {}, [])
        assert result == pytest.approx(10.0)

    def test_inverse_single_boxcox_nonzero_lambda(self):
        from polars_ts.transforms._inverse import inverse_single

        # boxcox with lam=0.5: forward = (v^0.5 - 1) / 0.5
        v = 4.0
        lam = 0.5
        transformed = (v**lam - 1) / lam  # = (2 - 1) / 0.5 = 2.0
        result = inverse_single(transformed, "boxcox", {"lam": lam}, [])
        assert result == pytest.approx(v)

    def test_inverse_single_boxcox_zero_lambda(self):
        from polars_ts.transforms._inverse import inverse_single

        v = 4.0
        transformed = float(np.log(v))
        result = inverse_single(transformed, "boxcox", {"lam": 0}, [])
        assert result == pytest.approx(v)

    def test_inverse_single_difference(self):
        from polars_ts.transforms._inverse import inverse_single

        # diff prediction=2.0, previous value (period=1 ago) was 10.0 → 12.0
        result = inverse_single(2.0, "difference", {"period": 1}, [8.0, 10.0])
        assert result == pytest.approx(12.0)

    def test_inverse_single_difference_short_history(self):
        from polars_ts.transforms._inverse import inverse_single

        # Not enough history → return pred as-is
        result = inverse_single(2.0, "difference", {"period": 3}, [10.0])
        assert result == pytest.approx(2.0)

    # -- transform_buffer tests --

    def test_transform_buffer_no_transform(self):
        from polars_ts.transforms._inverse import transform_buffer

        values = [1.0, 2.0, 3.0]
        assert transform_buffer(values, None, {}) == [1.0, 2.0, 3.0]

    def test_transform_buffer_log(self):
        from polars_ts.transforms._inverse import transform_buffer

        values = [1.0, 2.0, 3.0]
        result = transform_buffer(values, "log", {})
        expected = [float(np.log1p(v)) for v in values]
        assert result == pytest.approx(expected)

    def test_transform_buffer_boxcox(self):
        from polars_ts.transforms._inverse import transform_buffer

        values = [1.0, 4.0, 9.0]
        lam = 0.5
        result = transform_buffer(values, "boxcox", {"lam": lam})
        expected = [float((v**lam - 1) / lam) for v in values]
        assert result == pytest.approx(expected)

    def test_transform_buffer_boxcox_zero_lambda(self):
        from polars_ts.transforms._inverse import transform_buffer

        values = [1.0, 2.0, 3.0]
        result = transform_buffer(values, "boxcox", {"lam": 0})
        expected = [float(np.log(v)) for v in values]
        assert result == pytest.approx(expected)

    def test_transform_buffer_difference(self):
        from polars_ts.transforms._inverse import transform_buffer

        values = [10.0, 12.0, 15.0, 14.0]
        result = transform_buffer(values, "difference", {"period": 1, "order": 1})
        # diffs: [nan, 2.0, 3.0, -1.0] → drop nan → [2.0, 3.0, -1.0]
        assert result == pytest.approx([2.0, 3.0, -1.0])

    def test_transform_buffer_roundtrip_log(self):
        from polars_ts.transforms._inverse import inverse_single, transform_buffer

        values = [1.0, 5.0, 10.0]
        transformed = transform_buffer(values, "log", {})
        recovered = [inverse_single(t, "log", {}, []) for t in transformed]
        assert recovered == pytest.approx(values)


# ---------------------------------------------------------------------------
# T2.4: baselines.py, arima.py no longer define their own copies
# ---------------------------------------------------------------------------


class TestNoDuplicateDefinitions:
    """Verify that _infer_freq and _make_future_dates are not defined in arima.py."""

    def _get_function_defs(self, filepath: Path) -> list[str]:
        source = filepath.read_text()
        tree = ast.parse(source)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    def test_arima_does_not_define_infer_freq(self):
        arima = POLARS_TS_ROOT / "models" / "arima.py"
        defs = self._get_function_defs(arima)
        assert "_infer_freq" not in defs, "arima.py should import _infer_freq from _time_utils, not define its own"

    def test_arima_does_not_define_make_future_dates(self):
        arima = POLARS_TS_ROOT / "models" / "arima.py"
        defs = self._get_function_defs(arima)
        assert (
            "_make_future_dates" not in defs
        ), "arima.py should import _make_future_dates from _time_utils, not define its own"

    def test_pipeline_does_not_define_inverse_single(self):
        """pipeline.py should not have a standalone _inverse_single method duplicating global_model."""
        pipeline = POLARS_TS_ROOT / "pipeline.py"
        source = pipeline.read_text()
        tree = ast.parse(source)
        # Check for method definitions named _inverse_single
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_inverse_single":
                pytest.fail("pipeline.py should use inverse_single from transforms._inverse")

    def test_global_model_does_not_define_inverse_single(self):
        gm = POLARS_TS_ROOT / "global_model.py"
        source = gm.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_inverse_single":
                pytest.fail("global_model.py should use inverse_single from transforms._inverse")

    def test_global_model_does_not_define_transform_buffer(self):
        gm = POLARS_TS_ROOT / "global_model.py"
        source = gm.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_transform_buffer":
                pytest.fail("global_model.py should use transform_buffer from transforms._inverse")
