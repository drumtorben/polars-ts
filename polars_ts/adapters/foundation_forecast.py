"""Foundation model forecasting adapters (Chronos, TimesFM, Moirai).

Wraps pre-trained time series foundation models for direct zero-shot
forecasting, returning a polars DataFrame with point forecasts and
prediction intervals.

Extends the existing embedding adapters (issue #151) with a
``predict`` pipeline that produces future values rather than
fixed-length representations.

References
----------
Chronos-2 (Amazon, 2025). T5-based tokenized time series model.
TimesFM (Google, 2024). Pre-trained foundation model for forecasting.
Moirai (Salesforce, 2024). Universal time series forecasting model.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.adapters.embeddings import _extract_series
from polars_ts.models.baselines import _infer_freq, _make_future_dates


def _build_forecast_df(
    ids: list[str],
    forecasts: dict[str, np.ndarray],
    df: pl.DataFrame,
    h: int,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """Build output DataFrame with future dates and forecasts.

    Parameters
    ----------
    ids
        Series identifiers.
    forecasts
        Mapping with keys ``"y_hat"``, ``"y_hat_lower"``, ``"y_hat_upper"``,
        each of shape ``(n_series, h)``.
    df
        Original input DataFrame (for inferring future dates).
    h
        Forecast horizon.
    id_col, time_col
        Column names.

    """
    sorted_df = df.sort(id_col, time_col)
    rows: list[dict[str, Any]] = []

    for i, sid in enumerate(ids):
        series_df = sorted_df.filter(pl.col(id_col) == sid)
        times = series_df[time_col]
        freq = _infer_freq(times)
        last_time = times[-1]
        future_dates = _make_future_dates(last_time, freq, h)

        for t in range(h):
            row: dict[str, Any] = {
                id_col: sid,
                time_col: future_dates[t],
                "y_hat": float(forecasts["y_hat"][i, t]),
            }
            if "y_hat_lower" in forecasts:
                row["y_hat_lower"] = float(forecasts["y_hat_lower"][i, t])
            if "y_hat_upper" in forecasts:
                row["y_hat_upper"] = float(forecasts["y_hat_upper"][i, t])
            rows.append(row)

    return pl.DataFrame(rows)


class ChronosForecaster:
    """Zero-shot forecaster using Amazon Chronos models.

    Uses the Chronos pipeline to generate probabilistic forecasts
    via sample paths. Point forecast is the median; prediction intervals
    are derived from sample quantiles.

    Requires ``torch`` and ``chronos``.

    Parameters
    ----------
    model_name
        HuggingFace model identifier (e.g. ``"amazon/chronos-t5-small"``).
    device
        Torch device (``"cpu"``, ``"cuda"``).
    num_samples
        Number of sample paths for probabilistic forecasts.
    coverage
        Prediction interval coverage (e.g. 0.9 for 90%).
    id_col, time_col, target_col
        Column names.

    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cpu",
        num_samples: int = 20,
        coverage: float = 0.9,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.num_samples = num_samples
        self.coverage = coverage
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._pipeline: Any = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("torch is required for Chronos forecasting. " "Install with: pip install torch") from None

        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos is required for Chronos forecasting. " "Install with: pip install chronos-forecasting"
            ) from None

        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
        ).to(self.device)
        return self._pipeline

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate h-step-ahead forecasts for each series.

        Parameters
        ----------
        df
            Panel DataFrame with historical observations.
        h
            Forecast horizon.

        Returns
        -------
        pl.DataFrame
            Columns: ``[id_col, time_col, y_hat, y_hat_lower, y_hat_upper]``.

        """
        import torch

        pipeline = self._load_pipeline()
        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)

        alpha = (1 - self.coverage) / 2
        all_y_hat = np.zeros((len(ids), h))
        all_lower = np.zeros((len(ids), h))
        all_upper = np.zeros((len(ids), h))

        for i, arr in enumerate(arrays):
            context = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            samples = pipeline.predict(
                context,
                prediction_length=h,
                num_samples=self.num_samples,
            )
            # samples shape: (num_samples, h) or (1, num_samples, h)
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples_np = samples.numpy()
            all_y_hat[i] = np.median(samples_np, axis=0)
            all_lower[i] = np.quantile(samples_np, alpha, axis=0)
            all_upper[i] = np.quantile(samples_np, 1 - alpha, axis=0)

        forecasts = {
            "y_hat": all_y_hat,
            "y_hat_lower": all_lower,
            "y_hat_upper": all_upper,
        }
        return _build_forecast_df(ids, forecasts, df, h, self.id_col, self.time_col)


class TimesFMForecaster:
    """Zero-shot forecaster using Google TimesFM.

    Requires ``timesfm``.

    Parameters
    ----------
    model_name
        TimesFM model identifier.
    context_length
        Number of historical observations to use as context.
    id_col, time_col, target_col
        Column names.

    """

    def __init__(
        self,
        model_name: str = "google/timesfm-2.0-500m-pytorch",
        context_length: int = 512,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.model_name = model_name
        self.context_length = context_length
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate h-step-ahead forecasts for each series.

        Parameters
        ----------
        df
            Panel DataFrame with historical observations.
        h
            Forecast horizon.

        Returns
        -------
        pl.DataFrame
            Columns: ``[id_col, time_col, y_hat]``.

        """
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "timesfm is required for TimesFM forecasting. " "Install with: pip install timesfm"
            ) from None

        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)

        # Truncate to context_length
        contexts = [arr[-self.context_length :] for arr in arrays]

        model = timesfm.TimesFm(
            context_len=self.context_length,
            horizon_len=h,
        )

        # TimesFM expects list of arrays
        point_forecasts, quantile_forecasts = model.forecast(contexts)

        all_y_hat = np.array(point_forecasts)  # (n_series, h)

        forecasts: dict[str, np.ndarray] = {"y_hat": all_y_hat}
        return _build_forecast_df(ids, forecasts, df, h, self.id_col, self.time_col)


class MoiraiForecaster:
    """Zero-shot forecaster using Salesforce Moirai models.

    Requires ``torch`` and ``uni2ts``.

    Parameters
    ----------
    model_name
        HuggingFace model identifier for Moirai.
    device
        Torch device.
    num_samples
        Number of sample paths for probabilistic forecasts.
    coverage
        Prediction interval coverage.
    id_col, time_col, target_col
        Column names.

    """

    def __init__(
        self,
        model_name: str = "salesforce/moirai-1.1-R-small",
        device: str = "cpu",
        num_samples: int = 20,
        coverage: float = 0.9,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.num_samples = num_samples
        self.coverage = coverage
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate h-step-ahead forecasts for each series.

        Parameters
        ----------
        df
            Panel DataFrame with historical observations.
        h
            Forecast horizon.

        Returns
        -------
        pl.DataFrame
            Columns: ``[id_col, time_col, y_hat, y_hat_lower, y_hat_upper]``.

        """
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("torch is required for Moirai forecasting. " "Install with: pip install torch") from None

        try:
            from uni2ts.model.moirai_forecast import MoiraiForecast
        except ImportError:
            raise ImportError(
                "uni2ts is required for Moirai forecasting. " "Install with: pip install uni2ts"
            ) from None

        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)

        pipeline = MoiraiForecast.from_pretrained(self.model_name).to(self.device)

        alpha = (1 - self.coverage) / 2
        all_y_hat = np.zeros((len(ids), h))
        all_lower = np.zeros((len(ids), h))
        all_upper = np.zeros((len(ids), h))

        for i, arr in enumerate(arrays):
            context = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            samples, point = pipeline(
                context,
                prediction_length=h,
                num_samples=self.num_samples,
            )
            # samples: (1, num_samples, h), point: (1, h)
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples_np = samples.cpu().numpy()
            all_y_hat[i] = np.median(samples_np, axis=0)
            all_lower[i] = np.quantile(samples_np, alpha, axis=0)
            all_upper[i] = np.quantile(samples_np, 1 - alpha, axis=0)

        forecasts = {
            "y_hat": all_y_hat,
            "y_hat_lower": all_lower,
            "y_hat_upper": all_upper,
        }
        return _build_forecast_df(ids, forecasts, df, h, self.id_col, self.time_col)


_MODEL_ALIASES: dict[str, type] = {
    "chronos": ChronosForecaster,
    "chronos-2": ChronosForecaster,
    "timesfm": TimesFMForecaster,
    "moirai": MoiraiForecaster,
    "moirai-2": MoiraiForecaster,
}


def foundation_forecast(
    df: pl.DataFrame,
    model: str,
    h: int,
    model_name: str | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Unified foundation model forecasting interface.

    Parameters
    ----------
    df
        Panel DataFrame with historical observations.
    model
        Model family: ``"chronos"``, ``"timesfm"``, or ``"moirai"``.
    h
        Forecast horizon.
    model_name
        Override the default HuggingFace model identifier.
    **kwargs
        Additional arguments passed to the forecaster constructor.

    Returns
    -------
    pl.DataFrame
        Forecast DataFrame with ``[id_col, time_col, y_hat, ...]``.

    """
    model_lower = model.lower()
    if model_lower not in _MODEL_ALIASES:
        raise ValueError(f"Unknown model {model!r}. " f"Supported: {sorted(_MODEL_ALIASES.keys())}")

    cls = _MODEL_ALIASES[model_lower]
    if model_name is not None:
        kwargs["model_name"] = model_name
    forecaster = cls(**kwargs)
    return forecaster.predict(df, h=h)
