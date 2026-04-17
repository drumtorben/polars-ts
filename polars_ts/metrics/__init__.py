from dataclasses import dataclass

import polars as pl

from polars_ts.metrics.forecast import (
    crps,
    mae,
    mape,
    mase,
    rmse,
    smape,
)

try:
    from statsforecast import StatsForecast as _StatsForecast
except ImportError:
    _StatsForecast = None


@dataclass
@pl.api.register_dataframe_namespace("pts")
class Metrics:
    _df: pl.DataFrame

    def mae(
        self,
        actual_col: str = "y",
        predicted_col: str = "y_hat",
        id_col: str | None = None,
    ) -> pl.DataFrame | float:
        """Mean Absolute Error. See :func:`polars_ts.metrics.forecast.mae`."""
        return mae(self._df, actual_col, predicted_col, id_col)

    def rmse(
        self,
        actual_col: str = "y",
        predicted_col: str = "y_hat",
        id_col: str | None = None,
    ) -> pl.DataFrame | float:
        """Root Mean Squared Error. See :func:`polars_ts.metrics.forecast.rmse`."""
        return rmse(self._df, actual_col, predicted_col, id_col)

    def mape(
        self,
        actual_col: str = "y",
        predicted_col: str = "y_hat",
        id_col: str | None = None,
    ) -> pl.DataFrame | float:
        """Mean Absolute Percentage Error. See :func:`polars_ts.metrics.forecast.mape`."""
        return mape(self._df, actual_col, predicted_col, id_col)

    def smape(
        self,
        actual_col: str = "y",
        predicted_col: str = "y_hat",
        id_col: str | None = None,
    ) -> pl.DataFrame | float:
        """Symmetric MAPE. See :func:`polars_ts.metrics.forecast.smape`."""
        return smape(self._df, actual_col, predicted_col, id_col)

    def mase(
        self,
        actual_col: str = "y",
        predicted_col: str = "y_hat",
        id_col: str = "unique_id",
        time_col: str = "ds",
        season_length: int = 1,
    ) -> pl.DataFrame | float:
        """Mean Absolute Scaled Error. See :func:`polars_ts.metrics.forecast.mase`."""
        return mase(self._df, actual_col, predicted_col, id_col, time_col, season_length)

    def crps(
        self,
        actual_col: str = "y",
        quantile_cols: list[str] | None = None,
        quantiles: list[float] | None = None,
        id_col: str | None = None,
    ) -> pl.DataFrame | float:
        """CRPS (quantile approximation). See :func:`polars_ts.metrics.forecast.crps`."""
        return crps(self._df, actual_col, quantile_cols, quantiles, id_col)

    def kaboudan(
        self,
        sf: object,
        block_size: int = 0,
        backtesting_start: float = 0.0,
        n_folds: int = 0,
        seed: int = 42,
        modified: bool = True,
        agg: bool = False,
    ) -> pl.DataFrame:
        if _StatsForecast is None:
            raise ImportError(
                "statsforecast is required for Metrics.kaboudan(). "
                "Install it with: pip install polars-timeseries[forecast]"
            )
        from polars_ts.metrics.kaboudan import Kaboudan

        kaboudan = Kaboudan(
            sf=sf,
            block_size=block_size,
            backtesting_start=backtesting_start,
            n_folds=n_folds,
            seed=seed,
            modified=modified,
            agg=agg,
        )
        return kaboudan.kaboudan_metric(self._df)

    def lag_features(
        self,
        lags: list[int],
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> pl.DataFrame:
        """Create lag features. See :func:`polars_ts.features.lags.lag_features`."""
        from polars_ts.features.lags import lag_features

        return lag_features(self._df, lags, target_col, id_col, time_col)

    def rolling_features(
        self,
        windows: list[int],
        aggs: list[str] | None = None,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
        center: bool = False,
        min_samples: int | None = None,
    ) -> pl.DataFrame:
        """Create rolling features. See :func:`polars_ts.features.rolling.rolling_features`."""
        from polars_ts.features.rolling import rolling_features

        return rolling_features(self._df, windows, aggs, target_col, id_col, time_col, center, min_samples)

    def calendar_features(
        self,
        features: list[str] | None = None,
        time_col: str = "ds",
    ) -> pl.DataFrame:
        """Extract calendar features. See :func:`polars_ts.features.calendar.calendar_features`."""
        from polars_ts.features.calendar import calendar_features

        return calendar_features(self._df, features, time_col)

    def fourier_features(
        self,
        period: float,
        n_harmonics: int = 1,
        time_col: str = "ds",
        id_col: str = "unique_id",
    ) -> pl.DataFrame:
        """Generate Fourier features. See :func:`polars_ts.features.fourier.fourier_features`."""
        from polars_ts.features.fourier import fourier_features

        return fourier_features(self._df, period, n_harmonics, time_col, id_col)

    def log_transform(
        self,
        target_col: str = "y",
    ) -> pl.DataFrame:
        """Apply log1p transform. See :func:`polars_ts.transforms.log.log_transform`."""
        from polars_ts.transforms.log import log_transform

        return log_transform(self._df, target_col)

    def inverse_log_transform(
        self,
        target_col: str = "y",
    ) -> pl.DataFrame:
        """Invert log transform. See :func:`polars_ts.transforms.log.inverse_log_transform`."""
        from polars_ts.transforms.log import inverse_log_transform

        return inverse_log_transform(self._df, target_col)

    def boxcox_transform(
        self,
        lam: float,
        target_col: str = "y",
    ) -> pl.DataFrame:
        """Apply Box-Cox transform. See :func:`polars_ts.transforms.boxcox.boxcox_transform`."""
        from polars_ts.transforms.boxcox import boxcox_transform

        return boxcox_transform(self._df, lam, target_col)

    def inverse_boxcox_transform(
        self,
        lam: float | None = None,
        target_col: str = "y",
    ) -> pl.DataFrame:
        """Invert Box-Cox transform. See :func:`polars_ts.transforms.boxcox.inverse_boxcox_transform`."""
        from polars_ts.transforms.boxcox import inverse_boxcox_transform

        return inverse_boxcox_transform(self._df, lam, target_col)

    def difference(
        self,
        order: int = 1,
        period: int = 1,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> pl.DataFrame:
        """Apply differencing. See :func:`polars_ts.transforms.differencing.difference`."""
        from polars_ts.transforms.differencing import difference

        return difference(self._df, order, period, target_col, id_col, time_col)

    def undifference(
        self,
        order: int = 1,
        period: int = 1,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> pl.DataFrame:
        """Invert differencing. See :func:`polars_ts.transforms.differencing.undifference`."""
        from polars_ts.transforms.differencing import undifference

        return undifference(self._df, order, period, target_col, id_col, time_col)

    def expanding_window_cv(
        self,
        n_splits: int = 5,
        horizon: int = 1,
        step: int = 1,
        gap: int = 0,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ):
        """Expand-window CV. See :func:`polars_ts.validation.splits.expanding_window_cv`."""
        from polars_ts.validation.splits import expanding_window_cv

        return expanding_window_cv(self._df, n_splits, horizon, step, gap, id_col, time_col)

    def sliding_window_cv(
        self,
        n_splits: int = 5,
        train_size: int = 10,
        horizon: int = 1,
        step: int = 1,
        gap: int = 0,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ):
        """Slide-window CV. See :func:`polars_ts.validation.splits.sliding_window_cv`."""
        from polars_ts.validation.splits import sliding_window_cv

        return sliding_window_cv(self._df, n_splits, train_size, horizon, step, gap, id_col, time_col)

    def rolling_origin_cv(
        self,
        n_splits: int = 5,
        initial_train_size: int | None = None,
        horizon: int = 1,
        step: int = 1,
        gap: int = 0,
        fixed_train_size: int | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ):
        """Roll-origin CV. See :func:`polars_ts.validation.splits.rolling_origin_cv`."""
        from polars_ts.validation.splits import rolling_origin_cv

        return rolling_origin_cv(
            self._df, n_splits, initial_train_size, horizon, step, gap, fixed_train_size, id_col, time_col
        )
