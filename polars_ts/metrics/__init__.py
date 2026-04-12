from dataclasses import dataclass

import polars as pl

try:
    from statsforecast import StatsForecast as _StatsForecast
except ImportError:
    _StatsForecast = None


@dataclass
@pl.api.register_dataframe_namespace("pts")
class Metrics:
    _df: pl.DataFrame

    def kaboudan(
        self,
        sf: object,
        block_size: int = 0,
        backtesting_start: float = 0.0,
        n_folds: int = 0,
        seed: int = 42,
        modified: bool = True,
        agg: bool = False,
    ) -> pl.Expr:
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
