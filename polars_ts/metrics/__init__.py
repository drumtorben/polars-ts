from dataclasses import dataclass

import polars as pl
from statsforecast import StatsForecast

from polars_ts.metrics.kaboudan import Kaboudan


@dataclass
@pl.api.register_dataframe_namespace("pts")
class Metrics:
    _df: pl.DataFrame

    def kaboudan(
        self,
        sf: StatsForecast,
        block_size: int = 0,
        backtesting_start: float = 0.0,
        n_folds: int = 0,
        seed: int = 42,
        modified: bool = True,
        agg: bool = False,
    ) -> pl.Expr:
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
