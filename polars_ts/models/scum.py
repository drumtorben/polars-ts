from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import polars as pl
from statsforecast.models import _TS, AutoARIMA, AutoCES, AutoETS, DynamicOptimizedTheta


class SCUM(_TS):
    uses_exog = True

    def __init__(self, season_length: int = 1, alias: str = "SCUM") -> None:
        self.season_length = season_length
        self.sub_model_classes = [AutoARIMA, AutoETS, AutoCES, DynamicOptimizedTheta]
        self.n_models = len(self.sub_model_classes)
        self.sub_models: list[Any] | None = None
        self.alias = alias

    def fit(self, y: np.ndarray) -> SCUM:
        """StatsForecast will pass y as a 1D array or Series.

        If you want each sub-model to see a Polars DataFrame, you'll need to
        re-create it here or adapt your sub-model usage.
        """
        # Instantiate and fit each sub-model
        self.sub_models = []
        for ModelClass in self.sub_model_classes:
            model = ModelClass(season_length=self.season_length)
            model.fit(y)
            self.sub_models.append(model)
        return self

    def predict(
        self,
        h: int,
        level: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> dict[str, Any]:
        """StatsForecast calls this to produce out-of-sample forecasts.

        Return a dictionary that at least has a key 'mean' or <your_model_name>.
        If you produce multiple forecast columns, each can be put in the dictionary.
        """
        if self.sub_models is None:
            raise RuntimeError("Call fit() before predict()")

        # For example, average predictions from each sub-model:
        dfs: list[pl.DataFrame] = []
        for model in self.sub_models:
            # The built-in sub-models also have a predict(...) method
            fcst = model.predict(h=h, level=level)
            dfs.append(pl.from_dict(fcst))
        ensemble: pl.DataFrame = sum(dfs) / len(dfs)  # type: ignore[assignment]

        res: dict[str, Any] = {"mean": ensemble["mean"]}
        # If you want intervals, compute them and add to `res` with keys like 'mean-lo-95', 'mean-hi-95'
        if level is None:
            return res
        sorted_level: list[int] = sorted(level) if isinstance(level, tuple) else [level]
        if self.prediction_intervals is not None:
            res = self._add_predict_conformal_intervals(res, sorted_level)
        else:
            res = {
                "mean": ensemble["mean"],
                **{f"lo-{quantile}": ensemble[f"lo-{quantile}"] for quantile in reversed(sorted_level)},
                **{f"hi-{quantile}": ensemble[f"hi-{quantile}"] for quantile in sorted_level},
            }
        return res
