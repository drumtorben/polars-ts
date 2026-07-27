"""Agents for hierarchical energy/demand forecasting.

- :class:`DemandForecastAgent` — per-node seasonal demand forecasting.
- :class:`WeatherContextAgent` — weather-driven demand adjustment (degree-day).
- :class:`RenewableAgent` — net demand after intermittent renewable generation.
- :class:`DemandResponseAgent` — peak-shaving / load-shifting optimisation.
"""

from __future__ import annotations

import numpy as np


class DemandForecastAgent:
    """Seasonal-naive demand forecaster for a single node.

    Repeats the most recent seasonal cycle; falls back to the historical mean
    when history is shorter than one season.

    Parameters
    ----------
    season
        Seasonal period in steps (e.g. 24 for hourly-with-daily-cycle).

    """

    def __init__(self, season: int = 24) -> None:
        if season < 1:
            raise ValueError("season must be >= 1")
        self.season = season

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Return an ``horizon``-step demand forecast for one node."""
        h = np.asarray(history, dtype=np.float64)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if h.size < self.season:
            return np.full(horizon, float(h.mean()) if h.size else 0.0)
        last_cycle = h[-self.season :]
        reps = int(np.ceil(horizon / self.season))
        return np.tile(last_cycle, reps)[:horizon]


class WeatherContextAgent:
    """Adjust a base demand forecast for weather via a degree-day response.

    Demand rises with both heating (cold) and cooling (hot) load relative to a
    comfort temperature.

    Parameters
    ----------
    comfort_temp
        Temperature (deg C) of minimal weather-driven load.
    cooling_coef, heating_coef
        Additional demand per degree above / below ``comfort_temp``.

    """

    def __init__(self, comfort_temp: float = 18.0, cooling_coef: float = 2.0, heating_coef: float = 3.0) -> None:
        self.comfort_temp = comfort_temp
        self.cooling_coef = cooling_coef
        self.heating_coef = heating_coef

    def adjust(self, base_forecast: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Return the weather-adjusted forecast for the given temperature path."""
        base = np.asarray(base_forecast, dtype=np.float64)
        temp = np.asarray(temperature, dtype=np.float64)
        if temp.shape != base.shape:
            raise ValueError("temperature must match the forecast horizon")
        cooling = np.clip(temp - self.comfort_temp, 0.0, None) * self.cooling_coef
        heating = np.clip(self.comfort_temp - temp, 0.0, None) * self.heating_coef
        return base + cooling + heating


class RenewableAgent:
    """Compute net demand after subtracting intermittent renewable generation.

    Parameters
    ----------
    curtail
        When ``True``, net demand is floored at zero (excess generation is
        curtailed rather than exported).

    """

    def __init__(self, curtail: bool = False) -> None:
        self.curtail = curtail

    def net_demand(self, demand: np.ndarray, generation: np.ndarray) -> np.ndarray:
        """Return ``demand - generation`` (floored at 0 when ``curtail``)."""
        demand = np.asarray(demand, dtype=np.float64)
        generation = np.asarray(generation, dtype=np.float64)
        if generation.shape != demand.shape:
            raise ValueError("generation must match the demand horizon")
        net = demand - generation
        return np.clip(net, 0.0, None) if self.curtail else net


class DemandResponseAgent:
    """Peak-shaving / load-shifting optimiser over a demand profile.

    Energy above ``capacity`` is shed from peak periods and shifted into the
    lowest-demand troughs, conserving total energy while flattening peaks.

    Parameters
    ----------
    capacity
        Maximum demand target; peaks above it are shifted to troughs.

    """

    def __init__(self, capacity: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity

    def optimize(self, profile: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(shifted_profile, energy_shifted)``.

        Total energy is always preserved. When the profile *can* fit under
        ``capacity`` (total energy <= ``capacity`` * n), peaks are clipped to
        ``capacity`` and the shed energy is water-filled into the lowest
        periods without exceeding ``capacity``. When it cannot (an inherently
        over-loaded window), the profile is flattened to its mean — the closest
        feasible approximation.
        """
        prof = np.asarray(profile, dtype=np.float64).copy()
        n = prof.size
        shifted = float(np.clip(prof - self.capacity, 0.0, None).sum())
        if shifted == 0.0:
            return prof, 0.0

        total = float(prof.sum())
        if total <= self.capacity * n:
            prof = np.minimum(prof, self.capacity)
            deficit = shifted
            for i in np.argsort(prof, kind="stable"):
                if deficit <= 1e-12:
                    break
                add = min(self.capacity - float(prof[i]), deficit)
                prof[i] += add
                deficit -= add
        else:
            prof = np.full(n, total / n)
        return prof, shifted
