"""Predictive-maintenance agents: spectral features, health, RUL, scheduling.

The agents form a pipeline: :class:`SpectralFeatureAgent` extracts vibration
band energies, :class:`HealthIndexAgent` fuses multi-sensor signals into a
health index, :class:`RULEstimator` projects Remaining Useful Life, and the
reinforcement-learning :class:`MaintenanceSchedulerAgent` decides when to
operate, inspect, or maintain.
"""

from __future__ import annotations

import numpy as np

from polars_ts.iiot_agents.env import MAINTAIN, OPERATE


class SpectralFeatureAgent:
    """Extract vibration band-energy features from a sensor window via FFT.

    A dependency-free complement to the imaging module's ``to_spectrogram`` /
    ``to_scalogram``: computes RMS amplitude and the fraction of spectral
    energy in low/mid/high frequency bands, which shift as bearings and gears
    degrade.

    Parameters
    ----------
    n_bands
        Number of equal-width frequency bands to summarise energy over.

    """

    def __init__(self, n_bands: int = 3) -> None:
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        self.n_bands = n_bands

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Return ``[rms, band_0_frac, …, band_{n_bands-1}_frac]`` for a 1D window."""
        window = np.asarray(window, dtype=np.float64)
        rms = float(np.sqrt(np.mean(window**2)))
        centered = window - window.mean()
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        total = float(spectrum.sum()) + 1e-12
        bands = np.array_split(spectrum, self.n_bands)
        fracs = [float(b.sum()) / total for b in bands]
        return np.array([rms, *fracs], dtype=np.float64)


class HealthIndexAgent:
    """Fuse multi-sensor readings into a health index in ``[0, 1]``.

    Degradation is modelled as growth in each sensor's RMS amplitude relative
    to a healthy baseline; per-sensor degradation scores are fused by weighted
    mean and mapped to a health index (1 = healthy, 0 = failed).

    Parameters
    ----------
    baseline
        Per-sensor healthy RMS baseline. Inferred from the first ``warmup``
        steps when omitted.
    warmup
        Number of initial steps used to infer ``baseline`` when not supplied.
    fail_ratio
        RMS ratio (current / baseline) at which health reaches 0.
    weights
        Optional per-sensor fusion weights (defaults to equal weighting).

    """

    def __init__(
        self,
        baseline: np.ndarray | None = None,
        warmup: int = 5,
        fail_ratio: float = 3.0,
        weights: np.ndarray | None = None,
    ) -> None:
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=np.float64)
        self.warmup = warmup
        self.fail_ratio = fail_ratio
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    def fit_baseline(self, sensors: np.ndarray) -> None:
        """Infer the healthy per-sensor RMS baseline from initial observations."""
        head = np.asarray(sensors, dtype=np.float64)[: self.warmup]
        self.baseline = np.sqrt(np.mean(head**2, axis=0)) + 1e-12

    def score(self, window: np.ndarray) -> float:
        """Return a fused health index in ``[0, 1]`` for a multi-sensor window.

        ``window`` is ``(w, n_sensors)`` or a single ``(n_sensors,)`` row.
        """
        window = np.atleast_2d(np.asarray(window, dtype=np.float64))
        rms = np.sqrt(np.mean(window**2, axis=0)) + 1e-12
        if self.baseline is None:
            self.baseline = rms
        ratio = rms / self.baseline
        # Per-sensor degradation in [0, 1]: 0 at baseline, 1 at fail_ratio.
        degradation = np.clip((ratio - 1.0) / (self.fail_ratio - 1.0), 0.0, 1.0)
        w = self.weights if self.weights is not None else np.ones(degradation.shape[0])
        fused = float(np.average(degradation, weights=w))
        return float(np.clip(1.0 - fused, 0.0, 1.0))


class RULEstimator:
    """Estimate Remaining Useful Life by extrapolating the health trend.

    Fits a linear trend to the recent health-index history and projects the
    number of steps until it reaches ``failure_threshold``.

    Parameters
    ----------
    failure_threshold
        Health level defining failure.
    min_history
        Minimum number of points before a finite RUL is returned.

    """

    def __init__(self, failure_threshold: float = 0.2, min_history: int = 3) -> None:
        self.failure_threshold = failure_threshold
        self.min_history = min_history

    def estimate(self, health_history: list[float] | np.ndarray) -> float:
        """Return estimated steps until failure (``inf`` if not yet declining)."""
        h = np.asarray(health_history, dtype=np.float64)
        if h.size < self.min_history:
            return float("inf")
        x = np.arange(h.size, dtype=np.float64)
        slope, intercept = np.polyfit(x, h, 1)
        current = float(intercept + slope * (h.size - 1))
        if current <= self.failure_threshold:
            return 0.0
        if slope >= -1e-9:  # stable or improving
            return float("inf")
        return float((current - self.failure_threshold) / (-slope))


class MaintenanceSchedulerAgent:
    """Tabular Q-learning agent that schedules maintenance from health state.

    The health index is discretised into ``n_states`` buckets; the agent learns
    a Q-value for each ``(health_bucket, action)`` pair from environment reward,
    balancing uptime against maintenance cost and failure risk.

    Parameters
    ----------
    n_states
        Number of health buckets (finer = more granular timing).
    n_actions
        Size of the maintenance action set.
    alpha, gamma, epsilon
        Learning rate, discount factor, and epsilon-greedy exploration rate.
    seed
        Seed for the exploration RNG.

    """

    def __init__(
        self,
        n_states: int = 10,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        # Optimistic prior: prefer operating while healthy, maintaining while degraded.
        self.q[-1, OPERATE] = 0.1
        self.q[0, MAINTAIN] = 0.1

    def bucket(self, health: float) -> int:
        """Map a health index in ``[0, 1]`` to a discrete state bucket."""
        b = int(np.clip(health, 0.0, 1.0) * (self.n_states - 1) + 0.5)
        return int(min(max(b, 0), self.n_states - 1))

    def act(self, state: int, explore: bool = False) -> int:
        """Return an action for ``state`` (epsilon-greedy when ``explore``)."""
        if explore and float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Apply a Q-learning temporal-difference update."""
        best_next = float(np.max(self.q[next_state]))
        td_target = reward + self.gamma * best_next
        self.q[state, action] += self.alpha * (td_target - self.q[state, action])
