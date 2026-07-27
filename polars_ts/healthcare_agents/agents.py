"""Specialized clinical agents for EHR vital-sign monitoring.

The agents operate on a single vital-sign observation (one row of
``ClinicalEnv``) in :data:`~polars_ts.healthcare_agents.env.VITAL_CHANNELS`
order: ``(heart_rate, systolic_bp, respiratory_rate, temperature, spo2)``.

Scoring heuristics are grounded in widely used bedside instruments — qSOFA and
SIRS for sepsis risk and a NEWS-style aggregate for escalation — kept
dependency-free and deterministic so they are testable and auditable.
"""

from __future__ import annotations

import numpy as np

# Channel indices within a vitals row.
_HR, _SBP, _RR, _TEMP, _SPO2 = 0, 1, 2, 3, 4


class SepsisWarningAgent:
    """Early sepsis-risk scoring from vital signs (qSOFA + SIRS heuristics).

    qSOFA awards a point each for respiratory rate >= 22, systolic BP <= 100,
    and (unavailable here) altered mentation. SIRS awards points for
    temperature, heart rate and respiratory-rate derangement. The agent flags
    sepsis risk when the combined score meets ``threshold``.

    Parameters
    ----------
    threshold
        Combined qSOFA+SIRS score at or above which sepsis risk is flagged.

    """

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = threshold

    def score(self, vitals: np.ndarray) -> tuple[float, bool]:
        """Return ``(risk_score, is_at_risk)`` for one vital-sign row."""
        hr, sbp, rr, temp = (
            float(vitals[_HR]),
            float(vitals[_SBP]),
            float(vitals[_RR]),
            float(vitals[_TEMP]),
        )
        qsofa = (rr >= 22.0) + (sbp <= 100.0)
        sirs = (temp > 38.0 or temp < 36.0) + (hr > 90.0) + (rr > 20.0)
        risk = float(qsofa + sirs)
        return risk, bool(risk >= self.threshold)


class VitalMonitorAgent:
    """Per-channel physiological range monitor.

    Flags any vital sign outside its normal reference band and reports how many
    channels are deranged.

    Parameters
    ----------
    bounds
        Optional mapping ``channel_index -> (low, high)`` overriding the
        default adult reference ranges.

    """

    #: Default adult normal reference ranges per channel index.
    DEFAULT_BOUNDS: dict[int, tuple[float, float]] = {
        _HR: (60.0, 100.0),
        _SBP: (100.0, 140.0),
        _RR: (12.0, 20.0),
        _TEMP: (36.0, 38.0),
        _SPO2: (94.0, 100.0),
    }

    def __init__(self, bounds: dict[int, tuple[float, float]] | None = None) -> None:
        self.bounds = bounds or dict(self.DEFAULT_BOUNDS)

    def score(self, vitals: np.ndarray) -> tuple[float, bool]:
        """Return ``(n_deranged, any_deranged)`` for one vital-sign row."""
        deranged = 0
        for c, (low, high) in self.bounds.items():
            if c < len(vitals):
                v = float(vitals[c])
                if v < low or v > high:
                    deranged += 1
        return float(deranged), bool(deranged > 0)


class EscalationAgent:
    """Map a clinical picture to a discrete escalation tier (NEWS-style).

    Aggregates a NEWS-like severity score from the vital-sign row and combines
    it with upstream sepsis and monitoring signals to pick an escalation tier
    in ``[0, n_tiers)``:

    - ``0`` routine monitoring
    - ``1`` increased observation frequency
    - ``2`` urgent clinical review
    - ``3`` rapid-response / ICU transfer

    """

    def __init__(self, n_tiers: int = 4) -> None:
        self.n_tiers = n_tiers

    def news_score(self, vitals: np.ndarray) -> int:
        """Compute a simplified National Early Warning Score (0-3 per channel)."""
        hr, sbp, rr, temp, spo2 = (float(vitals[i]) for i in range(5))
        score = 0
        # Respiratory rate.
        score += 3 if rr <= 8 or rr >= 25 else 2 if rr >= 21 else 1 if rr <= 11 else 0
        # SpO2.
        score += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
        # Systolic BP.
        score += 3 if sbp <= 90 or sbp >= 220 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0
        # Heart rate.
        score += 3 if hr <= 40 or hr >= 131 else 2 if hr >= 111 else 1 if hr >= 91 or hr <= 50 else 0
        # Temperature.
        score += 3 if temp <= 35 else 2 if temp >= 39.1 else 1 if temp >= 38.1 or temp <= 36 else 0
        return score

    def decide(self, vitals: np.ndarray, sepsis_risk: bool, n_deranged: float) -> int:
        """Choose an escalation tier from NEWS score and upstream signals."""
        news = self.news_score(vitals)
        # NEWS thresholds: >=7 high, 5-6 medium, 1-4 low, 0 none.
        tier = 3 if news >= 7 else 2 if news >= 5 else 1 if news >= 1 else 0
        # Sepsis risk forces at least urgent review; many deranged vitals bump a tier.
        if sepsis_risk:
            tier = max(tier, 2)
        if n_deranged >= 3:
            tier = min(tier + 1, self.n_tiers - 1)
        return int(min(tier, self.n_tiers - 1))


class TreatmentAgent:
    """Contextual-bandit treatment recommender over escalation tiers.

    Learns a per-tier preference for a small action set via a simple
    reward-averaging (bandit) update, letting the recommended intervention
    adapt to observed outcomes without any heavyweight RL dependency.

    Parameters
    ----------
    actions
        Ordered intervention labels; the index is the action id.
    seed
        Seed for the exploration RNG (``numpy.random.default_rng``).

    """

    DEFAULT_ACTIONS: tuple[str, ...] = (
        "continue_monitoring",
        "administer_fluids",
        "start_antibiotics",
        "transfer_icu",
    )

    def __init__(self, actions: tuple[str, ...] | None = None, seed: int = 0) -> None:
        self.actions = actions or self.DEFAULT_ACTIONS
        self._rng = np.random.default_rng(seed)
        n_tiers, n_actions = 4, len(self.actions)
        # Optimistic prior nudges each tier toward its namesake action.
        self._value = np.zeros((n_tiers, n_actions), dtype=np.float64)
        for t in range(n_tiers):
            self._value[t, min(t, n_actions - 1)] = 0.1
        self._counts = np.zeros((n_tiers, n_actions), dtype=np.int64)

    def recommend(self, tier: int, explore: float = 0.0) -> int:
        """Return an action id for ``tier``; ``explore`` is the epsilon-greedy rate."""
        tier = int(min(max(tier, 0), self._value.shape[0] - 1))
        if explore > 0.0 and float(self._rng.random()) < explore:
            return int(self._rng.integers(len(self.actions)))
        return int(np.argmax(self._value[tier]))

    def update(self, tier: int, action: int, reward: float) -> None:
        """Incremental sample-average update of the tier/action value estimate."""
        tier = int(min(max(tier, 0), self._value.shape[0] - 1))
        self._counts[tier, action] += 1
        n = self._counts[tier, action]
        self._value[tier, action] += (reward - self._value[tier, action]) / n


def federated_average(
    values: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Privacy-preserving FedAvg of per-site agent parameters.

    Combines locally trained parameter arrays (e.g. ``TreatmentAgent`` value
    tables) from multiple sites into a single global array by weighted mean,
    without any site sharing its raw patient data.

    Parameters
    ----------
    values
        Per-site parameter arrays, all of identical shape.
    weights
        Optional per-site weights (e.g. local sample counts). Defaults to
        equal weighting.

    Returns
    -------
    numpy.ndarray
        The aggregated global parameter array.

    """
    if not values:
        raise ValueError("values must contain at least one site's parameters")
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("all site parameter arrays must share the same shape")
    w = np.ones(len(arrays)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape[0] != len(arrays):
        raise ValueError("weights length must match number of sites")
    if float(w.sum()) == 0.0:
        raise ValueError("weights must not sum to zero")
    stacked = np.stack(arrays, axis=0)
    return np.tensordot(w, stacked, axes=([0], [0])) / float(w.sum())
