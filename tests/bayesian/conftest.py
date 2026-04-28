import numpy as np


def local_level_matrices(q: float = 1.0, r: float = 1.0):
    """Return system matrices (F, H, Q, R) for a local-level model.

    State: x_t (scalar level)
    Observation: y_t = x_t + v_t
    Transition: x_t = x_{t-1} + w_t
    """
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[q]])
    R = np.array([[r]])
    return F, H, Q, R
