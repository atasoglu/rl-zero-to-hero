import numpy as np
from rl_common import make_env

# Observation bounds for CartPole's 4 dimensions:
# [cart position, cart velocity, pole angle, pole angular velocity]
_OBS_BOUNDS = [
    (-2.4, 2.4),
    (-3.0, 3.0),
    (-0.21, 0.21),
    (-2.5, 2.5),
]

_DEFAULT_N_BINS = 10
_bins: list[np.ndarray] = []
_n_bins_per_dim: list[int] = []


def _build_bins(n_bins: int):
    global _bins, _n_bins_per_dim
    _bins = [np.linspace(lo, hi, n_bins - 1) for lo, hi in _OBS_BOUNDS]
    _n_bins_per_dim = [n_bins] * len(_OBS_BOUNDS)


_build_bins(_DEFAULT_N_BINS)


def discretize(obs: np.ndarray) -> int:
    indices = tuple(np.digitize(obs[i], _bins[i]) for i in range(len(_bins)))
    state = 0
    for idx, n in zip(indices, _n_bins_per_dim):
        state = state * n + idx
    return state


def n_states(n_bins: int = _DEFAULT_N_BINS) -> int:
    if n_bins != _DEFAULT_N_BINS:
        _build_bins(n_bins)
    return n_bins ** len(_OBS_BOUNDS)


__all__ = ["make_env", "discretize", "n_states"]
