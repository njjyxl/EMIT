"""Small helpers shared across the package."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("emit")


def setup_logging(verbose: bool = True) -> None:
    """Send the package's log records to the console (used by the command line)."""
    if logger.handlers:
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False


def resolve_device(device=None):
    """A ``torch.device``; ``None`` or ``"auto"`` picks CUDA when it is available."""
    import torch

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def as_str_array(values) -> np.ndarray:
    """A 1-D array of Python strings, decoding bytes where needed."""
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def lognorm(counts, scale: float = 1e4) -> np.ndarray:
    """Library-size normalization to ``scale`` counts per row followed by ``log1p``."""
    a = np.clip(np.asarray(counts, np.float32), 0, None)
    s = np.clip(a.sum(1, keepdims=True), 1e-12, None)
    return np.log1p(a / s * scale)


def group_mean(values, groups, n_groups: int) -> np.ndarray:
    """Mean of the rows of ``values`` within each group (``groups`` holds row -> group id)."""
    total = np.zeros((n_groups, values.shape[1]), np.float64)
    n = np.zeros(n_groups)
    np.add.at(total, groups, values)
    np.add.at(n, groups, 1.0)
    return (total / np.clip(n, 1, None)[:, None]).astype(np.float32)
