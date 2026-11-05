"""Measures of how well a per-cell output agrees with the data.

Two kinds are provided.  Measures against per-cell truth (:func:`within_spot_r`,
:func:`gene_correlation`) apply to benchmarks in which spots are simulated from
single-cell-resolution data.  Measures that need no truth (:func:`conservation_error`,
:func:`heldout_relative_distance`, :func:`fit_diagnostics`) apply to any section.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ._utils import group_mean, lognorm

__all__ = [
    "within_spot_r",
    "gene_correlation",
    "relative_distance",
    "conservation_error",
    "heldout_baselines",
    "heldout_relative_distance",
    "heldout_nll",
    "fit_diagnostics",
]


# ---------------------------------------------------------------------- against truth
def within_spot_r(pred, truth, cell_spot, n_spots: int | None = None) -> float:
    """Correlation between predicted and true deviation of each cell from its spot mean.

    Expression is library-size normalized and log-transformed, the mean over the cells of
    each spot is removed from both matrices, and the correlation is taken over all cells
    and genes.  A method that gives every cell the profile of its spot scores zero, so
    the value isolates what was resolved inside spots.
    """
    cell_spot = np.asarray(cell_spot)
    inside = cell_spot >= 0
    spot = cell_spot[inside]
    n_spots = int(spot.max()) + 1 if n_spots is None else n_spots
    t, p = lognorm(np.asarray(truth)[inside]), lognorm(np.asarray(pred)[inside])
    dt = t - group_mean(t, spot, n_spots)[spot]
    dp = p - group_mean(p, spot, n_spots)[spot]
    norm = np.linalg.norm(dt) * np.linalg.norm(dp)
    return float(np.dot(dt.ravel(), dp.ravel()) / max(norm, 1e-12))


def gene_correlation(pred, truth) -> np.ndarray:
    """Pearson correlation across cells of each gene, on log-normalized expression."""
    p, t = lognorm(pred), lognorm(truth)
    p = p - p.mean(0)
    t = t - t.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (p * t).sum(0) / den, np.nan)


def relative_distance(reference, estimate) -> float:
    """``||reference - estimate|| / ||reference||`` on log-normalized expression."""
    a, b = lognorm(reference), lognorm(estimate)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


# ---------------------------------------------------------------------- without truth
def conservation_error(pred, counts, cell_spot) -> np.ndarray:
    """Relative difference, per spot, between the summed cells of a spot and its counts.

    ``||sum_i pred_i - y_b||_1 / ||y_b||_1`` for every spot that contains a cell.  It is
    zero (to floating-point error) for the cells EMIT allocates.
    """
    cell_spot = np.asarray(cell_spot)
    counts = np.asarray(counts, np.float64)
    inside = cell_spot >= 0
    total = np.zeros_like(counts)
    np.add.at(total, cell_spot[inside], np.asarray(pred, np.float64)[inside])
    occupied = np.bincount(cell_spot[inside], minlength=len(counts)) > 0
    err = np.abs(total - counts).sum(1) / np.clip(counts.sum(1), 1e-12, None)
    return np.where(occupied, err, np.nan)


def heldout_baselines(counts, spot_xy, heldout) -> tuple[float, float]:
    """Relative distance on the held-out spots of two estimates that use no image.

    ``nearest`` copies the closest training spot, ``mean`` uses the mean profile of all
    training spots.  Held-out reconstruction should be read against these two numbers.
    """
    counts = np.asarray(counts)
    train = np.where(~heldout)[0]
    if not len(train) or not heldout.any():
        return float("nan"), float("nan")
    nearest = train[cKDTree(spot_xy[train]).query(spot_xy[heldout], k=1)[1]]

    def ln(a):
        a = np.clip(np.asarray(a, np.float64), 0, None)
        return np.log1p(a / np.clip(a.sum(1, keepdims=True), 1e-9, None) * 1e4)

    y = ln(counts[heldout])
    norm = max(np.linalg.norm(y), 1e-12)
    near = float(np.linalg.norm(y - ln(counts[nearest])) / norm)
    mean = float(np.linalg.norm(
        y - ln(np.repeat(counts[train].mean(0, keepdims=True), int(heldout.sum()), 0))) / norm)
    return near, mean


def heldout_relative_distance(pi, lam, cell_spot, counts, heldout) -> float:
    """Relative distance between held-out spot counts and the cells predicted inside them.

    The expected counts ``lam * pi`` of the cells of each held-out spot are summed and
    compared with the measurement of that spot, which the model was not trained on.
    Arguments are torch tensors; ``heldout`` is a boolean mask over spots.
    """
    import torch

    keep = (cell_spot >= 0) & heldout[cell_spot.clamp_min(0)]
    if not bool(keep.any()):
        return float("nan")
    spot = cell_spot[keep]
    total = torch.zeros_like(counts).index_add_(0, spot, lam[keep].unsqueeze(-1) * pi[keep])
    rows = torch.unique(spot)
    y, p = counts[rows], total[rows]
    y = torch.log1p(y / y.sum(1, keepdim=True).clamp_min(1e-9) * 1e4)
    p = torch.log1p(p / p.sum(1, keepdim=True).clamp_min(1e-9) * 1e4)
    return float(torch.linalg.norm(y - p) / torch.linalg.norm(y).clamp_min(1e-12))


def heldout_nll(pi, lam, cell_spot, counts, heldout) -> float:
    """Poisson negative log-likelihood of the held-out spot counts (torch tensors)."""
    import torch

    keep = (cell_spot >= 0) & heldout[cell_spot.clamp_min(0)]
    if not bool(keep.any()):
        return float("nan")
    spot = cell_spot[keep]
    mu = torch.zeros_like(counts).index_add_(0, spot, lam[keep].unsqueeze(-1) * pi[keep])
    rows = torch.unique(spot)
    y, mu = counts[rows], mu[rows].clamp_min(1e-9)
    return float((mu - y * mu.log()).mean())


def fit_diagnostics(pi, lam, cell_spot, counts, expression, n_spots: int) -> dict:
    """Truth-free descriptions of how the counts were arranged within spots.

    concentration
        ``1 - H / log(n)`` of the shares of each spot and gene over the cells of the
        spot, weighted by the counts: 0 when molecules are spread evenly over the cells
        of a spot, 1 when one cell receives them all.
    within_spot_share
        Median over genes of the fraction of compositional variance that lies within
        spots; 0 for an output that copies the spot profile to every cell.
    drift
        Mean ``KL(pi_i || mean profile of the spot of i)``.  A steady rise over rounds
        signals that self-training is moving cells away from what was measured.

    ``pi``, ``lam``, ``cell_spot`` and ``counts`` are torch tensors; ``expression`` is the
    ``(N, G)`` numpy array of allocated counts.
    """
    import torch

    inside = cell_spot >= 0
    spot = cell_spot[inside]
    cells = torch.nonzero(inside, as_tuple=True)[0]
    # Two chunked passes avoid several cells x genes temporaries at once.
    chunk = max(1, int(2.5e8) // (4 * max(pi.shape[1], 1)))

    total = torch.zeros_like(counts)
    mean_profile = torch.zeros_like(counts)
    n_cells = torch.zeros(counts.shape[0], device=counts.device)
    for a in range(0, len(cells), chunk):
        rows, b = cells[a:a + chunk], spot[a:a + chunk]
        total.index_add_(0, b, (lam[rows].unsqueeze(-1) * pi[rows]).clamp_min(1e-12))
        mean_profile.index_add_(0, b, pi[rows])
        n_cells.index_add_(0, b, torch.ones_like(b, dtype=counts.dtype))
    mean_profile = mean_profile / n_cells.clamp_min(1.0).unsqueeze(-1)

    entropy = torch.zeros_like(counts)
    drift = 0.0
    for a in range(0, len(cells), chunk):
        rows, b = cells[a:a + chunk], spot[a:a + chunk]
        share = (lam[rows].unsqueeze(-1) * pi[rows]).clamp_min(1e-12) / total[b].clamp_min(1e-12)
        entropy.index_add_(0, b, -share * share.clamp_min(1e-12).log())
        p_cell, p_spot = pi[rows].clamp_min(1e-12), mean_profile[b].clamp_min(1e-12)
        drift += float((p_cell * (p_cell.log() - p_spot.log())).sum())
    drift /= max(len(cells), 1)

    concentration = 1.0 - entropy / n_cells.clamp_min(2.0).log().unsqueeze(-1)
    weight = counts.clamp_min(0)
    concentration = float((concentration * weight).sum() / weight.sum().clamp_min(1.0))

    spot_np = cell_spot.cpu().numpy()
    mask = spot_np >= 0
    comp = np.clip(expression[mask], 0, None)
    comp = np.log(comp / np.clip(comp.sum(1, keepdims=True), 1e-12, None) + 1e-6)
    centred = comp - _group_mean64(comp, spot_np[mask], n_spots)[spot_np[mask]]
    share = float(np.median(centred.var(0) / np.clip(comp.var(0), 1e-12, None)))
    return {"concentration": concentration, "within_spot_share": share, "drift": drift}


def _group_mean64(values, groups, n_groups):
    total = np.zeros((n_groups, values.shape[1]), np.float64)
    n = np.zeros(n_groups)
    np.add.at(total, groups, values)
    np.add.at(n, groups, 1.0)
    return total / np.clip(n, 1, None)[:, None]
