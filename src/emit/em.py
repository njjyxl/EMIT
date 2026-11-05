"""The two steps of EMIT: apportioning spot counts to cells, and the training objective.

Each spot is treated as a set of molecules to be divided among the cells that produced
them.  With the count of gene ``g`` in cell ``i`` modelled as
``Poisson(lam[i] * pi[i, g])``, the counts of the cells of a spot, given the spot total,
are multinomial with probabilities proportional to ``lam[i] * pi[i, g]``.  The E-step
assigns to each cell its expected share, and the allocated counts of every spot sum
exactly to its measurement.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["e_step", "e_step_diffuse", "m_loss"]


def e_step(pi, lam, cell_spot, counts, shuffle: bool = False, generator=None):
    """Expected counts of each cell given the counts of its spot.

    Parameters
    ----------
    pi : (N, G) tensor
        Expression profile of each cell.
    lam : (N,) tensor
        Library size of each cell.
    cell_spot : (N,) long tensor
        Spot of each cell, ``-1`` for a cell outside every spot.
    counts : (B, G) tensor
        Measured spot counts.
    shuffle
        Control only: permute the shares among the cells of each spot, which keeps every
        spot total and destroys which cell receives what.

    Returns
    -------
    allocated : (n, G) tensor or None
        Allocated counts of the ``n`` cells inside spots; ``allocated`` summed over the
        cells of a spot equals the counts of that spot.
    inside : (N,) bool tensor
        The cells the rows of ``allocated`` refer to.
    """
    inside = cell_spot >= 0
    if not bool(inside.any()):
        return None, inside
    spot = cell_spot[inside]
    rate = (lam[inside].unsqueeze(-1) * pi[inside]).clamp_min(1e-12)
    total = torch.zeros_like(counts).index_add_(0, spot, rate)
    share = rate / total[spot].clamp_min(1e-12)
    if shuffle:
        share = share[_within_spot_permutation(spot, generator)]
        norm = torch.zeros_like(counts).index_add_(0, spot, share)
        share = share / norm[spot].clamp_min(1e-12)
    return counts[spot] * share, inside


def e_step_diffuse(pi, lam, cell_id, spot_id, weight, counts, chunk: int = 200_000):
    """E-step that lets a cell contribute to every spot within reach.

    ``(cell_id, spot_id, weight)`` are the pairs from
    :func:`emit.geometry.capture_pairs`.  The counts of spot ``b`` are divided among all
    cells in proportion to ``weight[i, b] * lam[i] * pi[i, g]``, so a cell may draw
    counts from several spots while the counts of each spot are still distributed
    exactly.

    Returns the ``(N, G)`` allocated counts of all cells; cells without a pair get zero.
    """
    n_pairs = len(cell_id)

    def rate(sl):
        return (lam[cell_id[sl]].unsqueeze(-1) * pi[cell_id[sl]]) * weight[sl].unsqueeze(-1)

    # Both passes are chunked over pairs: the (pairs, G) intermediate is the memory cost,
    # and the number of pairs grows with the square of the diffusion width.
    total = torch.zeros_like(counts)
    for start in range(0, n_pairs, chunk):
        sl = slice(start, min(start + chunk, n_pairs))
        total.index_add_(0, spot_id[sl], rate(sl))

    allocated = torch.zeros(pi.shape[0], counts.shape[1], device=pi.device, dtype=pi.dtype)
    for start in range(0, n_pairs, chunk):
        sl = slice(start, min(start + chunk, n_pairs))
        share = rate(sl) / total[spot_id[sl]].clamp_min(1e-12)
        allocated.index_add_(0, cell_id[sl], counts[spot_id[sl]] * share)
    return allocated


def _within_spot_permutation(spot, generator=None):
    """A permutation of the rows that moves a row only within its own spot."""
    order = torch.argsort(spot, stable=True)
    bounds = torch.searchsorted(spot[order], torch.arange(int(spot.max()) + 2, device=spot.device))
    perm = order.clone()
    for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        if hi - lo > 1:
            perm[lo:hi] = order[lo:hi][
                torch.randperm(hi - lo, device=spot.device, generator=generator)]
    out = torch.empty_like(perm)
    out[order] = perm
    return out


def m_loss(pi, log_lam, allocated, inside, cell_spot=None, n_spots: int | None = None,
           w_depth: float = 0.1, w_spread: float = 0.05, w_drift: float = 0.0):
    """Training objective on the allocated counts, which are held fixed.

    * a per-cell multinomial log-likelihood, normalized by the total allocated mass;
    * ``w_depth``: squared error between ``log lam`` and the log of the allocated total;
    * ``w_spread``: negative entropy of the mean profile, which keeps the profiles from
      collapsing onto a few genes;
    * ``w_drift``: mean ``KL(pi[i] || mean profile of the spot of i)``, which limits how
      far self-training moves a cell from what was measured in its spot.  The spot mean
      is a constant in this term, so the gradient moves the cells and not the mean.

    Returns ``(loss, parts)`` with the unweighted value of each term in ``parts``.
    """
    log_pi = pi[inside].clamp_min(1e-12).log()
    mass = allocated.sum().clamp_min(1.0)
    ce = -(allocated * log_pi).sum() / mass
    depth = F.mse_loss(log_lam[inside], allocated.sum(1).clamp_min(1.0).log())
    mean_profile = pi[inside].mean(0)
    spread = (mean_profile * mean_profile.clamp_min(1e-12).log()).sum()
    total = ce + w_depth * depth + w_spread * spread
    parts = {"ce": float(ce.detach()), "depth": float(depth.detach()),
             "spread": float(spread.detach()), "drift": 0.0}
    if w_drift and cell_spot is not None:
        spot = cell_spot[inside]
        p = pi[inside]
        n_cells = torch.zeros(n_spots, device=p.device, dtype=p.dtype).index_add_(
            0, spot, torch.ones_like(spot, dtype=p.dtype))
        spot_mean = torch.zeros(n_spots, p.shape[1], device=p.device, dtype=p.dtype).index_add_(
            0, spot, p) / n_cells.clamp_min(1.0).unsqueeze(-1)
        p_cell = p.clamp_min(1e-12)
        p_spot = spot_mean[spot].detach().clamp_min(1e-12)
        drift = (p_cell * (p_cell.log() - p_spot.log())).sum(1).mean()
        total = total + w_drift * drift
        parts["drift"] = float(drift.detach())
    return total, parts
