"""Spot-based measurements simulated from data with single-cell resolution.

Aggregating the cells of an imaging-based section (for example Xenium) into the geometry
of a coarser platform gives spots whose per-cell truth is known exactly, which is how a
per-cell output can be scored.  A summed spot is an ideal instrument; ``capture`` and
``diffusion_um`` degrade it towards a real one.

:func:`toy_section` generates a small synthetic section for tests and tutorials.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .sample import Sample

__all__ = ["hex_grid", "visium_spots", "square_bins", "toy_section"]

VISIUM_SPOT_UM = 55.0
VISIUM_PITCH_UM = 100.0
_ROW = np.sqrt(3) / 2  # row spacing of a hexagonal lattice, in units of the pitch


def hex_grid(xy, pitch_px: float, margin: float = 0.5) -> np.ndarray:
    """Centres of a hexagonal lattice that covers the bounding box of ``xy``."""
    x0, y0 = xy.min(0) - margin * pitch_px
    x1, y1 = xy.max(0) + margin * pitch_px
    row_height = pitch_px * _ROW
    n_cols = int(np.ceil((x1 - x0) / pitch_px)) + 1
    centres = []
    for row in range(int(np.ceil((y1 - y0) / row_height)) + 1):
        offset = 0.5 * pitch_px if row % 2 else 0.0
        centres.append(np.stack([x0 + offset + np.arange(n_cols) * pitch_px,
                                 np.full(n_cols, y0 + row * row_height)], axis=1))
    return np.concatenate(centres, 0)


def _aggregate(counts, index, n_spots: int) -> np.ndarray:
    out = np.zeros((n_spots, counts.shape[1]), np.float64)
    inside = index >= 0
    np.add.at(out, index[inside], counts[inside])
    return out


def _soft_aggregate(counts, xy, centres, radius_px, sigma_px, chunk: int = 20_000) -> np.ndarray:
    """Spots in which each cell contributes to nearby spots with a capture weight.

    The weight is 1 inside the spot and decays as a Gaussian of the distance beyond its
    edge.  Weights are not normalized: what diffuses outside every spot is lost.
    """
    tree = cKDTree(centres)
    out = np.zeros((len(centres), counts.shape[1]), np.float64)
    reach = radius_px + 3 * sigma_px
    for start in range(0, len(xy), chunk):
        stop = min(start + chunk, len(xy))
        for i, near in zip(range(start, stop), tree.query_ball_point(xy[start:stop], reach)):
            if not near:
                continue
            near = np.asarray(near)
            dist = np.linalg.norm(centres[near] - xy[i], axis=1)
            weight = np.exp(-0.5 * (np.maximum(dist - radius_px, 0.0) / sigma_px) ** 2)
            keep = weight > 1e-3
            if keep.any():
                out[near[keep]] += np.outer(weight[keep], counts[i])
    return out


def _thin(counts, capture: float, rng) -> np.ndarray:
    if capture >= 1.0:
        return counts
    return rng.binomial(np.rint(counts).astype(np.int64), capture).astype(np.float64)


def visium_spots(counts, xy, px_um: float, spot_um: float = VISIUM_SPOT_UM,
                 pitch_um: float = VISIUM_PITCH_UM, capture: float = 1.0,
                 diffusion_um: float = 0.0, seed: int = 0) -> dict:
    """Aggregate single cells into Visium-like spots on a hexagonal lattice.

    Parameters
    ----------
    counts : (n_cells, n_genes) array
        Measured counts of each cell.
    xy : (n_cells, 2) array
        Cell centroids in pixels.
    px_um
        Micrometres per pixel.
    spot_um, pitch_um
        Spot diameter and centre-to-centre distance.
    capture
        Probability that a transcript is captured (binomial thinning).
    diffusion_um
        Width of lateral diffusion before capture; 0 sums the cells inside each spot.

    Returns
    -------
    dict with ``counts`` (spots x genes), ``spot_xy``, ``cell_spot`` (-1 outside every
    spot), ``spot_diameter_px`` and ``cells_per_spot``.  Spots without counts are dropped.

    Notes
    -----
    Discs of 55 um on a 100 um pitch cover about 27% of the tissue, so most cells lie
    outside every spot and should be scored separately from the cells inside spots.
    """
    counts = np.asarray(counts)
    xy = np.asarray(xy, float)
    rng = np.random.default_rng(seed)
    radius_px = 0.5 * spot_um / px_um
    centres = hex_grid(xy, pitch_um / px_um)
    dist, nearest = cKDTree(centres).query(xy, k=1)
    index = np.where(dist <= radius_px, nearest, -1).astype(np.int64)

    if diffusion_um > 0:
        spots = _soft_aggregate(counts, xy, centres, radius_px, diffusion_um / px_um)
    else:
        spots = _aggregate(counts, index, len(centres))
    spots = _thin(spots, capture, rng)

    occupied = spots.sum(1) > 0
    lut = np.full(len(centres), -1, np.int64)
    lut[np.where(occupied)[0]] = np.arange(int(occupied.sum()))
    cell_spot = np.where(index >= 0, lut[np.clip(index, 0, None)], -1)
    per_spot = np.bincount(index[index >= 0], minlength=len(centres))
    return {"counts": spots[occupied], "spot_xy": centres[occupied], "cell_spot": cell_spot,
            "spot_diameter_px": 2 * radius_px, "cells_per_spot": per_spot[occupied]}


def square_bins(counts, xy, px_um: float, bin_um: float = 8.0, capture: float = 1.0,
                seed: int = 0) -> dict:
    """Aggregate single cells into square bins that tile the tissue (Visium HD style)."""
    counts = np.asarray(counts)
    xy = np.asarray(xy, float)
    rng = np.random.default_rng(seed)
    step = bin_um / px_um
    ij = np.floor((xy - xy.min(0)) / step).astype(np.int64)
    n_x = ij[:, 0].max() + 1
    key, index = np.unique(ij[:, 1] * n_x + ij[:, 0], return_inverse=True)
    centres = np.stack([(key % n_x + 0.5) * step + xy.min(0)[0],
                        (key // n_x + 0.5) * step + xy.min(0)[1]], axis=1)
    return {"counts": _thin(_aggregate(counts, index, len(key)), capture, rng),
            "spot_xy": centres, "cell_spot": index.astype(np.int64),
            "cells_per_spot": np.bincount(index, minlength=len(key))}


def toy_section(n_cells: int = 6000, n_genes: int = 200, n_types: int = 5, width_um: float = 1500.0,
                px_um: float = 0.5, depth: float = 300.0, morph_noise: float = 0.6,
                seed: int = 0) -> tuple[Sample, dict]:
    """A small synthetic section with known per-cell expression.

    Cells of ``n_types`` types are scattered over a square of tissue.  The type of a cell
    depends on where it lies (smooth spatial domains) with some intermixing, its
    morphometrics depend on its type, and its counts are drawn from the profile of its
    type.  The cells are then aggregated into Visium-like spots.

    Returns
    -------
    sample : Sample
        What EMIT is given: spot counts, cell positions and morphometrics.
    truth : dict
        ``counts`` (n_cells x n_genes) and ``cell_type`` of every cell.
    """
    rng = np.random.default_rng(seed)
    width_px = width_um / px_um
    xy = rng.uniform(0, width_px, size=(n_cells, 2))

    # spatial domains: each type has a few centres, and a cell usually takes the type of
    # the nearest centre, sometimes a random one
    centres = rng.uniform(0, width_px, size=(3 * n_types, 2))
    domain = np.tile(np.arange(n_types), 3)[cKDTree(centres).query(xy)[1]]
    cell_type = np.where(rng.random(n_cells) < 0.35, rng.integers(0, n_types, n_cells), domain)

    profiles = rng.gamma(0.3, 1.0, size=(n_types, n_genes)) + 1e-3
    for k in range(n_types):  # marker genes
        profiles[k, rng.choice(n_genes, max(3, n_genes // (4 * n_types)), replace=False)] *= 12
    profiles /= profiles.sum(1, keepdims=True)
    library = rng.gamma(4.0, depth / 4.0, size=n_cells)
    counts = rng.poisson(library[:, None] * profiles[cell_type]).astype(np.float32)

    signature = rng.normal(0, 1, size=(n_types, 8))
    morph = signature[cell_type] + rng.normal(0, morph_noise, size=(n_cells, 8))
    density = np.array([len(v) - 1 for v in cKDTree(xy).query_ball_point(xy, 30.0 / px_um)])
    morph = np.column_stack([morph, density]).astype(np.float32)

    spots = visium_spots(counts, xy, px_um, seed=seed)
    sample = Sample(counts=spots["counts"], genes=[f"gene{g}" for g in range(n_genes)],
                    spot_xy=spots["spot_xy"], cell_xy=xy, morph=morph, px_um=px_um,
                    spot_diameter_px=spots["spot_diameter_px"], cell_spot=spots["cell_spot"],
                    morph_names=[f"shape{j}" for j in range(8)] + ["neigh_density"],
                    name="toy")
    return sample, {"counts": counts, "cell_type": cell_type}
