"""Spatial relations between cells and spots: membership, graphs and training tiles.

All coordinates are full-resolution image pixels, ``x`` (column) first and ``y`` (row)
second.  Every length that has a physical meaning is given in micrometres and converted
with the pixel size of the section, so one setting means the same amount of tissue on
every slide.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree

__all__ = [
    "assign_cells_to_spots",
    "delaunay_edges",
    "spot_window_edges",
    "capture_pairs",
    "spatial_tiles",
]


def assign_cells_to_spots(cell_xy, spot_xy, spot_radius_px: float) -> np.ndarray:
    """Index of the spot whose capture area contains each cell centroid, ``-1`` for none.

    Spots of a Visium array do not overlap, so the nearest spot centre within one spot
    radius is the containing spot.
    """
    dist, nearest = cKDTree(np.asarray(spot_xy, float)).query(np.asarray(cell_xy, float), k=1)
    return np.where(dist <= spot_radius_px, nearest, -1).astype(np.int64)


def delaunay_edges(xy, px_um: float | None = None, max_um: float | None = None,
                   symmetric: bool = True) -> np.ndarray:
    """Edges of the Delaunay triangulation of the cell centroids, as a ``(2, E)`` array.

    A triangulation spans the convex hull of the points and therefore joins cells across
    lumens and gaps in the tissue.  ``max_um`` removes edges longer than that length.
    """
    xy = np.asarray(xy, float)
    tri = Delaunay(xy).simplices
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    if max_um is not None:
        if px_um is None:
            raise ValueError("max_um needs px_um to convert micrometres to pixels")
        length = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1) * px_um
        edges = edges[length <= max_um]
    if symmetric:
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    return np.ascontiguousarray(edges.T).astype(np.int64)


def spot_window_edges(spot_xy, k: int = 8, px_um: float | None = None,
                      window_um: float | None = None) -> np.ndarray:
    """Edges joining each spot to its ``k`` nearest spots, as a ``(2, E)`` array.

    Row 0 is the receiving spot and row 1 the neighbour.  When ``px_um`` and
    ``window_um`` are given, neighbours farther than ``window_um`` are dropped, unless
    that would leave fewer edges than spots.
    """
    spot_xy = np.asarray(spot_xy, float)
    k = int(min(k + 1, len(spot_xy)))
    dist, idx = cKDTree(spot_xy).query(spot_xy, k=k)
    dst = np.repeat(np.arange(len(spot_xy)), k - 1)
    src = idx[:, 1:].reshape(-1)  # the first neighbour is the spot itself
    if window_um is not None and px_um:
        keep = (dist[:, 1:] * float(px_um) <= window_um).reshape(-1)
        if keep.sum() >= len(spot_xy):
            dst, src = dst[keep], src[keep]
    return np.stack([dst, src]).astype(np.int64)


def capture_pairs(cell_xy, spot_xy, px_um: float, spot_radius_um: float, sigma_um: float,
                  cutoff: float = 3.0, min_weight: float = 1e-3):
    """Cell-spot pairs and capture weights for the diffusion-aware E-step.

    A cell at distance ``d`` from the centre of a spot of radius ``r`` contributes to
    that spot with weight ``exp(-0.5 * (max(d - r, 0) / sigma) ** 2)``: fully when it
    lies inside the spot and decaying outside.  The weights of a cell are not
    normalized, because molecules that diffuse outside every capture area are lost.

    Returns ``(cell_id, spot_id, weight)``.
    """
    cell_xy = np.asarray(cell_xy, float)
    spot_xy = np.asarray(spot_xy, float)
    radius = spot_radius_um / px_um
    sigma = max(sigma_um, 1e-6) / px_um
    reach = radius + cutoff * sigma
    hits = cKDTree(spot_xy).query_ball_point(cell_xy, r=reach)
    cell_id, spot_id = [], []
    for i, spots in enumerate(hits):
        if spots:
            cell_id.extend([i] * len(spots))
            spot_id.extend(spots)
    cell_id = np.asarray(cell_id, np.int64)
    spot_id = np.asarray(spot_id, np.int64)
    dist = np.linalg.norm(cell_xy[cell_id] - spot_xy[spot_id], axis=1)
    weight = np.exp(-0.5 * (np.maximum(dist - radius, 0.0) / sigma) ** 2)
    keep = weight > min_weight
    return cell_id[keep], spot_id[keep], weight[keep].astype(np.float32)


def spatial_tiles(spot_xy, cell_spot, nearest_spot, edge_index, counts, n_tiles: int = 3,
                  shuffle: bool = True, seed: int = 0, spot_neighbours: int = 8) -> list[dict]:
    """Split a section into ``n_tiles x n_tiles`` training batches of whole spots.

    The likelihood is defined per spot, so a batch has to hold whole spots together with
    all of their cells.  Tiles are cut at the quantiles of the spot coordinates.  A cell
    outside every spot goes to the tile of its nearest spot, where it contributes graph
    edges and context but no likelihood term.

    Each tile is a dictionary with the cell rows (``cells``), the tile-local spot index
    of every cell (``cell_spot``), the spot counts (``counts``), the original spot ids
    (``spot_ids``) and the tile-local graphs (``edge_index``, ``spot_edges``,
    ``nearest_spot``).
    """
    spot_xy = np.asarray(spot_xy, float)
    rng = np.random.default_rng(seed)
    q = np.quantile(spot_xy, np.linspace(0, 1, n_tiles + 1), axis=0)
    tiles = []
    for a in range(n_tiles):
        for b in range(n_tiles):
            sel = ((spot_xy[:, 0] >= q[a, 0]) & (spot_xy[:, 0] <= q[a + 1, 0])
                   & (spot_xy[:, 1] >= q[b, 1]) & (spot_xy[:, 1] <= q[b + 1, 1]))
            if sel.sum() >= 4:
                tiles.append(np.where(sel)[0])
    if shuffle:
        rng.shuffle(tiles)
    return [_tile(spot_xy, cell_spot, nearest_spot, edge_index, counts, ids, spot_neighbours)
            for ids in tiles]


def _tile(spot_xy, cell_spot, nearest_spot, edge_index, counts, spot_ids, spot_neighbours):
    """Restrict a section to a set of spots: their cells and the uncovered cells nearest them."""
    n_spots, n_cells = len(spot_xy), len(cell_spot)
    in_tile = np.zeros(n_spots, bool)
    in_tile[spot_ids] = True
    covered = cell_spot >= 0
    cells = np.where((covered & in_tile[np.clip(cell_spot, 0, None)])
                     | (~covered & in_tile[nearest_spot]))[0]
    spot_lut = -np.ones(n_spots, np.int64)
    spot_lut[spot_ids] = np.arange(len(spot_ids))
    local_spot = np.where(covered[cells], spot_lut[np.clip(cell_spot[cells], 0, None)], -1)

    cell_lut = -np.ones(n_cells, np.int64)
    cell_lut[cells] = np.arange(len(cells))
    kept = (cell_lut[edge_index[0]] >= 0) & (cell_lut[edge_index[1]] >= 0)
    return {
        "cells": cells,
        "cell_spot": local_spot,
        "counts": np.asarray(counts, np.float32)[spot_ids],
        "n_spots": len(spot_ids),
        "spot_ids": np.asarray(spot_ids),
        "nearest_spot": spot_lut[nearest_spot[cells]].clip(0),
        "edge_index": cell_lut[edge_index[:, kept]],
        "spot_edges": spot_window_edges(spot_xy[spot_ids], spot_neighbours),
    }
