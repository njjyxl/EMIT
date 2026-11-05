"""Single cells from subcellular bins (Visium HD).

Every bin is assigned to at most one cell.  A bin whose centre lies inside a segmented
nucleus belongs to that cell.  Any other bin goes to the nearest nucleus, the distance
being measured to the nearest bin of that nucleus, unless that distance exceeds
``d_max_um``, in which case the bin is left unassigned.  The counts of a cell are the
sum over the bins it owns, so every transcript is assigned to exactly one cell or to
none.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ._utils import logger

__all__ = ["nuclei_on_bins", "assign_bins", "aggregate_bins", "bins_to_cells"]


def nuclei_on_bins(labels, bin_xy_px, centroids_px=None) -> np.ndarray:
    """Nucleus under the centre of each bin, ``-1`` where there is none.

    Parameters
    ----------
    labels : (H, W) int array
        Label image of the segmentation: 0 is background, nucleus ``k`` has value
        ``k + 1``.
    bin_xy_px : (n_bins, 2) array
        Bin centres in pixels of the same image, ``x`` first.
    centroids_px : (n_nuclei, 2) array, optional
        Nucleus centroids.  A nucleus smaller than a bin may cover no bin centre; when
        centroids are given, such a nucleus takes the free bin nearest to its centroid.
    """
    labels = np.asarray(labels)
    bin_xy_px = np.asarray(bin_xy_px, float)
    col = np.rint(bin_xy_px[:, 0]).astype(np.int64)
    row = np.rint(bin_xy_px[:, 1]).astype(np.int64)
    ok = (row >= 0) & (row < labels.shape[0]) & (col >= 0) & (col < labels.shape[1])
    nucleus = np.full(len(bin_xy_px), -1, np.int64)
    nucleus[ok] = labels[row[ok], col[ok]].astype(np.int64) - 1
    if centroids_px is not None and len(bin_xy_px) > 1:
        centroids_px = np.asarray(centroids_px, float)
        missing = np.setdiff1d(np.arange(len(centroids_px)), nucleus[nucleus >= 0])
        if len(missing):
            tree = cKDTree(bin_xy_px)
            pitch = float(np.median(tree.query(bin_xy_px[:2000], k=2)[0][:, 1]))
            dist, nearest = tree.query(centroids_px[missing], k=1)
            for k, d, b in zip(missing, dist, nearest):
                if d <= pitch and nucleus[b] < 0:
                    nucleus[b] = k
    return nucleus


def assign_bins(bin_xy_um, bin_nucleus, d_max_um: float = 4.0, ref_xy_um=None,
                ref_nucleus=None, k_ties: int = 16, chunk: int = 500_000) -> np.ndarray:
    """Owner cell of each bin, ``-1`` for unassigned bins.

    Parameters
    ----------
    bin_xy_um : (n_bins, 2) array
        Centres of the bins to assign, in micrometres.
    bin_nucleus : (n_bins,) int array
        Nucleus under each bin, ``-1`` for none (see :func:`nuclei_on_bins`).
    d_max_um
        A bin outside every nucleus is assigned only if its nearest nucleus lies within
        this distance.
    ref_xy_um, ref_nucleus
        Positions and labels of *all* nucleus bins when ``bin_xy_um`` holds only part of
        the grid, for example only the bins with counts.  Default: the nucleus bins among
        ``bin_xy_um``.
    k_ties
        Number of nearest nucleus bins examined per bin to resolve ties.
    chunk
        Bins per query, which bounds memory.

    Notes
    -----
    When two nuclei are equally near, the bin goes to the nucleus with the smaller label,
    so the result does not depend on the order of the bins.
    """
    bin_xy_um = np.asarray(bin_xy_um, float)
    bin_nucleus = np.asarray(bin_nucleus, np.int64)
    if ref_xy_um is None:
        in_nucleus = bin_nucleus >= 0
        ref_xy_um, ref_nucleus = bin_xy_um[in_nucleus], bin_nucleus[in_nucleus]
    ref_xy_um = np.asarray(ref_xy_um, float)
    ref_nucleus = np.asarray(ref_nucleus, np.int64)

    owner = bin_nucleus.copy()
    outside = np.where(bin_nucleus < 0)[0]
    if not len(outside) or not len(ref_xy_um):
        return owner
    tree = cKDTree(ref_xy_um)
    k = min(k_ties, len(ref_xy_um))
    big = np.iinfo(np.int64).max
    for start in range(0, len(outside), chunk):
        rows = outside[start:start + chunk]
        # the upper bound of the query is exclusive; a bin exactly d_max away is assigned
        dist, near = tree.query(bin_xy_um[rows], k=k, distance_upper_bound=d_max_um + 1e-6,
                                workers=-1)
        if k == 1:
            dist, near = dist[:, None], near[:, None]
        found = np.isfinite(dist) & (dist <= d_max_um)
        label = np.where(found, ref_nucleus[np.where(found, near, 0)], big)
        # the nearest nucleus; among equally near nuclei, the smaller label
        tied = found & (dist <= dist[:, :1] + 1e-9)
        choice = np.where(tied, label, big).min(1)
        hit = found[:, 0]
        owner[rows[hit]] = choice[hit]
    return owner


def aggregate_bins(counts, owner, n_cells: int):
    """Sum the counts of the bins of each cell: ``(n_bins, G)`` -> ``(n_cells, G)`` CSR."""
    owner = np.asarray(owner, np.int64)
    counts = sp.csr_matrix(counts)
    assigned = np.where(owner >= 0)[0]
    membership = sp.csr_matrix((np.ones(len(assigned), np.float32), (owner[assigned], assigned)),
                               shape=(n_cells, counts.shape[0]))
    return (membership @ counts).tocsr()


def bins_to_cells(counts, bin_xy_um, bin_nucleus, n_cells: int | None = None,
                  d_max_um: float = 4.0, **kwargs):
    """Assign bins to cells and sum their counts.

    Returns ``(cell_counts, owner)``: a ``(n_cells, G)`` CSR matrix and the owner of
    every bin.  Further keyword arguments are passed to :func:`assign_bins`.
    """
    owner = assign_bins(bin_xy_um, bin_nucleus, d_max_um=d_max_um, **kwargs)
    if n_cells is None:
        n_cells = int(max(owner.max(initial=-1), np.max(kwargs.get("ref_nucleus", [-1]),
                                                        initial=-1))) + 1
    cells = aggregate_bins(counts, owner, n_cells)
    total = counts.sum()
    logger.info("%s of %s bins assigned (%s in a nucleus) | %.1f%% of counts assigned | %s cells",
                f"{int((owner >= 0).sum()):,}", f"{len(owner):,}",
                f"{int((np.asarray(bin_nucleus) >= 0).sum()):,}",
                100 * cells.sum() / max(total, 1), f"{n_cells:,}")
    return cells, owner
