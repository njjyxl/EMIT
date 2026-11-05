"""Nucleus segmentation with StarDist and the morphometrics of each nucleus.

Each nucleus defines one cell.  Thirteen morphometrics are computed: seven describe the
shape of the nucleus, five are read from the image around it and one counts the nuclei
nearby.
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .._utils import logger

__all__ = ["tissue_mask", "morphometrics", "stain_features", "neighbour_density",
           "segment_nuclei", "MORPH_NAMES"]

SHAPE_NAMES = ["area", "perimeter", "eccentricity", "solidity", "major_axis_length",
               "minor_axis_length", "circularity"]
STAIN_NAMES = ["hematoxylin", "eosin", "gray_mean", "gray_std", "texture"]
MORPH_NAMES = SHAPE_NAMES + STAIN_NAMES + ["neigh_density"]

# optical-density vectors of the two stains (Ruifrok and Johnston)
_HEMATOXYLIN = np.array([0.65, 0.70, 0.29], np.float32)
_EOSIN = np.array([0.07, 0.99, 0.11], np.float32)


def tissue_mask(spot_xy, px_um: float, height: int, width: int, reach: float = 0.75,
                grid_um: float = 8.0) -> np.ndarray:
    """Pixels within ``reach`` spot pitches of a spot, as an ``(H, W)`` boolean image.

    Segmentation is confined to this region, which removes detections off the tissue.
    The distance to the nearest spot is exact and is evaluated on a coarse grid.
    """
    spot_xy = np.asarray(spot_xy, float)
    tree = cKDTree(spot_xy)
    pitch = float(np.median(tree.query(spot_xy, k=2)[0][:, 1]))
    step = max(1, int(round(grid_um / px_um)))
    grid_h, grid_w = height // step + 1, width // step + 1
    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    dist = tree.query(np.column_stack([xx.ravel() * step, yy.ravel() * step]), k=1)[0]
    coarse = ndi.binary_fill_holes(dist.reshape(grid_h, grid_w) <= reach * pitch)
    rows = np.clip(np.arange(height) // step, 0, grid_h - 1)
    cols = np.clip(np.arange(width) // step, 0, grid_w - 1)
    return coarse[rows][:, cols]


def stain_features(image, xy, px_um: float, patch_um: float = 8.8) -> np.ndarray:
    """Stain and texture of a square window around each centroid: ``(n, 5)`` float32.

    Columns: mean hematoxylin and eosin optical density, mean and standard deviation of
    grey intensity, and the variance of the Laplacian of the grey image.
    """
    half = max(2, int(round(patch_um / px_um)) // 2 * 2) // 2
    hem = _HEMATOXYLIN / np.linalg.norm(_HEMATOXYLIN)
    eos = _EOSIN / np.linalg.norm(_EOSIN)
    out = np.full((len(xy), len(STAIN_NAMES)), np.nan, np.float32)
    for i, (x, y) in enumerate(np.asarray(xy).astype(int)):
        patch = image[max(0, y - half):y + half, max(0, x - half):x + half]
        if patch.size < 48:
            continue
        density = -np.log10(np.clip(patch.astype(np.float32) / 255.0, 1e-3, 1.0))
        gray = patch.mean(2)
        if min(gray.shape) > 2:
            laplacian = (gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2]
                         - 4 * gray[1:-1, 1:-1])
        else:
            laplacian = np.zeros(1)
        out[i] = ((density @ hem).mean(), (density @ eos).mean(), gray.mean(), gray.std(),
                  laplacian.var())
    return out


def neighbour_density(xy, px_um: float, radius_um: float = 30.0) -> np.ndarray:
    """Number of other nuclei within ``radius_um`` of each nucleus."""
    xy = np.asarray(xy, float)
    return np.array([len(v) - 1 for v in cKDTree(xy).query_ball_point(xy, radius_um / px_um)],
                    np.float32)


def morphometrics(labels, image, px_um: float, patch_um: float = 8.8,
                  neighbour_um: float = 30.0):
    """Centroids and morphometrics of the nuclei of a label image.

    Parameters
    ----------
    labels : (H, W) int array
        0 is background; every nucleus has its own positive value.
    image : (H, W, 3) uint8 array
        The H&E image the labels refer to.
    px_um
        Micrometres per pixel.
    patch_um
        Side of the window around each centroid from which stain and texture are read.
    neighbour_um
        Radius within which neighbouring nuclei are counted.

    Returns
    -------
    xy : (n, 2) array
        Centroids, ``x`` first.
    morph : (n, 13) float32 array
        Columns as in ``MORPH_NAMES``.  Circularity is NaN for a nucleus without a
        perimeter and the image features are NaN for a nucleus at the very edge.
    label_ids : (n,) int array
        Value of each nucleus in ``labels``.
    """
    from skimage.measure import regionprops

    props = regionprops(labels)
    if not props:
        return np.zeros((0, 2)), np.zeros((0, len(MORPH_NAMES)), np.float32), np.zeros(0, np.int64)
    xy = np.array([[p.centroid[1], p.centroid[0]] for p in props])
    shape = np.array([[p.area, p.perimeter, p.eccentricity, p.solidity, p.axis_major_length,
                       p.axis_minor_length,
                       min(4 * np.pi * p.area / p.perimeter ** 2, 1.0) if p.perimeter > 0
                       else np.nan] for p in props], np.float32)

    stain = stain_features(image, xy, px_um, patch_um)
    neighbours = neighbour_density(xy, px_um, neighbour_um)
    morph = np.hstack([shape, stain, neighbours[:, None]]).astype(np.float32)
    return xy, morph, np.array([p.label for p in props], np.int64)


def segment_nuclei(image, px_um: float, spot_xy=None, model: str = "2D_versatile_he",
                   prob_thresh: float = 0.05, nms_thresh: float = 0.5, block_size: int = 4096,
                   overlap: int = 128, n_tiles=(4, 4, 1), mask_reach: float = 0.75,
                   return_labels: bool = False) -> dict:
    """Segment the nuclei of an H&E image with StarDist and describe each of them.

    Parameters
    ----------
    image : (H, W, 3) uint8 array
    px_um
        Micrometres per pixel.
    spot_xy : (n_spots, 2) array, optional
        Spot centres.  When given, segmentation is confined to the region covered by the
        spot lattice (:func:`tissue_mask`).
    model
        Name of a pretrained StarDist model, or the directory of a trained one.
    prob_thresh, nms_thresh
        StarDist probability and non-maximum-suppression thresholds.
    block_size, overlap, n_tiles
        Tiling of the prediction over large images.
    return_labels
        Also return the label image (needed to assign Visium HD bins to nuclei).

    Returns
    -------
    dict with ``xy``, ``morph``, ``morph_names``, ``px_um``, ``hw`` and, if requested,
    ``labels`` in which nucleus ``k`` has value ``k + 1``.

    Notes
    -----
    Pixel values are normalized with percentiles computed once over the tissue (1% and
    99.8% per channel), not per block: a block that contains white background would
    otherwise be normalized differently from one that does not.
    """
    try:
        from csbdeep.data import Normalizer
        from stardist.models import StarDist2D
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('nucleus segmentation needs StarDist: '
                          'pip install "emit-spatial[segment]"') from exc

    height, width = image.shape[:2]
    mask = (tissue_mask(spot_xy, px_um, height, width, mask_reach) if spot_xy is not None
            else np.ones((height, width), bool))
    pixels = image[mask][::37].astype(np.float32)
    low = np.percentile(pixels, 1, axis=0).astype(np.float32)
    high = np.percentile(pixels, 99.8, axis=0).astype(np.float32)

    class GlobalNormalizer(Normalizer):
        def before(self, x, axes):
            return (x.astype(np.float32) - low) / (high - low + 1e-8)

        def after(self, mean, scale, axes):
            raise NotImplementedError

        @property
        def do_after(self):
            return False

    if os.path.isdir(model):
        net = StarDist2D(None, name=os.path.basename(os.path.normpath(model)),
                         basedir=os.path.dirname(os.path.normpath(model)))
    else:
        net = StarDist2D.from_pretrained(model)

    work = image.copy()
    work[~mask] = 255  # the background of an H&E image is white
    t0 = time.time()
    if max(height, width) > block_size:
        labels, _ = net.predict_instances_big(
            work, axes="YXC", block_size=block_size, min_overlap=overlap, context=overlap,
            normalizer=GlobalNormalizer(), n_tiles=n_tiles, prob_thresh=prob_thresh,
            nms_thresh=nms_thresh, labels_out_dtype=np.int32)
    else:
        labels, _ = net.predict_instances(
            work, axes="YXC", normalizer=GlobalNormalizer(), prob_thresh=prob_thresh,
            nms_thresh=nms_thresh)
    labels = np.asarray(labels, np.int32)
    labels[~mask] = 0
    del work

    xy, morph, ids = morphometrics(labels, image, px_um)
    logger.info("%s nuclei in %.1f min | %.3f um/px | tissue mask %.1f%% of the image",
                f"{len(xy):,}", (time.time() - t0) / 60, px_um, 100 * mask.mean())
    out = {"xy": xy, "morph": morph, "morph_names": np.array(MORPH_NAMES, dtype=object),
           "px_um": float(px_um), "hw": np.array([height, width])}
    if return_labels:
        relabel = np.zeros(int(labels.max()) + 1, np.int32)
        relabel[ids] = np.arange(1, len(ids) + 1)
        out["labels"] = relabel[labels]
    return out
