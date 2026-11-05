"""Readers for Space Ranger output and for the files written by the ``emit`` commands.

A standard Visium directory is expected to hold::

    [<prefix>_]filtered_feature_bc_matrix.h5
    spatial/tissue_positions.csv | tissue_positions_list.csv | tissue_positions.parquet
    spatial/scalefactors_json.json

Two properties of this format are handled here because they are easy to get wrong:

* the position columns are (row, col), that is (y, x); this package uses (x, y);
* ``spot_diameter_fullres`` is the *fitted* spot extent, which need not equal the nominal
  55 um.  The pixel size is therefore taken from the spot pitch, which is 100 um by
  design, and the fitted diameter is used only to decide which cells a spot contains.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ._utils import as_str_array, logger
from .sample import Sample

__all__ = ["read_visium", "read_visium_hd", "read_nuclei", "read_image_features", "build_sample"]

VISIUM_SPOT_UM = 55.0
VISIUM_PITCH_UM = 100.0


def _find(root: str, *patterns: str, required: bool = True) -> str | None:
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(f"none of {patterns} under {root}")
    return None


def _read_matrix(path: str):
    """Counts as a ``spots x genes`` CSR matrix, with barcodes, gene names and gene ids."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(v) for v in g["shape"][:])  # stored as genes x barcodes
        matrix = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        barcodes = as_str_array(g["barcodes"][:])
        names = as_str_array(g["features"]["name"][:])
        ids = as_str_array(g["features"]["id"][:]) if "id" in g["features"] else names.copy()
    return matrix.T.tocsr(), barcodes, names, ids


def _read_positions(root: str):
    """Barcodes and full-resolution (x, y) pixel coordinates of the in-tissue spots or bins."""
    path = _find(root, "spatial/tissue_positions.parquet", "spatial/tissue_positions.csv",
                 "spatial/tissue_positions_list.csv", "tissue_positions.parquet",
                 "tissue_positions.csv", "tissue_positions_list.csv")
    if path.endswith(".parquet"):
        table = pd.read_parquet(path)
        table.columns = [c.strip().lower() for c in table.columns]
    else:
        first = pd.read_csv(path, nrows=1, header=None)
        if str(first.iloc[0, 0]).lower() in ("barcode", "barcodes"):  # Space Ranger >= 2.0
            table = pd.read_csv(path)
            table.columns = [c.strip().lower() for c in table.columns]
        else:
            table = pd.read_csv(path, header=None, names=[
                "barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres"])
    table = table[table["in_tissue"] == 1]
    xy = np.column_stack([table["pxl_col_in_fullres"].to_numpy(float),
                          table["pxl_row_in_fullres"].to_numpy(float)])
    return as_str_array(table["barcode"]), xy


def _scalefactors(root: str) -> dict:
    with open(_find(root, "spatial/scalefactors_json.json", "scalefactors_json.json")) as f:
        return json.load(f)


def read_visium(path: str, n_top_genes: int | None = 2000, min_counts: int = 100,
                spot_um: float = VISIUM_SPOT_UM, pitch_um: float = VISIUM_PITCH_UM) -> dict:
    """Read the spots of a Space Ranger (standard Visium) output directory.

    Parameters
    ----------
    path
        The Space Ranger ``outs`` directory.
    n_top_genes
        Keep this many genes with the highest variance of log-normalized expression;
        ``None`` keeps all genes.
    min_counts
        Spots with fewer counts are removed.
    spot_um, pitch_um
        Nominal spot diameter and centre-to-centre distance of the array.

    Returns
    -------
    dict with ``counts`` (spots x genes, float32), ``genes``, ``gene_ids``, ``spot_xy``,
    ``spot_names``, ``px_um``, ``spot_diameter_px``, ``pitch_px`` and ``image`` (path of the
    full-resolution image if one is found next to the matrix).
    """
    matrix, barcodes, names, ids = _read_matrix(
        _find(path, "*filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"))
    spot_names, xy = _read_positions(path)
    order = pd.Index(barcodes).get_indexer(spot_names)
    found = order >= 0
    spot_names, xy, order = spot_names[found], xy[found], order[found]

    # The matrix stays sparse until the genes are chosen.
    matrix = matrix[order]
    deep = np.asarray(matrix.sum(1)).ravel() >= min_counts
    matrix, spot_names, xy = matrix[deep], spot_names[deep], xy[deep]

    scale = 1e4 / np.clip(np.asarray(matrix.sum(1)).ravel(), 1e-9, None)
    lognorm = matrix.multiply(scale[:, None]).log1p().tocsc()
    mean = np.asarray(lognorm.sum(0)).ravel() / lognorm.shape[0]
    var = np.asarray(lognorm.multiply(lognorm).sum(0)).ravel() / lognorm.shape[0] - mean ** 2

    spot_diameter_px = float(_scalefactors(path)["spot_diameter_fullres"])
    pitch_px = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    px_um = pitch_um / pitch_px
    nominal_px = spot_um / px_um
    if not 0.9 < spot_diameter_px / nominal_px < 1.1:
        logger.info("  fitted spot diameter %.1f px = %.0f um against a nominal %.0f um; the pixel "
                    "size is taken from the %.0f um pitch and spot membership from the fitted "
                    "diameter", spot_diameter_px, spot_diameter_px * px_um, spot_um, pitch_um)

    if n_top_genes is None or n_top_genes >= matrix.shape[1]:
        keep = np.arange(matrix.shape[1])
    else:
        keep = np.sort(np.argsort(-var)[:n_top_genes])
    counts = np.asarray(matrix[:, keep].todense(), np.float32)
    logger.info("%s: %s spots x %s of %s genes | %.4f um/px | pitch %.1f px",
                os.path.basename(os.path.normpath(path)), f"{len(xy):,}", f"{len(keep):,}",
                f"{len(names):,}", px_um, pitch_px)
    return {
        "counts": counts, "genes": names[keep], "gene_ids": ids[keep], "spot_xy": xy,
        "spot_names": spot_names, "px_um": px_um, "spot_diameter_px": spot_diameter_px,
        "pitch_px": pitch_px,
        "image": _find(path, "*image.tif", "*tissue_image.tif", "*.btf", "../../*.btf",
                       "../../*image.tif", required=False),
    }


def read_visium_hd(path: str) -> dict:
    """Read one bin size of a Visium HD run, e.g. ``binned_outputs/square_002um``.

    Returns a dict with ``counts`` (bins x genes, sparse CSR), ``genes``, ``gene_ids``,
    ``bin_xy`` (full-resolution pixels), ``bin_names``, ``px_um`` and ``bin_um``.
    """
    matrix, barcodes, names, ids = _read_matrix(
        _find(path, "filtered_feature_bc_matrix.h5", "*filtered_feature_bc_matrix.h5"))
    bin_names, xy = _read_positions(path)
    order = pd.Index(bin_names).get_indexer(barcodes)
    if (order < 0).any():
        raise ValueError("some barcodes of the count matrix have no tissue position")
    factors = _scalefactors(path)
    bin_um = float(factors.get("bin_size_um", 2.0))
    px_um = float(factors["microns_per_pixel"]) if "microns_per_pixel" in factors else (
        bin_um / float(factors["spot_diameter_fullres"]))
    logger.info("%s: %s bins x %s genes | %.4f um/px | %g um bins",
                os.path.basename(os.path.normpath(path)), f"{matrix.shape[0]:,}",
                f"{matrix.shape[1]:,}", px_um, bin_um)
    return {"counts": matrix.astype(np.float32), "genes": names, "gene_ids": ids,
            "bin_xy": xy[order], "bin_names": barcodes, "px_um": px_um, "bin_um": bin_um}


def read_nuclei(path: str) -> dict:
    """Read the nuclei written by ``emit segment``.

    The ``.npz`` file holds ``xy`` (n, 2), ``morph`` (n, n_morph), ``morph_names`` and
    ``px_um``.  Any segmentation can be used as long as it is stored this way.
    """
    with np.load(path, allow_pickle=True) as z:
        out = {"xy": np.asarray(z["xy"], float), "morph": np.asarray(z["morph"], np.float32),
               "morph_names": as_str_array(z["morph_names"]) if "morph_names" in z.files else None,
               "px_um": float(z["px_um"]) if "px_um" in z.files else None}
        if "cell_names" in z.files:
            out["cell_names"] = as_str_array(z["cell_names"])
    return out


def read_image_features(path: str, n_cells: int | None = None) -> np.ndarray:
    """Read the per-cell image embeddings written by ``emit features`` as float32."""
    with np.load(path, allow_pickle=True) as z:
        feats = z["feats"].astype(np.float32)
        if "idx" in z.files:  # rows may have been written in another order
            order = np.asarray(z["idx"], np.int64)
            if not np.array_equal(order, np.arange(len(order))):
                reordered = np.zeros_like(feats)
                reordered[order] = feats
                feats = reordered
    if n_cells is not None and len(feats) != n_cells:
        raise ValueError(f"{path} holds features of {len(feats):,} cells, expected {n_cells:,}")
    return feats


def build_sample(spots: dict, nuclei: dict, image_features: np.ndarray | None = None,
                 name: str = "sample") -> Sample:
    """Combine spots (:func:`read_visium`) and nuclei (:func:`read_nuclei`) into a sample."""
    if nuclei.get("px_um") is not None and abs(nuclei["px_um"] - spots["px_um"]) > 1e-3:
        raise ValueError(f"the segmentation is at {nuclei['px_um']:.4f} um/px and the spots at "
                         f"{spots['px_um']:.4f} um/px: they do not refer to the same image")
    sample = Sample(
        counts=spots["counts"], genes=spots["genes"], spot_xy=spots["spot_xy"],
        cell_xy=nuclei["xy"], morph=nuclei["morph"], px_um=spots["px_um"],
        spot_diameter_px=spots["spot_diameter_px"], image_features=image_features,
        morph_names=nuclei.get("morph_names"), spot_names=spots.get("spot_names"),
        cell_names=nuclei.get("cell_names"), name=name,
        meta={"gene_ids": spots.get("gene_ids")})
    stats = sample.check_registration()
    logger.info("  %s cells | %.1f%% inside a spot | %.1f per spot", f"{sample.n_cells:,}",
                100 * stats["fraction_of_cells_inside_spots"], stats["median_cells_per_spot"])
    return sample
