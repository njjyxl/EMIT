"""Command line interface: ``emit <command> --help``.

    emit segment   H&E image -> nuclei and their morphometrics
    emit features  H&E image + nuclei -> image embedding of every cell
    emit fit       spots + nuclei (+ embeddings) -> expression of every cell
    emit hd-cells  Visium HD bins + nucleus labels -> single cells
    emit toy       write a small synthetic sample for trying the commands
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __version__
from ._utils import logger, setup_logging


# ---------------------------------------------------------------------- commands
def _cmd_segment(a) -> int:
    from .image import read_image
    from .image.segment import segment_nuclei
    from .io import _read_positions, _scalefactors

    image = read_image(a.image)
    spot_xy, px_um = None, a.px_um
    if a.spaceranger:
        from scipy.spatial import cKDTree

        _, spot_xy = _read_positions(a.spaceranger)
        if a.hd:
            px_um = px_um or float(_scalefactors(a.spaceranger)["microns_per_pixel"])
        else:
            pitch_px = float(np.median(cKDTree(spot_xy).query(spot_xy, k=2)[0][:, 1]))
            px_um = px_um or a.pitch_um / pitch_px
    if px_um is None:
        raise SystemExit("give --px-um or --spaceranger so that the pixel size is known")
    logger.info("%s: %d x %d px at %.4f um/px", os.path.basename(a.image), image.shape[1],
                image.shape[0], px_um)
    out = segment_nuclei(image, px_um, spot_xy=None if a.no_mask else spot_xy, model=a.model,
                         prob_thresh=a.prob_thresh, nms_thresh=a.nms_thresh,
                         return_labels=a.labels is not None)
    labels = out.pop("labels", None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez(a.out, **out)
    logger.info("nuclei -> %s", a.out)
    if labels is not None:
        np.save(a.labels, labels)
        logger.info("label image -> %s", a.labels)
    return 0


def _cmd_features(a) -> int:
    from .image import read_image
    from .image.features import embed_cells
    from .io import read_nuclei

    nuclei = read_nuclei(a.nuclei)
    px_um = a.px_um or nuclei["px_um"]
    if px_um is None:
        raise SystemExit("the nuclei file holds no pixel size: give --px-um")
    embed_cells(read_image(a.image), nuclei["xy"], px_um, patch_um=a.patch_um, model=a.model,
                batch_size=a.batch_size, cache=a.out, device=a.device,
                half_precision=a.half_precision)
    return 0


def _load_sample(a):
    from .io import build_sample, read_image_features, read_nuclei, read_visium
    from .sample import Sample

    if a.sample:
        sample = Sample.load(a.sample)
    elif a.spaceranger and a.nuclei:
        spots = read_visium(a.spaceranger, n_top_genes=a.n_top_genes or None,
                            min_counts=a.min_counts, spot_um=a.spot_um, pitch_um=a.pitch_um)
        name = a.name or os.path.basename(os.path.normpath(a.spaceranger))
        sample = build_sample(spots, read_nuclei(a.nuclei), name=name)
    else:
        raise SystemExit("give --sample, or --spaceranger together with --nuclei")
    if a.features:
        sample.image_features = read_image_features(a.features, sample.n_cells)
    return sample


def _cmd_fit(a) -> int:
    from .fitting import fit

    sample = _load_sample(a)
    result = fit(sample, n_types=a.n_types, n_states=a.n_states, dim=a.dim, rounds=a.rounds,
                 steps=a.steps, lr=a.lr, tiles=a.tiles, holdout=a.holdout, w_drift=a.w_drift,
                 diffusion_um=a.diffusion_um, anchors=a.anchors, anchor_weight=a.anchor_weight,
                 predict_between=not a.no_between, monitor=not a.no_monitor, seed=a.seed,
                 device=a.device, eval_chunk=a.eval_chunk, eval_placement=a.eval_placement)
    result.save(a.out, save_model=not a.no_model)
    logger.info("%s cells (%s allocated, %s predicted) -> %s", f"{len(result.allocated):,}",
                f"{int(result.allocated.sum()):,}",
                f"{0 if a.no_between else int((~result.allocated).sum()):,}",
                os.path.join(a.out, "cells.h5ad"))
    return 0


def _cmd_hd_cells(a) -> int:
    import anndata
    import pandas as pd

    from .hd import bins_to_cells, nuclei_on_bins
    from .io import read_visium_hd

    bins = read_visium_hd(a.bins)
    labels = np.load(a.labels, mmap_mode="r")
    centroids = None
    if a.nuclei:
        from .io import read_nuclei

        centroids = read_nuclei(a.nuclei)["xy"]
    bin_nucleus = nuclei_on_bins(labels, bins["bin_xy"], centroids)
    n_cells = int(labels.max()) if centroids is None else len(centroids)
    cells, owner = bins_to_cells(bins["counts"], bins["bin_xy"] * bins["px_um"], bin_nucleus,
                                 n_cells=n_cells, d_max_um=a.d_max_um)
    obs = pd.DataFrame({"n_bins": np.bincount(owner[owner >= 0], minlength=n_cells),
                        "n_counts": np.asarray(cells.sum(1)).ravel()},
                       index=pd.Index([f"cell{i}" for i in range(n_cells)], name="cell"))
    adata = anndata.AnnData(X=cells, obs=obs,
                            var=pd.DataFrame({"gene_ids": bins["gene_ids"]},
                                             index=pd.Index([str(g) for g in bins["genes"]])))
    adata.var_names_make_unique()
    if centroids is not None:
        adata.obsm["spatial"] = np.asarray(centroids, float)
    adata.uns["emit"] = {"d_max_um": a.d_max_um, "bin_um": bins["bin_um"], "px_um": bins["px_um"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    adata.write_h5ad(a.out)
    logger.info("cells -> %s", a.out)
    if a.owner:
        np.save(a.owner, owner)
    return 0


def _cmd_toy(a) -> int:
    from .simulate import toy_section

    sample, truth = toy_section(n_cells=a.n_cells, n_genes=a.n_genes, seed=a.seed)
    sample.save(a.out)
    logger.info("%s -> %s", sample.summary(), a.out)
    if a.truth:
        np.savez_compressed(a.truth, **truth)
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emit", description="EMIT: single-cell expression from spot-based spatial "
                                 "transcriptomics and histology.")
    parser.add_argument("--version", action="version", version=f"emit {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="print warnings only")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("segment", help="segment nuclei in the H&E image (needs StarDist)")
    p.add_argument("--image", required=True, help="full-resolution H&E image")
    p.add_argument("--out", required=True, help="output .npz with xy, morph, morph_names, px_um")
    p.add_argument("--spaceranger", help="Space Ranger directory: gives the pixel size and "
                                         "confines segmentation to the spot lattice")
    p.add_argument("--hd", action="store_true", help="the directory is a Visium HD bin directory")
    p.add_argument("--px-um", type=float, help="micrometres per pixel (overrides the estimate)")
    p.add_argument("--pitch-um", type=float, default=100.0, help="spot pitch of the array")
    p.add_argument("--no-mask", action="store_true", help="segment the whole image")
    p.add_argument("--labels", help="also write the label image to this .npy (for hd-cells)")
    p.add_argument("--model", default="2D_versatile_he", help="StarDist model name or directory")
    p.add_argument("--prob-thresh", type=float, default=0.05)
    p.add_argument("--nms-thresh", type=float, default=0.5)
    p.set_defaults(func=_cmd_segment)

    p = sub.add_parser("features", help="embed a patch around every cell (needs transformers)")
    p.add_argument("--image", required=True)
    p.add_argument("--nuclei", required=True, help=".npz written by `emit segment`")
    p.add_argument("--out", required=True, help="output .npz with feats")
    p.add_argument("--model", default="owkin/phikon", help="Hugging Face name or local directory")
    p.add_argument("--patch-um", type=float, default=26.0)
    p.add_argument("--px-um", type=float)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--half-precision", action="store_true", help="store the features as float16")
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("fit", help="infer the expression of every cell of a section")
    g = p.add_argument_group("input")
    g.add_argument("--spaceranger", help="Space Ranger output directory of a Visium section")
    g.add_argument("--nuclei", help=".npz written by `emit segment`")
    g.add_argument("--features", help=".npz written by `emit features` (optional)")
    g.add_argument("--sample", help="a sample saved with Sample.save(), instead of the above")
    g.add_argument("--name", help="name of the section")
    g.add_argument("--n-top-genes", type=int, default=2000,
                   help="most variable genes to keep; 0 keeps all")
    g.add_argument("--min-counts", type=int, default=100, help="drop spots with fewer counts")
    g.add_argument("--spot-um", type=float, default=55.0)
    g.add_argument("--pitch-um", type=float, default=100.0)
    p.add_argument("--out", required=True, help="output directory")
    g = p.add_argument_group("model")
    g.add_argument("--n-types", type=int, default=10, help="latent types K")
    g.add_argument("--n-states", type=int, default=20, help="state dimensions m")
    g.add_argument("--dim", type=int, default=256, help="width of the cell tokens")
    g = p.add_argument_group("training")
    g.add_argument("--rounds", type=int, default=20, help="EM rounds")
    g.add_argument("--steps", type=int, default=3, help="passes over the tiles per round")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--tiles", type=int, default=3)
    g.add_argument("--holdout", type=float, default=0.15, help="fraction of spots held out")
    g.add_argument("--w-drift", type=float, default=0.1, help="weight of the drift term")
    g.add_argument("--diffusion-um", type=float, default=None,
                   help="diffusion width for the diffusion-aware E-step")
    g.add_argument("--anchors", help=".npz with rows and counts of cells with per-cell counts")
    g.add_argument("--anchor-weight", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("run")
    g.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    g.add_argument("--no-between", action="store_true",
                   help="do not predict the cells outside every spot")
    g.add_argument("--no-monitor", action="store_true", help="evaluate only the final round")
    g.add_argument("--no-model", action="store_true", help="do not write model.pt")
    g.add_argument("--eval-chunk", type=int, default=40_000)
    g.add_argument("--eval-placement", default="auto", choices=["auto", "0", "1", "2"])
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("hd-cells", help="assign Visium HD bins to cells")
    p.add_argument("--bins", required=True, help="bin directory, e.g. binned_outputs/square_002um")
    p.add_argument("--labels", required=True,
                   help="label image (.npy) written by `emit segment --labels`")
    p.add_argument("--nuclei", help=".npz written by `emit segment` (centroids of the nuclei)")
    p.add_argument("--out", required=True, help="output .h5ad (cells x genes)")
    p.add_argument("--owner", help="also write the owner of every bin to this .npy")
    p.add_argument("--d-max-um", type=float, default=4.0,
                   help="largest distance from a nucleus at which a bin is assigned")
    p.set_defaults(func=_cmd_hd_cells)

    p = sub.add_parser("toy", help="write a small synthetic sample")
    p.add_argument("--out", required=True, help="output .npz, readable by `emit fit --sample`")
    p.add_argument("--truth", help="also write the per-cell truth to this .npz")
    p.add_argument("--n-cells", type=int, default=6000)
    p.add_argument("--n-genes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_toy)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbose=not args.quiet)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
