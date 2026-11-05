"""Fitting EMIT to one section.

The model is fitted per section and needs no single-cell reference, cell-type labels or
per-cell training data:

1. each cell is encoded from its image features and morphometrics, and exchanges
   information with neighbouring cells and, through its spot, with neighbouring spots;
2. the network predicts an expression profile and a library size for every cell;
3. **E-step** - every measured spot count is apportioned among the cells of that spot in
   proportion to their expected counts;
4. **M-step** - the network is trained on the allocated counts, cell by cell;
5. after the last round, the allocation is the expression of the cells inside spots, and
   the expected counts of the network are the prediction for the cells between spots.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._utils import logger, resolve_device
from .em import e_step, e_step_diffuse, m_loss
from .geometry import capture_pairs, delaunay_edges, spatial_tiles, spot_window_edges
from .metrics import fit_diagnostics, heldout_baselines, heldout_nll, heldout_relative_distance
from .sample import Sample

__all__ = ["fit", "EMITResult", "initial_type_profiles"]


@dataclass
class EMITResult:
    """Per-cell output of :func:`fit`.

    Attributes
    ----------
    expression : (n_cells, n_genes) float32
        Counts of each cell.  For cells with ``allocated`` set these are the allocated
        spot counts; for the other cells they are predictions of the network (zero when
        ``predict_between`` was off).
    allocated : (n_cells,) bool
        Cells that received a share of the measured counts.
    type_prob : (n_cells, K) float32
        Mixture of each cell over the ``K`` latent types.
    library_size : (n_cells,) float32
        Library size ``lam`` predicted by the network.
    type_profiles : (K, n_genes) float32
        Expression profile of each latent type.
    history : DataFrame
        One row per monitored round: loss terms, held-out reconstruction and diagnostics.
    """

    expression: np.ndarray
    allocated: np.ndarray
    type_prob: np.ndarray
    library_size: np.ndarray
    type_profiles: np.ndarray
    history: pd.DataFrame
    genes: np.ndarray
    cell_names: np.ndarray
    cell_xy: np.ndarray
    cell_spot: np.ndarray
    spot_names: np.ndarray
    px_um: float
    params: dict = field(default_factory=dict)
    state_dict: dict | None = None

    @property
    def latent_type(self) -> np.ndarray:
        """Most probable latent type of each cell."""
        return self.type_prob.argmax(1)

    def to_anndata(self):
        """The cells as an ``AnnData`` object (cells x genes)."""
        import anndata

        obs = pd.DataFrame({
            "x": self.cell_xy[:, 0], "y": self.cell_xy[:, 1],
            "spot": np.where(self.cell_spot >= 0,
                             self.spot_names[np.clip(self.cell_spot, 0, None)], ""),
            "in_spot": self.cell_spot >= 0,
            "source": np.where(self.allocated, "allocated", "predicted"),
            "latent_type": pd.Categorical([f"T{k + 1}" for k in self.latent_type]),
            "library_size": self.library_size,
        }, index=pd.Index([str(c) for c in self.cell_names], name="cell"))
        adata = anndata.AnnData(X=self.expression, obs=obs,
                                var=pd.DataFrame(index=pd.Index([str(g) for g in self.genes])))
        adata.var_names_make_unique()
        adata.obsm["spatial"] = np.asarray(self.cell_xy, float)
        adata.obsm["type_prob"] = self.type_prob
        adata.varm["type_profiles"] = self.type_profiles.T
        adata.uns["emit"] = {"params": json.dumps(self.params), "px_um": self.px_um}
        return adata

    def save(self, directory: str, save_model: bool = True) -> str:
        """Write ``cells.h5ad``, ``history.csv`` and, optionally, ``model.pt``."""
        os.makedirs(directory, exist_ok=True)
        self.to_anndata().write_h5ad(os.path.join(directory, "cells.h5ad"))
        self.history.to_csv(os.path.join(directory, "history.csv"), index=False)
        if save_model and self.state_dict is not None:
            import torch

            torch.save({"state_dict": self.state_dict, "params": self.params},
                       os.path.join(directory, "model.pt"))
        return directory


def initial_type_profiles(counts, n_types: int = 10, seed: int = 0) -> np.ndarray:
    """Initial latent type profiles: mean counts of ``n_types`` k-means clusters of the spots.

    Spot profiles are log-normalized, standardized and reduced to 50 principal components
    before clustering.  Each row of the result sums to one.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    counts = np.asarray(counts, np.float32)
    logn = np.log1p(counts / np.clip(counts.sum(1, keepdims=True), 1e-9, None) * 1e4)
    scaled = (logn - logn.mean(0)) / np.clip(logn.std(0), 1e-6, None)
    pcs = PCA(n_components=min(50, min(scaled.shape) - 1), random_state=seed).fit_transform(scaled)
    label = KMeans(n_types, n_init=10, random_state=seed).fit_predict(pcs)
    profiles = np.vstack([counts[label == k].mean(0) if (label == k).any() else counts.mean(0)
                          for k in range(n_types)])
    return (profiles / np.clip(profiles.sum(1, keepdims=True), 1e-12, None)).astype(np.float32)


def _clean_morph(sample: Sample) -> np.ndarray:
    """Morphometrics with non-finite values replaced by the column median."""
    morph = np.asarray(sample.morph, np.float32)
    bad = ~np.isfinite(morph)
    if not bad.any():
        return morph
    median = np.nanmedian(np.where(np.isfinite(morph), morph, np.nan), axis=0)
    median = np.nan_to_num(median, nan=0.0)
    columns = [str(sample.morph_names[j]) for j in np.where(bad.any(0))[0]]
    logger.info("  %s of %s cells carry a non-finite morphometric (%s): set to the column median",
                f"{int(bad.any(1).sum()):,}", f"{len(morph):,}", ", ".join(columns))
    return np.where(bad, median[None, :], morph).astype(np.float32)


def _image_view(sample: Sample, morph: np.ndarray) -> np.ndarray:
    """The image view of each cell: standardized morphometrics and image embedding, or,
    without an embedding, the morphometrics themselves."""
    if sample.image_features is None:
        return morph.copy()
    emb = np.asarray(sample.image_features, np.float32)
    emb = (emb - emb.mean(0)) / np.clip(emb.std(0), 1e-6, None)
    scaled = (morph - morph.mean(0)) / np.clip(morph.std(0), 1e-6, None)
    return np.hstack([np.nan_to_num(scaled), emb]).astype(np.float32)


def _load_anchors(anchors, n_cells: int):
    """Anchor cells as ``(lookup, profiles)``: row of each cell in ``profiles`` or -1."""
    if isinstance(anchors, (str, os.PathLike)):
        with np.load(anchors) as z:
            rows, counts = z["rows"], z["counts"]
    else:
        rows, counts = anchors
    rows = np.asarray(rows, np.int64)
    counts = np.asarray(counts, np.float32)
    lookup = np.full(n_cells, -1, np.int64)
    lookup[rows] = np.arange(len(rows))
    return lookup, counts / np.clip(counts.sum(1, keepdims=True), 1.0, None)


def _oom_errors():
    import torch

    # The caching allocator raises OutOfMemoryError, a raw allocation AcceleratorError.
    return tuple(e for e in (getattr(torch.cuda, "OutOfMemoryError", None),
                             getattr(torch, "AcceleratorError", None),
                             getattr(torch, "OutOfMemoryError", None)) if e)


def _empty_cache(device) -> None:
    import torch

    if getattr(device, "type", device) != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - a failed allocation can leave the context unusable
        pass


def fit(sample: Sample, *, n_types: int = 10, n_states: int = 20, dim: int = 256,
        rounds: int = 20, steps: int = 3, lr: float = 1e-3, tiles: int = 3,
        holdout: float = 0.15, w_depth: float = 0.1, w_spread: float = 0.05,
        w_drift: float = 0.1, max_edge_um: float = 40.0, spot_neighbours: int = 8,
        diffusion_um: float | None = None, anchors=None, anchor_weight: float = 1.0,
        predict_between: bool = True, monitor: bool = True, diagnostics: bool = True,
        shuffle_allocation: bool = False, seed: int = 0, device=None,
        eval_chunk: int = 40_000, eval_placement: str | int = "auto",
        callback=None) -> EMITResult:
    """Fit EMIT to one section and return the expression of every cell.

    Parameters
    ----------
    sample
        Spots and cells of the section.
    n_types, n_states, dim
        Number of latent types ``K``, dimension ``m`` of the cell-state vector and width
        of the cell tokens.
    rounds, steps
        Number of EM rounds and of passes over the training tiles per round.
    lr
        Learning rate of the Adam optimizer.
    tiles
        The section is divided into ``tiles x tiles`` batches of whole spots.
    holdout
        Fraction of spots withheld from training and used only to monitor how well the
        cells predicted inside them reproduce their measurement.
    w_depth, w_spread, w_drift
        Weights of the depth, spread and drift terms of the objective (see
        :func:`emit.em.m_loss`).
    max_edge_um
        Longest edge kept in the Delaunay graph of the cells.
    spot_neighbours
        Number of neighbouring spots each spot attends to.
    diffusion_um
        If given, the width ``sigma`` of lateral diffusion before capture.  The E-step
        then lets a cell contribute to every spot within reach.
    anchors
        Optional per-cell counts for some cells, for example cells assigned from Visium
        HD bins: a ``(rows, counts)`` pair or the path of an ``.npz`` file holding both.
        They add a per-cell term to the objective and leave the E-step unchanged.
    anchor_weight
        Weight of the anchor term.
    predict_between
        Fill the cells that received no allocation with the expected counts of the
        network.  These are predictions and are not constrained by any measurement.
    monitor
        Evaluate the whole section after every round (and before the first).  With
        ``False`` only the final round is evaluated, which is faster on large sections.
    diagnostics
        Add the truth-free diagnostics of :func:`emit.metrics.fit_diagnostics` to the
        history.
    shuffle_allocation
        Control only: permute the allocation among the cells of each spot during training.
    seed
        Seed of all random number generators.
    device
        ``"cuda"``, ``"cpu"`` or ``None`` to choose automatically.
    eval_chunk
        Cells per chunk when the profile matrix of the whole section is built.
    eval_placement
        Where that matrix lives: ``0`` on the compute device, ``1`` in host memory with
        the network on the device, ``2`` everything on the CPU, ``"auto"`` to decide from
        its size.  The result is the same at every level.
    callback
        ``callback(round, expression, allocated)`` is called after every monitored round;
        a dict it returns is added to that row of the history.

    Returns
    -------
    EMITResult
    """
    import torch

    from .model import EMITNet

    dev = resolve_device(device)
    torch.manual_seed(seed)
    params = dict(n_types=n_types, n_states=n_states, dim=dim, rounds=rounds, steps=steps, lr=lr,
                  tiles=tiles, holdout=holdout, w_depth=w_depth, w_spread=w_spread,
                  w_drift=w_drift, max_edge_um=max_edge_um, spot_neighbours=spot_neighbours,
                  diffusion_um=diffusion_um, anchor_weight=anchor_weight if anchors is not None
                  else None, predict_between=predict_between, seed=seed)

    # ------------------------------------------------------------- inputs and graphs
    n_cells, n_spots, n_genes = sample.n_cells, sample.n_spots, sample.n_genes
    cell_spot = sample.cell_spot
    morph = _clean_morph(sample)
    features = _image_view(sample, morph)
    profiles0 = initial_type_profiles(sample.counts, n_types, seed)
    nearest_spot = cKDTree(sample.spot_xy).query(sample.cell_xy, k=1)[1]
    edge_index = delaunay_edges(sample.cell_xy, sample.px_um, max_edge_um)
    spot_edges = spot_window_edges(sample.spot_xy, spot_neighbours)

    model = EMITNet(features.shape[1], morph.shape[1], profiles0, n_states=n_states, dim=dim)
    model = model.to(dev)
    model.encoder.fit_morph_stats(morph)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    feat_t = torch.as_tensor(features, device=dev)
    morph_t = torch.as_tensor(np.nan_to_num(morph, nan=0.0), device=dev)
    batches = spatial_tiles(sample.spot_xy, cell_spot, nearest_spot, edge_index, sample.counts,
                            n_tiles=tiles, seed=seed, spot_neighbours=spot_neighbours)

    anchor_lookup = anchor_profiles = None
    if anchors is not None:
        anchor_lookup, anchor_profiles = _load_anchors(anchors, n_cells)
        anchor_profiles = torch.as_tensor(anchor_profiles, device=dev)
        logger.info("anchors: %s cells with per-cell counts (weight %g)",
                    f"{int((anchor_lookup >= 0).sum()):,}", anchor_weight)
    generator = torch.Generator(device=dev)
    generator.manual_seed(seed)

    rng = np.random.default_rng(seed)
    heldout = np.zeros(n_spots, bool)
    if holdout > 0:
        heldout[rng.choice(n_spots, int(holdout * n_spots), replace=False)] = True
    heldout_t = torch.as_tensor(heldout, device=dev)
    base_nearest, base_mean = heldout_baselines(sample.counts, sample.spot_xy, heldout)

    # the diffusion-aware E-step needs the cell-spot pairs, per tile and for the section
    tile_pairs, all_pairs = {}, None
    if diffusion_um is not None:
        for k, b in enumerate(batches):
            ci, si, w = capture_pairs(sample.cell_xy[b["cells"]], sample.spot_xy[b["spot_ids"]],
                                      sample.px_um, sample.spot_radius_um, diffusion_um)
            tile_pairs[k] = (torch.as_tensor(ci, device=dev), torch.as_tensor(si, device=dev),
                             torch.as_tensor(w, device=dev))
        all_pairs = capture_pairs(sample.cell_xy, sample.spot_xy, sample.px_um,
                                  sample.spot_radius_um, diffusion_um)

    logger.info("%s | %d tiles | %s", sample.summary(), len(batches), dev)
    if holdout > 0:
        logger.info("  %s spots held out | relative distance of the baselines on them: nearest "
                    "training spot %.4f, section mean %.4f", f"{int(heldout.sum()):,}",
                    base_nearest, base_mean)

    # ------------------------------------------------------------- where inference runs
    # Decided up front: on Windows an over-large CUDA allocation spills into host memory
    # instead of failing, and after a failed allocation the model cannot always be moved.
    level = 0
    if eval_placement != "auto":
        level = int(eval_placement)
    elif dev.type == "cuda":
        need = n_cells * n_genes * 4
        vram = torch.cuda.get_device_properties(dev).total_memory
        if need > 0.35 * vram:
            level = 1
            logger.info("  the profile matrix is %.1f GB against %.1f GB of GPU memory: it is "
                        "assembled in host memory", need / 1e9, vram / 1e9)
    cpu = torch.device("cpu")
    oom = _oom_errors()

    def full_pass(compute, host, final):
        """Inference on the whole section and everything read from it."""
        model.to(compute)
        with torch.no_grad():
            out = model(torch.as_tensor(features, device=compute),
                        torch.as_tensor(np.nan_to_num(morph, nan=0.0), device=compute),
                        torch.as_tensor(edge_index, device=compute),
                        torch.as_tensor(cell_spot, device=compute), n_spots,
                        torch.as_tensor(spot_edges, device=compute),
                        torch.as_tensor(nearest_spot, device=compute),
                        chunk=eval_chunk, out_device=None if host is compute else host)
            pi, lam = out["pi"], out["lam"].to(host)
            type_prob = out["log_p"].exp().float().cpu().numpy() if final else None
            del out
            spot_t = torch.as_tensor(cell_spot, device=host)
            counts_t = torch.as_tensor(sample.counts, device=host)
            if all_pairs is None:
                allocated, inside = e_step(pi, lam, spot_t, counts_t)
                mask = inside.cpu().numpy()
                expression = np.zeros((n_cells, n_genes), np.float32)
                expression[mask] = allocated.cpu().numpy()
                del allocated, inside
            else:
                ci, si, w = (torch.as_tensor(v, device=host) for v in all_pairs)
                expression = e_step_diffuse(pi, lam, ci, si, w, counts_t).cpu().numpy()
                mask = expression.sum(1) > 0
            info = {}
            if holdout > 0:
                held = heldout_t.to(host)
                info["heldout_reldist"] = heldout_relative_distance(pi, lam, spot_t, counts_t, held)
                info["heldout_nll"] = heldout_nll(pi, lam, spot_t, counts_t, held)
            if diagnostics:
                info.update(fit_diagnostics(pi, lam, spot_t, counts_t, expression, n_spots))
            if final and predict_between:
                rest = np.where(~mask)[0]
                for a in range(0, len(rest), eval_chunk):
                    rows = torch.as_tensor(rest[a:a + eval_chunk], device=host)
                    expression[rest[a:a + eval_chunk]] = (
                        lam[rows].unsqueeze(-1) * pi[rows]).float().cpu().numpy()
            library = lam.float().cpu().numpy()
        return expression, mask, type_prob, library, info

    def evaluate(final):
        nonlocal level
        _empty_cache(dev)
        while True:
            try:
                if level == 0:
                    result = full_pass(dev, dev, final)
                elif level == 1:
                    result = full_pass(dev, cpu, final)
                else:
                    result = full_pass(cpu, cpu, final)
                break
            except oom:
                if level >= 2:
                    raise
                level += 1
                _empty_cache(dev)
                logger.info("    out of GPU memory in the inference pass; continuing with %s",
                            "the profile matrix in host memory" if level == 1 else "the CPU")
        model.to(dev)
        _empty_cache(dev)
        return result

    # ------------------------------------------------------------- EM rounds
    def to_dev(b):
        return {k: (torch.as_tensor(v, device=dev) if isinstance(v, np.ndarray) else v)
                for k, v in b.items()}

    dev_batches = [to_dev(b) for b in batches]
    history = []
    parts = dict(ce=np.nan, depth=np.nan, spread=np.nan, drift=np.nan)
    result = None
    logger.info("%5s %8s %9s %7s %5s", "round", "ce", "held-out", "drift", "min")
    for rd in range(rounds + 1):
        t0 = time.time()
        if rd:
            model.train()
            for _ in range(steps):
                for k, (batch, b) in enumerate(zip(batches, dev_batches)):
                    out = model(feat_t[b["cells"]], morph_t[b["cells"]], b["edge_index"],
                                b["cell_spot"], b["n_spots"], b["spot_edges"], b["nearest_spot"])
                    spot, counts = b["cell_spot"], b["counts"]
                    # Held-out spots are removed by marking their cells as uncovered: they
                    # leave both steps but stay in the graph and the context.
                    held = torch.as_tensor(heldout[batch["spot_ids"]], device=dev)
                    train_spot = torch.where(held[spot.clamp_min(0)] & (spot >= 0),
                                             torch.full_like(spot, -1), spot)
                    with torch.no_grad():
                        if diffusion_um is None:
                            allocated, inside = e_step(out["pi"].detach(), out["lam"].detach(),
                                                       train_spot, counts,
                                                       shuffle=shuffle_allocation,
                                                       generator=generator)
                            loss_spot = train_spot
                        else:
                            ci, si, w = tile_pairs[k]
                            use = ~held[si]
                            alloc_all = e_step_diffuse(out["pi"].detach(), out["lam"].detach(),
                                                       ci[use], si[use], w[use], counts)
                            inside = alloc_all.sum(1) > 0
                            allocated = alloc_all[inside]
                            # a cell outside every spot can receive counts here; for the
                            # drift term it is compared with its nearest spot
                            loss_spot = torch.where(spot >= 0, spot, b["nearest_spot"])
                    if allocated is None or not bool(inside.any()):
                        continue
                    loss, parts = m_loss(out["pi"], out["log_lam"], allocated, inside,
                                         cell_spot=loss_spot, n_spots=b["n_spots"],
                                         w_depth=w_depth, w_spread=w_spread, w_drift=w_drift)
                    if anchor_lookup is not None:
                        rows = anchor_lookup[batch["cells"]]
                        sel = np.where(rows >= 0)[0]
                        if len(sel):
                            pi_anchor = out["pi"][torch.as_tensor(sel, device=dev)]
                            log_pi = pi_anchor.clamp_min(1e-12).log()
                            target = anchor_profiles[torch.as_tensor(rows[sel], device=dev)]
                            loss = loss + anchor_weight * (-(target * log_pi).sum(1).mean())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

        final = rd == rounds
        if not (monitor or final):
            continue
        model.eval()
        result = evaluate(final)
        row = dict(round=rd, loss_ce=parts["ce"], loss_depth=parts["depth"],
                   loss_spread=parts["spread"], loss_drift=parts["drift"], **result[4],
                   baseline_nearest=base_nearest, baseline_mean=base_mean)
        if callback is not None:
            extra = callback(rd, result[0], result[1])
            if isinstance(extra, dict):
                row.update(extra)
        row["minutes"] = round((time.time() - t0) / 60, 2)
        history.append(row)
        logger.info("%5d %8.4f %9.4f %7.3f %5.2f", rd, row["loss_ce"],
                    row.get("heldout_reldist", np.nan), row.get("drift", np.nan), row["minutes"])

    expression, mask, type_prob, library, _ = result
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    profiles = model.type_profiles.detach().float().cpu().numpy()
    return EMITResult(
        expression=expression, allocated=mask, type_prob=type_prob, library_size=library,
        type_profiles=profiles, history=pd.DataFrame(history), genes=sample.genes,
        cell_names=sample.cell_names, cell_xy=sample.cell_xy, cell_spot=cell_spot,
        spot_names=sample.spot_names, px_um=sample.px_um, params=params, state_dict=state)
