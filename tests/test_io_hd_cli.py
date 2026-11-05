import json

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import emit
from emit.cli import main
from emit.hd import aggregate_bins, assign_bins, bins_to_cells, nuclei_on_bins
from emit.simulate import hex_grid


def _write_matrix(path, counts, barcodes, genes):
    matrix = sp.csc_matrix(counts.T)  # stored as genes x barcodes
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = matrix.data, matrix.indices, matrix.indptr
        g["shape"] = np.array(matrix.shape)
        g["barcodes"] = np.array(barcodes, dtype="S")
        feat = g.create_group("features")
        feat["name"] = np.array(genes, dtype="S")
        feat["id"] = np.array([f"ENSG{i:05d}" for i in range(len(genes))], dtype="S")


@pytest.fixture()
def spaceranger_dir(tmp_path):
    """A minimal Space Ranger directory: 100 um hexagonal lattice at 0.5 um/px."""
    rng = np.random.default_rng(0)
    xy = hex_grid(np.array([[0.0, 0.0], [2400.0, 2000.0]]), 200.0)
    barcodes = [f"BC{i:04d}-1" for i in range(len(xy))]
    in_tissue = np.ones(len(xy), int)
    in_tissue[:3] = 0
    counts = rng.poisson(2.0, size=(len(xy), 40))
    counts[5] = 0  # a spot below min_counts
    root = tmp_path / "outs"
    (root / "spatial").mkdir(parents=True)
    _write_matrix(root / "filtered_feature_bc_matrix.h5", counts[in_tissue == 1],
                  np.array(barcodes)[in_tissue == 1], [f"g{i}" for i in range(40)])
    pd.DataFrame({"barcode": barcodes, "in_tissue": in_tissue, "array_row": 0, "array_col": 0,
                  "pxl_row_in_fullres": xy[:, 1], "pxl_col_in_fullres": xy[:, 0]}).to_csv(
        root / "spatial" / "tissue_positions.csv", index=False)
    (root / "spatial" / "scalefactors_json.json").write_text(
        json.dumps({"spot_diameter_fullres": 110.0, "tissue_hires_scalef": 0.1}))
    return root, xy, counts, in_tissue


def test_read_visium(spaceranger_dir):
    root, xy, counts, in_tissue = spaceranger_dir
    spots = emit.read_visium(str(root), n_top_genes=20, min_counts=10)
    kept = (in_tissue == 1) & (counts.sum(1) >= 10)
    assert spots["counts"].shape == (kept.sum(), 20)
    assert np.allclose(spots["spot_xy"], xy[kept])  # x is the column, y the row
    assert abs(spots["px_um"] - 0.5) < 1e-6 and spots["spot_diameter_px"] == 110.0
    genes = [int(g[1:]) for g in spots["genes"]]
    assert np.array_equal(spots["counts"], counts[kept][:, genes])

    everything = emit.read_visium(str(root), n_top_genes=None, min_counts=0)
    assert everything["counts"].shape == ((in_tissue == 1).sum(), 40)


def test_read_visium_without_header(spaceranger_dir):
    root, xy, _, in_tissue = spaceranger_dir
    table = pd.read_csv(root / "spatial" / "tissue_positions.csv")
    (root / "spatial" / "tissue_positions.csv").unlink()
    table.to_csv(root / "spatial" / "tissue_positions_list.csv", index=False, header=False)
    spots = emit.read_visium(str(root), n_top_genes=None, min_counts=0)
    assert np.allclose(spots["spot_xy"], xy[in_tissue == 1])


def test_build_sample_and_registration(spaceranger_dir, tmp_path):
    root, xy, _, in_tissue = spaceranger_dir
    spots = emit.read_visium(str(root), n_top_genes=None, min_counts=10)
    rng = np.random.default_rng(1)
    cell_xy = rng.uniform(xy.min(0), xy.max(0), size=(4000, 2))
    np.savez(tmp_path / "nuclei.npz", xy=cell_xy, morph=rng.normal(size=(4000, 5)),
             morph_names=np.array(list("abcde"), dtype=object), px_um=spots["px_um"])
    nuclei = emit.read_nuclei(str(tmp_path / "nuclei.npz"))
    sample = emit.build_sample(spots, nuclei, name="fixture")
    assert sample.n_cells == 4000 and 0.1 < sample.inside.mean() < 0.4
    dist = np.linalg.norm(sample.cell_xy[sample.inside]
                          - sample.spot_xy[sample.cell_spot[sample.inside]], axis=1)
    assert dist.max() <= 55.0
    with pytest.raises(ValueError):
        emit.build_sample(spots, {**nuclei, "px_um": 0.25})


def test_read_image_features_restores_the_order(tmp_path):
    feats = np.arange(12, dtype=np.float16).reshape(6, 2)
    order = np.array([3, 0, 5, 1, 4, 2])
    np.savez(tmp_path / "f.npz", feats=feats, idx=order)
    out = emit.read_image_features(str(tmp_path / "f.npz"), n_cells=6)
    assert out.dtype == np.float32 and np.array_equal(out[order], feats)
    with pytest.raises(ValueError):
        emit.read_image_features(str(tmp_path / "f.npz"), n_cells=7)


# ---------------------------------------------------------------------- Visium HD
def _grid(n=20, bin_um=2.0):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n))
    return np.column_stack([(ii.ravel() + 0.5) * bin_um, (jj.ravel() + 0.5) * bin_um])


def test_bins_go_to_the_nearest_nucleus_within_reach():
    xy = _grid()
    nucleus = np.full(len(xy), -1)
    nucleus[(np.abs(xy[:, 0] - 9) <= 2) & (np.abs(xy[:, 1] - 9) <= 2)] = 0
    nucleus[(np.abs(xy[:, 0] - 31) <= 2) & (np.abs(xy[:, 1] - 9) <= 2)] = 1
    owner = assign_bins(xy, nucleus, d_max_um=4.0)
    assert (owner[nucleus >= 0] == nucleus[nucleus >= 0]).all()
    assigned = (owner >= 0) & (nucleus < 0)
    assert assigned.any()
    for cell, centre in ((0, 9), (1, 31)):
        mine = xy[owner == cell]
        core = xy[nucleus == cell]
        d = np.min(np.linalg.norm(mine[:, None] - core[None], axis=2), axis=1)
        assert d.max() <= 4.0 + 1e-9
        assert np.abs(mine[:, 0] - centre).max() < 11
    # a bin far from both nuclei stays unassigned, a larger reach assigns more
    assert owner[np.argmin(np.linalg.norm(xy - [20, 35], axis=1))] == -1
    assert (assign_bins(xy, nucleus, d_max_um=8.0) >= 0).sum() > (owner >= 0).sum()
    # ties go to the smaller label, whatever the order of the bins
    perm = np.random.default_rng(0).permutation(len(xy))
    assert np.array_equal(assign_bins(xy[perm], nucleus[perm], d_max_um=20.0),
                          assign_bins(xy, nucleus, d_max_um=20.0)[perm])


def test_distance_bound_is_inclusive_and_chunking_does_not_matter():
    xy = _grid()
    nucleus = np.full(len(xy), -1)
    nucleus[np.argmin(np.linalg.norm(xy - [9, 9], axis=1))] = 0
    owner = assign_bins(xy, nucleus, d_max_um=4.0)
    dist = np.linalg.norm(xy - [9, 9], axis=1)
    assert (owner[dist <= 4.0] == 0).all() and (owner[dist > 4.0] == -1).all()
    assert (dist == 4.0).any()  # bins exactly two bins away are assigned
    assert np.array_equal(assign_bins(xy, nucleus, d_max_um=4.0, chunk=7), owner)
    # nucleus bins given separately from the bins to assign
    part = np.where(nucleus < 0)[0][::3]
    sub = assign_bins(xy[part], nucleus[part], d_max_um=4.0, ref_xy_um=xy[nucleus >= 0],
                      ref_nucleus=nucleus[nucleus >= 0])
    assert np.array_equal(sub, owner[part])


def test_every_transcript_goes_to_one_cell_or_none():
    rng = np.random.default_rng(0)
    xy = _grid()
    nucleus = np.full(len(xy), -1)
    nucleus[:40] = np.repeat(np.arange(4), 10)
    counts = sp.csr_matrix(rng.poisson(0.5, size=(len(xy), 7)).astype(np.float32))
    cells, owner = bins_to_cells(counts, xy, nucleus, n_cells=4, d_max_um=4.0)
    assert cells.shape == (4, 7)
    assert np.isclose(cells.sum(), counts[owner >= 0].sum())
    assert np.allclose(aggregate_bins(counts, owner, 4).toarray(), cells.toarray())


def test_nuclei_on_bins():
    labels = np.zeros((40, 40), np.int32)
    labels[5:15, 5:15] = 1
    labels[20, 30] = 2  # smaller than a bin
    xy = np.column_stack([np.tile(np.arange(2, 40, 4), 10), np.repeat(np.arange(2, 40, 4), 10)])
    nucleus = nuclei_on_bins(labels, xy.astype(float))
    assert set(nucleus) == {-1, 0}
    with_centroids = nuclei_on_bins(labels, xy.astype(float),
                                    centroids_px=np.array([[9.5, 9.5], [30.0, 20.0]]))
    assert (with_centroids == 1).sum() == 1


# ---------------------------------------------------------------------- command line
def test_cli_toy_and_fit(tmp_path):
    sample_path = str(tmp_path / "toy.npz")
    assert main(["-q", "toy", "--out", sample_path, "--n-cells", "2500", "--n-genes", "50",
                 "--truth", str(tmp_path / "truth.npz")]) == 0
    out = str(tmp_path / "fit")
    assert main(["-q", "fit", "--sample", sample_path, "--out", out, "--rounds", "1", "--dim",
                 "32", "--n-types", "4", "--n-states", "4", "--device", "cpu",
                 "--no-monitor"]) == 0
    import anndata

    adata = anndata.read_h5ad(f"{out}/cells.h5ad")
    assert adata.n_obs == 2500 and adata.n_vars == 50
    assert (tmp_path / "fit" / "model.pt").exists()


def test_cli_needs_an_input(tmp_path):
    with pytest.raises(SystemExit):
        main(["-q", "fit", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        main([])
