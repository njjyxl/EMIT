import numpy as np
import pytest

from emit.image import read_image
from emit.image.segment import (
    MORPH_NAMES,
    morphometrics,
    neighbour_density,
    stain_features,
    tissue_mask,
)
from emit.simulate import hex_grid

pytest.importorskip("skimage")


def _disc(labels, cx, cy, r, value):
    yy, xx = np.ogrid[:labels.shape[0], :labels.shape[1]]
    labels[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = value


def test_morphometrics_of_known_shapes():
    labels = np.zeros((200, 300), np.int32)
    _disc(labels, 60, 50, 12, 1)
    _disc(labels, 90, 50, 8, 2)
    labels[120:130, 200:260] = 5  # an elongated nucleus with a non-consecutive label
    image = np.full((200, 300, 3), 255, np.uint8)
    image[labels > 0] = (90, 40, 150)  # purple nuclei on a white background
    xy, morph, ids = morphometrics(labels, image, px_um=0.5)
    assert morph.shape == (3, len(MORPH_NAMES)) and ids.tolist() == [1, 2, 5]
    assert np.allclose(xy[0], [60, 50], atol=0.5) and np.allclose(xy[2], [229.5, 124.5], atol=0.5)
    col = {n: morph[:, i] for i, n in enumerate(MORPH_NAMES)}
    assert abs(col["area"][0] - np.pi * 12 ** 2) < 20
    assert col["eccentricity"][0] < 0.2 < 0.9 < col["eccentricity"][2]
    assert col["circularity"][0] > col["circularity"][2] and col["circularity"].max() <= 1.0
    assert col["hematoxylin"][0] > 0.2 and col["gray_mean"][0] < 150
    # the two discs are 15 um apart, the bar is far from both
    assert col["neigh_density"].tolist() == [1.0, 1.0, 0.0]


def test_empty_label_image():
    labels, image = np.zeros((20, 20), np.int32), np.zeros((20, 20, 3), np.uint8)
    xy, morph, ids = morphometrics(labels, image, 0.5)
    assert xy.shape == (0, 2) and morph.shape == (0, len(MORPH_NAMES)) and len(ids) == 0


def test_stain_features_at_the_image_edge():
    image = np.random.default_rng(0).integers(0, 255, size=(60, 60, 3), dtype=np.uint8)
    out = stain_features(image, np.array([[30.0, 30.0], [0.0, 0.0]]), px_um=0.5)
    assert np.isfinite(out[0]).all() and out.shape == (2, 5)
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0]])
    assert neighbour_density(xy, 1.0).tolist() == [1, 1, 0]


def test_tissue_mask_follows_the_spot_lattice():
    spots = hex_grid(np.array([[400.0, 400.0], [1200.0, 1000.0]]), 200.0)
    mask = tissue_mask(spots, px_um=0.5, height=1500, width=2000)
    assert mask.shape == (1500, 2000)
    assert mask[spots[:, 1].astype(int), spots[:, 0].astype(int)].all()
    assert not mask[10, 10] and not mask[1490, 1990]
    assert 0.1 < mask.mean() < 0.6


def test_read_image(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    array = np.random.default_rng(0).integers(0, 255, size=(30, 40, 3), dtype=np.uint8)
    pil.fromarray(array).save(tmp_path / "a.png")
    assert np.array_equal(read_image(str(tmp_path / "a.png")), array)
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "a.tif", array)
    assert np.array_equal(read_image(str(tmp_path / "a.tif")), array)


# ---------------------------------------------------------------------- segmentation plumbing
@pytest.fixture()
def fake_stardist(monkeypatch):
    """Stand-ins for StarDist and CSBDeep that label dark blobs, so that everything
    around the network (normalization, masking, relabelling, files) can be tested."""
    import sys
    import types

    from scipy import ndimage as ndi

    class Normalizer:
        pass

    class StarDist2D:
        calls = []

        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def _predict(self, img, normalizer):
            x = normalizer.before(img, "YXC")
            assert not normalizer.do_after and x.dtype == np.float32
            labels, _ = ndi.label(x.mean(-1) < 0.5)
            return labels.astype(np.int32) * 3, {}  # non-consecutive label values

        def predict_instances(self, img, axes, normalizer, prob_thresh, nms_thresh):
            self.calls.append("small")
            return self._predict(img, normalizer)

        def predict_instances_big(self, img, axes, block_size, min_overlap, context, normalizer,
                                  n_tiles, prob_thresh, nms_thresh, labels_out_dtype):
            self.calls.append("big")
            return self._predict(img, normalizer)

    data = types.ModuleType("csbdeep.data")
    data.Normalizer = Normalizer
    models = types.ModuleType("stardist.models")
    models.StarDist2D = StarDist2D
    for name, module in (("csbdeep", types.ModuleType("csbdeep")), ("csbdeep.data", data),
                         ("stardist", types.ModuleType("stardist")), ("stardist.models", models)):
        monkeypatch.setitem(sys.modules, name, module)
    return StarDist2D


def _tissue_image(height=600, width=800, seed=0):
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 235, np.uint8)
    centres = rng.uniform([40, 40], [width - 40, height - 40], size=(60, 2))
    labels = np.zeros((height, width), np.int32)
    for k, (cx, cy) in enumerate(centres):
        _disc(labels, cx, cy, 7, k + 1)
    image[labels > 0] = (70, 40, 130)
    return image, centres


def test_segment_nuclei_plumbing(fake_stardist):
    from emit.image.segment import segment_nuclei

    image, centres = _tissue_image()
    out = segment_nuclei(image, px_um=0.5, return_labels=True)
    n = len(out["xy"])
    assert 40 <= n <= 60 and out["morph"].shape == (n, len(MORPH_NAMES))
    assert out["labels"].shape == image.shape[:2] and out["labels"].max() == n
    # nucleus k of the table has value k + 1 in the label image
    rows = out["labels"][out["xy"][:, 1].astype(int), out["xy"][:, 0].astype(int)]
    assert (rows == np.arange(1, n + 1)).all()
    assert fake_stardist.calls[-1] == "small"

    # confined to the spot lattice: nuclei far from every spot are dropped
    spots = hex_grid(np.array([[100.0, 100.0], [300.0, 300.0]]), 200.0)
    masked = segment_nuclei(image, px_um=0.5, spot_xy=spots, block_size=256)
    assert 0 < len(masked["xy"]) < n and fake_stardist.calls[-1] == "big"
    assert "labels" not in masked


def test_cli_segment_then_hd_cells(fake_stardist, tmp_path):
    import json

    import anndata
    import h5py
    import pandas as pd
    import scipy.sparse as sp

    pytest.importorskip("pyarrow")
    pil = pytest.importorskip("PIL.Image")
    from emit.cli import main

    image, _ = _tissue_image()
    pil.fromarray(image).save(tmp_path / "he.png")

    # a Visium HD bin directory: 2 um bins at 0.5 um/px over the whole image
    px_um, bin_px = 0.5, 4
    ii, jj = np.meshgrid(np.arange(image.shape[1] // bin_px), np.arange(image.shape[0] // bin_px))
    xy = np.column_stack([(ii.ravel() + 0.5) * bin_px, (jj.ravel() + 0.5) * bin_px])
    barcodes = np.array([f"s_002um_{j:05d}_{i:05d}-1" for i, j in zip(ii.ravel(), jj.ravel())])
    rng = np.random.default_rng(0)
    counts = sp.csc_matrix(rng.poisson(0.3, size=(len(xy), 12)).astype(np.int32).T)
    shuffled = rng.permutation(len(xy))  # the matrix lists the bins in another order
    root = tmp_path / "square_002um"
    (root / "spatial").mkdir(parents=True)
    with h5py.File(root / "filtered_feature_bc_matrix.h5", "w") as f:
        g = f.create_group("matrix")
        m = counts[:, shuffled]
        g["data"], g["indices"], g["indptr"], g["shape"] = m.data, m.indices, m.indptr, m.shape
        g["barcodes"] = barcodes[shuffled].astype("S")
        g.create_group("features")["name"] = np.array([f"g{i}" for i in range(12)], dtype="S")
    pd.DataFrame({"barcode": barcodes, "in_tissue": 1, "array_row": jj.ravel(),
                  "array_col": ii.ravel(), "pxl_row_in_fullres": xy[:, 1],
                  "pxl_col_in_fullres": xy[:, 0]}).to_parquet(
        root / "spatial" / "tissue_positions.parquet")
    (root / "spatial" / "scalefactors_json.json").write_text(json.dumps(
        {"bin_size_um": 2.0, "microns_per_pixel": px_um, "spot_diameter_fullres": 4.0}))

    nuclei, labels = str(tmp_path / "nuclei.npz"), str(tmp_path / "labels.npy")
    assert main(["-q", "segment", "--image", str(tmp_path / "he.png"), "--spaceranger", str(root),
                 "--hd", "--out", nuclei, "--labels", labels]) == 0
    seg = np.load(nuclei, allow_pickle=True)
    assert float(seg["px_um"]) == px_um and len(seg["xy"]) > 30

    out, owner_path = str(tmp_path / "cells.h5ad"), str(tmp_path / "owner.npy")
    assert main(["-q", "hd-cells", "--bins", str(root), "--labels", labels, "--nuclei", nuclei,
                 "--out", out, "--owner", owner_path]) == 0
    cells = anndata.read_h5ad(out)
    owner = np.load(owner_path)
    assert cells.shape == (len(seg["xy"]), 12) and len(owner) == len(xy)
    total = counts.sum()
    assert 0 < cells.X.sum() < total
    assert np.isclose(cells.X.sum(), counts[:, shuffled].T[owner >= 0].sum())
    assert (cells.obs["n_bins"] > 0).mean() > 0.9
    # bins are matched to their positions by barcode: every owned bin is near its nucleus
    bin_xy = xy[shuffled]
    d = np.linalg.norm(bin_xy[owner >= 0] - seg["xy"][owner[owner >= 0]], axis=1) * px_um
    assert np.median(d) < 8.0 and d.max() < 25.0  # a mismatch would give hundreds of um
