import numpy as np
import pytest
import torch

import emit
from emit.metrics import conservation_error, gene_correlation, within_spot_r
from emit.model import EMITNet

FAST = dict(rounds=4, steps=2, dim=64, n_types=4, n_states=8, device="cpu")


@pytest.fixture(scope="module")
def fitted(toy):
    sample, _ = toy
    return emit.fit(sample, seed=0, **FAST)


def test_output_shapes(toy, fitted):
    sample, _ = toy
    assert fitted.expression.shape == (sample.n_cells, sample.n_genes)
    assert fitted.expression.dtype == np.float32
    assert fitted.type_prob.shape == (sample.n_cells, 4)
    assert np.allclose(fitted.type_prob.sum(1), 1.0, atol=1e-4)
    assert fitted.type_profiles.shape == (4, sample.n_genes)
    assert np.allclose(fitted.type_profiles.sum(1), 1.0, atol=1e-4)
    assert (fitted.allocated == sample.inside).all()
    assert np.isfinite(fitted.expression).all() and (fitted.expression >= 0).all()
    assert len(fitted.history) == FAST["rounds"] + 1
    assert {"heldout_reldist", "drift", "baseline_nearest"} <= set(fitted.history.columns)


def test_allocation_returns_the_spot_counts(toy, fitted):
    sample, _ = toy
    err = conservation_error(np.where(fitted.allocated[:, None], fitted.expression, 0),
                             sample.counts, sample.cell_spot)
    assert np.nanmax(err) < 1e-4


def test_cells_between_spots_are_predicted(toy, fitted):
    sample, _ = toy
    between = ~sample.inside
    assert between.any()
    assert (fitted.expression[between].sum(1) > 0).all()
    without = emit.fit(sample, seed=0, predict_between=False, monitor=False, **FAST)
    assert float(without.expression[between].sum()) == 0.0
    assert len(without.history) == 1


def test_fit_resolves_cells_inside_spots(toy, fitted):
    sample, truth = toy
    r = within_spot_r(fitted.expression, truth["counts"], sample.cell_spot)
    # the spot mean scores exactly zero on this measure
    spot_mean = sample.counts[np.clip(sample.cell_spot, 0, None)]
    assert abs(within_spot_r(spot_mean, truth["counts"], sample.cell_spot)) < 1e-6
    assert r > 0.1
    assert np.nanmedian(gene_correlation(fitted.expression[sample.inside],
                                         truth["counts"][sample.inside])) > 0.2


def test_fit_is_reproducible(toy, fitted):
    sample, _ = toy
    again = emit.fit(sample, seed=0, **FAST)
    assert np.allclose(again.expression, fitted.expression, rtol=1e-4, atol=1e-4)
    other = emit.fit(sample, seed=1, **FAST)
    assert not np.allclose(other.expression, fitted.expression, rtol=1e-4, atol=1e-4)


def test_image_features_are_used(toy):
    sample, _ = toy
    rng = np.random.default_rng(0)
    with_features = emit.Sample(
        counts=sample.counts, genes=sample.genes, spot_xy=sample.spot_xy, cell_xy=sample.cell_xy,
        morph=sample.morph, px_um=sample.px_um, spot_diameter_px=sample.spot_diameter_px,
        image_features=rng.normal(size=(sample.n_cells, 16)).astype(np.float16))
    result = emit.fit(with_features, seed=0, monitor=False, **{**FAST, "rounds": 1})
    assert np.isfinite(result.expression).all()


def test_diffusion_aware_fit(toy):
    sample, _ = toy
    result = emit.fit(sample, seed=0, diffusion_um=10.0, monitor=False, **{**FAST, "rounds": 2})
    # cells within reach of a spot receive counts, including some outside every spot
    assert result.allocated[sample.inside].all()
    assert (result.allocated & ~sample.inside).any()
    assert np.isfinite(result.expression).all()


def test_anchors_and_callback(toy):
    sample, truth = toy
    rows = np.where(sample.inside)[0][:200]
    seen = []

    def callback(rd, expression, allocated):
        seen.append(rd)
        return {"r": within_spot_r(expression, truth["counts"], sample.cell_spot)}

    result = emit.fit(sample, seed=0, anchors=(rows, truth["counts"][rows]), callback=callback,
                      **{**FAST, "rounds": 2})
    assert seen == [0, 1, 2]
    assert "r" in result.history.columns and result.history.r.notna().all()


def test_non_finite_morphometrics_are_tolerated(toy):
    sample, _ = toy
    morph = sample.morph.copy()
    morph[::17, 2] = np.nan
    morph[5, 0] = np.inf
    damaged = emit.Sample(counts=sample.counts, genes=sample.genes, spot_xy=sample.spot_xy,
                          cell_xy=sample.cell_xy, morph=morph, px_um=sample.px_um,
                          spot_diameter_px=sample.spot_diameter_px)
    result = emit.fit(damaged, seed=0, monitor=False, **{**FAST, "rounds": 1})
    assert np.isfinite(result.expression).all()


def test_save_and_anndata(toy, fitted, tmp_path):
    sample, _ = toy
    out = fitted.save(str(tmp_path / "out"))
    import anndata

    adata = anndata.read_h5ad(f"{out}/cells.h5ad")
    assert adata.shape == (sample.n_cells, sample.n_genes)
    assert set(adata.obs["source"]) == {"allocated", "predicted"}
    assert adata.obsm["spatial"].shape == (sample.n_cells, 2)
    assert (tmp_path / "out" / "history.csv").exists()
    state = torch.load(f"{out}/model.pt", weights_only=False)
    net = EMITNet(sample.morph.shape[1], sample.morph.shape[1],
                  np.ones((4, sample.n_genes)), n_states=8, dim=64)
    net.load_state_dict(state["state_dict"])


def test_sample_roundtrip_and_validation(toy, tmp_path):
    sample, _ = toy
    path = sample.save(str(tmp_path / "sample.npz"))
    loaded = emit.Sample.load(path)
    assert np.array_equal(loaded.counts, sample.counts)
    assert np.array_equal(loaded.cell_spot, sample.cell_spot)
    assert loaded.name == "toy" and list(loaded.genes) == list(sample.genes)
    stats = loaded.check_registration()
    assert 0.15 < stats["fraction_of_cells_inside_spots"] < 0.4
    with pytest.raises(ValueError):
        emit.Sample(counts=sample.counts, genes=sample.genes[:-1], spot_xy=sample.spot_xy,
                    cell_xy=sample.cell_xy, morph=sample.morph, px_um=sample.px_um,
                    spot_diameter_px=sample.spot_diameter_px)
    with pytest.raises(ValueError):  # cells and spots in different frames
        emit.Sample(counts=sample.counts, genes=sample.genes, spot_xy=sample.spot_xy + 1e6,
                    cell_xy=sample.cell_xy, morph=sample.morph, px_um=sample.px_um,
                    spot_diameter_px=sample.spot_diameter_px)
