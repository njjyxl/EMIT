import numpy as np

from emit.geometry import assign_cells_to_spots, delaunay_edges, spatial_tiles, spot_window_edges
from emit.simulate import hex_grid, square_bins, visium_spots


def test_cells_are_assigned_to_the_spot_that_contains_them():
    spots = np.array([[0.0, 0.0], [100.0, 0.0]])
    cells = np.array([[5.0, 0.0], [95.0, 3.0], [50.0, 0.0]])
    assert assign_cells_to_spots(cells, spots, 27.5).tolist() == [0, 1, -1]


def test_delaunay_edges_are_symmetric_and_pruned():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 500, size=(400, 2))
    edges = delaunay_edges(xy, px_um=1.0, max_um=40.0)
    assert edges.shape[0] == 2 and edges.dtype == np.int64
    length = np.linalg.norm(xy[edges[0]] - xy[edges[1]], axis=1)
    assert length.max() <= 40.0
    pairs = set(map(tuple, edges.T))
    assert all((b, a) in pairs for a, b in pairs)
    assert len(delaunay_edges(xy)[0]) > len(edges[0])


def test_spot_window_edges():
    spots = hex_grid(np.array([[0.0, 0.0], [600.0, 600.0]]), 100.0)
    edges = spot_window_edges(spots, k=6)
    assert edges.shape == (2, 6 * len(spots))
    assert (edges[0] != edges[1]).all()
    near = spot_window_edges(spots, k=8, px_um=1.0, window_um=101.0)
    assert np.linalg.norm(spots[near[0]] - spots[near[1]], axis=1).max() <= 101.0


def test_tiles_hold_whole_spots(toy):
    sample, _ = toy
    from scipy.spatial import cKDTree

    nearest = cKDTree(sample.spot_xy).query(sample.cell_xy)[1]
    edges = delaunay_edges(sample.cell_xy, sample.px_um, 40.0)
    tiles = spatial_tiles(sample.spot_xy, sample.cell_spot, nearest, edges, sample.counts,
                          n_tiles=3, seed=0)
    assert 1 <= len(tiles) <= 9
    seen = np.concatenate([t["spot_ids"] for t in tiles])
    assert set(seen) == set(range(sample.n_spots))
    for t in tiles:
        # every cell of a spot of the tile is in the tile
        in_tile = np.isin(sample.cell_spot, t["spot_ids"])
        assert np.isin(np.where(in_tile)[0], t["cells"]).all()
        assert t["counts"].shape == (t["n_spots"], sample.n_genes)
        assert t["cell_spot"].max() < t["n_spots"]
        assert t["edge_index"].max() < len(t["cells"])
        inside = t["cell_spot"] >= 0
        assert (t["spot_ids"][t["cell_spot"][inside]] == sample.cell_spot[t["cells"]][inside]).all()


def test_simulated_spots_conserve_counts():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 2000, size=(2000, 2))
    counts = rng.poisson(3.0, size=(2000, 15)).astype(float)
    spots = visium_spots(counts, xy, px_um=0.5)
    inside = spots["cell_spot"] >= 0
    assert np.isclose(spots["counts"].sum(), counts[inside].sum())
    # discs of 55 um on a 100 um pitch cover about 27% of the tissue
    assert 0.20 < inside.mean() < 0.34
    assert spots["cell_spot"].max() == len(spots["counts"]) - 1

    thinned = visium_spots(counts, xy, px_um=0.5, capture=0.2)
    assert 0.1 < thinned["counts"].sum() / spots["counts"].sum() < 0.3
    spilled = visium_spots(counts, xy, px_um=0.5, diffusion_um=10.0)
    assert spilled["counts"].sum() > spots["counts"].sum()

    bins = square_bins(counts, xy, px_um=0.5, bin_um=8.0)
    assert np.isclose(bins["counts"].sum(), counts.sum())
    assert (bins["cell_spot"] >= 0).all()
