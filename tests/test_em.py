import numpy as np
import pytest
import torch

from emit.em import e_step, e_step_diffuse, m_loss
from emit.geometry import capture_pairs


def _random_problem(n_cells=200, n_spots=12, n_genes=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    pi = torch.softmax(torch.randn(n_cells, n_genes, generator=g), dim=-1)
    lam = torch.rand(n_cells, generator=g) * 100 + 1
    cell_spot = torch.randint(-1, n_spots, (n_cells,), generator=g)
    counts = torch.poisson(torch.rand(n_spots, n_genes, generator=g) * 20)
    return pi, lam, cell_spot, counts


def test_e_step_returns_the_measurement_exactly():
    pi, lam, cell_spot, counts = _random_problem()
    allocated, inside = e_step(pi, lam, cell_spot, counts)
    total = torch.zeros_like(counts).index_add_(0, cell_spot[inside], allocated)
    occupied = torch.bincount(cell_spot[inside], minlength=len(counts)) > 0
    assert torch.allclose(total[occupied], counts[occupied], rtol=1e-5, atol=1e-4)
    assert bool((allocated >= 0).all())
    assert inside.sum() == (cell_spot >= 0).sum()


def test_e_step_shares_follow_the_expected_counts():
    # two cells in one spot, one gene: the split is lam * pi
    pi = torch.tensor([[1.0], [1.0]])
    lam = torch.tensor([1.0, 3.0])
    allocated, _ = e_step(pi, lam, torch.tensor([0, 0]), torch.tensor([[8.0]]))
    assert torch.allclose(allocated.ravel(), torch.tensor([2.0, 6.0]))


def test_e_step_without_covered_cells():
    pi, lam, _, counts = _random_problem()
    allocated, inside = e_step(pi, lam, torch.full((len(lam),), -1), counts)
    assert allocated is None and not bool(inside.any())


def test_shuffled_allocation_keeps_spot_totals():
    pi, lam, cell_spot, counts = _random_problem()
    g = torch.Generator().manual_seed(1)
    plain, inside = e_step(pi, lam, cell_spot, counts)
    shuffled, _ = e_step(pi, lam, cell_spot, counts, shuffle=True, generator=g)
    spot = cell_spot[inside]
    a = torch.zeros_like(counts).index_add_(0, spot, plain)
    b = torch.zeros_like(counts).index_add_(0, spot, shuffled)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-3)
    assert not torch.allclose(plain, shuffled)


def test_diffuse_e_step_distributes_each_spot_exactly():
    rng = np.random.default_rng(0)
    cell_xy = rng.uniform(0, 400, size=(300, 2))
    spot_xy = np.array([[100.0, 100.0], [300.0, 100.0], [100.0, 300.0], [300.0, 300.0]])
    ci, si, w = capture_pairs(cell_xy, spot_xy, px_um=1.0, spot_radius_um=27.5, sigma_um=20.0)
    assert len(ci) and w.max() <= 1.0 and w.min() > 1e-3
    pi, lam, _, _ = _random_problem(n_cells=300, n_genes=10)
    counts = torch.poisson(torch.full((4, 10), 30.0))
    allocated = e_step_diffuse(pi, lam, torch.as_tensor(ci), torch.as_tensor(si),
                               torch.as_tensor(w), counts, chunk=97)
    assert allocated.shape == (300, 10)
    # the counts of every spot with a contributing cell are distributed completely
    assert torch.allclose(allocated.sum(0), counts.sum(0), rtol=1e-4)
    untouched = np.setdiff1d(np.arange(300), ci)
    assert float(allocated[untouched].abs().sum()) == 0.0


def test_diffuse_e_step_reduces_to_the_plain_one_without_spill():
    pi, lam, cell_spot, counts = _random_problem()
    inside = cell_spot >= 0
    cells = torch.nonzero(inside).ravel()
    plain, _ = e_step(pi, lam, cell_spot, counts)
    diffuse = e_step_diffuse(pi, lam, cells, cell_spot[inside], torch.ones(len(cells)), counts)
    assert torch.allclose(diffuse[inside], plain, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("w_drift", [0.0, 0.1])
def test_m_loss_is_finite_and_differentiable(w_drift):
    pi, lam, cell_spot, counts = _random_problem()
    logits = torch.randn(pi.shape, requires_grad=True)
    log_lam = torch.zeros(len(lam), requires_grad=True)
    allocated, inside = e_step(pi, lam, cell_spot, counts)
    loss, parts = m_loss(torch.softmax(logits, -1), log_lam, allocated, inside,
                         cell_spot=cell_spot, n_spots=len(counts), w_drift=w_drift)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert set(parts) == {"ce", "depth", "spread", "drift"}
    assert (parts["drift"] > 0) == (w_drift > 0)
