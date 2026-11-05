import pytest

from emit.simulate import toy_section


@pytest.fixture(scope="session")
def toy():
    """A small synthetic section and its per-cell truth."""
    return toy_section(n_cells=3000, n_genes=80, n_types=4, width_um=1000.0, seed=0)
