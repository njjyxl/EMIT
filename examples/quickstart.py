"""EMIT on a synthetic section, scored against the known expression of every cell.

    python examples/quickstart.py
"""
import numpy as np

import emit
from emit.metrics import conservation_error, gene_correlation, within_spot_r
from emit.simulate import toy_section


def main():
    emit.setup_logging()
    sample, truth = toy_section(n_cells=6000, n_genes=200, seed=0)
    print(sample.check_registration())

    result = emit.fit(sample, rounds=8, seed=0)

    inside = sample.inside
    allocated = np.where(result.allocated[:, None], result.expression, 0)
    print(f"\nlargest relative difference between a spot and the sum of its cells: "
          f"{np.nanmax(conservation_error(allocated, sample.counts, sample.cell_spot)):.1e}")
    print(f"within-spot correlation with the truth: "
          f"{within_spot_r(result.expression, truth['counts'], sample.cell_spot):.3f} "
          f"(a copy of the spot profile scores 0)")
    for name, cells in (("inside spots", inside), ("between spots", ~inside)):
        r = gene_correlation(result.expression[cells], truth["counts"][cells])
        print(f"median gene correlation, cells {name}: {np.nanmedian(r):.3f}")

    adata = result.to_anndata()
    print(f"\n{adata}")


if __name__ == "__main__":
    main()
