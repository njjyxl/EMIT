"""EMIT: single-cell expression from spot-based spatial transcriptomics and histology.

EMIT (emission-tomography mapping of individual-cell transcriptomes) treats each spot as
a set of molecules to be apportioned among the cells that produced them.  The counts of
the cells inside a spot always sum to the measurement of that spot; cells between spots
are predicted from the histology image.

Typical use::

    import emit

    spots = emit.read_visium("path/to/spaceranger/outs")
    nuclei = emit.read_nuclei("nuclei.npz")            # written by `emit segment`
    sample = emit.build_sample(spots, nuclei)
    result = emit.fit(sample)
    result.to_anndata().write_h5ad("cells.h5ad")

``torch`` is imported only when a model is built, so the readers, the simulation and the
Visium HD assignment work without it.
"""
from ._utils import setup_logging
from ._version import __version__
from .io import build_sample, read_image_features, read_nuclei, read_visium, read_visium_hd
from .sample import Sample

_LAZY = {
    "fit": "fitting", "EMITResult": "fitting", "initial_type_profiles": "fitting",
    "EMITNet": "model",
    "e_step": "em", "e_step_diffuse": "em", "m_loss": "em",
}

__all__ = [
    "EMITNet", "EMITResult", "Sample", "__version__", "build_sample", "e_step", "e_step_diffuse",
    "fit", "initial_type_profiles", "m_loss", "read_image_features", "read_nuclei", "read_visium",
    "read_visium_hd", "setup_logging",
]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
