"""Cells from the histology image: nucleus segmentation, morphometrics and embeddings.

These steps need optional dependencies (``pip install "emit-spatial[segment,features]"``)
and are imported lazily, so the rest of the package works without them.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["read_image", "segment_nuclei", "morphometrics", "tissue_mask", "embed_cells"]


def read_image(path: str) -> np.ndarray:
    """Read the full-resolution H&E image as an ``(H, W, 3)`` uint8 array.

    TIFF files (``.tif``, ``.tiff``, ``.btf``) are read with ``tifffile`` at their first,
    full-resolution level; other formats with Pillow.  A compressed TIFF that ``tifffile``
    cannot decode without the ``imagecodecs`` package is read with Pillow instead.
    """
    img = None
    if os.path.splitext(path)[1].lower() in (".tif", ".tiff", ".btf"):
        import tifffile

        try:
            with tifffile.TiffFile(path) as tif:
                img = tif.pages[0].asarray()
        except ValueError as exc:  # compression needs imagecodecs
            if "imagecodecs" not in str(exc):
                raise
            try:
                img = _read_with_pillow(path)
            except Exception:
                message = f"{path}: {exc}; install it with `pip install imagecodecs`"
                raise ValueError(message) from exc
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)  # channels first
    else:
        img = _read_with_pillow(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.ascontiguousarray(img[..., :3])


def _read_with_pillow(path: str) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path).convert("RGB"))


def __getattr__(name):
    if name in ("segment_nuclei", "morphometrics", "tissue_mask"):
        from . import segment

        return getattr(segment, name)
    if name == "embed_cells":
        from .features import embed_cells

        return embed_cells
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
