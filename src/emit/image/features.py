"""Embedding of an image patch around each cell with a frozen pathology foundation model.

The model is never trained here: the embeddings are computed once, written to disk and
reused by every fit.
"""
from __future__ import annotations

import os

import numpy as np

from .._utils import logger, resolve_device

__all__ = ["embed_cells"]


def _key(xy, px_um, patch_um, model) -> dict:
    return {"n": len(xy), "patch_um": float(patch_um), "model": str(model),
            "px_um": round(float(px_um), 6), "xy_hash": int(np.abs(xy).sum() * 1000) % (2 ** 31)}


def embed_cells(image, xy, px_um: float, patch_um: float = 26.0, model: str = "owkin/phikon",
                batch_size: int = 128, cache: str | None = None, device=None,
                half_precision: bool = False) -> np.ndarray:
    """Class-token embedding of a square patch centred on every cell.

    Parameters
    ----------
    image : (H, W, 3) uint8 array
        The full-resolution H&E image.
    xy : (n_cells, 2) array
        Cell centroids in pixels, ``x`` first.
    px_um
        Micrometres per pixel.  The patch is defined in micrometres, so it covers the
        same amount of tissue on slides scanned at different resolutions.
    patch_um
        Side of the patch.
    model
        A Hugging Face model name or a local directory (default: Phikon, a ViT-B).
    cache
        Path of an ``.npz`` file.  It is reused only if it was computed for the same
        cells, pixel size, patch size and model.
    half_precision
        Store the cache as float16, which halves its size.

    Returns
    -------
    (n_cells, D) float32 array
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('image embeddings need transformers: '
                          'pip install "emit-spatial[features]"') from exc

    xy = np.asarray(xy, float)
    key = _key(xy, px_um, patch_um, model)
    if cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=True) as z:
            if all(z["key"].item().get(k) == v for k, v in key.items()):
                logger.info("reusing the embeddings in %s", cache)
                return z["feats"].astype(np.float32)
        logger.info("%s was computed with other settings; recomputing", cache)

    dev = resolve_device(device)
    net = AutoModel.from_pretrained(model).to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(model)
    size = processor.size.get("shortest_edge") or processor.size.get("height") or 224
    mean = torch.tensor(processor.image_mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=dev).view(1, 3, 1, 1)

    patch = max(2, int(round(patch_um / px_um)) // 2 * 2)
    half = patch // 2
    height, width = image.shape[:2]
    logger.info("%s on %s | patch %g um = %d px -> %d px input", model, dev, patch_um, patch, size)
    feats = None
    with torch.no_grad():
        for start in range(0, len(xy), batch_size):
            stop = min(start + batch_size, len(xy))
            crops = np.zeros((stop - start, patch, patch, 3), np.uint8)
            for j, i in enumerate(range(start, stop)):
                x, y = int(xy[i, 0]), int(xy[i, 1])
                r0 = int(np.clip(y - half, 0, max(0, height - patch)))
                c0 = int(np.clip(x - half, 0, max(0, width - patch)))
                crop = image[r0:r0 + patch, c0:c0 + patch]
                crops[j, :crop.shape[0], :crop.shape[1]] = crop
            t = torch.from_numpy(crops).to(dev).permute(0, 3, 1, 2).float() / 255.0
            t = (F.interpolate(t, size=size, mode="bilinear", align_corners=False) - mean) / std
            f = net(pixel_values=t).last_hidden_state[:, 0].float().cpu().numpy()
            if feats is None:
                feats = np.zeros((len(xy), f.shape[1]), np.float32)
            feats[start:stop] = f
            if (start // batch_size) % 50 == 0:
                logger.info("  %s / %s cells", f"{stop:,}", f"{len(xy):,}")
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        np.savez(cache, feats=feats.astype(np.float16 if half_precision else np.float32), key=key)
        logger.info("embeddings -> %s", cache)
    return feats
