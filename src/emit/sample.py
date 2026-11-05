"""The input of EMIT: the spots of one section and the cells seen in its histology image."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ._utils import as_str_array, logger
from .geometry import assign_cells_to_spots

__all__ = ["Sample"]

_ARRAYS = ("counts", "genes", "spot_xy", "cell_xy", "morph", "cell_spot", "image_features",
           "morph_names", "spot_names", "cell_names")


@dataclass
class Sample:
    """One section: spot counts plus the position and appearance of every cell.

    Coordinates are full-resolution image pixels, ``x`` (column) first.

    Parameters
    ----------
    counts : (n_spots, n_genes) array
        Raw spot counts.
    genes : (n_genes,) array of str
    spot_xy : (n_spots, 2) array
        Spot centres.
    cell_xy : (n_cells, 2) array
        Cell (nucleus) centroids.
    morph : (n_cells, n_morph) array
        Morphometrics of each cell.  Non-finite values are replaced by the column median
        when the sample is fitted.
    px_um : float
        Micrometres per pixel.
    spot_diameter_px : float
        Diameter of the capture area of a spot.
    cell_spot : (n_cells,) int array, optional
        Spot containing each cell, ``-1`` for none.  Computed from the geometry if omitted.
    image_features : (n_cells, D) array, optional
        Embedding of an image patch around each cell from a frozen pathology model.
    """

    counts: np.ndarray
    genes: np.ndarray
    spot_xy: np.ndarray
    cell_xy: np.ndarray
    morph: np.ndarray
    px_um: float
    spot_diameter_px: float
    cell_spot: np.ndarray | None = None
    image_features: np.ndarray | None = None
    morph_names: np.ndarray | None = None
    spot_names: np.ndarray | None = None
    cell_names: np.ndarray | None = None
    name: str = "sample"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.counts = np.asarray(self.counts, np.float32)
        self.genes = as_str_array(self.genes)
        self.spot_xy = np.asarray(self.spot_xy, float)
        self.cell_xy = np.asarray(self.cell_xy, float)
        self.morph = np.asarray(self.morph, np.float32)
        self.px_um = float(self.px_um)
        self.spot_diameter_px = float(self.spot_diameter_px)
        if self.morph.ndim == 1:
            self.morph = self.morph[:, None]
        if self.cell_spot is None:
            self.cell_spot = assign_cells_to_spots(self.cell_xy, self.spot_xy,
                                                   self.spot_diameter_px / 2)
        self.cell_spot = np.asarray(self.cell_spot, np.int64)
        if self.image_features is not None:
            self.image_features = np.asarray(self.image_features)
        if self.morph_names is None:
            self.morph_names = np.array([f"morph{i}" for i in range(self.morph.shape[1])],
                                        dtype=object)
        if self.spot_names is None:
            self.spot_names = np.array([f"spot{i}" for i in range(len(self.counts))], dtype=object)
        if self.cell_names is None:
            self.cell_names = np.array([f"cell{i}" for i in range(len(self.cell_xy))], dtype=object)
        self.morph_names = as_str_array(self.morph_names)
        self.spot_names = as_str_array(self.spot_names)
        self.cell_names = as_str_array(self.cell_names)
        self.validate()

    # ------------------------------------------------------------------ description
    @property
    def n_cells(self) -> int:
        return len(self.cell_xy)

    @property
    def n_spots(self) -> int:
        return len(self.counts)

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]

    @property
    def inside(self) -> np.ndarray:
        """Cells whose centroid lies inside a spot."""
        return self.cell_spot >= 0

    @property
    def spot_radius_um(self) -> float:
        return 0.5 * self.spot_diameter_px * self.px_um

    def validate(self) -> None:
        """Raise ``ValueError`` if the arrays are inconsistent with each other."""
        n_spots, n_genes = self.counts.shape
        checks = [
            (len(self.genes) == n_genes, "genes does not match the columns of counts"),
            (self.spot_xy.shape == (n_spots, 2), "spot_xy must be (n_spots, 2)"),
            (self.cell_xy.ndim == 2 and self.cell_xy.shape[1] == 2, "cell_xy must be (n_cells, 2)"),
            (len(self.morph) == self.n_cells, "morph does not match the number of cells"),
            (len(self.cell_spot) == self.n_cells, "cell_spot does not match the number of cells"),
            (self.cell_spot.max(initial=-1) < n_spots, "cell_spot refers to a missing spot"),
            (self.px_um > 0 and self.spot_diameter_px > 0,
             "px_um and spot_diameter_px must be positive"),
            (len(self.spot_names) == n_spots, "spot_names does not match the number of spots"),
            (len(self.cell_names) == self.n_cells, "cell_names does not match the number of cells"),
            (len(self.morph_names) == self.morph.shape[1], "morph_names does not match morph"),
            (self.image_features is None or len(self.image_features) == self.n_cells,
             "image_features does not match the number of cells"),
            (bool(np.isfinite(self.counts).all()) and float(self.counts.min(initial=0)) >= 0,
             "counts must be finite and non-negative"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        if n_spots < 16:
            raise ValueError(f"only {n_spots} spots; EMIT needs a section, not a handful of spots")
        if not self.inside.any():
            raise ValueError(
                "no cell lies inside a spot: check that cells and spots share one coordinate frame")

    def check_registration(self) -> dict:
        """Summary statistics that reveal cells and spots in different coordinate frames."""
        from scipy.spatial import cKDTree

        dist = cKDTree(self.spot_xy).query(self.cell_xy, k=1)[0]
        per_spot = np.bincount(self.cell_spot[self.inside], minlength=self.n_spots)
        stats = {
            "fraction_of_cells_inside_spots": float(self.inside.mean()),
            "median_cells_per_spot": float(np.median(per_spot)),
            "fraction_of_empty_spots": float((per_spot == 0).mean()),
            "median_cell_to_spot_um": float(np.median(dist) * self.px_um),
        }
        if np.median(dist) > self.spot_diameter_px:
            logger.warning(
                "%s: the median cell is %.0f um from the nearest spot centre; cells and spots "
                "may not share a coordinate frame", self.name, stats["median_cell_to_spot_um"])
        return stats

    def summary(self) -> str:
        features = "none" if self.image_features is None else self.image_features.shape[1]
        return (f"{self.name}: {self.n_cells:,} cells ({int(self.inside.sum()):,} inside spots), "
                f"{self.n_spots:,} spots, {self.n_genes:,} genes, {self.px_um:.4f} um/px, "
                f"spot diameter {self.spot_diameter_px * self.px_um:.0f} um, "
                f"image features: {features}")

    # ------------------------------------------------------------------ files
    def save(self, path: str) -> str:
        """Write the sample to one compressed ``.npz`` file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        arrays = {k: getattr(self, k) for k in _ARRAYS if getattr(self, k) is not None}
        np.savez_compressed(path, px_um=self.px_um, spot_diameter_px=self.spot_diameter_px,
                            name=self.name, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> Sample:
        """Read a sample written by :meth:`save`."""
        with np.load(path, allow_pickle=True) as z:
            kwargs = {k: z[k] for k in _ARRAYS if k in z.files}
            return cls(px_um=float(z["px_um"]), spot_diameter_px=float(z["spot_diameter_px"]),
                       name=str(z["name"]) if "name" in z.files else "sample", **kwargs)
