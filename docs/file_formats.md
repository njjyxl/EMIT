# File formats

All coordinates are **full-resolution image pixels, x (column) first, y (row) second**.

## Nuclei: `nuclei.npz`

Written by `emit segment`, read by `emit.read_nuclei`.

| key | shape | content |
|---|---|---|
| `xy` | (n, 2) | nucleus centroids |
| `morph` | (n, n_morph) float32 | morphometrics; NaN is allowed and is replaced by the column median at fit time |
| `morph_names` | (n_morph,) | column names |
| `px_um` | scalar | micrometres per pixel of the image that was segmented |
| `hw` | (2,) | height and width of that image |

The morphometrics written by `emit segment` are, in order: `area`, `perimeter`,
`eccentricity`, `solidity`, `major_axis_length`, `minor_axis_length`, `circularity`
(nucleus shape); `hematoxylin`, `eosin`, `gray_mean`, `gray_std`, `texture` (an 8.8 um
window around the centroid); `neigh_density` (nuclei within 30 um).

To use another segmentation, write a file with at least `xy`, `morph` and `px_um`. Any
set of per-cell descriptors can serve as `morph`.

With `--labels labels.npy`, `emit segment` also writes the label image as an int32 array
of the size of the H&E image: 0 is background and nucleus `k` (row `k` of `xy`) has value
`k + 1`.

## Image embeddings: `features.npz`

Written by `emit features`, read by `emit.read_image_features`.

| key | content |
|---|---|
| `feats` | (n, D) float32 or float16, row `i` for nucleus `i` |
| `key` | the cells, pixel size, patch size and model they were computed for |
| `idx` | optional: the nucleus of each row, if rows are not in order |

## Sample: `sample.npz`

Written by `Sample.save`, read by `Sample.load` and `emit fit --sample`. Holds `counts`,
`genes`, `spot_xy`, `cell_xy`, `morph`, `cell_spot`, `px_um`, `spot_diameter_px`, the
names of spots, cells and morphometrics and, if present, `image_features`.

## Anchors: `anchors.npz`

Read by `emit fit --anchors`.

| key | content |
|---|---|
| `rows` | (m,) indices of the anchored cells among the cells of the sample |
| `counts` | (m, genes) their counts, genes in the order of the sample |

## Output of `emit fit`

`cells.h5ad` (AnnData, cells x genes):

| field | content |
|---|---|
| `X` | counts of every cell (float32, dense) |
| `obs["x"]`, `obs["y"]`, `obsm["spatial"]` | cell position |
| `obs["spot"]`, `obs["in_spot"]` | name of the containing spot (empty outside spots) |
| `obs["source"]` | `allocated`: a share of measured spot counts; `predicted`: expected counts of the network |
| `obs["latent_type"]`, `obsm["type_prob"]` | most probable latent type, and the mixture over latent types |
| `obs["library_size"]` | library size predicted by the network |
| `varm["type_profiles"]` | (genes, K) expression profile of each latent type |
| `uns["emit"]` | settings of the run and the pixel size |

Latent types are components of the expression model. They are not annotated cell types;
cluster and annotate the cells from `X` as for any single-cell dataset.

`history.csv`: see [usage.md](usage.md#monitoring-a-fit).

`model.pt`: `torch.load` gives `{"state_dict": ..., "params": ...}` for `emit.EMITNet`.

## Output of `emit hd-cells`

`hd_cells.h5ad`: cells x genes (sparse), `obs["n_bins"]`, `obs["n_counts"]`,
`obsm["spatial"]` (when `--nuclei` is given) and `var["gene_ids"]`. `--owner owner.npy`
additionally writes the owner cell of every bin (`-1` for unassigned).
