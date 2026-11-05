# EMIT

**E**mission-tomography **m**apping of **i**ndividual-cell **t**ranscriptomes: single-cell
expression from spot-based spatial transcriptomics and the matched histology image.

A Visium spot counts the molecules of the handful of cells it covers without telling them
apart. EMIT treats each spot as a set of molecules to be *apportioned* among the cells
that produced them, the way emission tomography assigns detected events to the sources it
cannot resolve, with the H&E image guiding the reconstruction.

- **Inside spots, counts are allocated, not predicted.** The counts of the cells of a spot
  always sum exactly to what the spot measured.
- **Between spots, cells are predicted** from the histology by the network trained on the
  allocated counts. These cells are flagged, because no measurement constrains them.
- **No reference is needed**: no single-cell atlas, cell-type labels or per-cell training
  data. The model is fitted on the section itself.
- **Visium HD**: subcellular bins are assigned to single cells, and the resulting cells
  can anchor the fit of a standard Visium section.

## Installation

```bash
git clone <repository-url> EMIT
cd EMIT
pip install -e .                      # core: fitting, readers, Visium HD, simulation
pip install -e ".[segment,features]"  # + StarDist segmentation and image embeddings
pip install -e ".[all,dev]"           # everything, with the test tools
```

Python 3.9 or later and PyTorch 2.0 or later are required. Install the PyTorch build
that matches your CUDA version first (see <https://pytorch.org>); EMIT also runs on the
CPU.

StarDist depends on TensorFlow. If TensorFlow and PyTorch conflict on your system, run
`emit segment` from a separate environment: that command does not import PyTorch, so
`pip install --no-deps -e .` plus `numpy scipy pandas h5py stardist tensorflow
scikit-image tifffile pillow` is enough there. If `import tensorflow` fails with a NumPy
error, that TensorFlow build needs `numpy<2`.

## Quick start

A synthetic section is included so that the whole pipeline can be tried in a minute:

```bash
emit toy --out toy.npz
emit fit --sample toy.npz --out toy_out --rounds 5
```

```python
import emit
from emit.simulate import toy_section
from emit.metrics import within_spot_r

sample, truth = toy_section()
result = emit.fit(sample, rounds=5)

result.expression      # (cells, genes) counts of every cell
result.allocated       # True for cells whose counts were allocated from a spot
result.type_prob       # mixture of every cell over the latent types
result.history         # held-out reconstruction and diagnostics per round
within_spot_r(result.expression, truth["counts"], sample.cell_spot)
```

## A Visium section

EMIT needs three things: the Space Ranger output, the nuclei seen in the H&E image, and,
optionally, an image embedding of every cell.

```bash
# 1. nuclei and their morphometrics (StarDist, confined to the spot lattice)
#    --model takes the name of a pretrained StarDist model (downloaded on first use)
#    or the directory of a model on disk; the default is 2D_versatile_he
emit segment --image tissue_image.tif --spaceranger outs/ --out nuclei.npz

# 2. optional: embedding of a 26 um patch around every cell (Phikon by default)
emit features --image tissue_image.tif --nuclei nuclei.npz --out features.npz

# 3. fit
emit fit --spaceranger outs/ --nuclei nuclei.npz --features features.npz --out emit_out
```

`emit_out/` then holds

| file | content |
|---|---|
| `cells.h5ad` | cells x genes. `obs`: position, spot, `in_spot`, `source` (`allocated` or `predicted`), `latent_type`, `library_size`; `obsm["spatial"]`, `obsm["type_prob"]`; `varm["type_profiles"]` |
| `history.csv` | one row per round: loss terms, held-out reconstruction against two image-free baselines, and diagnostics |
| `model.pt` | weights of the fitted network and the settings of the run |

The same in Python:

```python
import emit

emit.setup_logging()
spots = emit.read_visium("outs/")                 # 2,000 most variable genes by default
nuclei = emit.read_nuclei("nuclei.npz")
sample = emit.build_sample(spots, nuclei, emit.read_image_features("features.npz"))
result = emit.fit(sample)
adata = result.to_anndata()
```

Any segmentation can be used in place of `emit segment`: write an `.npz` file with `xy`
(centroids in full-resolution pixels, x first), `morph` (one row of morphometrics per
nucleus), `morph_names` and `px_um`. See [docs/file_formats.md](docs/file_formats.md).

### Reading the result

- Use `source == "allocated"` to select the cells whose expression is tied to a
  measurement; treat `predicted` cells as predictions.
- `history.csv` reports how well the cells predicted inside spots that were *withheld*
  from training reproduce the counts of those spots (`heldout_reldist`, lower is
  better). Read it against `baseline_nearest` and `baseline_mean` in the same row.
- A steadily rising `drift` signals that training is moving cells away from what was
  measured; see [docs/usage.md](docs/usage.md).

## Visium HD

```bash
emit segment --image tissue_image.btf --spaceranger binned_outputs/square_002um --hd \
             --out nuclei.npz --labels labels.npy
emit hd-cells --bins binned_outputs/square_002um --labels labels.npy --nuclei nuclei.npz \
              --out hd_cells.h5ad
```

A bin inside a nucleus belongs to that cell; any other bin goes to the nearest nucleus
within `--d-max-um` (4 um by default) or stays unassigned, so every transcript is assigned
to one cell or to none. Cells obtained this way can anchor a standard Visium fit of the
same tissue through `emit fit --anchors` (see [docs/usage.md](docs/usage.md)).

## Options worth knowing

| option | default | meaning |
|---|---|---|
| `--rounds` | 20 | EM rounds |
| `--n-types`, `--n-states` | 10, 20 | latent types and state dimensions of the expression model |
| `--w-drift` | 0.1 | limits how far a cell may move from the mean profile of its spot |
| `--holdout` | 0.15 | fraction of spots withheld to monitor the fit |
| `--diffusion-um` | off | width of lateral diffusion; enables the diffusion-aware E-step |
| `--n-top-genes` | 2000 | most variable genes kept (`0` keeps all) |
| `--no-between` | off | do not predict the cells outside every spot |
| `--device` | auto | `cuda` or `cpu` |

The defaults are the settings used for all analyses in the paper. `emit <command> --help`
lists every option.

**Memory.** The output is a dense cells x genes matrix (about 1 GB for 135,000 cells and
2,000 genes). When it does not fit on the GPU it is assembled in host memory
automatically, with the network still running on the GPU.

## Package layout

```
src/emit/
  sample.py      Sample: the spots and cells of one section
  io.py          Space Ranger and Visium HD readers
  geometry.py    spot membership, cell graph, spot graph, training tiles
  model.py       EMITNet: cell encoder, graph attention, spot context, decoder
  em.py          E-step (exact and diffusion-aware) and the training objective
  fitting.py     fit() and EMITResult
  hd.py          Visium HD bins -> single cells
  simulate.py    spots simulated from single-cell-resolution data; toy section
  metrics.py     accuracy against per-cell truth, conservation, held-out reconstruction
  image/         nucleus segmentation, morphometrics, image embeddings
  cli.py         the `emit` command
tests/           pytest suite on synthetic data
docs/            usage notes and file formats
examples/        a runnable example on synthetic data
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

The manuscript describing EMIT is in preparation; citation details will be added here on
publication.
