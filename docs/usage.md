# Using EMIT

## How the method works

For every cell `i` the network predicts an expression profile `pi[i]` (a probability
vector over genes) and a library size `lam[i]`; the expected count of gene `g` is
`lam[i] * pi[i, g]`. The profile is a mixture of `K` latent type profiles `R` plus a
low-dimensional state term:

```
pi[i] = softmax( log(p[i] @ R) + v[i] @ U )
```

One EM round consists of

- **E-step.** With the count of gene `g` in cell `i` modelled as
  `Poisson(lam[i] * pi[i, g])`, the counts of the cells of a spot given the spot total
  are multinomial. Each cell receives its expected share
  `y[b, g] * lam[i] * pi[i, g] / sum_j lam[j] * pi[j, g]`, the sum running over the cells
  of spot `b`. The allocated counts of a spot sum exactly to its measurement.
- **M-step.** The network is trained on the allocated counts cell by cell: a multinomial
  log-likelihood, a term tying `lam` to the allocated total, a term that keeps the
  profiles from collapsing, and a *drift* term that limits how far a cell may move from
  the mean profile of its spot.

After the last round, the allocation is the output for cells inside spots. Cells outside
every spot receive the expected counts of the network, `lam[i] * pi[i]`.

The cell tokens are built from the morphometrics and, if provided, the image embedding of
each cell. Cells exchange information over a Delaunay graph (edges up to 40 um) and,
through their spot, with the eight neighbouring spots.

## Inputs

| input | how to obtain it |
|---|---|
| spot counts and positions | `emit.read_visium(dir)` from Space Ranger output, or your own arrays |
| nuclei and morphometrics | `emit segment`, or any segmentation saved in the same format |
| image embeddings (optional) | `emit features` |

Everything is combined in an `emit.Sample`. It can also be built directly:

```python
sample = emit.Sample(
    counts=counts,            # (spots, genes) raw counts
    genes=genes,
    spot_xy=spot_xy,          # full-resolution pixels, x first
    cell_xy=cell_xy,
    morph=morph,              # (cells, n_morph)
    px_um=0.345,              # micrometres per pixel
    spot_diameter_px=160.0,
    image_features=None,      # or (cells, D)
)
sample.check_registration()   # fraction of cells inside spots, cells per spot, ...
sample.save("sample.npz")     # usable with `emit fit --sample sample.npz`
```

`cell_spot` (the spot containing each cell, `-1` for none) is computed from the geometry
unless you pass it. On a Visium array about a quarter to a third of the cells lie inside a
spot. If `check_registration()` reports almost none, or a median distance to the nearest
spot of several spot diameters, the cells and the spots are not in the same coordinate
frame.

### Pixel size

`read_visium` derives the pixel size from the spot pitch, which is 100 um by design, and
uses the fitted `spot_diameter_fullres` only to decide which cells a spot contains. The
fitted diameter often differs from the nominal 55 um, so deriving the scale from it would
shift every length defined in micrometres. For arrays with another pitch pass `pitch_um`
(and `spot_um`).

## Monitoring a fit

`result.history` (written as `history.csv`) has one row per round.

| column | meaning |
|---|---|
| `heldout_reldist` | relative distance between the counts of the withheld spots and the cells predicted inside them; lower is better |
| `baseline_nearest`, `baseline_mean` | the same distance for the nearest training spot and for the mean of all training spots, which use no image |
| `drift` | mean KL divergence of a cell's profile from the mean profile of its spot |
| `concentration` | 0 when the counts of a spot are spread evenly over its cells, 1 when one cell takes them all |
| `within_spot_share` | fraction of compositional variance that lies within spots |
| `loss_*` | the terms of the training objective |

Held-out reconstruction measures spot-level fit and cannot see how counts are arranged
*within* a spot. `drift` describes that arrangement: if it keeps rising from round to
round while `heldout_reldist` barely changes, the fit is over-training and `--w-drift`
should be raised (or the number of rounds reduced). With the default `--w-drift 0.1` this
did not occur on the sections analysed in the paper.

Evaluating the whole section after every round costs time on large sections.
`--no-monitor` (`monitor=False`) evaluates only the final round; the result is identical.

If per-cell truth is available (a benchmark with simulated spots), a callback can score
every round:

```python
from emit.metrics import within_spot_r

result = emit.fit(sample, callback=lambda rd, expression, allocated:
                  {"r": within_spot_r(expression, truth, sample.cell_spot)})
```

## Diffusion-aware fitting

Transcripts can diffuse before capture and be recorded by a neighbouring spot. With
`diffusion_um=sigma` the E-step lets a cell at distance `d` from the centre of a spot of
radius `r` contribute to that spot with weight `exp(-0.5 * (max(d - r, 0) / sigma)^2)`.
A cell may then draw counts from several spots, and cells just outside a spot receive an
allocation too (`result.allocated` marks them), while the counts of each spot are still
distributed exactly.

`sigma` is not known on a real section. It can be chosen by fitting a few values and
comparing `heldout_reldist`.

## Visium HD

`emit.hd` assigns bins to cells from a label image of the nuclei:

```python
from emit import read_visium_hd
from emit.hd import nuclei_on_bins, bins_to_cells

bins = read_visium_hd("binned_outputs/square_002um")
bin_nucleus = nuclei_on_bins(labels, bins["bin_xy"], centroids_px=nuclei["xy"])
cells, owner = bins_to_cells(bins["counts"], bins["bin_xy"] * bins["px_um"], bin_nucleus,
                             n_cells=len(nuclei["xy"]), d_max_um=4.0)
```

`labels` is the label image written by `emit segment --labels`, in which nucleus `k` has
value `k + 1`. `cells` is a sparse cells x genes matrix.

### HD-derived cells as anchors

When part of the tissue of a standard Visium section has per-cell counts, for example
cells assigned from Visium HD bins of the same block registered to the same image, they
can be added to the objective:

```python
result = emit.fit(sample, anchors=(rows, anchor_counts))   # rows index sample cells
```

`anchor_counts` must have the genes of `sample` in the same order. On the command line,
`--anchors anchors.npz` reads `rows` and `counts` from a file. Anchors enter the M-step
only; the allocation and its exact conservation are unchanged.

## Benchmarking against per-cell truth

`emit.simulate.visium_spots` aggregates the cells of an imaging-based section into
Visium-like spots (optionally with a capture efficiency and lateral diffusion), so that a
per-cell output can be scored against the cells it was never shown:

```python
from emit.simulate import visium_spots
from emit.metrics import within_spot_r, gene_correlation, conservation_error

spots = visium_spots(cell_counts, cell_xy, px_um)
sample = emit.Sample(counts=spots["counts"], genes=genes, spot_xy=spots["spot_xy"],
                     cell_xy=cell_xy, morph=morph, px_um=px_um,
                     spot_diameter_px=spots["spot_diameter_px"], cell_spot=spots["cell_spot"])
result = emit.fit(sample)
within_spot_r(result.expression, cell_counts, sample.cell_spot)
```

`within_spot_r` removes the mean of each spot before correlating, so an output that gives
every cell the profile of its spot scores zero. Score the cells inside spots and the cells
between spots separately: the first are allocations, the second predictions.

## Reproducibility and performance

- `seed` fixes every random number generator. On the CPU, two runs with the same seed
  agree to floating-point precision; on a GPU, scatter operations are not deterministic
  and results agree to about six significant digits.
- Runtime is dominated by the number of cells and genes: a few minutes for a section of
  some 10^5 cells and 2,000 genes on a single GPU.
- `eval_placement` controls where the cells x genes profile matrix is built (`0` GPU,
  `1` host memory, `2` CPU only). The default picks from its size; the numbers are the
  same at every level.
