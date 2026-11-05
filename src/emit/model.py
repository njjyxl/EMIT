"""The EMIT network: cell tokens, graph attention among cells and spots, and the decoder.

For every cell the network predicts an expression profile ``pi`` (a probability vector
over genes) and a library size ``lam``; the expected count of gene ``g`` in cell ``i`` is
``lam[i] * pi[i, g]``.  The profile is a mixture of ``K`` latent type profiles plus a
low-dimensional state term that lets expression vary within a type::

    pi[i] = softmax( log(p[i] @ R) + v[i] @ U )

Graph attention is written edge-wise with scatter reductions, so the package does not
depend on a graph library and the cost is linear in the number of edges.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EMITNet", "CellEncoder", "CellGraphEncoder", "SpotContext", "CellHeads", "SparseGAT"]


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of ``src`` within the groups given by ``index``, stable against overflow."""
    peak = src.new_full((n,), float("-inf"))
    peak = peak.scatter_reduce(0, index, src, reduce="amax", include_self=True)
    ex = (src - peak[index]).exp()
    den = torch.zeros(n, dtype=src.dtype, device=src.device).index_add_(0, index, ex)
    return ex / den[index].clamp_min(1e-12)


class SparseGAT(nn.Module):
    """Multi-head graph attention over an explicit edge list.

    ``edge_index`` is ``(2, E)`` with the receiving node in row 0 and the sending node in
    row 1.  Every node also receives a self-loop, so its own token survives aggregation.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        loop = torch.arange(n, device=x.device)
        edges = torch.cat([edge_index, torch.stack([loop, loop])], dim=1)
        dst, src = edges[0], edges[1]

        h = self.lin(x).view(n, self.heads, self.out_dim)
        # Reduce over the feature dimension before gathering by edge: the per-edge tensor
        # is then (E, heads) instead of (E, heads, out_dim).
        score_src = (h * self.att_src).sum(-1)
        score_dst = (h * self.att_dst).sum(-1)
        score = F.leaky_relu(score_src[src] + score_dst[dst], 0.2)

        out = x.new_zeros(n, self.heads, self.out_dim)
        for k in range(self.heads):  # one head at a time keeps peak memory at O(E)
            alpha = self.drop(scatter_softmax(score[:, k], dst, n))
            out[:, k].index_add_(0, dst, h[src, k] * alpha.unsqueeze(-1))
        return out.reshape(n, self.heads * self.out_dim) + self.bias


class CellGraphEncoder(nn.Module):
    """Residual graph-attention layers over the cell graph.

    Kept shallow: aggregation over many hops makes the cells of a spot converge on one
    representation, which would erase the within-spot differences to be recovered.
    """

    def __init__(self, dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by the number of heads")
        self.layers = nn.ModuleList(
            [SparseGAT(dim, dim // heads, heads, dropout) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for gat, norm in zip(self.layers, self.norms):
            x = norm(x + F.silu(gat(x, edge_index)))
        return x


def pool_to_spots(x: torch.Tensor, cell_spot: torch.Tensor, n_spots: int) -> torch.Tensor:
    """Mean of the cell tokens of each spot; cells outside every spot contribute nothing."""
    inside = cell_spot >= 0
    out = x.new_zeros(n_spots, x.shape[1])
    n = x.new_zeros(n_spots)
    out.index_add_(0, cell_spot[inside], x[inside])
    n.index_add_(0, cell_spot[inside],
                 torch.ones(int(inside.sum()), device=x.device, dtype=x.dtype))
    return out / n.clamp_min(1.0).unsqueeze(-1)


class SpotContext(nn.Module):
    """Context beyond a cell's neighbours, computed at spot resolution.

    Cell tokens are mean-pooled into their spot, spots attend to their neighbouring spots,
    and the result is returned to each cell through a gate.  Cells outside every spot
    receive the context of their nearest spot.
    """

    def __init__(self, dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by the number of heads")
        self.fp_layers = nn.ModuleList(
            [SparseGAT(dim, dim // heads, heads, dropout) for _ in range(layers)])
        self.fp_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())

    def forward(self, x, cell_spot, n_spots: int, spot_edges, nearest_spot=None):
        cell_spot = torch.as_tensor(cell_spot, device=x.device, dtype=torch.long)
        spots = pool_to_spots(x, cell_spot, n_spots)
        for gat, norm in zip(self.fp_layers, self.fp_norms):
            spots = norm(spots + F.silu(gat(spots, spot_edges)))
        if nearest_spot is not None:
            nearest = torch.as_tensor(nearest_spot, device=x.device, dtype=torch.long)
            take = torch.where(cell_spot >= 0, cell_spot, nearest)
        else:
            take = cell_spot.clamp_min(0)
        context = spots[take]
        gate = self.gate(torch.cat([x, context], dim=-1))
        return self.out(x + gate * context)


class CellEncoder(nn.Module):
    """One token per cell from its image features and its morphometrics.

    The two views are projected separately, each with its own normalization, and then
    fused.  Morphometrics are standardized with statistics fixed once by
    :meth:`fit_morph_stats`.
    """

    def __init__(self, feature_dim: int, morph_dim: int, out_dim: int = 256, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.fm_proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.SiLU(),
            nn.Dropout(dropout))
        self.morph_proj = nn.Sequential(
            nn.Linear(morph_dim, hidden // 2), nn.LayerNorm(hidden // 2), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden // 2, out_dim), nn.LayerNorm(out_dim), nn.SiLU())
        self.register_buffer("morph_mean", torch.zeros(morph_dim))
        self.register_buffer("morph_std", torch.ones(morph_dim))
        self.register_buffer("stats_fitted", torch.tensor(False))

    def fit_morph_stats(self, morph) -> CellEncoder:
        """Fix the standardization of the morphometrics from all cells of the section."""
        m = torch.as_tensor(np.asarray(morph, np.float32))
        ok = torch.isfinite(m)
        n = ok.sum(0).clamp_min(1)
        mean = torch.where(ok, m, torch.zeros_like(m)).sum(0) / n
        var = torch.where(ok, (m - mean) ** 2, torch.zeros_like(m)).sum(0) / n
        self.morph_mean.copy_(mean)
        self.morph_std.copy_(var.sqrt().clamp_min(1e-6))
        self.stats_fitted.fill_(True)
        return self

    def forward(self, features: torch.Tensor, morph: torch.Tensor) -> torch.Tensor:
        if not bool(self.stats_fitted):
            raise RuntimeError("call fit_morph_stats() before the first forward pass")
        m = (morph - self.morph_mean) / self.morph_std
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        return self.fuse(torch.cat([self.fm_proj(features), self.morph_proj(m)], dim=-1))


class CellHeads(nn.Module):
    """Type mixture, library size and state vector of a cell, from one shared trunk."""

    def __init__(self, in_dim: int, n_types: int, n_states: int = 20, hidden: int = 256,
                 lam_init: float = 100.0, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.type_head = nn.Linear(hidden, n_types)
        self.program_head = nn.Linear(hidden, n_states)
        # The library size is predicted as a deviation from ``lam_init`` on the log scale.
        self.library_head = nn.Linear(hidden, 1)
        self.register_buffer("log_lam0", torch.tensor(float(np.log(lam_init))))
        nn.init.zeros_(self.library_head.weight)
        nn.init.zeros_(self.library_head.bias)
        # The state head starts at zero, so the model begins as a pure type mixture.
        nn.init.zeros_(self.program_head.weight)
        nn.init.zeros_(self.program_head.bias)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(h)
        log_p = F.log_softmax(self.type_head(z), dim=-1)
        log_lam = self.log_lam0 + self.library_head(z).squeeze(-1)
        return {"log_p": log_p, "lam": log_lam.clamp(-8.0, 12.0).exp(),
                "v": self.program_head(z), "log_lam": log_lam}


class EMITNet(nn.Module):
    """Cell encoder, cell graph, spot context and expression heads.

    Parameters
    ----------
    feature_dim, morph_dim
        Width of the image view and of the morphometric view of a cell.
    type_profiles
        ``(K, G)`` initial latent type profiles; rows are normalized to sum to one.
    n_states
        Dimension ``m`` of the cell-state vector.
    dim, heads
        Token width and number of attention heads.
    """

    def __init__(self, feature_dim: int, morph_dim: int, type_profiles, n_states: int = 20,
                 dim: int = 256, heads: int = 4):
        super().__init__()
        profiles = torch.as_tensor(np.asarray(type_profiles), dtype=torch.float32)
        profiles = profiles / profiles.sum(1, keepdim=True).clamp_min(1e-12)
        self.encoder = CellEncoder(feature_dim, morph_dim, out_dim=dim)
        self.graph = CellGraphEncoder(dim, heads=heads, layers=2)
        self.context = SpotContext(dim, heads=heads, layers=2)
        self.heads = CellHeads(dim, profiles.shape[0], n_states=n_states)
        self.log_R = nn.Parameter(profiles.clamp_min(1e-12).log())
        self.U = nn.Parameter(torch.randn(n_states, profiles.shape[1]) * 0.01)

    @property
    def type_profiles(self) -> torch.Tensor:
        """The latent type profiles ``R``; every row is a probability vector over genes."""
        return F.softmax(self.log_R, dim=-1)

    def profiles(self, log_p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Expression profile ``pi`` of each cell: ``softmax(log(p @ R) + v @ U)``."""
        log_mix = (log_p.exp() @ self.type_profiles).clamp_min(1e-12).log()
        return F.softmax(log_mix + v @ self.U, dim=-1)

    def forward(self, features, morph, edge_index, cell_spot, n_spots: int, spot_edges,
                nearest_spot=None, chunk: int | None = None, out_device=None):
        """Predict the type mixture, library size and expression profile of every cell.

        ``chunk`` builds the ``cells x genes`` profile matrix in pieces, and
        ``out_device`` moves each piece off the compute device as it is made, so that
        the matrix can live in host memory when it does not fit on the GPU.
        """
        h = self.encoder(features, morph)
        h = self.graph(h, edge_index)
        h = self.context(h, cell_spot, n_spots, spot_edges, nearest_spot)
        out = self.heads(h)
        if chunk is None:
            pi = self.profiles(out["log_p"], out["v"])
            if out_device is not None:
                pi = pi.to(out_device)
        else:
            parts = []
            for i in range(0, h.shape[0], chunk):
                part = self.profiles(out["log_p"][i:i + chunk], out["v"][i:i + chunk])
                parts.append(part if out_device is None else part.to(out_device))
                del part
            pi = torch.cat(parts, 0)
            del parts
        return {**out, "pi": pi}
