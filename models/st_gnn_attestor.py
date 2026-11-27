"""Spatio-Temporal Graph Neural Network attestor (no physics-informed terms).

Graph: nodes = cameras, edges = overlapping fields of view (visibility graph).
Per frame, each node carries transport telemetry + optical-flow statistics + an
untrusted ADS-B/Remote-ID prior; each edge carries cross-view epipolar /
homography consistency residuals. A causal GRU keeps per-camera temporal state;
two edge-conditioned message-passing layers let a camera's score depend on how
its *neighbours'* views agree with it. Output: per-camera compromise probability.

The model is causal (frame t uses only data up to t) so it can run online; the
streaming ``step`` path is what latency is measured on.

Untrusted-prior safety: the ADS-B input passes through a bounded gate
(|contribution| <= beta on the logit), so a spoofed prior can shift a score by at
most ``beta`` and never by itself flip a camera with strong geometric evidence.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeConv(nn.Module):
    def __init__(self, d: int, de: int):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * d + de, d), nn.ReLU(), nn.Linear(d, d))
        self.gate = nn.Sequential(nn.Linear(de, 1), nn.Sigmoid())
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, e: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, C, d), e: (B, C, C, de), adj: (B, C, C) in {0,1}
        B, C, d = h.shape
        hi = h.unsqueeze(2).expand(B, C, C, d)
        hj = h.unsqueeze(1).expand(B, C, C, d)
        m = self.msg(torch.cat([hi, hj, e], -1)) * self.gate(e) * adj.unsqueeze(-1)
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        agg = m.sum(2) / deg
        return self.norm(h + self.upd(torch.cat([h, agg], -1)))


class STGNNAttestor(nn.Module):
    def __init__(self, n_node: int, n_edge: int, d: int = 32, beta: float = 0.5,
                 use_epipolar: bool = True, use_telemetry: bool = True, use_adsb: bool = True,
                 node_names: list[str] | None = None):
        super().__init__()
        self.beta = beta
        self.use_adsb = use_adsb
        mask = torch.ones(n_node)
        emask = torch.ones(n_edge)
        names = node_names or []
        if names:
            geo = {"incons_mean", "incons_max", "incons_frac", "resid_mean", "epi_mean"}
            tel = {"delivered", "lat_z", "jitter", "gap"}
            for i, nm in enumerate(names):
                if (nm in geo and not use_epipolar) or (nm in tel and not use_telemetry) or nm == "adsb_prior":
                    mask[i] = 0
            if not use_epipolar:
                emask[:4] = 0
        self.register_buffer("nmask", mask)
        self.register_buffer("emask", emask)
        self.adsb_idx = names.index("adsb_prior") if "adsb_prior" in names else None
        self.enc = nn.Sequential(nn.Linear(n_node, d), nn.ReLU())
        self.gru = nn.GRUCell(d, d)
        self.g1 = EdgeConv(d, n_edge)
        self.g2 = EdgeConv(d, n_edge)
        self.head = nn.Linear(d, 1)
        self.prior = nn.Linear(1, 1)

    def init_state(self, B: int, C: int, device=None) -> torch.Tensor:
        return torch.zeros(B * C, self.gru.hidden_size, device=device)

    def step(self, xn: torch.Tensor, xe: torch.Tensor, adj: torch.Tensor, state: torch.Tensor):
        """One causal frame. xn (B,C,Fn), xe (B,C,C,Fe). Returns logits (B,C), new state."""
        B, C, _ = xn.shape
        z = self.enc(xn * self.nmask)
        s = self.gru(z.reshape(B * C, -1), state)
        h = s.view(B, C, -1)
        e = xe * self.emask
        h = self.g2(self.g1(h, e, adj), e, adj)
        logit = self.head(h).squeeze(-1)
        if self.use_adsb and self.adsb_idx is not None:
            logit = logit + self.beta * torch.tanh(self.prior(xn[..., self.adsb_idx : self.adsb_idx + 1])).squeeze(-1)
        return logit, s

    def forward(self, Xn: torch.Tensor, Xe: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Xn (B,T,C,Fn), Xe (B,T,C,C,Fe), adj (B,C,C) -> logits (B,T,C)."""
        B, T, C, _ = Xn.shape
        state = self.init_state(B, C, Xn.device)
        out = []
        for t in range(T):
            lg, state = self.step(Xn[:, t], Xe[:, t], adj, state)
            out.append(lg)
        return torch.stack(out, 1)
