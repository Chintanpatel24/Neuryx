"""
core/lattice.py
Decoder-only sequence model built on top of core.flux.Flux.

The model is intentionally general: it consumes integer token sequences
and emits logits over the next token.  The caller decides what tokens mean.

Key names (internal glossary):
  manifold   — weight registry (dict of 2-D Flux matrices)
  horizon    — context length (max sequence window)
  depth      — hidden-state dimension
  rifts      — number of transformer blocks
  streams    — number of attention heads
  channel    — per-head dimension  (depth // streams)
  emit()     — forward pass
  _weave()   — matrix-vector multiply (linear layer)
  _scatter() — numerically-stable softmax
  _norm()    — RMSNorm
"""

import random as _random
from .flux import Flux


# ── Lattice hyperparameters (overridable by caller) ──────────────────────────

DEFAULT_CFG = {
    "rifts":   2,
    "depth":   32,
    "horizon": 64,
    "streams": 4,
}


def _slab(nrow: int, ncol: int, spread: float = 0.05) -> list[list[Flux]]:
    """Allocate a (nrow × ncol) matrix of Flux nodes initialised ~ N(0, spread)."""
    return [
        [Flux(_random.gauss(0.0, spread)) for _ in range(ncol)]
        for _ in range(nrow)
    ]


class Lattice:
    """
    General-purpose causal transformer.

    Parameters
    ----------
    vocab_sz : int
        Total number of distinct tokens.
    cfg : dict
        Hyper-parameters; keys: rifts, depth, horizon, streams.
    """

    def __init__(self, vocab_sz: int, cfg: dict | None = None):
        self.vocab_sz = vocab_sz
        cfg = {**DEFAULT_CFG, **(cfg or {})}

        self.rifts   = cfg["rifts"]
        self.depth   = cfg["depth"]
        self.horizon = cfg["horizon"]
        self.streams = cfg["streams"]
        self.channel = self.depth // self.streams   # per-head width

        # ── Weight registry ───────────────────────────────────────────────────
        self.manifold: dict[str, list[list[Flux]]] = {
            "tok_slab": _slab(vocab_sz,      self.depth),       # token embeddings
            "pos_slab": _slab(self.horizon,  self.depth),       # position embeddings
            "proj_out": _slab(vocab_sz,      self.depth),       # prediction head
        }
        for idx in range(self.rifts):
            pfx = f"r{idx}"
            self.manifold[f"{pfx}.Wq"] = _slab(self.depth, self.depth)
            self.manifold[f"{pfx}.Wk"] = _slab(self.depth, self.depth)
            self.manifold[f"{pfx}.Wv"] = _slab(self.depth, self.depth)
            self.manifold[f"{pfx}.Wo"] = _slab(self.depth, self.depth)
            self.manifold[f"{pfx}.Wu"] = _slab(4 * self.depth, self.depth)
            self.manifold[f"{pfx}.Wd"] = _slab(self.depth, 4 * self.depth)

        # Flat list of all learnable Flux nodes
        self.params: list[Flux] = [
            p for mat in self.manifold.values()
            for row in mat for p in row
        ]

    # ── Primitives ────────────────────────────────────────────────────────────

    @staticmethod
    def _weave(vec: list[Flux], slab: list[list[Flux]]) -> list[Flux]:
        """Linear layer: out = slab @ vec."""
        return [sum(w * x for w, x in zip(row, vec)) for row in slab]

    @staticmethod
    def _scatter(logits: list[Flux]) -> list[Flux]:
        """Softmax with max-subtraction for numerical stability."""
        peak = max(z.val for z in logits)
        exps = [(z - peak).expe() for z in logits]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _norm(vec: list[Flux]) -> list[Flux]:
        """RMSNorm: divide each element by the root mean square."""
        ms    = sum(h * h for h in vec) / len(vec)
        scale = (ms + 1e-6) ** -0.5
        return [h * scale for h in vec]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def emit(
        self,
        token_idx: int,
        pos_idx:   int,
        k_shelf:   list[list[list[Flux]]],
        v_shelf:   list[list[list[Flux]]],
    ) -> list[Flux]:
        """
        Process one token at position pos_idx.

        k_shelf / v_shelf are the KV caches — one list-of-timesteps per rift.
        They are mutated in-place (new keys / values appended).

        Returns raw logits over the vocabulary.
        """
        # Embedding lookup + positional bias
        h = [t + p for t, p in zip(
            self.manifold["tok_slab"][token_idx],
            self.manifold["pos_slab"][pos_idx],
        )]
        h = self._norm(h)

        for idx in range(self.rifts):
            pfx = f"r{idx}"

            # ── Multi-stream attention (causal) ───────────────────────────────
            residual = h
            h = self._norm(h)

            q_vec = self._weave(h, self.manifold[f"{pfx}.Wq"])
            k_vec = self._weave(h, self.manifold[f"{pfx}.Wk"])
            v_vec = self._weave(h, self.manifold[f"{pfx}.Wv"])

            k_shelf[idx].append(k_vec)
            v_shelf[idx].append(v_vec)
            T = len(k_shelf[idx])           # current sequence length

            heads_concat: list[Flux] = []
            for s in range(self.streams):
                lo, hi = s * self.channel, (s + 1) * self.channel

                q_h = q_vec[lo:hi]
                k_h = [k_shelf[idx][t][lo:hi] for t in range(T)]
                v_h = [v_shelf[idx][t][lo:hi] for t in range(T)]

                # Scaled dot-product attention
                scores = [
                    sum(q_h[d] * k_h[t][d] for d in range(self.channel))
                    / self.channel ** 0.5
                    for t in range(T)
                ]
                weights = self._scatter(scores)

                ctx = [
                    sum(weights[t] * v_h[t][d] for t in range(T))
                    for d in range(self.channel)
                ]
                heads_concat.extend(ctx)

            h = self._weave(heads_concat, self.manifold[f"{pfx}.Wo"])
            h = [a + b for a, b in zip(h, residual)]

            # ── Position-wise feed-forward ────────────────────────────────────
            residual = h
            h = self._norm(h)
            h = self._weave(h, self.manifold[f"{pfx}.Wu"])
            h = [z.thresh() for z in h]          # ReLU gate
            h = self._weave(h, self.manifold[f"{pfx}.Wd"])
            h = [a + b for a, b in zip(h, residual)]

        return self._weave(h, self.manifold["proj_out"])
