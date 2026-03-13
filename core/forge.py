"""
core/forge.py
Training loop that feeds tokenised sequences through a Lattice model,
computes cross-entropy loss, and calls Apex to update weights.

Internal names:
  corpus     — list of token-ID sequences used for training
  chronicle  — list of loss values recorded during training
  anneal()   — execute training for N steps
  _ignite()  — single forward + backward pass over one sequence
"""

import random as _random
import time   as _time

from .lattice import Lattice
from .apex    import Apex
from .flux    import Flux


class Forge:
    """
    Manages the training lifecycle for a Lattice model.

    Parameters
    ----------
    model     : Lattice
    optimizer : Apex
    corpus    : list[list[int]] — pre-built training sequences
    """

    def __init__(self, model: Lattice, optimizer: Apex, corpus: list[list[int]]):
        self.model     = model
        self.optimizer = optimizer
        self.corpus    = corpus
        self.chronicle: list[float] = []   # loss per step

    # ── Public API ────────────────────────────────────────────────────────────

    def anneal(
        self,
        steps:       int,
        on_step_cb=None,   # optional callable(step, total, loss)
    ) -> list[float]:
        """
        Train for `steps` gradient steps.

        Returns the full loss chronicle (one float per step).
        """
        _random.shuffle(self.corpus)
        t0 = _time.time()

        for step in range(steps):
            seq  = self.corpus[step % len(self.corpus)]
            loss = self._ignite(seq)

            decay = 1.0 - step / steps      # linear LR warmdown
            self.optimizer.step(decay)

            self.chronicle.append(loss)

            if on_step_cb:
                elapsed = _time.time() - t0
                eta     = elapsed / (step + 1) * (steps - step - 1)
                on_step_cb(step, steps, loss, eta)

        return self.chronicle

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ignite(self, seq: list[int]) -> float:
        """
        Run one training step on a single sequence.

        Forward → loss → backward → return scalar loss value.
        """
        model  = self.model
        L      = min(model.horizon, len(seq) - 1)
        k_shelf = [[] for _ in range(model.rifts)]
        v_shelf = [[] for _ in range(model.rifts)]
        pieces: list[Flux] = []

        for pos in range(L):
            inp    = seq[pos]
            target = seq[pos + 1]
            logits = model.emit(inp, pos, k_shelf, v_shelf)
            probs  = model._scatter(logits)
            pieces.append(-probs[target].loge())

        loss = (1.0 / L) * sum(pieces)
        loss.diffuse()
        return loss.val

    def infer(
        self,
        context:     list[int],
        n_steps:     int,
        temperature: float = 0.5,
        stop_token:  int | None = None,
    ) -> list[int]:
        """
        Auto-regressive inference: feed `context`, then predict `n_steps` tokens.

        Returns the list of predicted token IDs (not including the seed context).
        """
        import random as _r

        model   = self.model
        k_shelf = [[] for _ in range(model.rifts)]
        v_shelf = [[] for _ in range(model.rifts)]
        output  = []

        # Warm up KV cache with context
        for pos, tok in enumerate(context):
            last_logits = model.emit(tok, pos, k_shelf, v_shelf)

        cur_tok = context[-1]
        pos     = len(context)

        for _ in range(n_steps):
            if pos >= model.horizon:
                break
            logits  = model.emit(cur_tok, pos, k_shelf, v_shelf)
            tempered = [Flux(z.val / max(temperature, 1e-6)) for z in logits]
            probs    = model._scatter(tempered)
            weights  = [p.val for p in probs]
            cur_tok  = _r.choices(range(model.vocab_sz), weights=weights)[0]
            if stop_token is not None and cur_tok == stop_token:
                break
            output.append(cur_tok)
            pos += 1

        return output
