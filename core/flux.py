"""
core/flux.py
Reverse-mode automatic differentiation over scalar computation graphs.

Public surface: Flux  — the only class you need.
"""

import math as _math


class Flux:
    """
    A single node in a directed acyclic computation graph.

    Each Flux node stores:
      val    — the forward-pass scalar value
      delta  — the accumulated gradient from backward diffusion
      src    — parent nodes that produced this node
      jac    — local partial derivatives w.r.t. each parent
    """

    __slots__ = ("val", "delta", "src", "jac")

    def __init__(self, val, src=(), jac=()):
        self.val   = float(val)
        self.delta = 0.0
        self.src   = src          # tuple[Flux, ...]
        self.jac   = jac          # tuple[float, ...]  — ∂self / ∂src[i]

    # ── Arithmetic operators ─────────────────────────────────────────────────

    def __add__(self, rhs):
        rhs = rhs if isinstance(rhs, Flux) else Flux(rhs)
        return Flux(self.val + rhs.val, (self, rhs), (1.0, 1.0))

    def __mul__(self, rhs):
        rhs = rhs if isinstance(rhs, Flux) else Flux(rhs)
        return Flux(self.val * rhs.val, (self, rhs), (rhs.val, self.val))

    def __pow__(self, exp):
        out_val  = self.val ** exp
        slope    = exp * self.val ** (exp - 1)
        return Flux(out_val, (self,), (slope,))

    # ── Activation / element-wise ops ────────────────────────────────────────

    def loge(self):
        safe = max(self.val, 1e-12)
        return Flux(_math.log(safe), (self,), (1.0 / safe,))

    def expe(self):
        clamped = min(self.val, 88.72)           # guard against inf
        result  = _math.exp(clamped)
        return Flux(result, (self,), (result,))

    def thresh(self):                             # ReLU
        return Flux(max(0.0, self.val), (self,), (float(self.val > 0),))

    def swish(self):                              # x * sigmoid(x)
        sig = 1.0 / (1.0 + _math.exp(-self.val))
        return Flux(self.val * sig, (self,), (sig + self.val * sig * (1 - sig),))

    # ── Python dunder aliases ─────────────────────────────────────────────────

    def __neg__(self):           return self * -1
    def __radd__(self, lhs):     return self + lhs
    def __sub__(self, rhs):      return self + (-rhs)
    def __rsub__(self, lhs):     return Flux(lhs) + (-self)
    def __rmul__(self, lhs):     return self * lhs
    def __truediv__(self, rhs):  return self * rhs ** -1
    def __rtruediv__(self, lhs): return Flux(lhs) * self ** -1

    def __repr__(self):
        return f"Flux(val={self.val:.6g}, delta={self.delta:.6g})"

    # ── Backward diffusion ────────────────────────────────────────────────────

    def diffuse(self):
        """
        Back-propagate gradients through the entire computation graph.

        Algorithm:
          1. Topological sort starting from `self`
          2. Seed self.delta = 1.0
          3. Traverse in reverse topo order; accumulate chain-rule contributions
        """
        order: list["Flux"] = []
        seen:  set[int]     = set()

        def _topo(node: "Flux") -> None:
            nid = id(node)
            if nid not in seen:
                seen.add(nid)
                for parent in node.src:
                    _topo(parent)
                order.append(node)

        _topo(self)

        self.delta = 1.0
        for node in reversed(order):
            for parent, local_slope in zip(node.src, node.jac):
                parent.delta += local_slope * node.delta
