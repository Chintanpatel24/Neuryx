"""
core/apex.py
Adaptive momentum optimizer (Adam variant) for Flux parameter lists.

Internal names:
  pulse    — learning rate
  mu / nu  — β1 / β2  exponential decay rates
  inertia  — first-moment (momentum) buffer
  torque   — second-moment (variance) buffer
  tick     — step counter (for bias correction)
"""

import math as _math
from .flux import Flux


class Apex:
    """
    Adam-style adaptive gradient descent for a flat list of Flux params.

    Parameters
    ----------
    params  : list[Flux]
    pulse   : float   — base learning rate
    mu      : float   — β1, momentum decay
    nu      : float   — β2, variance decay
    epsilon : float   — numerical floor
    """

    def __init__(
        self,
        params:  list[Flux],
        pulse:   float = 0.005,
        mu:      float = 0.85,
        nu:      float = 0.99,
        epsilon: float = 1e-8,
    ):
        self.params  = params
        self.pulse   = pulse
        self.mu      = mu
        self.nu      = nu
        self.epsilon = epsilon
        self.tick    = 0

        n = len(params)
        self.inertia = [0.0] * n   # first-moment accumulators
        self.torque  = [0.0] * n   # second-moment accumulators

    def step(self, decay_factor: float = 1.0) -> None:
        """
        Consume accumulated .delta gradients, update .val of every param,
        then zero all gradients.

        decay_factor: multiplied against self.pulse (linear schedule support).
        """
        self.tick += 1
        lr = self.pulse * decay_factor

        mu, nu, eps = self.mu, self.nu, self.epsilon
        t = self.tick

        for i, p in enumerate(self.params):
            g = p.delta

            # Moment updates
            self.inertia[i] = mu * self.inertia[i] + (1.0 - mu) * g
            self.torque[i]  = nu * self.torque[i]  + (1.0 - nu) * g * g

            # Bias-corrected estimates
            m_hat = self.inertia[i] / (1.0 - mu ** t)
            v_hat = self.torque[i]  / (1.0 - nu ** t)

            # Descent step
            p.val   -= lr * m_hat / (_math.sqrt(v_hat) + eps)
            p.delta  = 0.0

    def zero_grad(self) -> None:
        """Manually zero all gradients without updating values."""
        for p in self.params:
            p.delta = 0.0
