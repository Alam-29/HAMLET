"""PyTorch optimizer implementations for the real-CNN benchmark.

These mirror the NumPy reference implementations in
src/hamiltonian_geometric.py and src/modern_optimizers.py exactly in their
update math, but as torch.optim.Optimizer subclasses so they work with
autograd on a real convolutional network. A full Hessian/Fisher metric is
infeasible at this parameter count, so Hamiltonian-geometric here uses the
same diagonal-metric-with-momentum reduction already validated on the spiral
MLP benchmark (main/run_modern_optimizer_benchmark.py), not the full
F_geo/F_mem/spectral architecture -- this is stated plainly, not implied.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class HamiltonianGeometricTorch(Optimizer):
    """Diagonal-metric-preconditioned momentum, the same reduction used in
    main/run_modern_optimizer_benchmark.py's train_hamiltonian_geometric."""

    def __init__(self, params, lr=0.19, beta=0.9, metric_decay=0.96, metric_epsilon=0.08):
        defaults = dict(lr=lr, beta=beta, metric_decay=metric_decay, metric_epsilon=metric_epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta = group["lr"], group["beta"]
            decay, eps = group["metric_decay"], group["metric_epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(p)
                    state["metric_accumulator"] = torch.zeros_like(p)
                velocity, metric_accumulator = state["velocity"], state["metric_accumulator"]
                metric_accumulator.mul_(decay).add_(grad * grad, alpha=1.0 - decay)
                inverse_diag_metric = 1.0 / (metric_accumulator.sqrt() + eps)
                velocity.mul_(beta).add_(grad, alpha=-lr)
                p.add_(lr * inverse_diag_metric * velocity)
        return loss


class LionTorch(Optimizer):
    """Lion (Chen et al. 2023): L1-kinetic sign dynamics, two-timescale momentum."""

    def __init__(self, params, lr=0.003, beta1=0.9, beta2=0.99, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta1, beta2, wd = group["lr"], group["beta1"], group["beta2"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "slow_momentum" not in state:
                    state["slow_momentum"] = torch.zeros_like(p)
                slow_momentum = state["slow_momentum"]
                fast_momentum = beta1 * slow_momentum + (1.0 - beta1) * grad
                p.add_(torch.sign(fast_momentum) + wd * p, alpha=-lr)
                slow_momentum.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class MuonTorch(Optimizer):
    """Muon (Jordan et al. 2024): momentum orthogonalized via exact SVD (the
    matrices here are small enough that this is a simpler, mathematically
    identical substitute for the original Newton-Schulz iteration -- see
    src/modern_optimizers.py's orthogonalize_via_polar_factor). Applied to
    every parameter with at least 2 dimensions (conv and linear weights);
    biases and 1D parameters fall back to plain SGD-with-momentum, matching
    Muon's documented practice of only orthogonalizing hidden-layer matrices.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, fallback_lr=0.02):
        defaults = dict(lr=lr, momentum=momentum, fallback_lr=fallback_lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, fallback_lr = group["lr"], group["momentum"], group["fallback_lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                if p.dim() >= 2:
                    shape = buf.shape
                    matrix = buf.reshape(shape[0], -1)
                    u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
                    orthogonalized = (u @ vh).reshape(shape)
                    p.add_(orthogonalized, alpha=-lr)
                else:
                    p.add_(buf, alpha=-fallback_lr)
        return loss
