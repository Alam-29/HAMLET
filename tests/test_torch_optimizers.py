import unittest

import torch

from src.torch_optimizers import HamiltonianGeometricTorch, LionTorch, MuonTorch


def _train_on_quadratic(optimizer_cls, param_shape, steps=200, **kwargs):
    torch.manual_seed(0)
    target = torch.randn(param_shape)
    theta = torch.zeros(param_shape, requires_grad=True)
    optimizer = optimizer_cls([theta], **kwargs)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((theta - target) ** 2).sum()
        loss.backward()
        optimizer.step()
    return ((theta - target) ** 2).sum().item()


class TorchOptimizerTests(unittest.TestCase):
    def test_hamiltonian_geometric_reduces_quadratic_loss(self) -> None:
        final_loss = _train_on_quadratic(HamiltonianGeometricTorch, (10,), lr=0.1)
        self.assertLess(final_loss, 1e-3)

    def test_lion_reduces_quadratic_loss(self) -> None:
        final_loss = _train_on_quadratic(LionTorch, (10,), lr=0.05)
        self.assertLess(final_loss, 1e-2)

    def test_muon_reduces_quadratic_loss_on_matrix_parameter(self) -> None:
        # Muon's orthogonalization branch only engages for >=2D parameters.
        final_loss = _train_on_quadratic(MuonTorch, (6, 4), lr=0.1, momentum=0.9)
        self.assertLess(final_loss, 1.0)

    def test_muon_falls_back_to_momentum_sgd_for_1d_parameters(self) -> None:
        final_loss = _train_on_quadratic(MuonTorch, (10,), lr=0.1, fallback_lr=0.1)
        self.assertLess(final_loss, 1e-3)


if __name__ == "__main__":
    unittest.main()
