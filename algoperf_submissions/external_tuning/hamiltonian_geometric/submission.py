"""AlgoPerf submission wrapper for Hamiltonian-Geometric optimization.

This file follows the MLCommons AlgoPerf submission API. It is intentionally
PyTorch-first because the target-setting baselines we are comparing against
include official PyTorch AdamW, Nesterov, and Heavy Ball implementations.

The update is a diagonal-metric, Adam-like version of the project's optimizer:

  memory_t = kappa memory_{t-1} + (1 - kappa) grad
  force_t = grad + mu (memory_t - grad)
  m_t = beta1 m_{t-1} + (1 - beta1) force_t
  g_t = beta2 g_{t-1} + (1 - beta2) grad^2
  theta_t = theta_{t-1} - lr m_hat_t / (sqrt(g_hat_t) + epsilon)

where g_t is a diagonal Adam-style metric estimate. This practical
AlgoPerf-compatible form keeps memory, momentum, and inverse-metric structure
but bounds the geometric memory term so the optimizer tracks useful descent
directions instead of exploring irrelevant sideways regions early in training.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from algoperf import spec
from algorithms.target_setting_algorithms.data_selection import (  # noqa: F401
  data_selection,
)
from algorithms.target_setting_algorithms.get_batch_size import (
  get_batch_size as target_setting_get_batch_size,
)
from algorithms.target_setting_algorithms.jax_momentum import (
  create_lr_schedule_fn,
)


def get_batch_size(workload_name: str) -> int:
  """Return global batch size, including AlgoPerf's MNIST dev workload."""
  if workload_name == 'mnist':
    return 128
  return target_setting_get_batch_size(workload_name)


class HamiltonianGeometricOptimizer(Optimizer):
  """Diagonal-metric Hamiltonian-Geometric optimizer for PyTorch modules."""

  def __init__(
    self,
    params: Iterator[torch.nn.Parameter],
    lr: float,
    beta: float = 0.9,
    metric_decay: float = 0.999,
    metric_epsilon: float = 1e-8,
    memory_decay: float = 0.9,
    memory_coupling: float = 0.003,
    weight_decay: float = 0.0,
  ) -> None:
    if lr <= 0.0:
      raise ValueError(f'lr must be positive, got {lr!r}')
    if not 0.0 <= beta < 1.0:
      raise ValueError(f'beta must be in [0, 1), got {beta!r}')
    if not 0.0 <= metric_decay < 1.0:
      raise ValueError(f'metric_decay must be in [0, 1), got {metric_decay!r}')
    if metric_epsilon <= 0.0:
      raise ValueError(f'metric_epsilon must be positive, got {metric_epsilon!r}')
    if not 0.0 <= memory_decay < 1.0:
      raise ValueError(f'memory_decay must be in [0, 1), got {memory_decay!r}')
    if memory_coupling < 0.0:
      raise ValueError(f'memory_coupling must be non-negative, got {memory_coupling!r}')

    defaults = {
      'lr': lr,
      'beta': beta,
      'metric_decay': metric_decay,
      'metric_epsilon': metric_epsilon,
      'memory_decay': memory_decay,
      'memory_coupling': memory_coupling,
      'weight_decay': weight_decay,
    }
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      beta = group['beta']
      metric_decay = group['metric_decay']
      metric_epsilon = group['metric_epsilon']
      memory_decay = group['memory_decay']
      memory_coupling = group['memory_coupling']
      weight_decay = group['weight_decay']

      for parameter in group['params']:
        if parameter.grad is None:
          continue
        grad = parameter.grad
        if grad.is_sparse:
          raise RuntimeError('HamiltonianGeometricOptimizer does not support sparse gradients')

        state = self.state[parameter]
        if len(state) == 0:
          state['step'] = 0
          state['momentum'] = torch.zeros_like(parameter)
          state['memory'] = torch.zeros_like(parameter)
          state['metric'] = torch.zeros_like(parameter)

        state['step'] += 1
        step = state['step']
        momentum = state['momentum']
        memory = state['memory']
        metric = state['metric']

        if weight_decay != 0.0:
          parameter.mul_(1.0 - lr * weight_decay)

        metric.mul_(metric_decay).addcmul_(grad, grad, value=1.0 - metric_decay)
        memory.mul_(memory_decay).add_(grad, alpha=1.0 - memory_decay)

        force = grad.add(memory - grad, alpha=memory_coupling)
        momentum.mul_(beta).add_(force, alpha=1.0 - beta)

        momentum_hat = momentum / (1.0 - beta**step)
        metric_hat = metric / (1.0 - metric_decay**step)
        parameter.addcdiv_(momentum_hat, metric_hat.sqrt().add_(metric_epsilon), value=-lr)

    return loss


def _get_hparam(hyperparameters: spec.Hyperparameters, name: str, default: float) -> float:
  return getattr(hyperparameters, name) if hasattr(hyperparameters, name) else default


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Create the Hamiltonian-Geometric optimizer and learning-rate schedule."""
  del model_state
  del rng

  optimizer_state = {
    'optimizer': HamiltonianGeometricOptimizer(
      model_params.parameters(),
      lr=hyperparameters.learning_rate,
      beta=_get_hparam(hyperparameters, 'beta1', 0.9),
      metric_decay=_get_hparam(hyperparameters, 'metric_decay', 0.999),
      metric_epsilon=_get_hparam(hyperparameters, 'metric_epsilon', 1e-8),
      memory_decay=_get_hparam(hyperparameters, 'memory_decay', 0.9),
      memory_coupling=_get_hparam(hyperparameters, 'memory_coupling', 0.003),
      weight_decay=_get_hparam(hyperparameters, 'weight_decay', 0.0),
    ),
  }

  target_setting_step_hint = int(0.75 * workload.step_hint)
  lr_schedule_fn = create_lr_schedule_fn(target_setting_step_hint, hyperparameters)

  def _lr_lambda(step: int) -> float:
    return lr_schedule_fn(step).item() / hyperparameters.learning_rate

  optimizer_state['scheduler'] = LambdaLR(
    optimizer_state['optimizer'], lr_lambda=_lr_lambda
  )
  return optimizer_state


def update_params(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  batch: Dict[str, spec.Tensor],
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
  train_state: Optional[Dict[str, Any]] = None,
) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()

  logits_batch, new_model_state = workload.model_fn(
    params=current_model,
    augmented_and_preprocessed_input_batch=batch,
    model_state=model_state,
    mode=spec.ForwardPassMode.TRAIN,
    rng=rng,
    update_batch_norm=True,
  )

  label_smoothing = _get_hparam(hyperparameters, 'label_smoothing', 0.0)
  loss_dict = workload.loss_fn(
    label_batch=batch['targets'],
    logits_batch=logits_batch,
    mask_batch=batch.get('weights'),
    label_smoothing=label_smoothing,
  )
  loss = loss_dict['summed'] / loss_dict['n_valid_examples']
  loss.backward()

  grad_clip = getattr(hyperparameters, 'grad_clip', None)
  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=grad_clip)

  optimizer_state['optimizer'].step()
  if 'scheduler' in optimizer_state:
    optimizer_state['scheduler'].step()

  if workload.metrics_logger is not None and (global_step <= 100 or global_step % 500 == 0):
    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
      )
    workload.metrics_logger.append_scalar_metrics(
      {
        'loss': loss.item(),
        'grad_norm': grad_norm.item(),
      },
      global_step,
    )

  return (optimizer_state, current_param_container, new_model_state)


def prepare_for_eval(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
) -> spec.UpdateReturn:
  """Return parameters unchanged for evaluation."""
  del workload
  del current_params_types
  del hyperparameters
  del loss_type
  del eval_results
  del global_step
  del rng
  return (optimizer_state, current_param_container, model_state)
