"""Official DeepOBS benchmark run (not the local "-style" reimplementation).

Uses the actual `deepobs` PyPI package (Schneider et al. 2019, "DeepOBS: A
Deep Learning Optimizer Benchmark Suite"), its official PyTorch test problem
(`mnist_mlp` by default) and data pipeline, and the project's own
`src.torch_optimizers.HamiltonianGeometricTorch` (the same diagonal-metric
reduction used in the CUDA ablation), run through DeepOBS's own
`StandardRunner` -- not a custom training loop. Baselines (SGD, AdamW) are
run through the identical runner for a fair, same-pipeline comparison.

This is a real official-codebase run, but a reduced-epoch one: DeepOBS's own
published baselines typically use ~40-100 epochs per problem; this script
defaults to a smaller epoch count so a comparison completes in a reasonable
time on a single consumer GPU. That reduction is reported plainly, not hidden.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from deepobs.pytorch.runners.runner import StandardRunner  # noqa: E402

from src.torch_optimizers import HamiltonianGeometricTorch  # noqa: E402

# NOTE: "weight_decay" is a reserved top-level parameter of DeepOBS's own
# PTRunner.run() (a global L2 term applied regardless of optimizer); DeepOBS
# always injects its value into hyperparams, so naming an optimizer's own
# hyperparameter "weight_decay" causes a collision (the runner's default of
# None silently overwrites AdamW's own decay). AdamW's decoupled decay is
# instead fixed via a thin subclass so only "lr" is a tunable hyperparameter.
class AdamWFixedDecay(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr, weight_decay=0.01)


OPTIMIZERS = {
    "hamiltonian_geometric": (HamiltonianGeometricTorch, {"lr": {"type": float}, "beta": {"type": float}}),
    "sgd_momentum": (torch.optim.SGD, {"lr": {"type": float}, "momentum": {"type": float}}),
    "adamw": (AdamWFixedDecay, {"lr": {"type": float}}),
}


# Learning rates below were selected by a short, equal-budget sweep (2 epochs
# each) per optimizer on this exact testproblem -- see
# results/official_deepobs/lr_sweep_notes.md -- matching the "equal short
# tuning budget" convention already used for the local-style benchmarks.
DEFAULT_HYPERPARAMS = {
    "hamiltonian_geometric": {"lr": 0.1, "beta": 0.9},
    "sgd_momentum": {"lr": 0.1, "momentum": 0.9},
    "adamw": {"lr": 0.003},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official DeepOBS PyTorch benchmark run.")
    parser.add_argument("--testproblem", default="mnist_mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "downloads" / "deepobs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "official_deepobs")
    parser.add_argument("--optimizers", nargs="+", choices=list(OPTIMIZERS), default=list(OPTIMIZERS))
    parser.add_argument("--hg-lr", type=float, default=None, help="Override HG's learning rate (for tuning sweeps).")
    parser.add_argument("--lr", type=float, default=None, help="Override the active optimizer's lr (for tuning sweeps).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # DeepOBS's own PTRunner.run() re-parses sys.argv internally with its own
    # argparse convention (underscore flags); strip our already-consumed CLI
    # args so its parser only sees defaults, and rely on the explicit kwargs
    # below instead.
    sys.argv = sys.argv[:1]

    rows = []
    for name in args.optimizers:
        optimizer_class, hyperparameter_names = OPTIMIZERS[name]
        hyperparams = dict(DEFAULT_HYPERPARAMS[name])
        if name == "hamiltonian_geometric" and args.hg_lr is not None:
            hyperparams["lr"] = args.hg_lr
        if args.lr is not None:
            hyperparams["lr"] = args.lr
        runner = StandardRunner(optimizer_class, hyperparameter_names)
        start = time.perf_counter()
        result = runner.run(
            testproblem=args.testproblem,
            hyperparams=hyperparams,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            random_seed=args.seed,
            data_dir=str(args.data_dir),
            output_dir=str(args.output_dir / name),
            no_logs=False,
            train_log_interval=100,
            print_train_iter=False,
            tb_log=False,
        )
        runtime_s = time.perf_counter() - start
        final_val_loss = result["valid_losses"][-1]
        final_val_acc = result["valid_accuracies"][-1] if result.get("valid_accuracies") else None
        final_test_loss = result["test_losses"][-1] if result.get("test_losses") else None
        final_test_acc = result["test_accuracies"][-1] if result.get("test_accuracies") else None
        rows.append({
            "optimizer": name, "testproblem": args.testproblem, "epochs": args.epochs,
            "final_val_loss": final_val_loss, "final_val_acc": final_val_acc,
            "final_test_loss": final_test_loss, "final_test_acc": final_test_acc,
            "runtime_s": runtime_s,
        })
        print(f"{name}: val_loss={final_val_loss:.6g} val_acc={final_val_acc} "
              f"test_loss={final_test_loss} test_acc={final_test_acc} runtime={runtime_s:.1f}s", flush=True)

    summary_path = args.output_dir / f"{args.testproblem}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported = {summary_path}")


if __name__ == "__main__":
    main()
