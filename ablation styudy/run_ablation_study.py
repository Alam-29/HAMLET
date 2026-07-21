"""Reproducible multi-seed ablations and supplementary-paper generator.

The full Hamiltonian-geometric architecture is NumPy based, so its exact
factorial and quantum-control experiments run on CPU.  The neural scaling
experiment uses CUDA and the repository's explicitly documented diagonal
Torch reduction.  Results from these mechanisms are never conflated.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.hamiltonian_geometric import (  # noqa: E402
    HamiltonianGeometricConfig,
    HamiltonianGeometricState,
    hamiltonian_geometric_step,
    initial_state,
)
from src.quantum_chaos import (  # noqa: E402
    QuantumChaosConfig,
    build_problem,
    train_quantum_optimizer,
)


RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FACTORS = tuple(itertools.product((0, 1), repeat=3))
FACTOR_NAMES = ("geometric", "memory", "spectral")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--architecture-seeds", type=int)
    parser.add_argument("--architecture-steps", type=int)
    parser.add_argument("--quantum-seeds", type=int)
    parser.add_argument("--quantum-iterations", type=int)
    parser.add_argument("--gpu-seeds", type=int)
    parser.add_argument("--gpu-steps", type=int)
    parser.add_argument(
        "--sections", nargs="+", choices=("architecture", "quantum", "gpu", "report"),
        default=("architecture", "quantum", "gpu", "report"),
        help="Run selected sections; skipped sections reuse existing raw CSV files.",
    )
    return parser.parse_args()


def mode_value(args: argparse.Namespace, smoke: int, full: int, override: str) -> int:
    value = getattr(args, override)
    return value if value is not None else (smoke if args.mode == "smoke" else full)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def config_name(g: int, m: int, s: int) -> str:
    return f"G{g}M{m}S{s}"


def make_quartic(seed: int = 20260720):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(4, 4))
    q = 0.5 * (raw + raw.T) - 1.4 * np.eye(4)
    c = rng.normal(size=4)
    base = rng.normal(size=(4, 4))
    base = base.T @ base + 0.5 * np.eye(4)

    def loss(theta: np.ndarray) -> float:
        return float(0.5 * theta @ q @ theta + c @ theta + 0.1 * np.sum(theta**4))

    def grad(theta: np.ndarray) -> np.ndarray:
        return q @ theta + c + 0.4 * theta**3

    def hessian_metric(theta: np.ndarray) -> np.ndarray:
        return q + np.diag(1.2 * theta**2)

    def control_metric(theta: np.ndarray) -> np.ndarray:
        return base + np.diag(1.0 + 0.5 * np.sin(theta) ** 2 + 0.3 * theta**2)

    return loss, grad, hessian_metric, control_metric


def run_architecture(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    seeds = mode_value(args, 3, 40, "architecture_seeds")
    steps = mode_value(args, 40, 300, "architecture_steps")
    loss_fn, grad_fn, hessian_metric, control_metric = make_quartic()
    raw_rows: list[dict] = []
    history_rows: list[dict] = []
    for seed in range(seeds):
        theta0 = np.random.default_rng(10_000 + seed).normal(0.0, 0.6, size=4)
        for metric_name, metric_fn in (("control", control_metric), ("hessian", hessian_metric)):
            for g, m, s in FACTORS:
                state0 = initial_state(4)
                state = HamiltonianGeometricState(
                    parameters=theta0.copy(), momentum=state0.momentum,
                    memory=state0.memory, memory_metric=state0.memory_metric,
                )
                cfg = HamiltonianGeometricConfig(
                    learning_rate=0.12, beta=0.70, metric_regularization=0.08,
                    memory_coupling=0.02, memory_decay=0.85,
                    spectral_weight=0.05 if s else 0.0,
                    finite_difference_step=1e-5,
                    use_geometric_correction=bool(g),
                    use_memory_correction=bool(m),
                    max_energy_backtracks=0,
                )
                start = time.perf_counter()
                initial_loss = loss_fn(state.parameters)
                best_loss = initial_loss
                force_sums = np.zeros(3)
                status = "ok"
                for step in range(steps):
                    current = loss_fn(state.parameters)
                    if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
                        history_rows.append({
                            "seed": seed, "metric": metric_name,
                            "configuration": config_name(g, m, s),
                            "step": step, "loss": current,
                        })
                    if not np.isfinite(current) or abs(current) > 1e14:
                        status = "diverged"
                        break
                    best_loss = min(best_loss, current)
                    state = hamiltonian_geometric_step(state, grad_fn, metric_fn, cfg)
                    force_sums += (state.geometric_force_norm, state.memory_force_norm, state.spectral_force_norm)
                final_loss = loss_fn(state.parameters)
                diverged = int(status != "ok" or not np.isfinite(final_loss) or abs(final_loss) > 100.0)
                raw_rows.append({
                    "study": "architecture", "seed": seed, "metric": metric_name,
                    "configuration": config_name(g, m, s), "geometric": g,
                    "memory": m, "spectral": s, "steps_requested": steps,
                    "steps_completed": state.step, "initial_loss": initial_loss,
                    "final_loss": final_loss, "best_loss": best_loss,
                    "diverged": diverged, "runtime_s": time.perf_counter() - start,
                    "mean_geometric_force": force_sums[0] / max(state.step, 1),
                    "mean_memory_force": force_sums[1] / max(state.step, 1),
                    "mean_spectral_force": force_sums[2] / max(state.step, 1),
                })
                print(f"architecture {metric_name} {config_name(g,m,s)} seed={seed} loss={final_loss:.4g}", flush=True)
    return raw_rows, history_rows


def run_quantum(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    seeds = mode_value(args, 2, 20, "quantum_seeds")
    iterations = mode_value(args, 12, 80, "quantum_iterations")
    raw_rows: list[dict] = []
    history_rows: list[dict] = []
    variants: list[tuple[str, str, dict]] = []
    for g, m, s in FACTORS:
        variants.append((
            config_name(g, m, s), "hamiltonian_geometric",
            {"learning_rate": 0.16, "beta": 0.30, "memory_coupling": 0.02,
             "memory_decay": 0.85, "metric_regularization": 0.05,
             "spectral_weight": 0.01 if s else 0.0,
             "use_geometric_correction": bool(g), "use_memory_correction": bool(m),
             "max_energy_backtracks": 4},
        ))
    variants.extend([
        ("adamw", "adamw", {"learning_rate": 0.035, "beta1": 0.9, "beta2": 0.995, "weight_decay": 1e-4}),
        ("heavy_ball", "heavy_ball", {"learning_rate": 0.05, "momentum": 0.82}),
        ("entropy_descent", "entropy_descent", {"learning_rate": 0.08, "metric_regularization": 0.05}),
    ])
    for seed in range(seeds):
        config = QuantumChaosConfig(spin_j=4, steps=14, kick_strength=7.0, seed=41 + seed)
        problem = build_problem(config)
        for label, optimizer, hp in variants:
            result = train_quantum_optimizer(problem, optimizer, iterations, hp)
            g = int(label[1]) if label.startswith("G") else -1
            m = int(label[3]) if label.startswith("G") else -1
            s = int(label[5]) if label.startswith("G") else -1
            raw_rows.append({
                "study": "quantum", "seed": seed, "problem_seed": config.seed,
                "configuration": label, "optimizer": optimizer,
                "geometric": g, "memory": m, "spectral": s,
                "iterations": iterations, "final_loss": result.final_loss,
                "final_fidelity": result.final_fidelity,
                "best_gradient_norm": min(result.gradient_norm_history),
                "diverged": int(not np.isfinite(result.final_loss)),
                "runtime_s": result.runtime_s,
            })
            for step, (loss, fidelity, grad_norm) in enumerate(zip(
                result.loss_history, result.fidelity_history, result.gradient_norm_history
            )):
                history_rows.append({"seed": seed, "configuration": label, "step": step,
                                     "loss": loss, "fidelity": fidelity, "gradient_norm": grad_norm})
            print(f"quantum {label} seed={seed} fidelity={result.final_fidelity:.4f}", flush=True)
    return raw_rows, history_rows


def cuda_status() -> tuple[bool, dict, object | None]:
    try:
        import torch
        available = bool(torch.cuda.is_available())
        info = {
            "torch_version": torch.__version__, "cuda_available": available,
            "torch_cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if available else None,
        }
        if available:
            probe = (torch.ones(256, device="cuda") @ torch.ones(256, device="cuda")).item()
            info["probe"] = probe
        return available, info, torch
    except Exception as exc:
        return False, {"cuda_available": False, "error": repr(exc)}, None


def run_gpu(args: argparse.Namespace) -> list[dict]:
    available, info, torch = cuda_status()
    (RESULTS / "cuda_environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if args.skip_gpu:
        return []
    if not available:
        if args.require_cuda:
            raise RuntimeError(f"CUDA is required but unavailable: {info}")
        print(f"GPU study skipped: {info}", flush=True)
        return []

    from src.torch_optimizers import HamiltonianGeometricTorch

    seeds = mode_value(args, 2, 10, "gpu_seeds")
    steps = mode_value(args, 40, 600, "gpu_steps")
    device = torch.device("cuda")
    target_accuracy = 0.90
    eval_every = 25
    rows: list[dict] = []
    variants = (
        ("HG_metric_momentum", "hg", {"beta": 0.90, "metric_decay": 0.96}),
        ("HG_metric_no_momentum", "hg", {"beta": 0.0, "metric_decay": 0.96}),
        ("HG_scalar_momentum", "hg", {"beta": 0.90, "metric_decay": 1.0}),
        ("HG_scalar_no_momentum", "hg", {"beta": 0.0, "metric_decay": 1.0}),
        ("SGD_momentum", "sgd", {}),
        ("AdamW", "adamw", {}),
    )
    for seed in range(seeds):
        torch.manual_seed(20_000 + seed)
        torch.cuda.manual_seed_all(20_000 + seed)
        generator = torch.Generator(device=device).manual_seed(30_000 + seed)
        x = torch.randn(32768, 128, generator=generator, device=device)
        teacher = torch.randn(128, 10, generator=generator, device=device)
        y = (x @ teacher + 0.4 * torch.randn(32768, 10, generator=generator, device=device)).argmax(1)
        for label, kind, hp in variants:
            torch.manual_seed(40_000 + seed)
            # Reset the stream so every optimizer receives identical minibatches.
            batch_generator = torch.Generator(device=device).manual_seed(50_000 + seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 512), torch.nn.GELU(),
                torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 10),
            ).to(device)
            if kind == "hg":
                optimizer = HamiltonianGeometricTorch(model.parameters(), lr=0.12,
                    beta=hp["beta"], metric_decay=hp["metric_decay"], metric_epsilon=0.08)
            elif kind == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            initial_loss = math.nan
            final_loss = math.nan
            target_step = None
            target_time_s = None
            for step in range(steps):
                idx = torch.randint(0, x.shape[0] - 4096, (512,), generator=batch_generator, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[idx])
                loss = torch.nn.functional.cross_entropy(logits, y[idx])
                if step == 0:
                    initial_loss = float(loss.detach())
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
                if (step + 1) % eval_every == 0 or step + 1 == steps:
                    with torch.no_grad():
                        checkpoint_logits = model(x[-4096:])
                        checkpoint_accuracy = float((checkpoint_logits.argmax(1) == y[-4096:]).float().mean())
                    if target_step is None and checkpoint_accuracy >= target_accuracy:
                        torch.cuda.synchronize()
                        target_step = step + 1
                        target_time_s = time.perf_counter() - start
            torch.cuda.synchronize()
            runtime = time.perf_counter() - start
            with torch.no_grad():
                val_logits = model(x[-4096:])
                val_loss = float(torch.nn.functional.cross_entropy(val_logits, y[-4096:]))
                accuracy = float((val_logits.argmax(1) == y[-4096:]).float().mean())
            rows.append({
                "study": "gpu_neural", "seed": seed, "configuration": label,
                "steps": steps, "initial_train_loss": initial_loss,
                "final_train_loss": final_loss, "validation_loss": val_loss,
                "validation_accuracy": accuracy, "runtime_s": runtime,
                "examples_per_second": steps * 512 / runtime,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                "target_accuracy": target_accuracy,
                "updates_to_target": target_step if target_step is not None else "",
                "time_to_target_s": target_time_s if target_time_s is not None else "",
                "target_reached": target_step is not None,
                "device": info["device"], "torch_version": info["torch_version"],
            })
            print(f"gpu {label} seed={seed} accuracy={accuracy:.4f}", flush=True)
            del model, optimizer, val_logits
            torch.cuda.empty_cache()
    return rows


def bootstrap_ci(values: Iterable[float], seed: int = 7) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        value = float(np.median(x)) if len(x) else math.nan
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(5000, len(x)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, (0.025, 0.975)))


def factorial_effects(rows: list[dict], study: str, metric: str | None = None) -> list[dict]:
    subset = [r for r in rows if r["study"] == study and int(r.get("geometric", -1)) >= 0]
    if metric is not None:
        subset = [r for r in subset if r.get("metric") == metric]
    finite = [r for r in subset if np.isfinite(float(r["final_loss"]))]
    if not finite:
        return []
    losses = np.array([float(r["final_loss"]) for r in finite])
    offset = losses.min()
    y = np.log10(np.maximum(losses - offset, 0.0) + 1e-10)
    coded = np.array([[2 * int(r[n]) - 1 for n in FACTOR_NAMES] for r in finite], dtype=float)
    columns = [np.ones(len(finite)), coded[:, 0], coded[:, 1], coded[:, 2],
               coded[:, 0] * coded[:, 1], coded[:, 0] * coded[:, 2],
               coded[:, 1] * coded[:, 2], coded.prod(axis=1)]
    terms = ("intercept", "G", "M", "S", "G:M", "G:S", "M:S", "G:M:S")
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = np.divide(beta, standard_error, out=np.zeros_like(beta), where=standard_error > 0)
    p_values = 2.0 * stats.t.sf(np.abs(t_values), dof)
    return [{"study": study, "metric": metric or "quantum", "term": term,
             "coefficient": b, "factorial_effect": 2 * b if term != "intercept" else b,
             "standard_error": se, "t": t, "p": p, "n": len(y), "dof": dof}
            for term, b, se, t, p in zip(terms, beta, standard_error, t_values, p_values)]


def summarize(rows: list[dict], value_key: str, group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in group_keys), []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        values = np.array([float(r[value_key]) for r in group], dtype=float)
        finite = values[np.isfinite(values)]
        low, high = bootstrap_ci(finite)
        item = {k: v for k, v in zip(group_keys, key)}
        item.update({"n": len(group), "finite_n": len(finite),
                     "median": float(np.median(finite)) if len(finite) else math.nan,
                     "mean": float(np.mean(finite)) if len(finite) else math.nan,
                     "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                     "ci95_low": low, "ci95_high": high,
                     "divergence_count": sum(int(r.get("diverged", 0)) for r in group),
                     "median_runtime_s": float(np.median([float(r["runtime_s"]) for r in group]))})
        output.append(item)
    return output


def paired_tests(rows: list[dict], study: str, value_key: str, reference: str,
                 metric: str | None = None, higher_is_better: bool = False) -> list[dict]:
    subset = [r for r in rows if r.get("study") == study]
    if metric is not None:
        subset = [r for r in subset if r.get("metric") == metric]
    by_config: dict[str, dict[int, float]] = {}
    for row in subset:
        value = float(row[value_key])
        if np.isfinite(value):
            by_config.setdefault(str(row["configuration"]), {})[int(row["seed"])] = value
    if reference not in by_config:
        return []
    output: list[dict] = []
    for config in sorted(by_config):
        if config == reference:
            continue
        common = sorted(set(by_config[reference]) & set(by_config[config]))
        ref = np.array([by_config[reference][seed] for seed in common])
        candidate = np.array([by_config[config][seed] for seed in common])
        improvement = (candidate - ref) if higher_is_better else (ref - candidate)
        t_result = stats.ttest_rel(candidate, ref)
        try:
            wilcoxon_p = float(stats.wilcoxon(candidate, ref, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        sd = float(np.std(improvement, ddof=1)) if len(improvement) > 1 else 0.0
        output.append({
            "study": study, "metric": metric or "none", "reference": reference,
            "configuration": config, "n_pairs": len(common),
            "mean_improvement": float(np.mean(improvement)),
            "median_improvement": float(np.median(improvement)),
            "cohen_dz": float(np.mean(improvement) / sd) if sd > 0 else 0.0,
            "paired_t_p": float(t_result.pvalue), "wilcoxon_p": wilcoxon_p,
        })
    # Holm correction is applied separately within each predeclared comparison family.
    for key in ("paired_t_p", "wilcoxon_p"):
        order = np.argsort([float(r[key]) for r in output])
        adjusted = np.ones(len(output))
        running = 0.0
        for rank, index in enumerate(order):
            value = min(1.0, (len(output) - rank) * float(output[index][key]))
            running = max(running, value)
            adjusted[index] = running
        for row, value in zip(output, adjusted):
            row[f"holm_{key}"] = float(value)
    return output


def make_figures(architecture: list[dict], quantum: list[dict], gpu: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric in zip(axes, ("control", "hessian")):
        groups = {config: [float(r["final_loss"]) for r in architecture
                           if r["metric"] == metric and r["configuration"] == config and not int(r["diverged"])]
                  for config in (config_name(*f) for f in FACTORS)}
        ax.boxplot(list(groups.values()), tick_labels=list(groups), showfliers=True)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{metric.capitalize()} metric")
        ax.set_ylabel("final loss")
        ax.grid(alpha=0.2)
    fig.suptitle("Full $2^3$ architecture ablation (paired seeds)")
    fig.tight_layout()
    fig.savefig(FIGURES / "architecture_factorial.png", dpi=180)
    plt.close(fig)

    configs = [config_name(*f) for f in FACTORS] + ["adamw", "heavy_ball", "entropy_descent"]
    medians = [np.median([float(r["final_fidelity"]) for r in quantum if r["configuration"] == c]) for c in configs]
    lows, highs = zip(*(bootstrap_ci([float(r["final_fidelity"]) for r in quantum if r["configuration"] == c]) for c in configs))
    yerr = np.array([[m - lo for m, lo in zip(medians, lows)], [hi - m for m, hi in zip(medians, highs)]])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(configs, medians, yerr=yerr, capsize=3, color="#4c78a8")
    ax.set_ylabel("best fidelity (median, bootstrap 95% CI)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Multi-seed chaotic quantum ablation")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "quantum_ablation.png", dpi=180)
    plt.close(fig)

    if gpu:
        labels = sorted({r["configuration"] for r in gpu})
        accuracies = [np.median([float(r["validation_accuracy"]) for r in gpu if r["configuration"] == c]) for c in labels]
        runtimes = [np.median([float(r["runtime_s"]) for r in gpu if r["configuration"] == c]) for c in labels]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].bar(labels, accuracies, color="#59a14f")
        axes[0].set_ylabel("validation accuracy")
        axes[1].bar(labels, runtimes, color="#f28e2b")
        axes[1].set_ylabel("runtime (seconds)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("CUDA diagonal-metric component ablation")
        fig.tight_layout()
        fig.savefig(FIGURES / "gpu_ablation.png", dpi=180)
        plt.close(fig)


def tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def generate_tex(architecture_summary: list[dict], quantum_summary: list[dict],
                 gpu_summary: list[dict], effects: list[dict], paired: list[dict], metadata: dict,
                 gpu_raw: list[dict] | None = None) -> None:
    main_effects = [r for r in effects if r["term"] in ("G", "M", "S")]
    effect_lines = "\n".join(
        f"{tex_escape(r['study'])}/{tex_escape(r['metric'])} & {r['term']} & {r['factorial_effect']:.3g} & {r['p']:.3g} \\\\" for r in main_effects
    )
    arch_lines = "\n".join(
        f"{tex_escape(r['metric'])} & {tex_escape(r['configuration'])} & {r['n']} & {r['median']:.4g} & [{r['ci95_low']:.4g}, {r['ci95_high']:.4g}] & {r['divergence_count']} \\\\"
        for r in architecture_summary
    )
    quantum_lines = "\n".join(
        f"{tex_escape(r['configuration'])} & {r['n']} & {r['median']:.4f} & [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] & {r['median_runtime_s']:.3f} \\\\"
        for r in quantum_summary
    )
    if gpu_summary:
        gpu_raw = gpu_raw or []
        target_lines = []
        for summary_row in gpu_summary:
            label = summary_row["configuration"]
            trials = [r for r in gpu_raw if r["configuration"] == label]
            hits = [r for r in trials if str(r.get("target_reached", "")).lower() == "true"]
            mean_updates = np.mean([float(r["updates_to_target"]) for r in hits]) if hits else math.nan
            mean_target_s = np.mean([float(r["time_to_target_s"]) for r in hits]) if hits else math.nan
            updates_text = f"{mean_updates:.1f}" if hits else "--"
            time_text = f"{mean_target_s:.2f}" if hits else "--"
            target_lines.append(f"{tex_escape(label)} & {len(hits)}/{len(trials)} & {updates_text} & {time_text} \\\\")
        target_table_lines = "\n".join(target_lines)
        gpu_lines = "\n".join(
            f"{tex_escape(r['configuration'])} & {r['n']} & {r['median']:.4f} & [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] & {r['median_runtime_s']:.2f} \\\\"
            for r in gpu_summary
        )
        gpu_section = rf"""
\section{{CUDA neural scaling ablation}}
This experiment uses the repository's diagonal-metric Torch reduction; it is
not evidence that a full Hessian or finite-difference geometric force scales
to neural networks.  CUDA execution was verified on
\texttt{{{tex_escape(metadata['cuda'].get('device', 'unknown'))}}}.
\begin{{table}}[ht]
\centering\small
\begin{{tabular}}{{lrrrr}}\toprule
Configuration & $n$ & Accuracy & 95\% CI & Runtime (s) \\\midrule
{gpu_lines}
\bottomrule\end{{tabular}}
\caption{{CUDA component-ablation results.}}
\end{{table}}
The protocol holds out the final 4096 examples, resets the minibatch stream
for every optimizer within seed, and evaluates every 25 updates. Failures to
reach validation accuracy $0.90$ are right-censored at the final update.
\begin{{table}}[ht]\centering\small
\begin{{tabular}}{{lrrr}}\toprule
Configuration & Target hits & Updates$^\dagger$ & Time (s)$^\dagger$ \\\midrule
{target_table_lines}
\bottomrule\end{{tabular}}
\caption{{CUDA time-to-target; $^\dagger$means among successful trials.}}
\end{{table}}
\begin{{figure}}[ht]\centering
\includegraphics[width=.95\linewidth]{{figures/gpu_ablation.png}}
\caption{{Accuracy and compute cost for the CUDA study.}}
\end{{figure}}
"""
    else:
        gpu_section = rf"""
\section{{CUDA neural scaling ablation}}
No CUDA result is reported.  The run recorded the following environment
status: \texttt{{{tex_escape(metadata['cuda'].get('error', 'CUDA unavailable'))}}}.
This missing result is not replaced by CPU numbers or an inferred value.
"""
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath,siunitx}}
\usepackage[hidelinks]{{hyperref}}
\title{{Supplementary Material: Extensive Ablation Study of Hamiltonian--Geometric Optimization}}
\author{{Reproducible computational supplement}}
\date{{Generated {tex_escape(metadata['generated_at'])}}}
\begin{{document}}\maketitle
\section{{Scope and claims}}
This supplement reports raw multi-seed ablations generated by
\texttt{{run\_ablation\_study.py}}.  Paired initializations are retained across
configurations.  The exact full-matrix architecture is evaluated only where
its parameter-dependent metric and finite-difference forces are tractable.
The CUDA study evaluates the documented diagonal reduction and is labeled
accordingly.  No missing GPU run is presented as completed.

\section{{Experimental design}}
The architecture experiment crosses geometric force ($G$), memory force
($M$), and spectral-entropy force ($S$) in a complete $2^3$ factorial under a
curvature-derived Hessian metric and an unrelated positive-definite control
metric.  The chaotic quantum experiment repeats the same eight configurations
with paired problem seeds and adds AdamW, heavy-ball, and entropy-descent
references.  Medians and nonparametric bootstrap 95\% intervals use raw
per-seed values.  Diverged trials remain in the raw CSV and are counted.
The completed data contain {metadata['observed_design']['architecture_seeds']}
architecture seeds at {metadata['observed_design']['architecture_steps']} steps,
{metadata['observed_design']['quantum_seeds']} quantum seeds at
{metadata['observed_design']['quantum_iterations']} iterations, and
{metadata['observed_design']['gpu_seeds']} CUDA seeds at
{metadata['observed_design']['gpu_steps']} minibatch steps.

\section{{Full-architecture factorial results}}
\begin{{table}}[ht]\centering\scriptsize
\begin{{tabular}}{{llrrrr}}\toprule
Metric & Config. & $n$ & Median loss & 95\% CI & Diverged \\\midrule
{arch_lines}
\bottomrule\end{{tabular}}
\caption{{Paired multi-seed full-architecture results.}}
\end{{table}}
\begin{{figure}}[ht]\centering
\includegraphics[width=.95\linewidth]{{figures/architecture_factorial.png}}
\caption{{Final-loss distributions for the complete factorial.}}
\end{{figure}}

\section{{Factorial effect estimates}}
Effects are twice the $\pm1$-coded OLS coefficient on
$\log_{{10}}(L-L_{{\min}}+10^{{-10}})$.  They measure association within this
controlled design, not universal optimizer superiority.
\begin{{table}}[ht]\centering\small
\begin{{tabular}}{{llrr}}\toprule
Study/metric & Factor & Effect & $p$ \\\midrule
{effect_lines}
\bottomrule\end{{tabular}}
\caption{{Main factorial effects. Full interactions and standard errors are in \texttt{{results/factorial\_effects.csv}}.}}
\end{{table}}
Paired $t$ tests, Wilcoxon signed-rank tests, standardized paired effect sizes,
and within-family Holm corrections are reported in
\texttt{{results/paired\_tests.csv}} ({len(paired)} prespecified comparisons).

\section{{Chaotic quantum-control replication}}
\begin{{table}}[ht]\centering\small
\begin{{tabular}}{{lrrrr}}\toprule
Configuration & $n$ & Fidelity & 95\% CI & Runtime (s) \\\midrule
{quantum_lines}
\bottomrule\end{{tabular}}
\caption{{Best-seen fidelity across paired kicked-top problem seeds.}}
\end{{table}}
\begin{{figure}}[ht]\centering
\includegraphics[width=.95\linewidth]{{figures/quantum_ablation.png}}
\caption{{Median quantum fidelity with bootstrap confidence intervals.}}
\end{{figure}}

{gpu_section}
\section{{Reproducibility and limitations}}
Software and device metadata are stored in \texttt{{results/metadata.json}}.
Raw observations and histories are retained as CSV files.  Hyperparameters
are fixed before the seed sweep, so these results test robustness of those
settings rather than granting each configuration a different tuning budget.
The quartic task isolates mechanisms but cannot establish application-wide
generalization.  The quantum workload is small, and the CUDA benchmark uses a
diagonal approximation because a dense metric is infeasible at neural-network
scale.  Claims should therefore remain specific to the measured tasks.
\end{{document}}
"""
    (HERE / "supplementary_ablation.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if "architecture" in args.sections:
        architecture, architecture_history = run_architecture(args)
        write_csv(RESULTS / "architecture_raw.csv", architecture)
        write_csv(RESULTS / "architecture_history.csv", architecture_history)
    else:
        architecture = read_csv(RESULTS / "architecture_raw.csv")
    if "quantum" in args.sections:
        quantum, quantum_history = run_quantum(args)
        write_csv(RESULTS / "quantum_raw.csv", quantum)
        write_csv(RESULTS / "quantum_history.csv", quantum_history)
    else:
        quantum = read_csv(RESULTS / "quantum_raw.csv")
    if "gpu" in args.sections:
        gpu = run_gpu(args)
        write_csv(RESULTS / "gpu_raw.csv", gpu)
    else:
        gpu = read_csv(RESULTS / "gpu_raw.csv")

    architecture_summary = summarize(architecture, "final_loss", ("metric", "configuration"))
    quantum_summary = summarize(quantum, "final_fidelity", ("configuration",))
    gpu_summary = summarize(gpu, "validation_accuracy", ("configuration",)) if gpu else []
    effects = factorial_effects(architecture, "architecture", "control")
    effects += factorial_effects(architecture, "architecture", "hessian")
    effects += factorial_effects(quantum, "quantum")
    paired = paired_tests(architecture, "architecture", "final_loss", "G0M0S0", "control")
    paired += paired_tests(architecture, "architecture", "final_loss", "G0M0S0", "hessian")
    paired += paired_tests(quantum, "quantum", "final_fidelity", "G0M0S0", higher_is_better=True)
    paired += paired_tests(gpu, "gpu_neural", "validation_accuracy", "HG_metric_momentum", higher_is_better=True)
    write_csv(RESULTS / "architecture_summary.csv", architecture_summary)
    write_csv(RESULTS / "quantum_summary.csv", quantum_summary)
    write_csv(RESULTS / "gpu_summary.csv", gpu_summary)
    write_csv(RESULTS / "factorial_effects.csv", effects)
    write_csv(RESULTS / "paired_tests.csv", paired)
    make_figures(architecture, quantum, gpu)

    cuda_available, cuda_info, _torch = cuda_status()
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "mode": args.mode, "arguments": vars(args), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "scipy": stats.__version__ if hasattr(stats, "__version__") else "see scipy package",
        "cuda": cuda_info, "cuda_results_completed": bool(gpu),
        "observed_design": {
            "architecture_seeds": len({int(r["seed"]) for r in architecture}),
            "architecture_steps": max((int(r["steps_requested"]) for r in architecture), default=0),
            "quantum_seeds": len({int(r["seed"]) for r in quantum}),
            "quantum_iterations": max((int(r["iterations"]) for r in quantum), default=0),
            "gpu_seeds": len({int(r["seed"]) for r in gpu}),
            "gpu_steps": max((int(r["steps"]) for r in gpu), default=0),
        },
        "elapsed_s": time.perf_counter() - started,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
    }
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    generate_tex(architecture_summary, quantum_summary, gpu_summary, effects, paired, metadata, gpu)
    print(f"completed in {metadata['elapsed_s']:.1f}s; CUDA results={bool(gpu)}", flush=True)
    print(f"supplement = {HERE / 'supplementary_ablation.tex'}", flush=True)


if __name__ == "__main__":
    main()
