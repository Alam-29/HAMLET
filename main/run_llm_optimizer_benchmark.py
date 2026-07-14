"""Small character-level transformer LLM benchmark for optimizer comparison.

Extends the CNN benchmark's pattern (main/run_cnn_benchmark.py) to a language
modeling workload: a tiny GPT-style transformer trained on the Tiny Shakespeare
corpus, comparing Hamiltonian-geometric against SGD+momentum, AdamW, Lion, and
Muon.

A custom torch.optim.Optimizer needs direct gradient/weight access every step,
which a closed LLM API cannot provide, so training here is local and real
(PyTorch autograd, real backprop) -- there is no way to "train via API" with a
custom optimizer. The API is used for a different purpose: once each optimizer
has produced a trained model, its generated text sample is blind-scored by a
free-tier Hugging Face Inference Providers chat model (src/llm_judge.py) as a
qualitative complement to the quantitative loss/perplexity metrics. This needs
no local model storage beyond the ~1MB training corpus and only a free HF
account token -- no checkpoint download, no pretrained weights, no billing.

Given this project's CPU-only, memory-constrained environment (see
main/run_cnn_benchmark.py), the model and step count are kept small -- this is
a small-scale but real transformer LM benchmark, not a claim of matching
published language-modeling results.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.torch_optimizers import HamiltonianGeometricTorch, LionTorch, MuonTorch
from src.llm_judge import DEFAULT_JUDGE_MODEL, MissingAPIKeyError, judge_samples

OPTIMIZER_COLORS = {
    "sgd_momentum": "#9aa3ab",
    "adamw": "#e8a33d",
    "lion": "#d1495b",
    "muon": "#2ca089",
    "hamiltonian_geometric": "#6a3d9a",
}
OPTIMIZER_NAMES = ["sgd_momentum", "adamw", "lion", "muon", "hamiltonian_geometric"]


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int = 64, n_head: int = 4, n_layer: int = 2) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _batch, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def load_corpus(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[int, str], dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], itos, stoi


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=rng)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, block_size: int, batch_size: int, eval_batches: int, rng: torch.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(data, block_size, batch_size, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def build_optimizer(name: str, model: nn.Module):
    if name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    if name == "lion":
        return LionTorch(model.parameters(), lr=0.001, beta1=0.9, beta2=0.99, weight_decay=0.01)
    if name == "muon":
        return MuonTorch(model.parameters(), lr=0.02, momentum=0.9, fallback_lr=0.02)
    if name == "hamiltonian_geometric":
        return HamiltonianGeometricTorch(model.parameters(), lr=0.15, beta=0.9, metric_decay=0.96, metric_epsilon=0.08)
    raise ValueError(f"unknown optimizer {name!r}")


def train_one(name: str, train_data: torch.Tensor, val_data: torch.Tensor, vocab_size: int, args: argparse.Namespace):
    seed_offsets = {"sgd_momentum": 1, "adamw": 2, "lion": 3, "muon": 4, "hamiltonian_geometric": 5}
    torch.manual_seed(args.seed + seed_offsets[name])
    rng = torch.Generator().manual_seed(args.seed + seed_offsets[name])

    model = CharTransformer(vocab_size, args.block_size, n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
    optimizer = build_optimizer(name, model)

    history = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, rng)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            train_loss = estimate_loss(model, train_data, args.block_size, args.batch_size, args.eval_batches, rng)
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.eval_batches, rng)
            history.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": math.exp(min(val_loss, 20.0)),
                }
            )
            print(f"  [{name}] step {step}/{args.steps}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)
    runtime_s = time.perf_counter() - start
    return {"name": name, "history": history, "runtime_s": runtime_s, "model": model}


def generate_sample(model: CharTransformer, itos: dict[int, str], stoi: dict[str, int], prompt: str, max_new_tokens: int, seed: int) -> str:
    torch.manual_seed(seed)
    idx = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8)
    return "".join(itos[i] for i in out[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small char-level transformer LLM benchmark (PyTorch), judged qualitatively via a free Hugging Face Inference model.")
    parser.add_argument("--corpus-path", type=Path, default=PROJECT_ROOT / "data" / "tinyshakespeare" / "input.txt")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--sample-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-judge", action="store_true", help="Skip the Hugging Face LLM-judge scoring step.")
    parser.add_argument("--hf-judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Hugging Face model id used as the LLM judge.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations" / "llm_optimizer_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    train_data, val_data, itos, stoi = load_corpus(args.corpus_path)
    vocab_size = len(itos)

    results = []
    for name in OPTIMIZER_NAMES:
        print(f"Training {name}...")
        results.append(train_one(name, train_data, val_data, vocab_size, args))

    prompt = "ROMEO:"
    samples = {
        result["name"]: generate_sample(result["model"], itos, stoi, prompt, args.sample_tokens, seed=args.seed)
        for result in results
    }

    history_path = args.output_dir / "llm_benchmark_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "step", "train_loss", "val_loss", "val_perplexity"])
        for result in results:
            for row in result["history"]:
                writer.writerow([result["name"], row["step"], f"{row['train_loss']:.6f}", f"{row['val_loss']:.6f}", f"{row['val_perplexity']:.6f}"])

    summary_path = args.output_dir / "llm_benchmark_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["optimizer", "best_step", "best_val_loss", "best_val_perplexity", "final_val_loss", "runtime_s"])
        for result in sorted(results, key=lambda r: min(row["val_loss"] for row in r["history"])):
            best = min(result["history"], key=lambda row: row["val_loss"])
            final = result["history"][-1]
            writer.writerow([result["name"], best["step"], f"{best['val_loss']:.6f}", f"{best['val_perplexity']:.6f}", f"{final['val_loss']:.6f}", f"{result['runtime_s']:.3f}"])

    samples_path = args.output_dir / "generated_samples.txt"
    with samples_path.open("w", encoding="utf-8") as file:
        for name, text in samples.items():
            file.write(f"=== {name} ===\n{text}\n\n")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for result in results:
        steps = [row["step"] for row in result["history"]]
        val_loss = [row["val_loss"] for row in result["history"]]
        style = "-" if result["name"] == "hamiltonian_geometric" else "--"
        linewidth = 2.6 if result["name"] == "hamiltonian_geometric" else 1.6
        ax.plot(steps, val_loss, style, linewidth=linewidth, label=result["name"], color=OPTIMIZER_COLORS.get(result["name"]))
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss (nats/char)")
    ax.set_title("Char-transformer LLM benchmark")
    ax.legend()
    fig.tight_layout()
    loss_plot_path = args.output_dir / "llm_benchmark_loss.png"
    fig.savefig(loss_plot_path, dpi=170)
    plt.close(fig)

    judge_scores: dict[str, dict] | None = None
    judge_note = ""
    if args.skip_judge:
        judge_note = "Judge scoring skipped (--skip-judge)."
    else:
        corpus_excerpt = args.corpus_path.read_text(encoding="utf-8")[:800]
        try:
            judge_scores = judge_samples(samples, corpus_excerpt, model=args.hf_judge_model)
        except MissingAPIKeyError as exc:
            judge_note = str(exc)

    judge_path = args.output_dir / "llm_judge_scores.csv"
    if judge_scores is not None:
        with judge_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["optimizer", "coherence_score", "style_fidelity_score", "rationale"])
            for name, scores in sorted(judge_scores.items(), key=lambda kv: -(kv[1]["coherence_score"] + kv[1]["style_fidelity_score"])):
                writer.writerow([name, scores["coherence_score"], scores["style_fidelity_score"], scores["rationale"]])

    report_path = args.output_dir / "llm_benchmark_results.md"
    write_report(args, results, judge_scores, judge_note, report_path)

    print("\nFinal ranking by best validation loss:")
    print("optimizer,best_step,best_val_loss,best_val_perplexity,runtime_s")
    for result in sorted(results, key=lambda r: min(row["val_loss"] for row in r["history"])):
        best = min(result["history"], key=lambda row: row["val_loss"])
        print(f"{result['name']},{best['step']},{best['val_loss']:.6f},{best['val_perplexity']:.6f},{result['runtime_s']:.3f}")
    if judge_scores is not None:
        print("\nLLM-judge scores (coherence + style_fidelity, blind):")
        for name, scores in judge_scores.items():
            print(f"{name}: coherence={scores['coherence_score']} style_fidelity={scores['style_fidelity_score']}")
    elif judge_note:
        print(f"\n{judge_note}")

    exported_paths = [history_path, summary_path, samples_path, loss_plot_path, report_path]
    if judge_scores is not None:
        exported_paths.append(judge_path)
    for path in exported_paths:
        print(f"exported = {path}")


def write_report(args: argparse.Namespace, results: list[dict], judge_scores: dict[str, dict] | None, judge_note: str, path: Path) -> None:
    ordered = sorted(results, key=lambda r: min(row["val_loss"] for row in r["history"]))
    lines = [
        "# Char-Transformer LLM Optimizer Benchmark",
        "",
        "A small GPT-style character-level transformer trained on the Tiny Shakespeare "
        "corpus, comparing Hamiltonian-Geometric against SGD+momentum, AdamW, Lion, and Muon. "
        "Training is local, real PyTorch autograd -- a custom optimizer cannot run inside a "
        "closed LLM API's training loop, since that requires direct gradient/weight access. "
        "A free Hugging Face Inference Providers chat model is instead used to blind-judge "
        "each optimizer's generated text sample.",
        "",
        f"Steps: {args.steps}, block_size: {args.block_size}, n_embd: {args.n_embd}, "
        f"n_head: {args.n_head}, n_layer: {args.n_layer}, batch_size: {args.batch_size}",
        "",
        "## Quantitative results (validation loss)",
        "",
        "| optimizer | best step | best val loss | best val perplexity | runtime s |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in ordered:
        best = min(result["history"], key=lambda row: row["val_loss"])
        lines.append(
            f"| {result['name']} | {best['step']} | {best['val_loss']:.4f} | "
            f"{best['val_perplexity']:.4f} | {result['runtime_s']:.3f} |"
        )

    lines.extend(["", f"## Qualitative LLM-judge results (blind, via Hugging Face model `{args.hf_judge_model}`)", ""])
    if judge_scores is not None:
        lines.append("| optimizer | coherence (1-10) | style fidelity (1-10) |")
        lines.append("|---|---:|---:|")
        for name, scores in sorted(judge_scores.items(), key=lambda kv: -(kv[1]["coherence_score"] + kv[1]["style_fidelity_score"])):
            lines.append(f"| {name} | {scores['coherence_score']} | {scores['style_fidelity_score']} |")
    else:
        lines.append(judge_note or "Judge scoring not run.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
