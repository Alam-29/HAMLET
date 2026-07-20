import csv

import torch

from main.run_industry_llm_benchmark import (
    RunResult,
    get_batch,
    make_optimizer,
    parse_args,
    write_aggregate_outputs,
)


def test_seeded_batch_generators_produce_identical_batches() -> None:
    data = torch.arange(200)
    first = torch.Generator().manual_seed(42)
    second = torch.Generator().manual_seed(42)

    x1, y1 = get_batch(data, 8, 4, torch.device("cpu"), first)
    x2, y2 = get_batch(data, 8, 4, torch.device("cpu"), second)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_hamiltonian_cli_settings_reach_optimizer() -> None:
    args = parse_args(
        [
            "--hg-lr", "0.002",
            "--hg-beta", "0.8",
            "--hg-metric-decay", "0.95",
            "--hg-memory-decay", "0.85",
            "--hg-memory-coupling", "0.02",
            "--hg-weight-decay", "0.03",
        ]
    )
    model = torch.nn.Linear(2, 1)

    optimizer = make_optimizer("hamiltonian_geometric", model, args)
    group = optimizer.param_groups[0]

    assert group["lr"] == 0.002
    assert group["beta"] == 0.8
    assert group["metric_decay"] == 0.95
    assert group["memory_decay"] == 0.85
    assert group["memory_coupling"] == 0.02
    assert group["weight_decay"] == 0.03


def test_multi_seed_aggregate_is_ranked_by_mean_loss(tmp_path) -> None:
    def result(name: str, loss: float) -> RunResult:
        return RunResult(
            name,
            1.0,
            [{"step": 1.0, "train_loss": loss, "val_loss": loss, "val_perplexity": loss + 1.0}],
        )

    names = ("adamw", "adafactor", "lion", "muon_lite", "hamiltonian_geometric")
    seeded = [
        (1, [result(name, float(index + 1)) for index, name in enumerate(names)]),
        (2, [result(name, float(index + 2)) for index, name in enumerate(names)]),
    ]

    write_aggregate_outputs(seeded, tmp_path)

    with (tmp_path / "industry_llm_seed_aggregate.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["optimizer"] == "adamw"
    assert float(rows[0]["mean_val_loss"]) == 1.5
    assert (tmp_path / "industry_llm_seed_aggregate.md").exists()
