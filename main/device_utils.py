from __future__ import annotations

import argparse
from typing import Any


def add_torch_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch device for GPU-capable benchmarks: auto, cpu, cuda, or cuda:N. "
            "Default auto uses CUDA when PyTorch reports it is available."
        ),
    )


def resolve_torch_device(requested: str, torch_module: Any):
    requested = requested.strip().lower()
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch_module.cuda.is_available():
            raise ValueError(
                f"requested --device {requested!r}, but PyTorch cannot access CUDA"
            )
        if ":" in requested:
            _prefix, index_text = requested.split(":", 1)
            try:
                index = int(index_text)
            except ValueError as error:
                raise ValueError(
                    f"invalid CUDA device {requested!r}; use cuda or cuda:N"
                ) from error
            device_count = torch_module.cuda.device_count()
            if index < 0 or index >= device_count:
                raise ValueError(
                    f"requested --device {requested!r}, but only {device_count} CUDA device(s) are visible"
                )
        return torch_module.device(requested)
    raise ValueError(f"unsupported --device {requested!r}; use auto, cpu, cuda, or cuda:N")
