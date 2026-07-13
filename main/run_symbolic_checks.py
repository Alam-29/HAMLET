import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.symbolic_checks import run_cadabra_script, run_sympy_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SymPy and optional Cadabra symbolic checks.")
    parser.add_argument(
        "--cadabra-script",
        type=Path,
        default=PROJECT_ROOT / "cadabra" / "maxwell_field_symmetry.cdb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "symbolic_checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sympy_checks = run_sympy_checks()
    cadabra = run_cadabra_script(args.cadabra_script)

    json_path = args.output_dir / "symbolic_checks.json"
    md_path = args.output_dir / "symbolic_checks.md"

    payload = {
        "sympy": [check.__dict__ for check in sympy_checks],
        "cadabra": cadabra.__dict__,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Symbolic Checks",
        "",
        "## SymPy",
        "",
        "| check | simplified residual | passed |",
        "|---|---:|---:|",
    ]
    for check in sympy_checks:
        lines.append(f"| {check.name} | `{check.simplified}` | {check.passed} |")

    lines.extend(["", "## Cadabra", ""])
    if cadabra.available:
        lines.extend(
            [
                f"Executable: `{cadabra.executable}`",
                f"Return code: `{cadabra.returncode}`",
                "",
                "```text",
                cadabra.stdout.strip() or "(no stdout)",
                "```",
            ]
        )
        if cadabra.stderr.strip():
            lines.extend(["", "stderr:", "", "```text", cadabra.stderr.strip(), "```"])
    else:
        lines.extend(
            [
                "Cadabra executable was not found on `PATH`.",
                "",
                "Install the Windows Cadabra package, then make sure `cadabra2-cli` or `cadabra2` is available on `PATH`.",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Symbolic checks")
    for check in sympy_checks:
        print(f"{check.name}: {check.passed} residual={check.simplified}")
    print(f"cadabra_available = {cadabra.available}")
    print(f"exported = {json_path}")
    print(f"exported = {md_path}")


if __name__ == "__main__":
    main()
