from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_math_validation import run_all_checks
from src.paper_validation import validate_literature_review_math, write_math_validation_report


def main() -> None:
    output_path = PROJECT_ROOT / "docs" / "mathematical_validation.md"
    numeric_checks = run_all_checks()
    write_math_validation_report(str(output_path), numeric_checks=numeric_checks)

    findings = validate_literature_review_math()
    print("Mathematical validation findings (qualitative)")
    print("severity,status,identifier,section")
    for finding in findings:
        print(
            f"{finding.severity},"
            f"{finding.status},"
            f"{finding.identifier},"
            f"{finding.section}"
        )

    print()
    print("Numerical checks")
    print("passed,kind,identifier,max_error,tolerance,section")
    failed = 0
    for check in numeric_checks:
        if not check.passed:
            failed += 1
        print(
            f"{check.passed},"
            f"{check.kind},"
            f"{check.identifier},"
            f"{check.max_error:.3e},"
            f"{check.tolerance:.3e},"
            f"{check.section}"
        )
    print(f"exported = {output_path}")
    if failed:
        print(f"WARNING: {failed} numeric check(s) failed -- see the report for detail")


if __name__ == "__main__":
    main()
