"""Build a privacy-checked code/data archive for journal submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "hamiltonian_geometric_submission_archive.zip"
MANIFEST = ROOT / "results" / "submission_artifact_manifest.json"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_info(relative: str) -> zipfile.ZipInfo:
    """Return platform-independent metadata for a reproducible ZIP member."""
    info = zipfile.ZipInfo(relative.replace("\\", "/"), date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def verify_written_archive(expected: list[dict[str, object]]) -> None:
    """Reopen the ZIP and verify its file set, sizes, and internal hashes."""
    with zipfile.ZipFile(OUTPUT, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate member found in release archive")
        if names.count("RELEASE_MANIFEST.json") != 1:
            raise ValueError("release archive must contain exactly one internal manifest")
        embedded = json.loads(archive.read("RELEASE_MANIFEST.json"))
        if embedded != {"schema_version": 1, "files": expected}:
            raise ValueError("embedded release manifest differs from the build manifest")
        expected_names = {item["path"] for item in expected} | {"RELEASE_MANIFEST.json"}
        if set(names) != expected_names:
            raise ValueError("release archive member set differs from its manifest")
        for item in expected:
            data = archive.read(item["path"])
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise ValueError(f"release member failed integrity check: {item['path']}")


def main() -> None:
    evidence = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {item["path"] for item in evidence["authoritative_files"]}
    paths.update({"requirements.txt", "requirements-repro.txt", "README.md"})
    for directory in ("src", "main", "tests", "scripts", "algoperf_submissions"):
        paths.update(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".ps1", ".json", ".md", ".txt"}
            and "__pycache__" not in path.parts
        )
    paths.update({
        "ablation styudy/run_ablation_study.py",
        "ablation styudy/supplementary_ablation.tex",
        "ablation styudy/supplementary_ablation.pdf",
        "docs/claim_evidence_audit.md",
        "docs/citation_audit.md",
        "docs/q1_submission_readiness.md",
        "docs/journal_targeting.md",
        "docs/algoperf_official_runbook.md",
        "docs/submission/highlights.txt",
    })
    archive_manifest = []
    # Split the signatures so the scanner does not flag this scanner itself.
    forbidden = (b"C:" + b"\\Users\\", b"/" + b"Users/", b"file:" + b"//")
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(paths):
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(relative)
            data = path.read_bytes()
            if path.suffix.lower() in {".json", ".csv", ".md", ".tex", ".txt", ".py", ".ps1"}:
                if any(marker in data for marker in forbidden):
                    raise ValueError(f"absolute local path found in release file: {relative}")
            member = relative.replace("\\", "/")
            archive.writestr(deterministic_info(member), data)
            archive_manifest.append({"path": member, "bytes": len(data), "sha256": digest(data)})
        payload = (json.dumps({"schema_version": 1, "files": archive_manifest}, indent=2) + "\n").encode()
        archive.writestr(deterministic_info("RELEASE_MANIFEST.json"), payload)
    verify_written_archive(archive_manifest)
    archive_hash = digest(OUTPUT.read_bytes())
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{archive_hash}  {OUTPUT.name}\n", encoding="ascii"
    )
    print(f"wrote and verified {OUTPUT.relative_to(ROOT)} with {len(archive_manifest)} files")


if __name__ == "__main__":
    main()
