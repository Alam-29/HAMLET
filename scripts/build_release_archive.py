"""Build a privacy-checked code/data archive for journal submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "hamiltonian_geometric_submission_archive.zip"
MANIFEST = ROOT / "results" / "submission_artifact_manifest.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    evidence = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {item["path"] for item in evidence["authoritative_files"]}
    paths.update({"requirements.txt", "requirements-repro.txt", "README.md"})
    for directory in ("src", "main", "tests", "scripts"):
        paths.update(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".ps1", ".json"}
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
            archive.write(path, relative.replace("\\", "/"))
            archive_manifest.append({"path": relative.replace("\\", "/"), "bytes": len(data), "sha256": digest(data)})
        payload = (json.dumps({"schema_version": 1, "files": archive_manifest}, indent=2) + "\n").encode()
        archive.writestr("RELEASE_MANIFEST.json", payload)
    archive_hash = digest(OUTPUT.read_bytes())
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{archive_hash}  {OUTPUT.name}\n", encoding="ascii"
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(archive_manifest)} files")


if __name__ == "__main__":
    main()
