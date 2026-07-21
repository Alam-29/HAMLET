import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.build_release_archive import (
    ZIP_TIMESTAMP,
    deterministic_info,
    digest,
    verify_written_archive,
)


class ReleaseArchiveTests(unittest.TestCase):
    def test_member_metadata_is_deterministic(self):
        info = deterministic_info(r"folder\artifact.txt")
        self.assertEqual(info.filename, "folder/artifact.txt")
        self.assertEqual(info.date_time, ZIP_TIMESTAMP)
        self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
        self.assertEqual(info.create_system, 3)
        self.assertEqual(info.external_attr, 0o100644 << 16)

    def _write_archive(self, path: Path, data: bytes, manifest_sha: str) -> list[dict[str, object]]:
        expected = [{"path": "artifact.txt", "bytes": len(data), "sha256": manifest_sha}]
        payload = json.dumps({"schema_version": 1, "files": expected}, indent=2) + "\n"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(deterministic_info("artifact.txt"), data)
            archive.writestr(deterministic_info("RELEASE_MANIFEST.json"), payload.encode())
        return expected

    def test_written_archive_matches_internal_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            data = b"reproducible evidence\n"
            expected = self._write_archive(output, data, digest(data))
            verify_written_archive(output, expected)

    def test_written_archive_rejects_member_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            expected = self._write_archive(output, b"tampered\n", digest(b"expected\n"))
            with self.assertRaisesRegex(ValueError, "integrity check"):
                verify_written_archive(output, expected)


if __name__ == "__main__":
    unittest.main()
