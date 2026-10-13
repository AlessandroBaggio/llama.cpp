#!/usr/bin/env python3
"""Synthetic checks for external fixture manifest path, size, and hash validation."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile

SCRIPT = Path(__file__).with_name("verify-xdna-external-fixtures.py")
spec = importlib.util.spec_from_file_location("external_fixture_verifier", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load external fixture verifier")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def rejected(root, entries):
    try:
        verifier.verify_files(root, entries)
    except (OSError, ValueError):
        return
    raise AssertionError("invalid external fixture input was accepted")


with tempfile.TemporaryDirectory(prefix="xdna-fixture-manifest-") as temp:
    root = Path(temp)
    payload = root / "small.bin"
    payload.write_bytes(b"deterministic fixture")
    entry = {"path": "small.bin", "size_bytes": payload.stat().st_size,
             "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
    assert verifier.verify_files(root, [entry]) == {"files": 1, "size_bytes": entry["size_bytes"]}
    rejected(root, [{**entry, "sha256": "0" * 64}])
    rejected(root, [{**entry, "size_bytes": entry["size_bytes"] + 1}])
    rejected(root, [{**entry, "path": "../small.bin"}])
    rejected(root, [{**entry, "path": "..\\\\small.bin"}])
    rejected(root, [{**entry, "path": "C:/small.bin"}])
    rejected(root, [{**entry, "path": "./small.bin"}])
    rejected(root, [{**entry, "path": "missing.bin"}])
    (root / "unlisted.bin").write_bytes(b"extra")
    rejected(root, [entry])

print("XDNA_EXTERNAL_FIXTURE_MANIFEST_SYNTHETIC=PASS valid=PASS missing=REJECTED corrupted=REJECTED traversal=REJECTED unlisted=REJECTED")
