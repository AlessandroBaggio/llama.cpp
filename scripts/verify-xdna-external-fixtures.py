#!/usr/bin/env python3
"""Verify explicitly supplied BR1/GN1 diagnostic fixtures without copying them."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root, entries, exact=True):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    if not isinstance(entries, list):
        raise ValueError("file entries must be a list")
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("each file entry must contain a string path")
        raw_path = entry["path"]
        relative = PurePosixPath(raw_path)
        windows_relative = PureWindowsPath(raw_path)
        if (not relative.parts or relative.is_absolute() or windows_relative.is_absolute() or windows_relative.drive or
                "\\" in raw_path or ".." in relative.parts or relative.as_posix() != raw_path):
            raise ValueError(f"unsafe or non-canonical manifest path: {raw_path}")
        if raw_path in expected:
            raise ValueError(f"duplicate manifest path: {raw_path}")
        size_bytes = entry.get("size_bytes")
        sha256 = entry.get("sha256")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(f"invalid size for {raw_path}")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"invalid SHA256 for {raw_path}")
        expected[raw_path] = entry
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"fixture path must not contain symlinks: {raw_path}")
        path = path.resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError(f"fixture path escapes its root or is not a file: {raw_path}")
        if path.stat().st_size != size_bytes:
            raise ValueError(f"size mismatch for {raw_path}")
        if sha256_file(path) != sha256:
            raise ValueError(f"SHA256 mismatch for {raw_path}")
    if exact:
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise ValueError(f"fixture file set mismatch: missing={missing} extra={extra}")
    return {"files": len(expected), "size_bytes": sum(item["size_bytes"] for item in expected.values())}


def verify(manifest_path, repo_root, br1_dir, gn1_dir, gn1_config, artifact_root):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "XDNA_EXTERNAL_DIAGNOSTIC_FIXTURES_V1" or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported external-fixture manifest schema")
    fixture_items = manifest.get("fixtures")
    if not isinstance(fixture_items, list) or any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in fixture_items):
        raise ValueError("manifest fixtures must be identified objects")
    fixtures = {item["id"]: item for item in fixture_items}
    if len(fixtures) != len(fixture_items):
        raise ValueError("duplicate fixture identity")
    br1_id = "BR1_CF1_20260924_115708_53d167aa"
    gn1_id = "GN1_PREFIX_20260929_115413"
    if set(fixtures) != {br1_id, gn1_id}:
        raise ValueError("manifest fixture identities do not match the frozen BR1/GN1 set")
    br1 = verify_files(br1_dir, fixtures[br1_id]["files"])
    gn1 = verify_files(gn1_dir, fixtures[gn1_id]["files"])
    if br1["size_bytes"] != fixtures[br1_id]["size_bytes"] or gn1["size_bytes"] != fixtures[gn1_id]["size_bytes"]:
        raise ValueError("fixture aggregate size mismatch")

    repo_root = Path(repo_root).resolve(strict=True)
    config_spec = manifest.get("configuration")
    if not isinstance(config_spec, dict) or config_spec.get("path") != "repos/llama.cpp/experiments/xdna-gn1-causal-matrix-gp1.json":
        raise ValueError("unexpected GN1 configuration path")
    config_path = Path(gn1_config).resolve(strict=True)
    expected_config = (repo_root / config_spec["path"]).resolve(strict=True)
    if config_path != expected_config:
        raise ValueError("GN1 config must be the versioned repository config named by the manifest")
    config_entry = {key: config_spec[key] for key in ("path", "size_bytes", "sha256")}
    config_root = config_path.parent
    verify_files(config_root, [{**config_entry, "path": config_path.name}], exact=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("instructions"), dict) or not isinstance(config.get("xclbin"), dict):
        raise ValueError("invalid GN1 config schema")
    if config.get("execution_authorized") is not False or config.get("stop_policy") != "STOP_XDNA1_DECODE_ACCELERATION":
        raise ValueError("GN1 configuration no longer preserves host-only stop policy")

    artifact_root = Path(artifact_root).resolve(strict=True)
    artifacts = manifest["pinned_artifacts"]
    verify_files(artifact_root, artifacts, exact=False)
    declared = {item["path"]: item["sha256"] for item in artifacts}
    config_artifacts = [config["xclbin"], *config["instructions"].values()]
    if any(not isinstance(item, dict) or not isinstance(item.get("path_from_root"), str) or not isinstance(item.get("sha256"), str) for item in config_artifacts):
        raise ValueError("invalid pinned artifact identity in GN1 config")
    for item in config_artifacts:
        if declared.get(item["path_from_root"]) != item["sha256"]:
            raise ValueError(f"GN1 config artifact identity is not pinned by the manifest: {item['path_from_root']}")
    expected_capture = (artifact_root / "repos/llama.cpp" / config["gn1_capture_dir_from_repo_root"]).resolve()
    if expected_capture != Path(gn1_dir).resolve(strict=True):
        raise ValueError("GN1 capture path does not match the versioned config's relative path")
    return {"br1": br1, "gn1": gn1, "artifacts": len(artifacts), "config_sha256": config_spec["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--br1-capture-dir", type=Path, required=True)
    parser.add_argument("--gn1-capture-dir", type=Path, required=True)
    parser.add_argument("--gn1-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.manifest, args.repo_root, args.br1_capture_dir, args.gn1_capture_dir,
                        args.gn1_config, args.artifact_root)
    except FileNotFoundError as exc:
        print(f"EXTERNAL_FIXTURE_PREREQUISITE=NOT_RUN_EXTERNAL_FIXTURE_REQUIRED missing={exc}")
        return 3
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"EXTERNAL_FIXTURE_VERIFY=FAIL error={exc}")
        return 1
    print(f"BR1_EXTERNAL_FIXTURE=PASS files={result['br1']['files']} bytes={result['br1']['size_bytes']}")
    print(f"GN1_EXTERNAL_FIXTURE=PASS files={result['gn1']['files']} bytes={result['gn1']['size_bytes']}")
    print(f"GN1_PINNED_ARTIFACTS=PASS files={result['artifacts']} config_sha256={result['config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
