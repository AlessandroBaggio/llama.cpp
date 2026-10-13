#!/usr/bin/env python3
"""Small hermetic checks for GN1 selector and fail-closed input handling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

SCRIPT = Path(__file__).with_name("xdna-causal-matrix-plan.py")
spec = importlib.util.spec_from_file_location("causal_matrix_plan_synthetic", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load causal matrix planner")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)

assert planner.parse_prefix_ids("1,4,7,12") == [1, 4, 7, 12]
assert planner.canonical_cell_label("B", "CAPTURE_PREFIX", [7, 8, 9]) == "B_PREFIX_7_9"
assert planner.canonical_cell_label("B", "CAPTURE_PREFIX", [12]) == "B_PREFIX_12_ONLY"
assert planner.canonical_cell_label("A", "CLEAN", []) == "A_CLEAN"
for invalid in ("", "0", "13", "7,7,8", "8,7", "7,,8", "x", "7-8"):
    try:
        planner.parse_prefix_ids(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted invalid prefix selector: {invalid!r}")

with tempfile.TemporaryDirectory(prefix="xdna-gn1-synthetic-") as temp:
    root = Path(temp)
    missing = root / "missing-capture"
    try:
        planner.validate(missing, pair="B", config_path=planner.CONFIG)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing explicit capture directory was accepted")

    corrupted = root / "corrupted-capture"
    corrupted.mkdir()
    (corrupted / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        planner.validate(corrupted, pair="B", config_path=planner.CONFIG)
    except ValueError as exc:
        assert "manifest SHA256 mismatch" in str(exc)
    else:
        raise AssertionError("corrupted manifest was accepted")

    events = []
    complete_events = ("KERNEL_CALL_RETURNED", "WAIT_ENTER", "WAIT_RETURN", "OUTPUT_SYNC_RETURN", "RESULT_COPY_RETURN")
    for capture_id in range(1, 14):
        events.append({"event": "PREPARED", "capture_sequence_id": capture_id, "shape": {"m": 4096 if capture_id in (1, 4, 7, 12) else 9216}})
        kinds = complete_events if capture_id <= 12 else ("KERNEL_CALL_RETURNED", "WAIT_ENTER")
        if capture_id == 1:
            kinds = kinds[:-1]
        events.extend({"event": kind, "capture_sequence_id": capture_id} for kind in kinds)
    synthetic_manifest = corrupted / "manifest.jsonl"
    synthetic_manifest.write_text("".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events), encoding="utf-8")
    frozen_manifest_sha = planner.CAPTURE_MANIFEST_SHA256
    try:
        planner.CAPTURE_MANIFEST_SHA256 = hashlib.sha256(synthetic_manifest.read_bytes()).hexdigest()
        try:
            planner.validate(corrupted, pair="B", config_path=planner.CONFIG)
        except ValueError as exc:
            assert "completed capture prefix" in str(exc)
        else:
            raise AssertionError("incomplete synthetic prefix was accepted")
    finally:
        planner.CAPTURE_MANIFEST_SHA256 = frozen_manifest_sha

    bad_config = root / "bad-config.json"
    bad_config.write_bytes(planner.CONFIG.read_bytes() + b" ")
    original_config = planner.CONFIG
    try:
        planner.CONFIG = bad_config
        try:
            planner.validate(missing, pair="B", config_path=bad_config)
        except ValueError as exc:
            assert "config SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("modified config was accepted")
    finally:
        planner.CONFIG = original_config

print("XDNA_CAUSAL_MATRIX_SYNTHETIC=PASS selectors=PASS missing_capture=REJECTED corrupted_manifest=REJECTED incomplete_prefix=REJECTED modified_config=REJECTED")
