#!/usr/bin/env python3
"""Host-only checks for the causal matrix plan; never invokes a backend."""
import argparse
import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile

SCRIPT = Path(__file__).with_name("xdna-causal-matrix-plan.py")
spec = importlib.util.spec_from_file_location("causal_matrix_plan", SCRIPT)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    capture_dir = args.capture_dir.resolve(strict=True)
    artifact_root = args.artifact_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    if not capture_dir.is_dir() or not artifact_root.is_dir():
        parser.error("capture and artifact roots must be directories")
    if not config_path.is_file():
        parser.error("--config must name a file")
    planner.ROOT = artifact_root
    a = planner.validate(capture_dir, config_path=config_path, pair="A")
    b = planner.validate(capture_dir, config_path=config_path, pair="B")
    assert a["capture_prefix_mode"]["prefix_capture_ids"] == list(range(1, 13))
    assert a["prefix_counts"] == {"M4096": 4, "M9216": 8}
    assert a["fixed_target_node"] == b["fixed_target_node"] == "ffn_gate-4"
    assert a["pair_source_node"] == "ffn_gate-0" and not a["pair_source_node_is_target"]
    assert b["pair_source_node"] == "ffn_gate-4" and b["pair_source_node_is_target"]
    assert a["execution_records"][-1]["capture_id"] == 2 and b["execution_records"][-1]["capture_id"] == 13
    assert not a["execution_authorized"] and not a["xrt_or_device_touched"]
    assert a["requires_separate_cell_authorization"] and not a["global_runtime_or_device_reset_claimed"]

    suffix = planner.validate(capture_dir, config_path=config_path, pair="B", prefix_ids=planner.parse_prefix_ids("7,8,9,10,11,12"))
    assert suffix["cell_label"] == "B_PREFIX_7_12" and suffix["selected_prefix_ids"] == [7, 8, 9, 10, 11, 12]
    assert [r["capture_id"] for r in suffix["execution_records"]] == [7, 8, 9, 10, 11, 12, 13]
    assert [r["execution_ordinal"] for r in suffix["execution_records"]] == list(range(1, 8))
    singleton = planner.validate(capture_dir, config_path=config_path, pair="B", prefix_ids=[12])
    assert singleton["cell_label"] == "B_PREFIX_12_ONLY" and singleton["prefix_counts"] == {"M4096": 1, "M9216": 0}
    interleaved = planner.validate(capture_dir, config_path=config_path, pair="B", prefix_ids=[1, 4, 7, 12])
    assert interleaved["cell_label"] == "B_PREFIX_IDS_1_4_7_12"
    prefix_1_6 = planner.validate(capture_dir, config_path=config_path, pair="B", prefix_ids=list(range(1, 7)))
    assert prefix_1_6["cell_label"] == "B_PREFIX_1_6" and [r["capture_id"] for r in prefix_1_6["execution_records"]] == [1,2,3,4,5,6,13]
    prefix_10_12 = planner.validate(capture_dir, config_path=config_path, pair="B", prefix_ids=[10, 11, 12])
    assert prefix_10_12["cell_label"] == "B_PREFIX_10_12" and [r["capture_id"] for r in prefix_10_12["execution_records"]] == [10,11,12,13]
    assert planner.canonical_cell_label("B", "CAPTURE_PREFIX", [7,8,9,10,11,12]) == "B_PREFIX_7_12"
    assert planner.canonical_cell_label("B", "CAPTURE_PREFIX", [12]) == "B_PREFIX_12_ONLY"
    assert planner.canonical_cell_label("B", "CAPTURE_PREFIX", [1,4,7,12]) == "B_PREFIX_IDS_1_4_7_12"
    full = planner.validate(capture_dir, config_path=config_path, pair="B")
    assert full["selected_prefix_ids"] == list(range(1, 13)) and full["execution_records"][-1]["capture_id"] == 13
    for invalid in ("", "0", "13", "7,7,8", "8,7", "7,,8", "x", "7-8"):
        try:
            planner.parse_prefix_ids(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid prefix selector: {invalid!r}")

    records = [json.loads(line) for line in (capture_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    with tempfile.TemporaryDirectory() as tmp:
        altered = Path(tmp) / "capture"
        altered.mkdir()
        records = [dict(e) for e in records]
        # Invalidate prefix completion before any large payload reads.
        event = next(e for e in records if e.get("capture_sequence_id") == 1 and e.get("event") == "RESULT_COPY_RETURN")
        records.remove(event)
        manifest = altered / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(e) + "\n" for e in records), encoding="utf-8")
        try:
            planner.validate(altered, "B", config_path=config_path)
        except ValueError as exc:
            assert "manifest SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("modified manifest unexpectedly validated against frozen identity")
        # Pin this deliberately incomplete synthetic manifest to exercise structural validation too.
        planner.CAPTURE_MANIFEST_SHA256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        try:
            planner.validate(altered, "B", config_path=config_path)
        except ValueError as exc:
            assert "completed capture prefix" in str(exc)
        else:
            raise AssertionError("incomplete prefix unexpectedly validated")

    code = SCRIPT.read_text(encoding="utf-8")
    assert "import xrt" not in code and "subprocess" not in code
    print("PLANNER_SUBSET_TESTS=PASS causal matrix host plan: A/B compatibility, fixed target, requested subset families, fail-closed incomplete prefix; no executor/XRT/device path")


if __name__ == "__main__":
    main()
