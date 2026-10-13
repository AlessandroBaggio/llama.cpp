#!/usr/bin/env python3
"""Host-only validator/planner for the frozen causal matrix; never loads XRT."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "repos/llama.cpp/experiments/xdna-gn1-causal-matrix-gp1.json"
GP1_CONFIG_SHA256 = "de4fe161a3e256bce55939e837f77b527ab347222285fc7fb23be582bd1623df"
CAPTURE_MANIFEST_SHA256 = "9f4dc51960c6200c9a82e8b3902a5fa540dd5fc9e9a7ffdcfcc8201f67706c78"
PAIR_B_Q4_SHA256 = "b442633de4f499245e18224bc8da7da5defddd81974cf58ca4a2c8693f5f003a"
PAIR_B_Q8_SHA256 = "02876a59bd6251377c7d76806fd690b42aa0a579cf43e37399bcad3d0957332e"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_prefix_ids(raw):
    if not isinstance(raw, str) or not raw:
        raise ValueError("prefix IDs must be a non-empty comma-separated list")
    tokens = raw.split(",")
    ids = []
    for token in tokens:
        if not token or not token.isascii() or not token.isdecimal():
            raise ValueError("prefix IDs must contain strict decimal integers with no empty tokens")
        value = int(token, 10)
        if token != str(value):
            raise ValueError("prefix IDs must use canonical decimal integers")
        if not 1 <= value <= 12:
            raise ValueError("prefix IDs must be in 1..12")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise ValueError("prefix IDs must not contain duplicates")
    if ids != sorted(ids):
        raise ValueError("prefix IDs must be in original capture order")
    return ids


def canonical_cell_label(pair, mode, prefix_ids):
    if mode == "CLEAN":
        return f"{pair}_CLEAN"
    if not prefix_ids:
        raise ValueError("CAPTURE_PREFIX must select at least one prefix ID")
    if prefix_ids == list(range(prefix_ids[0], prefix_ids[-1] + 1)):
        if len(prefix_ids) == 1:
            return f"{pair}_PREFIX_{prefix_ids[0]}_ONLY"
        return f"{pair}_PREFIX_{prefix_ids[0]}_{prefix_ids[-1]}"
    return f"{pair}_PREFIX_IDS_" + "_".join(map(str, prefix_ids))


def validate(capture_dir=None, pair="B", mode="CAPTURE_PREFIX", config_path=CONFIG, prefix_ids=None):
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config_path != CONFIG.resolve():
        raise ValueError("executor accepts only the frozen GP1 matrix config")
    if sha(config_path) != GP1_CONFIG_SHA256:
        raise ValueError("frozen GP1 matrix config SHA256 mismatch")
    if config.get("schema_version") != 1 or config.get("fixed_target") != {"node": "ffn_gate-4", "m": 9216, "k": 2560}:
        raise ValueError("GP1 config schema or fixed target mismatch")
    if config.get("prefix_mode") != {"source": "GN1 captures 1-12 exactly", "m4096_count": 4, "m9216_count": 8, "exclude_target_capture": 13}:
        raise ValueError("GP1 prefix contract mismatch")
    if config.get("clean_mode") != {"prefix_capture_ids": []}:
        raise ValueError("GP1 CLEAN contract mismatch")
    if config["execution_authorized"] or config["stop_policy"] != "STOP_XDNA1_DECODE_ACCELERATION":
        raise ValueError("GP1 must remain host-only under the active stop policy")
    if mode not in ("CLEAN", "CAPTURE_PREFIX"):
        raise ValueError("mode must be CLEAN or CAPTURE_PREFIX")
    if pair not in ("A", "B"):
        raise ValueError("pair must be A or B")
    if mode == "CLEAN":
        if prefix_ids is not None:
            raise ValueError("--prefix-ids is valid only with CAPTURE_PREFIX")
        selected_ids = []
    elif prefix_ids is None:
        selected_ids = list(range(1, 13))  # Deliberate legacy default: full frozen prefix.
    else:
        if pair != "B":
            raise ValueError("explicit prefix subsets are supported only for Pair B")
        selected_ids = list(prefix_ids)
        if not selected_ids or any(type(i) is not int or not 1 <= i <= 12 for i in selected_ids):
            raise ValueError("prefix IDs must be a non-empty list in 1..12")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("prefix IDs must not contain duplicates")
        if selected_ids != sorted(selected_ids):
            raise ValueError("prefix IDs must be in original capture order")

    capture_dir = Path(capture_dir or ROOT / "repos/llama.cpp" / config["gn1_capture_dir_from_repo_root"]).resolve(strict=True)
    manifest_path = capture_dir / "manifest.jsonl"
    if sha(manifest_path) != CAPTURE_MANIFEST_SHA256:
        raise ValueError("frozen capture manifest SHA256 mismatch")
    events = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    prepared = [e for e in events if e.get("event") == "PREPARED"]
    if [e.get("capture_sequence_id") for e in prepared] != list(range(1, 14)):
        raise ValueError("XDNA_CAUSAL must contain exactly contiguous PREPARED captures 1..13")
    by_id = {e["capture_sequence_id"]: e for e in prepared}
    full_prefix = prepared[:12]
    if sum(e["shape"]["m"] == 4096 for e in full_prefix) != 4 or sum(e["shape"]["m"] == 9216 for e in full_prefix) != 8:
        raise ValueError("frozen capture prefix must contain four M4096 and eight M9216 records")
    completed_events = ["KERNEL_CALL_RETURNED", "WAIT_ENTER", "WAIT_RETURN", "OUTPUT_SYNC_RETURN", "RESULT_COPY_RETURN"]
    for sid in range(1, 13):
        kinds = [e["event"] for e in events if e.get("capture_sequence_id") == sid and e["event"] != "PREPARED"]
        if kinds != completed_events:
            raise ValueError(f"capture {sid} is not a completed capture prefix record")
    kinds13 = [e["event"] for e in events if e.get("capture_sequence_id") == 13 and e["event"] != "PREPARED"]
    if kinds13 != ["KERNEL_CALL_RETURNED", "WAIT_ENTER"]:
        raise ValueError("capture 13 must remain the stalled target boundary")

    # Validate immutable manifest metadata for all records; require payload files only
    # for selected IDs and the fixed pair source, so unrelated captures cannot block a subset.
    for sid, record in by_id.items():
        m = record["shape"]["m"]
        if record.get("schema_version") != 1 or record.get("op") != "MUL_MAT" or record["shape"].get("k") != 2560:
            raise ValueError(f"capture {sid}: unsupported operation/schema/shape")
        if record["src0"].get("ne") != [2560, m, 1, 1] or record["src1"].get("ne") != [2560, 1, 1, 1] or record["dst"].get("ne") != [m, 1, 1, 1]:
            raise ValueError(f"capture {sid}: tensor extents do not match MxK profile")
        if record["q4_native"]["bytes"] != m * 1440 or record["q8"]["bytes"] != 2920:
            raise ValueError(f"capture {sid}: incompatible native Q4/Q8 payload lengths")

    xclbin = config["xclbin"]
    if sha(ROOT / xclbin["path_from_root"]) != xclbin["sha256"]:
        raise ValueError("frozen XCLBIN unavailable or hash mismatch")
    for m, artifact in config["instructions"].items():
        if sha(ROOT / artifact["path_from_root"]) != artifact["sha256"]:
            raise ValueError(f"M{m} instruction artifact unavailable or hash mismatch")

    source_info = config[f"pair_{pair.lower()}"]
    source = by_id[source_info["capture_id"]]
    target = by_id[13]
    fixed = config["fixed_target"]
    if source["node_name"] != source_info["source_node"] or target["node_name"] != fixed["node"]:
        raise ValueError("XDNA_CAUSAL source/target node identity changed")
    if source["shape"] != target["shape"] or source["shape"] != {"m": fixed["m"], "k": fixed["k"]}:
        raise ValueError("selected pair is not shape-compatible with fixed target")
    if source["q4_native"]["bytes"] != target["q4_native"]["bytes"] or source["q8"]["bytes"] != target["q8"]["bytes"]:
        raise ValueError("selected pair is not byte-length compatible with fixed target")
    if source["q4_native"]["sha256"] != source_info["q4_native_sha256"] or source["q8"]["sha256"] != source_info["q8_sha256"]:
        raise ValueError("selected pair does not match its frozen causal identity")
    if pair == "B" and (source_info["capture_id"] != 13 or source_info["q4_native_sha256"] != PAIR_B_Q4_SHA256 or source_info["q8_sha256"] != PAIR_B_Q8_SHA256):
        raise ValueError("Pair B target source/hash identity changed")

    required_payload_ids = set(selected_ids) | {source_info["capture_id"]}
    for sid in sorted(required_payload_ids):
        record = by_id[sid]
        for key in ("q4_native", "q8"):
            ref = record[key]
            path = (capture_dir / ref["file"]).resolve(strict=True)
            if capture_dir not in path.parents or path.stat().st_size != ref["bytes"] or sha(path) != ref["sha256"]:
                raise ValueError(f"capture {sid}: invalid {key} artifact")

    def planned_record(ordinal, capture_id, record, target_record=False, payload_record=None):
        m = record["shape"]["m"]
        payload = payload_record or record
        inst = config["instructions"].get(str(m))
        if inst is None:
            raise ValueError(f"M{m} instruction artifact is unavailable")
        return {
            "execution_ordinal": ordinal, "sequence_index": ordinal, "capture_id": capture_id,
            "node_name": fixed["node"] if target_record else record["node_name"],
            "m": m, "k": record["shape"]["k"], "target": target_record,
            "q4_file": str((capture_dir / payload["q4_native"]["file"]).resolve()),
            "q4_bytes": payload["q4_native"]["bytes"], "q4_sha256": payload["q4_native"]["sha256"],
            "q8_file": str((capture_dir / payload["q8"]["file"]).resolve()),
            "q8_bytes": payload["q8"]["bytes"], "q8_sha256": payload["q8"]["sha256"],
            "instruction_file": str((ROOT / inst["path_from_root"]).resolve()),
            "instruction_sha256": inst["sha256"],
        }

    prefix_records = [planned_record(i, sid, by_id[sid]) for i, sid in enumerate(selected_ids, 1)]
    target_ordinal = len(prefix_records) + 1 if mode == "CAPTURE_PREFIX" else 1
    target_record = planned_record(target_ordinal, source_info["capture_id"], target, True, source)
    execution_records = prefix_records + [target_record]
    cell_label = canonical_cell_label(pair, mode, selected_ids)
    selected_counts = {"M4096": sum(by_id[sid]["shape"]["m"] == 4096 for sid in selected_ids),
                       "M9216": sum(by_id[sid]["shape"]["m"] == 9216 for sid in selected_ids)}
    return {
        "classification": "HOST_PLAN_VALID_NO_EXECUTOR_OR_AUTHORIZATION",
        "execution_authorized": False, "xrt_or_device_touched": False,
        "fixed_target_node": fixed["node"], "pair": pair, "mode": mode,
        "executor_route": "XDNA_CAUSAL_DIAGNOSTIC", "cell_label": cell_label,
        "selected_prefix_ids": selected_ids, "prefix_submission_count": len(prefix_records),
        "target_submission_count": 1, "total_submission_count": len(execution_records),
        "execution_records": execution_records, "execution_order": execution_records,
        "pair_source_capture": source_info["capture_id"], "pair_source_node": source_info["source_node"],
        "pair_source_node_is_target": source_info["source_node"] == fixed["node"],
        "clean_mode": {"prefix_capture_ids": [], "target": {"node": fixed["node"], "pair": pair}},
        "capture_prefix_mode": {"prefix_capture_ids": selected_ids, "target": {"node": fixed["node"], "pair": pair}},
        "prefix_counts": selected_counts, "target_capture_13_excluded_from_prefix": True,
        "xclbin_sha256": xclbin["sha256"],
        "instruction_sha256": {m: x["sha256"] for m, x in config["instructions"].items()},
        "payload_source": str(capture_dir), "output_reference_bytes_captured": config["output_reference_bytes_captured"],
        "runtime_creation_equivalence": "SAME_FACTORY_AND_ARTIFACT_SETTINGS_REQUIRED; object lifetimes/order not asserted equivalent",
        "per_cell_fresh_process_required": config["per_cell_fresh_process_required"],
        "global_runtime_or_device_reset_claimed": config["global_runtime_or_device_reset_claimed"],
        "requires_separate_cell_authorization": config["separate_cell_authorization_required"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("CLEAN", "CAPTURE_PREFIX"), required=True)
    parser.add_argument("--pair", choices=("A", "B"), required=True)
    parser.add_argument("--prefix-ids", help="strict comma-separated original capture IDs 1..12; explicit subsets are Pair B only")
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--plan-out", type=Path)
    args = parser.parse_args()
    ids = parse_prefix_ids(args.prefix_ids) if args.prefix_ids is not None else None
    plan = validate(args.capture_dir, args.pair, args.mode, args.config, ids)
    encoded = json.dumps(plan, indent=2) + "\n"
    if args.plan_out:
        args.plan_out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
