#!/usr/bin/env python3
"""Strictly certify XDNA capture datasets; no XRT access or replay."""
import argparse
import hashlib
import shutil
import json
import tempfile
from pathlib import Path, PureWindowsPath

SCHEMA_VERSION = 1
ERROR_PHASES = {"prepare", "submit_record", "wait", "output_sync", "output_write", "output_hash", "completion_record"}
SHAPES = (1024, 4096, 8192, 9216)
FROZEN_XCLBIN_SHA256 = "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83"
PROFILE = "q4k_q8k_k2560"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe_artifact(root, name, size, digest):
    if (not isinstance(name, str) or not name or Path(name).name != name or
            PureWindowsPath(name).is_absolute() or "/" in name or "\\\\" in name or name in (".", "..")):
        raise ValueError("unsafe artifact path")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(digest, str):
        raise ValueError("invalid artifact size/SHA metadata")
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        raise ValueError("artifact escapes capture root or is not a regular file")
    data = path.read_bytes()
    if len(data) != size or _sha(data) != digest:
        raise ValueError(f"artifact size/SHA mismatch: {name}")


def _safe_node(node):
    if not isinstance(node, str):
        raise ValueError("node must be a string")
    safe = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in node)
    return safe or "unnamed"


def _expected_name(event, kind):
    return f"{event['submission_id']:06d}_{_safe_node(event['node'])}_M{event['M']}_{kind}.bin"


def _same_submission(prepared, event):
    fields = ("submission_id", "node", "M", "K", "rhs", "opcode", "instruction_file", "instruction_bytes", "instruction_sha256", "q4_file", "q4_bytes", "q4_sha256", "q8_file", "q8_bytes", "q8_sha256")
    if any(event.get(field) != prepared.get(field) for field in fields):
        raise ValueError("submission event metadata changed")


def validate(root, allow_incomplete=False, expected_xclbin_sha256=FROZEN_XCLBIN_SHA256, expected_profile=PROFILE):
    root = Path(root).resolve()
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("run metadata must be a JSON object")
    version = metadata.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ValueError("invalid run metadata/schema version")
    if not isinstance(metadata.get("HEAD"), str) or not metadata["HEAD"] or not isinstance(metadata.get("xclbin_sha256"), str):
        raise ValueError("invalid run metadata identity/XCLBIN SHA")
    if not isinstance(metadata.get("xclbin_path"), str) or not metadata["xclbin_path"]:
        raise ValueError("invalid run metadata XCLBIN path")
    xclbin = Path(metadata.get("xclbin_path", ""))
    if not xclbin.is_file() or _sha(xclbin.read_bytes()) != metadata["xclbin_sha256"]:
        raise ValueError("XCLBIN missing or SHA mismatch")

    if expected_xclbin_sha256 is not None and metadata["xclbin_sha256"] != expected_xclbin_sha256:
        raise ValueError("XCLBIN SHA differs from required replay profile")
    if expected_profile is not None and metadata.get("profile") != expected_profile:
        raise ValueError("capture profile differs from required replay profile")
    states = {}
    last_id = 0
    active_id = None
    last_event_id = 0
    terminal_target_id = None
    saw_error = False
    lines = (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line:
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("manifest event must be a JSON object")
        sid = event.get("submission_id")
        kind = event.get("event")
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 0:
            raise ValueError("invalid submission ID")
        if saw_error:
            raise ValueError("event after terminal ERROR")
        if terminal_target_id is not None:
            raise ValueError("event after semantic target")
        last_event_id = sid

        if kind == "ERROR":
            if event.get("phase") not in ERROR_PHASES or not isinstance(event.get("error_kind"), str) or not event["error_kind"] or not isinstance(event.get("message"), str):
                raise ValueError("malformed ERROR event")
            phase = event["phase"]
            state = states.get(sid)
            if phase == "prepare":
                if active_id is not None or sid != last_id + 1:
                    raise ValueError("prepare ERROR ID/order invalid")
                last_id = sid
            elif phase == "submit_record":
                if state is None or active_id != sid or state["submitted"]:
                    raise ValueError("submit_record ERROR out of order")
            elif state is None or active_id != sid or not state["submitted"]:
                raise ValueError("post-submit ERROR out of order")
            saw_error = True
            if not allow_incomplete:
                raise ValueError(f"capture ERROR at submission {sid}: {event['phase']}")
            continue

        if sid <= 0:
            raise ValueError("submission ID must be positive")
        state = states.get(sid)
        if kind == "PREPARED":
            if active_id is not None or state is not None or sid != last_id + 1:
                raise ValueError("duplicate, overlapping, non-monotonic, or gapped submission ID")
            last_id = sid
            required = ("node", "M", "K", "rhs", "opcode", "instruction_file", "instruction_bytes", "instruction_sha256", "q4_file", "q4_bytes", "q4_sha256", "q8_file", "q8_bytes", "q8_sha256", "submitted", "completed", "output_file", "output_bytes", "output_sha256", "stop_reason")
            if any(key not in event for key in required):
                raise ValueError("missing required manifest field")
            if event["submitted"] is not False or event["completed"] is not False:
                raise ValueError("PREPARED state flags invalid")
            if event["M"] not in SHAPES or event["K"] != 2560 or event["rhs"] != 1 or event["opcode"] != 3:
                raise ValueError("unsupported shape/K/rhs/opcode")
            if (not isinstance(event["instruction_bytes"], int) or event["instruction_bytes"] <= 0 or event["instruction_bytes"] % 4 or
                    event["q4_bytes"] != event["M"] * 1520 or event["q8_bytes"] != 2920):
                raise ValueError("artifact size contract mismatch")
            if event["M"] == 9216 and event["instruction_bytes"] != 189108:
                raise ValueError("M9216 instruction size differs from frozen artifact")
            for label, suffix in (("instruction", "instr"), ("q4", "q4"), ("q8", "q8")):
                if event[f"{label}_file"] != _expected_name(event, suffix):
                    raise ValueError("unexpected artifact filename")
                _safe_artifact(root, event[f"{label}_file"], event[f"{label}_bytes"], event[f"{label}_sha256"])
            target = event["node"] == "ffn_gate-4" and event["K"] == 2560 and event["M"] == 9216 and event["rhs"] == 1
            expected_stop = "semantic_target" if target else ""
            if event["stop_reason"] != expected_stop:
                raise ValueError("semantic target predicate mismatch")
            if target and (event["output_file"] not in ("", None) or event["output_bytes"] not in (0, None) or event["output_sha256"] not in ("", None)):
                raise ValueError("semantic target must not contain output data")
            if not target and (event["output_file"] not in ("", None) or event["output_bytes"] not in (0, None) or event["output_sha256"] not in ("", None)):
                raise ValueError("PREPARED output placeholders must be empty")
            states[sid] = {"prepared": event, "submitted": False, "completed": False}
            active_id = sid
            if target:
                terminal_target_id = sid
                active_id = None
        elif kind == "SUBMITTED":
            if state is None or active_id != sid or state["submitted"] or state["prepared"]["stop_reason"] == "semantic_target":
                raise ValueError("SUBMITTED without immediately active PREPARED")
            _same_submission(state["prepared"], event)
            if event.get("submitted") is not True or event.get("completed") is not False:
                raise ValueError("SUBMITTED flags invalid")
            state["submitted"] = True
        elif kind == "COMPLETED":
            if state is None or active_id != sid or not state["submitted"] or state["completed"]:
                raise ValueError("COMPLETED without active SUBMITTED")
            _same_submission(state["prepared"], event)
            if event.get("submitted") is not True or event.get("completed") is not True:
                raise ValueError("COMPLETED flags invalid")
            if event["output_bytes"] != event["M"] * 4:
                raise ValueError("completed output size mismatch")
            if event.get("output_file") != _expected_name(event, "out"):
                raise ValueError("unexpected output filename")
            _safe_artifact(root, event["output_file"], event["output_bytes"], event["output_sha256"])
            state["completed"] = True
            active_id = None
        else:
            raise ValueError(f"unknown event: {kind}")

    ready = not saw_error and active_id is None and terminal_target_id is not None and last_event_id == terminal_target_id
    if ready:
        ready = all(s["submitted"] and s["completed"] for sid, s in states.items() if sid != terminal_target_id)
        ready = ready and len([s for s in states.values() if s["prepared"]["stop_reason"] == "semantic_target"]) == 1
    if not ready and not allow_incomplete:
        raise ValueError("dataset is incomplete or has no unique terminal semantic target")
    return {"submissions": len(states), "replay_ready": bool(ready and not allow_incomplete)}


def _event(kind, sid, node, m, stop="", submitted=False, completed=False, output="", output_size=0, output_sha=""):
    def artifact(label, data, suffix=None):
        return {f"{label}_file": f"{sid:06d}_{_safe_node(node)}_M{m}_{suffix or label}.bin", f"{label}_bytes": len(data), f"{label}_sha256": _sha(data)}
    ins = b"I" * (189108 if m == 9216 else 4)
    q4, q8 = b"Q" * (m * 1520), b"A" * 2920
    event = {"event": kind, "submission_id": sid, "node": node, "M": m, "K": 2560, "rhs": 1, "opcode": 3}
    event.update(artifact("instruction", ins, "instr"))
    event.update(artifact("q4", q4))
    event.update(artifact("q8", q8))
    event.update({"submitted": submitted, "completed": completed, "output_file": output,
                  "output_bytes": output_size, "output_sha256": output_sha, "stop_reason": stop})
    return event


def _fixture(root, xclbin_path):
    root.mkdir()
    xclbin = Path(xclbin_path).resolve(strict=True)
    meta = {"schema_version": SCHEMA_VERSION, "HEAD": "synthetic", "branch": None,
            "capture_mode": "GGML_XDNA_CAPTURE_EXACT_HISTORY=1", "profile": PROFILE,
            "xclbin_path": str(xclbin.resolve()), "xclbin_sha256": _sha(xclbin.read_bytes())}
    (root / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = []
    for sid, node, m in ((1, "ordinary", 1024), (2, "ffn_gate-4", 9216)):
        prepared = _event("PREPARED", sid, node, m, "semantic_target" if sid == 2 else "")
        for label, data in (("instruction", b"I" * (189108 if m == 9216 else 4)), ("q4", b"Q" * (m * 1520)), ("q8", b"A" * 2920)):
            (root / prepared[f"{label}_file"]).write_bytes(data)
        lines.append(prepared)
        if sid == 1:
            lines.append(_event("SUBMITTED", sid, node, m, submitted=True))
            output = b"O" * (m * 4)
            output_name = f"{sid:06d}_{_safe_node(node)}_M{m}_out.bin"
            (root / output_name).write_bytes(output)
            lines.append(_event("COMPLETED", sid, node, m, submitted=True, completed=True,
                                output=output_name, output_size=len(output), output_sha=_sha(output)))
    _write_lines(root, lines)


def _write_lines(root, events):
    (root / "manifest.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events), encoding="utf-8")


def _expect_invalid(root, label, allow_incomplete=False):
    try:
        result = validate(root, allow_incomplete)
        if result["replay_ready"]:
            raise AssertionError(f"{label}: invalid dataset marked replay-ready")
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError):
        pass


def self_test(xclbin_path):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "capture"
        _fixture(root, xclbin_path)
        result = validate(root)
        assert result == {"submissions": 2, "replay_ready": True}
        original = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]

        events = original + [dict(original[0])]
        _write_lines(root, events); _expect_invalid(root, "duplicate ID")
        events = [dict(e) for e in original]
        for group, new_id in ((events[:3], 2), (events[3:], 1)):
            for event in group:
                event["submission_id"] = new_id
                for label, suffix in (("instruction", "instr"), ("q4", "q4"), ("q8", "q8")):
                    old_name = event[f"{label}_file"]
                    new_name = _expected_name(event, suffix)
                    shutil.copyfile(root / old_name, root / new_name)
                    event[f"{label}_file"] = new_name
                if event.get("output_file"):
                    old_name = event["output_file"]
                    new_name = _expected_name(event, "out")
                    shutil.copyfile(root / old_name, root / new_name)
                    event["output_file"] = new_name
        _write_lines(root, events); _expect_invalid(root, "non-monotonic ID")
        events = [dict(e) for e in original]; events[0]["q4_file"] = "missing.bin"
        _write_lines(root, events); _expect_invalid(root, "missing input")
        events = [dict(e) for e in original]; events[0]["q4_sha256"] = "0" * 64
        _write_lines(root, events); _expect_invalid(root, "bad input SHA")
        events = [dict(e) for e in original]; events[2]["output_file"] = "missing.bin"
        _write_lines(root, events); _expect_invalid(root, "missing output")
        events = [dict(e) for e in original]; events[2]["output_sha256"] = "0" * 64
        _write_lines(root, events); _expect_invalid(root, "bad output SHA")
        events = [dict(original[1])]
        _write_lines(root, events); _expect_invalid(root, "SUBMITTED without PREPARED")
        events = [dict(original[0]), dict(original[2])]
        _write_lines(root, events); _expect_invalid(root, "COMPLETED without SUBMITTED")
        events = original + [{"event": "PREPARED", "submission_id": 3}]
        _write_lines(root, events); _expect_invalid(root, "target followed by later ID")
        events = [dict(e) for e in original]; events[-1]["submitted"] = True
        _write_lines(root, events); _expect_invalid(root, "target submitted")
        events = original + [{"event": "PREPARED", "submission_id": 3, "node": "ffn_gate-4", "M": 9216, "K": 2560, "rhs": 1, "opcode": 3, "stop_reason": "semantic_target"}]
        _write_lines(root, events); _expect_invalid(root, "multiple targets")

        # POST-SUBMISSION ERROR is inspectable only with --allow-incomplete and never ready.
        events = original[:2] + [{"event": "ERROR", "submission_id": 1, "phase": "output_write", "error_kind": "artifact_write", "message": "injected"}]
        _write_lines(root, events)
        _expect_invalid(root, "ERROR strict")
        assert validate(root, allow_incomplete=True) == {"submissions": 1, "replay_ready": False}
        events.append({"event": "PREPARED", "submission_id": 2})
        _write_lines(root, events); _expect_invalid(root, "submission after ERROR", allow_incomplete=True)
        # Pre-submit failures have no SUBMITTED event; post-submit failures remain invalid.
        for failed_events, label in (
            ([{"event": "ERROR", "submission_id": 1, "phase": "prepare", "error_kind": "artifact_write", "message": "injected"}], "prepare failure"),
            (original[:1] + [{"event": "ERROR", "submission_id": 1, "phase": "submit_record", "error_kind": "manifest_write", "message": "injected"}], "submit-record failure"),
            (original[:2] + [{"event": "ERROR", "submission_id": 1, "phase": "completion_record", "error_kind": "manifest_write", "message": "injected"}], "completion-record failure"),
        ):
            _write_lines(root, failed_events)
            _expect_invalid(root, label)
            assert validate(root, allow_incomplete=True)["replay_ready"] is False

        # PREPARED-only and SUBMITTED-only histories are never ready.
        _write_lines(root, original[:1]); assert not validate(root, allow_incomplete=True)["replay_ready"]
        _write_lines(root, original[:2]); assert not validate(root, allow_incomplete=True)["replay_ready"]
        # Non-empty directory contract used by the backend initializer.
        assert any(root.iterdir())
    print("XDNA capture validator self-test: PASS (valid, duplicate/order, artifacts/SHA, transitions, target, ERROR/no-later-submit, incomplete)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", nargs="?")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--xclbin-path", type=Path, help="explicit frozen XCLBIN required by --self-test")
    args = parser.parse_args()
    if args.self_test:
        if not args.xclbin_path:
            parser.error("--self-test requires --xclbin-path")
        self_test(args.xclbin_path)
    elif args.capture_dir:
        try:
            result = validate(args.capture_dir, args.allow_incomplete)
            print(f"validated submissions: {result['submissions']}")
            print(f"REPLAY_READY={'YES' if result['replay_ready'] else 'NO'}")
            if not result["replay_ready"]:
                raise SystemExit(1)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            print("REPLAY_READY=NO")
            print(f"validation failed: {exc}")
            raise SystemExit(1)
    else:
        parser.error("provide capture_dir or --self-test")
