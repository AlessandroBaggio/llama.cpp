#!/usr/bin/env python3
"""Build a strict, host-only replay plan from a frozen capture schema-v1."""
import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys

SCHEMA_VERSION = 1
PROFILE = "q4k_q8k_k2560"
FROZEN_XCLBIN_SHA256 = "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83"


def _load_capture_validator():
    path = Path(__file__).with_name("validate-xdna-capture.py")
    spec = importlib.util.spec_from_file_location("xdna_capture_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared XDNA capture validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_VALIDATOR = _load_capture_validator()


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    submission_id: int
    node: str
    M: int
    K: int
    rhs: int
    opcode: int
    instruction_file: str
    instruction_bytes: int
    instruction_sha256: str
    q4_file: str
    q4_bytes: int
    q4_sha256: str
    q8_file: str
    q8_bytes: int
    q8_sha256: str
    output_file: str | None
    output_bytes: int | None
    output_sha256: str | None
    semantic_target: bool


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    schema_version: int
    HEAD: str
    branch: str | None
    capture_mode: str | None
    profile: str
    xclbin_path: str
    xclbin_sha256: str
    source_capture_dir: str
    records: tuple[ReplayRecord, ...]

    def as_dict(self):
        value = asdict(self)
        value["records"] = list(value["records"])
        return value


def build_plan(capture_dir) -> ReplayPlan:
    root = Path(capture_dir).resolve(strict=True)
    # Invoke the authoritative strict validator, not a second looser parser.
    _VALIDATOR.validate(
        root,
        expected_xclbin_sha256=FROZEN_XCLBIN_SHA256,
        expected_profile=PROFILE,
    )
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    branch = metadata.get("branch")
    capture_mode = metadata.get("capture_mode")
    if branch is not None and not isinstance(branch, str):
        raise ValueError("run_metadata.branch must be a string or null")
    if capture_mode is not None and not isinstance(capture_mode, str):
        raise ValueError("run_metadata.capture_mode must be a string or null")

    records = []
    prepared = None
    submitted = False
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        event = json.loads(line)
        kind = event["event"]
        if kind == "PREPARED":
            if prepared is not None:
                raise ValueError("shared validator/parser transition disagreement")
            target = event["stop_reason"] == "semantic_target"
            prepared = {
                "submission_id": event["submission_id"],
                "node": event["node"],
                "M": event["M"],
                "K": event["K"],
                "rhs": event["rhs"],
                "opcode": event["opcode"],
                "instruction_file": event["instruction_file"],
                "instruction_bytes": event["instruction_bytes"],
                "instruction_sha256": event["instruction_sha256"],
                "q4_file": event["q4_file"],
                "q4_bytes": event["q4_bytes"],
                "q4_sha256": event["q4_sha256"],
                "q8_file": event["q8_file"],
                "q8_bytes": event["q8_bytes"],
                "q8_sha256": event["q8_sha256"],
                "output_file": None,
                "output_bytes": None,
                "output_sha256": None,
                "semantic_target": target,
            }
            submitted = False
            if target:
                records.append(ReplayRecord(**prepared))
                prepared = None
        elif kind == "SUBMITTED":
            if prepared is None or submitted:
                raise ValueError("shared validator/parser transition disagreement")
            submitted = True
        elif kind == "COMPLETED":
            if prepared is None or not submitted:
                raise ValueError("shared validator/parser transition disagreement")
            prepared["output_file"] = event["output_file"]
            prepared["output_bytes"] = event["output_bytes"]
            prepared["output_sha256"] = event["output_sha256"]
            records.append(ReplayRecord(**prepared))
            prepared = None
            submitted = False
        else:
            raise ValueError(f"unexpected validated event: {kind}")
    if prepared is not None or not records or not records[-1].semantic_target:
        raise ValueError("shared validator/parser terminal-state disagreement")
    ids = tuple(record.submission_id for record in records)
    if ids != tuple(range(1, len(records) + 1)):
        raise ValueError("submission IDs must be contiguous from 1")
    return ReplayPlan(
        schema_version=metadata["schema_version"],
        HEAD=metadata["HEAD"],
        branch=branch,
        capture_mode=capture_mode,
        profile=metadata["profile"],
        xclbin_path=metadata["xclbin_path"],
        xclbin_sha256=metadata["xclbin_sha256"],
        source_capture_dir=str(root),
        records=tuple(records),
    )


def _print_plan(plan):
    target = plan.records[-1]
    print("REPLAY_PLAN_VALID=YES")
    print(f"schema_version={plan.schema_version}")
    print(f"submission_count={len(plan.records)}")
    print(f"historical_completed={len(plan.records) - 1}")
    print(f"target_submission_id={target.submission_id}")
    print(f"target_node={target.node}")
    print("| ID | node | M | instr bytes | Q4 bytes | Q8 bytes | output bytes | target |")
    print("| --: | ---- | -: | ----------: | -------: | -------: | -----------: | :----: |")
    for record in plan.records:
        output = "—" if record.output_bytes is None else str(record.output_bytes)
        print(f"| {record.submission_id} | {record.node} | {record.M} | {record.instruction_bytes} | {record.q4_bytes} | {record.q8_bytes} | {output} | {'YES' if record.semantic_target else 'NO'} |")


def _fixture(root, xclbin_path, shapes=(4096, 9216, 1024, 8192), include_target=True):
    """Create a host-only schema-v1 fixture tied to an explicitly supplied XCLBIN."""
    import hashlib
    xclbin = Path(xclbin_path).resolve(strict=True)
    digest = hashlib.sha256(xclbin.read_bytes()).hexdigest()
    root.mkdir()
    meta = {
        "schema_version": 1, "HEAD": "synthetic", "branch": "host-test",
        "capture_mode": "GGML_XDNA_CAPTURE_EXACT_HISTORY=1",
        "profile": PROFILE, "xclbin_path": str(xclbin.resolve()), "xclbin_sha256": digest,
    }
    (root / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    events = []
    entries = [(m, f"node-{m}", False) for m in shapes]
    if include_target:
        entries.append((9216, "ffn_gate-4", True))
    for sid, (m, node, target) in enumerate(entries, 1):
        instruction = b"I" * (189108 if m == 9216 else 4)
        q4, q8 = b"Q" * (m * 1520), b"A" * 2920
        event = {"event": "PREPARED", "submission_id": sid, "node": node, "M": m,
                 "K": 2560, "rhs": 1, "opcode": 3, "submitted": False,
                 "completed": False, "output_file": "", "output_bytes": 0,
                 "output_sha256": "", "stop_reason": "semantic_target" if target else ""}
        for label, suffix, data in (("instruction", "instr", instruction), ("q4", "q4", q4), ("q8", "q8", q8)):
            name = f"{sid:06d}_{node}_M{m}_{suffix}.bin"
            (root / name).write_bytes(data)
            event.update({f"{label}_file": name, f"{label}_bytes": len(data),
                          f"{label}_sha256": hashlib.sha256(data).hexdigest()})
        events.append(event)
        if not target:
            submitted_event = dict(event, event="SUBMITTED", submitted=True)
            events.append(submitted_event)
            output = b"O" * (m * 4)
            output_name = f"{sid:06d}_{node}_M{m}_out.bin"
            (root / output_name).write_bytes(output)
            events.append(dict(event, event="COMPLETED", submitted=True, completed=True,
                               output_file=output_name, output_bytes=len(output),
                               output_sha256=hashlib.sha256(output).hexdigest()))
    (root / "manifest.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events), encoding="utf-8")
    return events


def _self_test(xclbin_path):
    import copy
    import hashlib
    import tempfile

    xclbin = Path(xclbin_path).resolve(strict=True)
    if not xclbin.is_file() or hashlib.sha256(xclbin.read_bytes()).hexdigest() != FROZEN_XCLBIN_SHA256:
        raise RuntimeError("explicit frozen XCLBIN unavailable for host-only positive fixture")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "multi"
        _fixture(root, xclbin)
        plan = build_plan(root)
        assert [r.M for r in plan.records] == [4096, 9216, 1024, 8192, 9216]
        from contextlib import redirect_stdout
        import io
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _print_plan(plan)
        assert "REPLAY_PLAN_VALID=YES" in rendered.getvalue() and "target_node=ffn_gate-4" in rendered.getvalue()
        assert [r.submission_id for r in plan.records] == [1, 2, 3, 4, 5]
        assert plan.source_capture_dir == str(root.resolve()) and plan.xclbin_sha256 == FROZEN_XCLBIN_SHA256
        assert plan.branch == "host-test" and plan.capture_mode == "GGML_XDNA_CAPTURE_EXACT_HISTORY=1"
        assert plan.records[0].output_bytes == plan.records[0].M * 4
        assert plan.records[-1].semantic_target and plan.records[-1].output_file is None
        assert plan.records[1].instruction_bytes == 189108
        assert plan.records[0].q4_bytes == 6225920 and plan.records[0].q8_bytes == 2920
        assert json.loads(json.dumps(plan.as_dict())) == plan.as_dict()
        try:
            plan.records[0].M = 1
            raise AssertionError("ReplayRecord is mutable")
        except AttributeError:
            pass
        assert _VALIDATOR.validate(root, expected_xclbin_sha256=FROZEN_XCLBIN_SHA256, expected_profile=PROFILE)["replay_ready"]

        root = Path(tmp) / "negative"
        baseline = _fixture(root, xclbin, shapes=(1024,))
        original_manifest = (root / "manifest.jsonl").read_bytes()
        original_metadata = (root / "run_metadata.json").read_bytes()

        def rejected(mutator, label):
            events = copy.deepcopy(baseline)
            metadata = json.loads(original_metadata)
            restore = []
            try:
                mutator(events, metadata, restore)
                (root / "manifest.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events), encoding="utf-8")
                (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                results = []
                for check in (
                    lambda: _VALIDATOR.validate(root, expected_xclbin_sha256=FROZEN_XCLBIN_SHA256, expected_profile=PROFILE),
                    lambda: build_plan(root),
                ):
                    try:
                        check()
                        results.append(True)
                    except (ValueError, OSError, KeyError, json.JSONDecodeError, TypeError, RuntimeError):
                        results.append(False)
                assert results == [False, False], f"{label}: validator/planner disagreement: {results}"
            finally:
                (root / "manifest.jsonl").write_bytes(original_manifest)
                (root / "run_metadata.json").write_bytes(original_metadata)
                for path, data in restore:
                    path.write_bytes(data)

        def change(fn):
            return lambda events, metadata, restore: fn(events)

        def delete_artifact(events, metadata, restore, field):
            event = next(item for item in events if item.get(field))
            path = root / event[field]
            restore.append((path, path.read_bytes()))
            path.unlink()

        cases = [
            (lambda e, m, r: m.update(schema_version=2), "schema version"),
            (lambda e, m, r: m.update(profile="other"), "profile mismatch"),
            (lambda e, m, r: e[0].update(submission_id=2), "gap/duplicate ID"),
            (lambda e, m, r: e[-1].update(submission_id=1), "non-monotonic/duplicate ID"),
            (lambda e, m, r: e[-1].update(submission_id=3), "ID gap"),
            (lambda e, m, r: e.pop(), "missing target"),
            (lambda e, m, r: e.append({"event": "PREPARED", "submission_id": 3}), "target not terminal"),
            (lambda e, m, r: e.insert(len(e) - 1, {"event": "ERROR", "submission_id": 2, "phase": "prepare", "error_kind": "test", "message": "test"}), "ERROR"),
            (lambda e, m, r: e.__delitem__(slice(1, 3)), "PREPARED without SUBMITTED/COMPLETED"),
            (lambda e, m, r: e.__delitem__(slice(2, 3)), "SUBMITTED without COMPLETED"),
            (lambda e, m, r: e.__delitem__(slice(1, 2)), "COMPLETED without SUBMITTED"),
            (lambda e, m, r: e[-1].update(node="wrong-node"), "wrong target node"),
            (lambda e, m, r: e[-1].update(M=1024), "wrong target M"),
            (lambda e, m, r: e[-1].update(K=1280), "wrong target K"),
            (lambda e, m, r: e[-1].update(rhs=2), "wrong target rhs"),
            (lambda e, m, r: e[-1].update(opcode=4), "wrong target opcode"),
            (lambda e, m, r: e[-1].update(submitted=True), "target marked submitted"),
            (lambda e, m, r: e[-1].update(output_file="junk.bin", output_bytes=4, output_sha256="0" * 64), "output present on target"),
            (lambda e, m, r: e[0].update(M=2048), "unsupported shape"),
            (lambda e, m, r: m.update(xclbin_sha256="0" * 64), "wrong XCLBIN SHA"),
            (lambda e, m, r: e[0].update(q4_file="../escape.bin"), "unsafe relative path"),
            (lambda e, m, r: e[0].update(q4_file=str((root / "x").resolve())), "absolute artifact path"),
            (lambda e, m, r: e[0].update(instruction_bytes=e[0]["instruction_bytes"] + 4), "artifact size mismatch"),
            (lambda e, m, r: e[0].update(q8_sha256="0" * 64), "artifact SHA mismatch"),
            (lambda e, m, r: delete_artifact(e, m, r, "q4_file"), "missing artifact"),
            (lambda e, m, r: delete_artifact(e, m, r, "output_file"), "missing output"),
            (lambda e, m, r: e[2].update(output_sha256="0" * 64), "output SHA mismatch"),
        ]
        for mutate, label in cases:
            rejected(mutate, label)

        # Explicit second-target and target-terminal checks on a two-M9216 history.
        target_root = Path(tmp) / "targets"
        target_events = _fixture(target_root, xclbin, shapes=(9216,))
        baseline_target_manifest = (target_root / "manifest.jsonl").read_bytes()
        target_copy = copy.deepcopy(target_events)
        target_copy.insert(-1, copy.deepcopy(target_copy[-1]))
        (target_root / "manifest.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in target_copy), encoding="utf-8")
        for check in (
            lambda: _VALIDATOR.validate(target_root, expected_xclbin_sha256=FROZEN_XCLBIN_SHA256, expected_profile=PROFILE),
            lambda: build_plan(target_root),
        ):
            try:
                check()
                raise AssertionError("multiple target accepted")
            except (ValueError, OSError, KeyError, json.JSONDecodeError, TypeError, RuntimeError):
                pass
        (target_root / "manifest.jsonl").write_bytes(baseline_target_manifest)
        print("XDNA capture replay planner host self-test: PASS (all shapes, integrity, ordering, target, validator parity)")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", nargs="?")
    parser.add_argument("--write-plan", metavar="PATH")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--xclbin-path", type=Path, help="explicit frozen XCLBIN required by --self-test")
    parser.add_argument("--quiet", action="store_true", help="write the plan without printing the full source-record table")
    args = parser.parse_args(argv)
    if args.self_test:
        if not args.xclbin_path:
            parser.error("--self-test requires --xclbin-path")
        _self_test(args.xclbin_path)
        return 0
    if not args.capture_dir:
        parser.error("provide capture_dir or --self-test")
    try:
        plan = build_plan(args.capture_dir)
        if args.write_plan:
            destination = Path(args.write_plan).resolve()
            source = Path(plan.source_capture_dir)
            if destination == source or source in destination.parents:
                raise ValueError("replay plan output must not modify the capture directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(plan.as_dict(), sort_keys=True, indent=2) + "\n"
            destination.write_text(encoded, encoding="utf-8")
        if not args.quiet:
            _print_plan(plan)
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        print("REPLAY_PLAN_VALID=NO")
        print(f"validation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
