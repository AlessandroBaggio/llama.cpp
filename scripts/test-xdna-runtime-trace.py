#!/usr/bin/env python3
"""Host-only contract tests for opt-in XDNA runtime trace records.

Synthetic events validate the log contract only; they do not simulate XRT or claim hardware equivalence.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TRACE_SCHEMA = "ggml-xdna-runtime-trace-v1"
EVENTS = {
    "KERNEL_CALL_ENTER", "KERNEL_CALL_RETURNED", "KERNEL_CALL_EXCEPTION",
    "WAIT_BEGIN", "WAIT_RETURN_STATE", "WAIT_EXCEPTION",
    "OUTPUT_SYNC_RETURNED", "OUTPUT_SYNC_EXCEPTION", "RESULT_COPY_DONE",
}
IDENTITY = (
    "graph_id", "node_index", "node_name", "ggml_op", "backend", "M", "K", "RHS",
    "submission_id", "kernel_name", "path", "row_index",
)


def validate_records(records):
    groups = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("trace record is not an object")
        required = set(IDENTITY) | {
            "schema", "event", "xrt_state", "error", "monotonic_timestamp_ns",
        }
        if set(record) != required:
            raise ValueError("record has missing or unexpected fields")
        if not isinstance(record["schema"], str) or not isinstance(record["event"], str):
            raise ValueError("schema and event must be strings")
        if not isinstance(record["xrt_state"], (str, type(None))) or not isinstance(record["error"], (str, type(None))):
            raise ValueError("XRT state and error must be strings or null")
        if record["schema"] != TRACE_SCHEMA or record["event"] not in EVENTS:
            raise ValueError("unknown trace schema or event")
        if record["backend"] != "XDNA" or record["ggml_op"] != "MUL_MAT":
            raise ValueError("trace does not identify an XDNA MUL_MAT node")
        if any(type(record[key]) is not int for key in ("M", "K", "RHS")):
            raise ValueError("Q4/K2560 dimensions must be integers")
        if record["M"] <= 0 or record["K"] != 2560 or record["RHS"] != 1:
            raise ValueError("unsupported Q4/K2560 dimensions")
        if type(record["graph_id"]) is not int or record["graph_id"] <= 0:
            raise ValueError("invalid graph_id")
        if type(record["node_index"]) is not int or record["node_index"] < 0:
            raise ValueError("invalid node_index")
        if type(record["submission_id"]) is not int or record["submission_id"] <= 0:
            raise ValueError("invalid submission_id")
        if type(record["monotonic_timestamp_ns"]) is not int or record["monotonic_timestamp_ns"] < 0:
            raise ValueError("invalid monotonic timestamp")
        if not isinstance(record["node_name"], str) or not isinstance(record["kernel_name"], str) or not record["node_name"] or not record["kernel_name"]:
            raise ValueError("missing logical node or kernel name")
        if record["path"] not in ("batched", "m9216", "per_row"):
            raise ValueError("unknown execution path")
        if record["path"] == "m9216" and record["M"] != 9216:
            raise ValueError("M=9216 path has the wrong shape")
        if record["path"] == "batched" and record["M"] not in (1024, 4096, 8192):
            raise ValueError("batched path has an unsupported shape")
        if (record["path"] == "per_row") != (type(record["row_index"]) is int):
            raise ValueError("row_index does not match the per-row path")
        if record["path"] != "per_row" and record["row_index"] is not None:
            raise ValueError("batched event unexpectedly has row_index")
        if record["path"] == "per_row" and not 0 <= record["row_index"] < record["M"]:
            raise ValueError("row_index is outside the output tensor")

        submission_id = record["submission_id"]
        group = groups.setdefault(submission_id, {"identity": None, "events": []})
        identity = tuple(record[key] for key in IDENTITY)
        if group["identity"] is None:
            group["identity"] = identity
        elif group["identity"] != identity:
            raise ValueError("submission_id reused for a different graph/node/run")
        group["events"].append(record)

    if not groups:
        raise ValueError("no runtime trace records")

    for submission_id, group in groups.items():
        records_for_run = group["events"]
        sequence = [record["event"] for record in records_for_run]
        timestamps = [record["monotonic_timestamp_ns"] for record in records_for_run]
        if timestamps != sorted(timestamps):
            raise ValueError(f"non-monotonic event timestamps for submission {submission_id}")
        if len(sequence) != len(set(sequence)):
            raise ValueError("duplicate event in one submission")
        if sequence[0] != "KERNEL_CALL_ENTER":
            raise ValueError("submission does not begin at kernel call")

        if sequence == ["KERNEL_CALL_ENTER", "KERNEL_CALL_EXCEPTION"]:
            if not records_for_run[-1]["error"] or records_for_run[-1]["xrt_state"] is not None:
                raise ValueError("kernel-call exception is missing its error")
            continue
        if len(sequence) < 2 or sequence[1] != "KERNEL_CALL_RETURNED":
            raise ValueError("kernel call did not return a run object")
        if len(sequence) < 3 or sequence[2] != "WAIT_BEGIN":
            raise ValueError("wait did not begin after run creation")

        if sequence == ["KERNEL_CALL_ENTER", "KERNEL_CALL_RETURNED", "WAIT_BEGIN", "WAIT_EXCEPTION"]:
            if not records_for_run[-1]["error"] or records_for_run[-1]["xrt_state"] is not None:
                raise ValueError("wait exception is missing its error")
            continue
        if len(sequence) < 4 or sequence[3] != "WAIT_RETURN_STATE":
            raise ValueError("wait return state missing or out of order")

        wait = records_for_run[3]
        if wait["xrt_state"] == "ERT_CMD_STATE_COMPLETED":
            if wait["error"] is not None:
                raise ValueError("completed wait has an error")
            if sequence == sequence[:4] + ["OUTPUT_SYNC_EXCEPTION"]:
                if not records_for_run[-1]["error"]:
                    raise ValueError("output-sync exception is missing its error")
            elif sequence != [
                "KERNEL_CALL_ENTER", "KERNEL_CALL_RETURNED", "WAIT_BEGIN", "WAIT_RETURN_STATE",
                "OUTPUT_SYNC_RETURNED", "RESULT_COPY_DONE",
            ]:
                raise ValueError("completed run lacks ordered sync and result-copy events")
            elif records_for_run[4]["xrt_state"] != "ERT_CMD_STATE_COMPLETED" or records_for_run[5]["xrt_state"] != "ERT_CMD_STATE_COMPLETED":
                raise ValueError("sync/copy is not correlated with a completed run")
        else:
            if not wait["xrt_state"] or not wait["error"]:
                raise ValueError("non-completed wait state is not explicit")
            if sequence != ["KERNEL_CALL_ENTER", "KERNEL_CALL_RETURNED", "WAIT_BEGIN", "WAIT_RETURN_STATE"]:
                raise ValueError("non-completed run must not claim output sync or copy")
    return groups


def parse_log_lines(lines):
    records = []
    for line in lines:
        if TRACE_SCHEMA not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("malformed runtime trace JSON line") from exc
        records.append(record)
    return validate_records(records)


def event(base, submission_id, name, timestamp, state=None, error=None):
    return {
        "schema": TRACE_SCHEMA, **base, "submission_id": submission_id,
        "event": name, "xrt_state": state, "error": error,
        "monotonic_timestamp_ns": timestamp,
    }


def success(base, submission_id, first_timestamp):
    return [
        event(base, submission_id, "KERNEL_CALL_ENTER", first_timestamp),
        event(base, submission_id, "KERNEL_CALL_RETURNED", first_timestamp + 1),
        event(base, submission_id, "WAIT_BEGIN", first_timestamp + 2),
        event(base, submission_id, "WAIT_RETURN_STATE", first_timestamp + 3, "ERT_CMD_STATE_COMPLETED"),
        event(base, submission_id, "OUTPUT_SYNC_RETURNED", first_timestamp + 4, "ERT_CMD_STATE_COMPLETED"),
        event(base, submission_id, "RESULT_COPY_DONE", first_timestamp + 5, "ERT_CMD_STATE_COMPLETED"),
    ]


def expect_reject(records):
    try:
        validate_records(records)
    except ValueError:
        return
    raise AssertionError("invalid synthetic trace was accepted")


def test_contract():
    cmake = (ROOT / "ggml/src/ggml-xdna/CMakeLists.txt").read_text(encoding="utf-8")
    backend = (ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp").read_text(encoding="utf-8")
    option = 'option(GGML_XDNA_RUNTIME_TRACE "Build opt-in production-linkable XDNA runtime tracing" OFF)'
    assert option in cmake
    trace_block = cmake.split("option(GGML_XDNA_RUNTIME_TRACE", 1)[1]
    assert "target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_RUNTIME_TRACE)" in trace_block
    assert "ggml-xdna-diagnostics.cpp" not in trace_block
    assert "ggml-xdna-causal-payload-capture.cpp" not in trace_block
    assert "GGML_BACKEND_DL" not in trace_block
    assert 'std::strcmp(value, "1") == 0' in backend
    for marker in EVENTS:
        assert f'"{marker}"' in backend, f"missing event marker: {marker}"
    for path in ('"batched"', '"m9216"', '"per_row"'):
        assert path in backend, f"missing execution path: {path}"
    assert "std::atomic<uint64_t> ggml_backend_xdna_runtime_trace_graph_id" in backend
    assert "std::atomic<uint64_t> ggml_backend_xdna_runtime_trace_submission_id" in backend
    assert "ggml_get_name(node), ggml_op_name(node->op)" in backend
    assert '"monotonic_timestamp_ns"' in backend
    fields = set(IDENTITY) | {"schema", "event", "xrt_state", "error", "monotonic_timestamp_ns"}
    assert all(f'"{field}"' in backend for field in fields)

    shapes = (1024, 4096, 8192, 9216)
    records = []
    for index, m in enumerate(shapes, 1):
        base = {
            "graph_id": index, "node_index": 3, "node_name": f"layer-{m}",
            "ggml_op": "MUL_MAT", "backend": "XDNA", "M": m, "K": 2560,
            "RHS": 1, "kernel_name": "MLIR_AIE", "path": "m9216" if m == 9216 else "batched",
            "row_index": None,
        }
        records.extend(success(base, index, index * 100))

    same_node = {
        "graph_id": 10, "node_index": 7, "node_name": "same-node", "ggml_op": "MUL_MAT",
        "backend": "XDNA", "M": 4096, "K": 2560, "RHS": 1, "kernel_name": "MLIR_AIE",
        "path": "per_row", "row_index": 0,
    }
    records.extend(success(same_node, 10, 1000))
    second_row = dict(same_node, row_index=1)
    records.extend(success(second_row, 11, 1100))
    fallback_row = dict(same_node, graph_id=12, node_name="per-row-fallback", M=7, row_index=6)
    records.extend(success(fallback_row, 13, 1300))
    next_graph = dict(same_node, graph_id=11, path="batched", row_index=None)
    records.extend(success(next_graph, 12, 1200))

    timeout = {
        "graph_id": 20, "node_index": 1, "node_name": "timeout", "ggml_op": "MUL_MAT",
        "backend": "XDNA", "M": 1024, "K": 2560, "RHS": 1, "kernel_name": "MLIR_AIE",
        "path": "batched", "row_index": None,
    }
    records.extend([
        event(timeout, 20, "KERNEL_CALL_ENTER", 2000),
        event(timeout, 20, "KERNEL_CALL_RETURNED", 2001),
        event(timeout, 20, "WAIT_BEGIN", 2002),
        event(timeout, 20, "WAIT_RETURN_STATE", 2003, "ERT_CMD_STATE_TIMEOUT", "non_completed_xrt_state"),
    ])
    wait_exception = dict(timeout, graph_id=21, node_name="wait-exception")
    records.extend([
        event(wait_exception, 21, "KERNEL_CALL_ENTER", 2100),
        event(wait_exception, 21, "KERNEL_CALL_RETURNED", 2101),
        event(wait_exception, 21, "WAIT_BEGIN", 2102),
        event(wait_exception, 21, "WAIT_EXCEPTION", 2103, error="synthetic exception"),
    ])
    call_exception = dict(timeout, graph_id=22, node_name="call-exception")
    records.extend([
        event(call_exception, 22, "KERNEL_CALL_ENTER", 2200),
        event(call_exception, 22, "KERNEL_CALL_EXCEPTION", 2201, error="synthetic exception"),
    ])
    sync_exception = dict(timeout, graph_id=23, node_name="sync-exception")
    records.extend([
        event(sync_exception, 23, "KERNEL_CALL_ENTER", 2300),
        event(sync_exception, 23, "KERNEL_CALL_RETURNED", 2301),
        event(sync_exception, 23, "WAIT_BEGIN", 2302),
        event(sync_exception, 23, "WAIT_RETURN_STATE", 2303, "ERT_CMD_STATE_COMPLETED"),
        event(sync_exception, 23, "OUTPUT_SYNC_EXCEPTION", 2304, "ERT_CMD_STATE_COMPLETED", "synthetic exception"),
    ])

    parsed = parse_log_lines(["unrelated supervised output", *(json.dumps(record) for record in records)])
    assert len(parsed) == 12
    assert {group["identity"][5] for group in parsed.values()} >= set(shapes)
    assert any(group["identity"][5] == 7 and group["identity"][10:] == ("per_row", 6) for group in parsed.values())

    expect_reject(records[:-1])
    unsupported_batched = dict(next_graph, graph_id=13, node_name="unsupported-batched", M=7)
    expect_reject(success(unsupported_batched, 14, 1400))
    out_of_order = success(timeout, 30, 3000)
    out_of_order[3], out_of_order[2] = out_of_order[2], out_of_order[3]
    expect_reject(out_of_order)
    reused_id = success(dict(timeout, graph_id=99, node_name="other-node"), 20, 4000)
    expect_reject(success(timeout, 20, 3900) + reused_id)
    false_completion = success(timeout, 31, 5000)
    false_completion[-1]["event"] = "RESULT_COPY_DONE"
    false_completion[3]["xrt_state"] = "ERT_CMD_STATE_TIMEOUT"
    false_completion[3]["error"] = "non_completed_xrt_state"
    expect_reject(false_completion)
    malformed = ["{\"schema\":\"ggml-xdna-runtime-trace-v1\",broken"]
    try:
        parse_log_lines(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed trace JSON was accepted")


def check_build(build, trace_enabled):
    build = Path(build).resolve(strict=True)
    cache = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            cache[key] = value
    expected = {
        "CMAKE_BUILD_TYPE:STRING": "Release",
        "CMAKE_GENERATOR:INTERNAL": "Ninja",
        "BUILD_SHARED_LIBS:BOOL": "ON",
        "GGML_XDNA:BOOL": "ON",
        "GGML_XDNA_BUILD_DIAGNOSTICS:BOOL": "OFF",
        "GGML_XDNA_RUNTIME_TRACE:BOOL": "ON" if trace_enabled else "OFF",
        "GGML_BACKEND_DL:BOOL": "ON",
        "GGML_CPU_ALL_VARIANTS:BOOL": "ON",
        "GGML_NATIVE:BOOL": "OFF",
        "GGML_OPENMP:BOOL": "ON",
        "XRT_ROOT:PATH": "C:/Xilinx/xrt",
        "LLAMA_BUILD_TESTS:BOOL": "ON",
        "LLAMA_BUILD_EXAMPLES:BOOL": "ON",
        "LLAMA_BUILD_SERVER:BOOL": "ON",
        "LLAMA_BUILD_UI:BOOL": "OFF",
        "LLAMA_USE_PREBUILT_UI:BOOL": "OFF",
    }
    for key, value in expected.items():
        assert cache.get(key) == value, f"{build}: expected {key}={value}, got {cache.get(key)}"
    expected_source = (ROOT.resolve())
    actual_source = Path(cache["CMAKE_HOME_DIRECTORY:INTERNAL"]).resolve(strict=True)
    assert actual_source == expected_source, f"build uses different sources: {actual_source}"
    for name in ("llama-cli.exe", "ggml-xdna.dll", "ggml.dll"):
        assert (build / "bin" / name).is_file(), f"missing build artifact: {name}"
    data = (build / "bin" / "ggml-xdna.dll").read_bytes()
    markers = (b"KERNEL_CALL_ENTER", b"KERNEL_CALL_RETURNED", b"WAIT_BEGIN", b"WAIT_RETURN_STATE", b"OUTPUT_SYNC_RETURNED", b"RESULT_COPY_DONE")
    if trace_enabled:
        for marker in markers:
            assert marker in data, f"trace backend missing marker {marker!r}"
    else:
        for marker in markers:
            assert marker not in data, f"trace-OFF backend contains marker {marker!r}"
    return cache


def main():
    test_contract()
    if len(sys.argv) == 3:
        check_build(sys.argv[1], False)
        check_build(sys.argv[2], True)
    elif len(sys.argv) != 1:
        raise SystemExit("usage: test-xdna-runtime-trace.py [reference-build trace-build]")
    print("XDNA runtime trace-only contract tests: PASS (synthetic contract only; no XRT execution)")


if __name__ == "__main__":
    main()
