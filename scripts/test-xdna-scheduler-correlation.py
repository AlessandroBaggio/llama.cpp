#!/usr/bin/env python3
"""Synthetic host-only tests for structured scheduler-to-XDNA trace joins."""
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ggml-xdna-scheduler-trace-v1"
RUNTIME_PATH = ROOT / "scripts" / "test-xdna-runtime-trace.py"

spec = importlib.util.spec_from_file_location("runtime_trace_contract", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

FIELDS = {
    "SCHEDULER_SPLIT_ASSIGNMENT": {
        "scheduler_graph_id", "scheduler_split_id", "split_global_node_start",
        "split_global_node_end_exclusive", "backend", "input_count",
    },
    "SCHEDULER_GRAPH_BEGIN": {"scheduler_graph_id", "graph_node_count", "split_count", "assignment_count"},
    "SCHEDULER_NODE_ASSIGNMENT": {
        "scheduler_graph_id", "scheduler_split_id", "global_node_index", "local_split_node_index",
        "split_global_node_start", "split_global_node_end_exclusive", "node_name", "ggml_op", "backend",
    },
    "SCHEDULER_SPLIT_DISPATCH": {
        "scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id", "backend",
        "split_global_node_start", "split_global_node_end_exclusive", "dispatch_global_node_start",
        "dispatch_global_node_end_exclusive", "dispatch_local_node_start",
        "dispatch_local_node_end_exclusive", "input_count",
    },
    "SCHEDULER_DISPATCH_FAILED": {"scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id", "status"},
    "XDNA_NODE_EXECUTE": {
        "scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id", "global_node_index",
        "local_split_node_index", "split_global_node_start", "split_global_node_end_exclusive",
        "trace_graph_id", "trace_node_index", "node_name", "ggml_op", "backend",
    },
    "XDNA_SUBMISSION_LINK": {
        "scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id", "global_node_index",
        "local_split_node_index", "split_global_node_start", "split_global_node_end_exclusive",
        "trace_graph_id", "trace_node_index", "submission_id", "node_name", "ggml_op", "backend",
        "M", "K", "RHS", "kernel_name", "path", "row_index",
    },
}


def integer(record, key, minimum=0):
    value = record[key]
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {key}")
    return value


def text(record, key):
    value = record[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {key}")
    return value


def parse_scheduler_records(records, runtime_groups):
    graphs = {}
    assignments = {}
    split_info = {}
    assignment_split_info = {}
    dispatches = {}
    failures = {}
    node_events = {}
    links = {}

    for record in records:
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError("invalid scheduler trace record")
        event = record.get("event")
        if event not in FIELDS or set(record) != {"schema", "event"} | FIELDS[event]:
            raise ValueError("unknown scheduler event or fields")

        if event == "SCHEDULER_GRAPH_BEGIN":
            graph_id = integer(record, "scheduler_graph_id", 1)
            if graph_id in graphs:
                raise ValueError("duplicate scheduler_graph_id")
            graphs[graph_id] = record
            for key in ("graph_node_count", "split_count", "assignment_count"):
                integer(record, key)
        elif event == "SCHEDULER_SPLIT_ASSIGNMENT":
            graph_id = integer(record, "scheduler_graph_id", 1)
            split_id = integer(record, "scheduler_split_id", 1)
            split_start = integer(record, "split_global_node_start")
            split_end = integer(record, "split_global_node_end_exclusive", 1)
            backend = text(record, "backend")
            integer(record, "input_count")
            if split_id in split_info:
                raise ValueError("duplicate scheduler split assignment")
            split_info[split_id] = (graph_id, split_start, split_end, backend)
        elif event == "SCHEDULER_NODE_ASSIGNMENT":
            graph_id = integer(record, "scheduler_graph_id", 1)
            split_id = integer(record, "scheduler_split_id", 1)
            global_index = integer(record, "global_node_index")
            local_index = integer(record, "local_split_node_index")
            split_start = integer(record, "split_global_node_start")
            split_end = integer(record, "split_global_node_end_exclusive", 1)
            name, op, backend = (text(record, key) for key in ("node_name", "ggml_op", "backend"))
            if not split_start <= global_index < split_end or local_index != global_index - split_start:
                raise ValueError("scheduler assignment index/range mismatch")
            key = (graph_id, global_index)
            if key in assignments:
                raise ValueError("duplicate scheduler node assignment")
            assignments[key] = record
            owner = (graph_id, split_start, split_end, backend)
            if split_id in assignment_split_info and assignment_split_info[split_id] != owner:
                raise ValueError("scheduler_split_id reused")
            assignment_split_info[split_id] = owner
        elif event == "SCHEDULER_SPLIT_DISPATCH":
            graph_id = integer(record, "scheduler_graph_id", 1)
            split_id = integer(record, "scheduler_split_id", 1)
            dispatch_id = integer(record, "scheduler_dispatch_id", 1)
            backend = text(record, "backend")
            split_start = integer(record, "split_global_node_start")
            split_end = integer(record, "split_global_node_end_exclusive", 1)
            global_start = integer(record, "dispatch_global_node_start")
            global_end = integer(record, "dispatch_global_node_end_exclusive", 1)
            local_start = integer(record, "dispatch_local_node_start")
            local_end = integer(record, "dispatch_local_node_end_exclusive", 1)
            integer(record, "input_count")
            if dispatch_id in dispatches:
                raise ValueError("duplicate scheduler_dispatch_id")
            if not split_start <= global_start < global_end <= split_end:
                raise ValueError("dispatch outside scheduler split")
            if global_start != split_start + local_start or global_end != split_start + local_end:
                raise ValueError("dispatch global/local range mismatch")
            dispatches[dispatch_id] = record
        elif event == "SCHEDULER_DISPATCH_FAILED":
            dispatch_id = integer(record, "scheduler_dispatch_id", 1)
            integer(record, "scheduler_graph_id", 1)
            integer(record, "scheduler_split_id", 1)
            integer(record, "status", 1)
            if dispatch_id in failures:
                raise ValueError("duplicate dispatch failure")
            failures[dispatch_id] = record
        elif event == "XDNA_NODE_EXECUTE":
            trace_graph = integer(record, "trace_graph_id", 1)
            trace_node = integer(record, "trace_node_index")
            if text(record, "backend") != "XDNA":
                raise ValueError("non-XDNA execution event")
            key = (trace_graph, trace_node)
            if key in node_events:
                raise ValueError("duplicate XDNA node execution event")
            node_events[key] = record
        else:
            submission_id = integer(record, "submission_id", 1)
            if submission_id in links:
                raise ValueError("duplicate XDNA submission link")
            if integer(record, "trace_graph_id", 1) <= 0:
                raise ValueError("invalid trace_graph_id")
            integer(record, "trace_node_index")
            for key in ("scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id", "global_node_index",
                        "local_split_node_index", "split_global_node_start", "split_global_node_end_exclusive"):
                integer(record, key, 1 if key.endswith("_id") else 0)
            if text(record, "backend") != "XDNA":
                raise ValueError("submission link backend is not XDNA")
            for key in ("M", "K", "RHS"):
                integer(record, key, 1)
            if record["K"] != 2560 or record["RHS"] != 1:
                raise ValueError("unsupported XDNA submission shape")
            if record["row_index"] is not None:
                integer(record, "row_index")
            text(record, "node_name")
            text(record, "ggml_op")
            text(record, "kernel_name")
            if record["path"] not in ("batched", "m9216", "per_row"):
                raise ValueError("unknown XDNA path")
            links[submission_id] = record

    if any(split_info.get(split_id) != owner for split_id, owner in assignment_split_info.items()):
        raise ValueError("node assignment does not match split assignment")
    if set(graphs) == set():
        raise ValueError("no scheduler graph records")
    for graph_id, graph in graphs.items():
        graph_assignments = [row for (owner, _), row in assignments.items() if owner == graph_id]
        if len(graph_assignments) != graph["assignment_count"]:
            raise ValueError("incomplete scheduler assignment map")
        if any(index >= graph["graph_node_count"] for owner, index in assignments if owner == graph_id):
            raise ValueError("assignment index outside graph")
        graph_splits = {split_id for split_id, owner in split_info.items() if owner[0] == graph_id}
        if len(graph_splits) != graph["split_count"]:
            raise ValueError("incomplete scheduler split map")

    for split_id, (graph_id, start, end, backend) in split_info.items():
        if graph_id not in graphs or end > graphs[graph_id]["graph_node_count"]:
            raise ValueError("split references an unknown graph/range")
    for dispatch_id, dispatch in dispatches.items():
        graph_id = dispatch["scheduler_graph_id"]
        split_id = dispatch["scheduler_split_id"]
        owner = split_info.get(split_id)
        if graph_id not in graphs or owner is None or owner != (
                graph_id, dispatch["split_global_node_start"],
                dispatch["split_global_node_end_exclusive"], dispatch["backend"]):
            raise ValueError("dispatch does not match scheduler split assignment")
    for dispatch_id, failure in failures.items():
        dispatch = dispatches.get(dispatch_id)
        if dispatch is None or any(failure[key] != dispatch[key] for key in (
                "scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id")):
            raise ValueError("dispatch failure has no matching dispatch")

    for key, execution in node_events.items():
        ids = ("scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id", "global_node_index",
               "local_split_node_index", "split_global_node_start", "split_global_node_end_exclusive")
        values = [execution[name] for name in ids]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError("partial scheduler context on XDNA node event")
        graph_id, split_id, dispatch_id = values[:3]
        dispatch = dispatches.get(dispatch_id)
        if dispatch is None or dispatch["scheduler_graph_id"] != graph_id or dispatch["scheduler_split_id"] != split_id:
            raise ValueError("XDNA node references unknown scheduler dispatch")
        if dispatch["backend"] != "XDNA":
            raise ValueError("XDNA execution contradicts scheduler backend assignment")
        local_index = execution["trace_node_index"]
        if local_index >= dispatch["dispatch_local_node_end_exclusive"] - dispatch["dispatch_local_node_start"]:
            raise ValueError("XDNA node index outside dispatched graph view")
        global_index = dispatch["dispatch_global_node_start"] + local_index
        split_local = dispatch["dispatch_local_node_start"] + local_index
        if execution["global_node_index"] != global_index or execution["local_split_node_index"] != split_local:
            raise ValueError("XDNA node global/local identity mismatch")
        if execution["split_global_node_start"] != dispatch["split_global_node_start"] or execution[
                "split_global_node_end_exclusive"] != dispatch["split_global_node_end_exclusive"]:
            raise ValueError("XDNA node split range mismatch")
        assignment = assignments.get((graph_id, global_index))
        if assignment is None or assignment["scheduler_split_id"] != split_id or assignment["backend"] != "XDNA":
            raise ValueError("XDNA node has no matching scheduler assignment")
        if execution["node_name"] != assignment["node_name"] or execution["ggml_op"] != assignment["ggml_op"]:
            raise ValueError("XDNA node metadata disagrees with scheduler assignment")

    if set(links) != set(runtime_groups):
        raise ValueError("runtime submissions and scheduler links are not one-to-one")
    joined = {}
    for submission_id, link in links.items():
        identity = dict(zip(runtime.IDENTITY, runtime_groups[submission_id]["identity"]))
        checks = {
            "trace_graph_id": "graph_id", "trace_node_index": "node_index", "node_name": "node_name",
            "ggml_op": "ggml_op", "backend": "backend", "M": "M", "K": "K", "RHS": "RHS",
            "submission_id": "submission_id", "kernel_name": "kernel_name", "path": "path", "row_index": "row_index",
        }
        if any(link[source] != identity[target] for source, target in checks.items()):
            raise ValueError("scheduler link does not match runtime submission identity")
        node = node_events.get((link["trace_graph_id"], link["trace_node_index"]))
        if node is None or any(node[key] != link[key] for key in (
                "scheduler_graph_id", "scheduler_split_id", "scheduler_dispatch_id", "global_node_index",
                "local_split_node_index", "split_global_node_start", "split_global_node_end_exclusive",
                "node_name", "ggml_op", "backend")):
            raise ValueError("submission link has no matching XDNA node event")
        graph_id = link["scheduler_graph_id"]
        assignment = assignments.get((graph_id, link["global_node_index"]))
        if assignment is None or assignment["scheduler_split_id"] != link["scheduler_split_id"] or assignment["backend"] != "XDNA":
            raise ValueError("submission link has no matching scheduler assignment")
        if assignment["node_name"] != link["node_name"] or assignment["ggml_op"] != link["ggml_op"]:
            raise ValueError("submission metadata differs from assigned node")
        group = runtime_groups[submission_id]
        joined[submission_id] = {
            "scheduler_graph_id": graph_id,
            "scheduler_split_id": link["scheduler_split_id"],
            "scheduler_dispatch_id": link["scheduler_dispatch_id"],
            "global_node_index": link["global_node_index"],
            "local_split_node_index": link["local_split_node_index"],
            "xrt_result_copy_confirmed": group["events"][-1]["event"] == "RESULT_COPY_DONE",
        }

    distribution = Counter(row["backend"] for row in assignments.values())
    return {"joined_submissions": joined, "scheduler_backend_node_counts": dict(distribution),
            "dispatch_failures": failures, "unjoined_xdna_nodes": sum(
                1 for row in node_events.values() if row["scheduler_graph_id"] is None)}


def parse_scheduler_correlation_lines(lines):
    lines = list(lines)
    runtime_groups = runtime.parse_log_lines(lines)
    records = []
    for line in lines:
        if SCHEMA not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("malformed scheduler trace JSON line") from exc
        records.append(record)
    return parse_scheduler_records(records, runtime_groups)


def scheduler(event, **fields):
    return {"schema": SCHEMA, "event": event, **fields}


def test_synthetic():
    records = [
        scheduler("SCHEDULER_GRAPH_BEGIN", scheduler_graph_id=501, graph_node_count=3, split_count=2, assignment_count=3),
        scheduler("SCHEDULER_SPLIT_ASSIGNMENT", scheduler_graph_id=501, scheduler_split_id=511,
                  split_global_node_start=0, split_global_node_end_exclusive=1, backend="CPU", input_count=0),
        scheduler("SCHEDULER_SPLIT_ASSIGNMENT", scheduler_graph_id=501, scheduler_split_id=512,
                  split_global_node_start=1, split_global_node_end_exclusive=3, backend="XDNA", input_count=1),
        scheduler("SCHEDULER_NODE_ASSIGNMENT", scheduler_graph_id=501, scheduler_split_id=511, global_node_index=0,
                  local_split_node_index=0, split_global_node_start=0, split_global_node_end_exclusive=1,
                  node_name="cpu.copy", ggml_op="DUP", backend="CPU"),
        scheduler("SCHEDULER_NODE_ASSIGNMENT", scheduler_graph_id=501, scheduler_split_id=512, global_node_index=1,
                  local_split_node_index=0, split_global_node_start=1, split_global_node_end_exclusive=3,
                  node_name="layer.ffn_gate", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("SCHEDULER_NODE_ASSIGNMENT", scheduler_graph_id=501, scheduler_split_id=512, global_node_index=2,
                  local_split_node_index=1, split_global_node_start=1, split_global_node_end_exclusive=3,
                  node_name="layer.ffn_gate", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("SCHEDULER_SPLIT_DISPATCH", scheduler_graph_id=501, scheduler_split_id=511, scheduler_dispatch_id=521,
                  backend="CPU", split_global_node_start=0, split_global_node_end_exclusive=1,
                  dispatch_global_node_start=0, dispatch_global_node_end_exclusive=1,
                  dispatch_local_node_start=0, dispatch_local_node_end_exclusive=1, input_count=0),
        scheduler("SCHEDULER_SPLIT_DISPATCH", scheduler_graph_id=501, scheduler_split_id=512, scheduler_dispatch_id=522,
                  backend="XDNA", split_global_node_start=1, split_global_node_end_exclusive=3,
                  dispatch_global_node_start=2, dispatch_global_node_end_exclusive=3,
                  dispatch_local_node_start=1, dispatch_local_node_end_exclusive=2, input_count=1),
        scheduler("XDNA_NODE_EXECUTE", scheduler_graph_id=501, scheduler_split_id=512, scheduler_dispatch_id=522,
                  global_node_index=2, local_split_node_index=1, split_global_node_start=1,
                  split_global_node_end_exclusive=3, trace_graph_id=501, trace_node_index=0,
                  node_name="layer.ffn_gate", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("XDNA_SUBMISSION_LINK", scheduler_graph_id=501, scheduler_split_id=512,
                  scheduler_dispatch_id=522, global_node_index=2, local_split_node_index=1,
                  split_global_node_start=1, split_global_node_end_exclusive=3, trace_graph_id=501,
                  trace_node_index=0, submission_id=900, node_name="layer.ffn_gate", ggml_op="MUL_MAT",
                  backend="XDNA", M=1024, K=2560, RHS=1, kernel_name="MLIR_AIE", path="batched", row_index=None),
        scheduler("SCHEDULER_GRAPH_BEGIN", scheduler_graph_id=502, graph_node_count=4, split_count=1, assignment_count=3),
        scheduler("SCHEDULER_SPLIT_ASSIGNMENT", scheduler_graph_id=502, scheduler_split_id=513,
                  split_global_node_start=1, split_global_node_end_exclusive=4, backend="XDNA", input_count=2),
        scheduler("SCHEDULER_NODE_ASSIGNMENT", scheduler_graph_id=502, scheduler_split_id=513, global_node_index=1,
                  local_split_node_index=0, split_global_node_start=1, split_global_node_end_exclusive=4,
                  node_name="layer.ffn_gate", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("SCHEDULER_NODE_ASSIGNMENT", scheduler_graph_id=502, scheduler_split_id=513, global_node_index=2,
                  local_split_node_index=1, split_global_node_start=1, split_global_node_end_exclusive=4,
                  node_name="layer.ffn_gate", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("SCHEDULER_NODE_ASSIGNMENT", scheduler_graph_id=502, scheduler_split_id=513, global_node_index=3,
                  local_split_node_index=2, split_global_node_start=1, split_global_node_end_exclusive=4,
                  node_name="tail", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("SCHEDULER_SPLIT_DISPATCH", scheduler_graph_id=502, scheduler_split_id=513, scheduler_dispatch_id=523,
                  backend="XDNA", split_global_node_start=1, split_global_node_end_exclusive=4,
                  dispatch_global_node_start=2, dispatch_global_node_end_exclusive=4,
                  dispatch_local_node_start=1, dispatch_local_node_end_exclusive=3, input_count=2),
        scheduler("SCHEDULER_DISPATCH_FAILED", scheduler_graph_id=502, scheduler_split_id=513,
                  scheduler_dispatch_id=523, status=1),
        scheduler("XDNA_NODE_EXECUTE", scheduler_graph_id=502, scheduler_split_id=513, scheduler_dispatch_id=523,
                  global_node_index=2, local_split_node_index=1, split_global_node_start=1,
                  split_global_node_end_exclusive=4, trace_graph_id=502, trace_node_index=0,
                  node_name="layer.ffn_gate", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("XDNA_NODE_EXECUTE", scheduler_graph_id=502, scheduler_split_id=513, scheduler_dispatch_id=523,
                  global_node_index=3, local_split_node_index=2, split_global_node_start=1,
                  split_global_node_end_exclusive=4, trace_graph_id=502, trace_node_index=1,
                  node_name="tail", ggml_op="MUL_MAT", backend="XDNA"),
        scheduler("XDNA_SUBMISSION_LINK", scheduler_graph_id=502, scheduler_split_id=513,
                  scheduler_dispatch_id=523, global_node_index=2, local_split_node_index=1,
                  split_global_node_start=1, split_global_node_end_exclusive=4, trace_graph_id=502,
                  trace_node_index=0, submission_id=901, node_name="layer.ffn_gate", ggml_op="MUL_MAT",
                  backend="XDNA", M=1024, K=2560, RHS=1, kernel_name="MLIR_AIE", path="batched", row_index=None),
        scheduler("XDNA_SUBMISSION_LINK", scheduler_graph_id=502, scheduler_split_id=513,
                  scheduler_dispatch_id=523, global_node_index=3, local_split_node_index=2,
                  split_global_node_start=1, split_global_node_end_exclusive=4, trace_graph_id=502,
                  trace_node_index=1, submission_id=902, node_name="tail", ggml_op="MUL_MAT",
                  backend="XDNA", M=2, K=2560, RHS=1, kernel_name="MLIR_AIE", path="per_row", row_index=0),
        scheduler("XDNA_SUBMISSION_LINK", scheduler_graph_id=502, scheduler_split_id=513,
                  scheduler_dispatch_id=523, global_node_index=3, local_split_node_index=2,
                  split_global_node_start=1, split_global_node_end_exclusive=4, trace_graph_id=502,
                  trace_node_index=1, submission_id=903, node_name="tail", ggml_op="MUL_MAT",
                  backend="XDNA", M=2, K=2560, RHS=1, kernel_name="MLIR_AIE", path="per_row", row_index=1),
    ]
    bases = {
        900: {"graph_id": 501, "node_index": 0, "node_name": "layer.ffn_gate", "ggml_op": "MUL_MAT",
             "backend": "XDNA", "M": 1024, "K": 2560, "RHS": 1, "kernel_name": "MLIR_AIE",
             "path": "batched", "row_index": None},
        901: {"graph_id": 502, "node_index": 0, "node_name": "layer.ffn_gate", "ggml_op": "MUL_MAT",
             "backend": "XDNA", "M": 1024, "K": 2560, "RHS": 1, "kernel_name": "MLIR_AIE",
             "path": "batched", "row_index": None},
        902: {"graph_id": 502, "node_index": 1, "node_name": "tail", "ggml_op": "MUL_MAT",
             "backend": "XDNA", "M": 2, "K": 2560, "RHS": 1, "kernel_name": "MLIR_AIE",
             "path": "per_row", "row_index": 0},
        903: {"graph_id": 502, "node_index": 1, "node_name": "tail", "ggml_op": "MUL_MAT",
             "backend": "XDNA", "M": 2, "K": 2560, "RHS": 1, "kernel_name": "MLIR_AIE",
             "path": "per_row", "row_index": 1},
    }
    runtime_records = []
    for submission_id, base in bases.items():
        if submission_id == 903:
            runtime_records.extend([
                runtime.event(base, submission_id, "KERNEL_CALL_ENTER", submission_id * 10),
                runtime.event(base, submission_id, "KERNEL_CALL_RETURNED", submission_id * 10 + 1),
                runtime.event(base, submission_id, "WAIT_BEGIN", submission_id * 10 + 2),
                runtime.event(base, submission_id, "WAIT_RETURN_STATE", submission_id * 10 + 3,
                              "ERT_CMD_STATE_TIMEOUT", "non_completed_xrt_state"),
            ])
        else:
            runtime_records.extend(runtime.success(base, submission_id, submission_id * 10))
    lines = []
    scheduler_records = list(reversed(records))
    for index in range(max(len(scheduler_records), len(runtime_records))):
        if index < len(scheduler_records):
            lines.append(json.dumps(scheduler_records[index]))
        if index < len(runtime_records):
            lines.append(json.dumps(runtime_records[index]))
    result = parse_scheduler_correlation_lines(lines)
    assert len(result["joined_submissions"]) == 4
    assert result["joined_submissions"][900]["scheduler_graph_id"] == 501
    assert result["joined_submissions"][901]["scheduler_graph_id"] == 502
    assert result["joined_submissions"][902]["local_split_node_index"] == 2
    assert result["joined_submissions"][903]["xrt_result_copy_confirmed"] is False
    assert result["scheduler_backend_node_counts"] == {"CPU": 1, "XDNA": 5}
    assert set(result["dispatch_failures"]) == {523}

    expect_reject = lambda bad: _expect_reject(records, runtime_records, bad)
    duplicate = records + [records[0]]
    expect_reject(duplicate)
    missing_id = [dict(row, scheduler_dispatch_id=None) if row["event"] == "XDNA_SUBMISSION_LINK" else row for row in records]
    expect_reject(missing_id)
    missing_assignment = [row for row in records if not (row["event"] == "SCHEDULER_NODE_ASSIGNMENT" and row["scheduler_graph_id"] == 501 and row["global_node_index"] == 2)]
    expect_reject(missing_assignment)
    bad_range = [dict(row, global_node_index=99) if row["event"] == "XDNA_NODE_EXECUTE" and row["trace_graph_id"] == 501 else row for row in records]
    expect_reject(bad_range)
    incomplete_runtime = [row for row in runtime_records if not (row["submission_id"] == 900 and row["event"] == "RESULT_COPY_DONE")]
    try:
        parse_scheduler_records(records, runtime.validate_records(incomplete_runtime))
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete runtime sequence was accepted")


def _expect_reject(records, runtime_records, bad_records):
    try:
        parse_scheduler_records(bad_records, runtime.validate_records(runtime_records))
    except (ValueError, KeyError):
        return
    raise AssertionError("ambiguous or inconsistent scheduler correlation was accepted")


def main():
    test_synthetic()
    backend_cpp = (ROOT / "ggml" / "src" / "ggml-backend.cpp").read_text(encoding="utf-8")
    backend_xdna = (ROOT / "ggml" / "src" / "ggml-xdna" / "ggml-xdna.cpp").read_text(encoding="utf-8")
    cmake = (ROOT / "ggml" / "src" / "ggml-xdna" / "CMakeLists.txt").read_text(encoding="utf-8")
    impl = (ROOT / "ggml" / "src" / "ggml-impl.h").read_text(encoding="utf-8")
    assert "GGML_XDNA_RUNTIME_TRACE" in backend_cpp and "SCHEDULER_NODE_ASSIGNMENT" in backend_cpp
    assert "ggml_backend_sched_trace_dispatch(sched, split_id, &gv, j0)" in backend_cpp
    assert "ggml_backend_sched_trace_context_set(&next)" in backend_cpp
    assert "XDNA_NODE_EXECUTE" in backend_xdna and "XDNA_SUBMISSION_LINK" in backend_xdna
    assert "ggml_backend_sched_trace_context" in impl
    trace_block = cmake.split("option(GGML_XDNA_RUNTIME_TRACE", 1)[1]
    assert "target_compile_definitions(ggml-base PRIVATE GGML_XDNA_RUNTIME_TRACE)" in trace_block
    assert "target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_BUILD_DIAGNOSTICS)" not in trace_block
    print("XDNA structured scheduler correlation tests: PASS (synthetic host-only; no XRT execution)")




if __name__ == "__main__":
    main()
