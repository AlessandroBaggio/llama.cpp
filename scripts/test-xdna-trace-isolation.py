#!/usr/bin/env python3
"""Static and binary guard for XDNA trace-only diagnostics."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DIAG = "GGML_XDNA_BUILD_DIAGNOSTICS"
TRACES = (
    "GGML_XDNA_BO_FORENSICS",
    "GGML_XDNA_ELIGIBILITY_TRACE",
    "GGML_XDNA_EXECUTION_TRACE",
    "GGML_XDNA_INSTANCE_TRACE",
    "GGML_XDNA_RUNTIME_BATCH_TRACE",
    "GGML_XDNA_RUN_STATE_TRACE",
    "GGML_XDNA_SCHED_PROVENANCE_TRACE",
    "GGML_XDNA_SPLIT_ORDER_TRACE",
    "GGML_XDNA_XRT_SEQUENCE_TRACE",
    "GGML_XDNA_XRT_CONTEXT_TRACE",
)
MARKERS = (
    b"XDNA_EXECUTION_TRACE ", b"XDNA_XRT_SEQUENCE", b"XDNA_BACKEND_INSTANCE REGISTER", b"XDNA_BACKEND_INSTANCE FREE", b"XDNA_BACKEND_INSTANCE RESOURCES", b"XDNA_BACKEND_INSTANCE INIT",
    b"XDNA_BUFFER_OBJECT BO", b"XDNA_RUN_STATE RUN", b"XDNA_BACKEND_TRACE ", b"XDNA_SCHEDULER_PROVENANCE ",
    b"XDNA_XRT_CONTEXT ",
    b"ggml_xdna: eligibility", b"ggml_xdna: graph_compute n_nodes", b"offload_decision",
)
SOURCES = (
    ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp",
    ROOT / "ggml/src/ggml-backend.cpp",
    ROOT / "src/llama-context.cpp",
)


def verify_source(path: Path) -> None:
    stack = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.lstrip()
        if re.match(r"#\s*if\b", stripped):
            stack.append((DIAG in stripped, True))
        elif re.match(r"#\s*else\b", stripped) and stack and stack[-1][0]:
            diag, selected = stack[-1]
            stack[-1] = (diag, not selected)
        elif re.match(r"#\s*endif\b", stripped) and stack:
            stack.pop()
        for name in TRACES:
            if name in line:
                assert any(is_diag and selected for is_diag, selected in stack), (
                    f"{path.relative_to(ROOT)}:{number}: {name} is not diagnostics-only"
                )


for source in SOURCES:
    verify_source(source)

def check_binary(build, diagnostics):
    build = Path(build)
    backend = build / "bin/ggml-xdna.dll"
    assert backend.is_file(), f"missing XDNA backend binary: {backend}"
    data = backend.read_bytes()
    if diagnostics:
        assert b"GGML_XDNA_XRT_CONTEXT_TRACE" in data, "diagnostics binary missing shared-context trace gate"
        assert b"XDNA_XRT_CONTEXT " in data, "diagnostics binary missing shared-context trace marker"
        return

    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    assert f"{DIAG}:BOOL=OFF" in cache, "expected diagnostics-OFF production build"
    for name in ("ggml-base.dll", "ggml-xdna.dll", "llama.dll", "llama-cli.exe"):
        binary = build / "bin" / name
        if not binary.exists():
            continue
        data = binary.read_bytes()
        for trace in TRACES:
            assert trace.encode("ascii") not in data, f"{trace} remains in {binary.name}"
        for marker in MARKERS:
            assert marker not in data, f"trace marker {marker!r} remains in {binary.name}"


if len(sys.argv) > 1:
    check_binary(sys.argv[1], False)
if len(sys.argv) > 2:
    check_binary(sys.argv[2], True)
print(f"XDNA trace-only diagnostics isolation: PASS traces={len(TRACES)}")
