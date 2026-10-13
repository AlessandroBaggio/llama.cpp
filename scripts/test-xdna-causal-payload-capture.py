#!/usr/bin/env python3
"""Static/build checks for the diagnostics-only bounded causal payload capture."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp"
DIAGNOSTICS = ROOT / "ggml/src/ggml-xdna/ggml-xdna-diagnostics.cpp"
WRITER = ROOT / "ggml/src/ggml-xdna/ggml-xdna-causal-payload-capture.cpp"
HEADER = ROOT / "ggml/src/ggml-xdna/ggml-xdna-internal.h"
MARKERS = (
    b"GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_DIR",
    b"XDNA_CAUSAL_CAPTURE_ERROR=",
    b"PREPARED",
    b"WAIT_ENTER",
    b"WAIT_RETURN",
    b"RESULT_COPY_RETURN",
    b"q4-native.bin",
    b".q8.bin",
)


def diag_guarded(text, position):
    stack = []
    for line in text[:position].splitlines():
        directive = line.strip()
        if directive.startswith(("#if", "#ifdef", "#ifndef")):
            stack.append(["GGML_XDNA_BUILD_DIAGNOSTICS" in directive, True])
        elif directive.startswith("#else") and stack and stack[-1][0]:
            stack[-1][1] = not stack[-1][1]
        elif directive.startswith("#endif") and stack:
            stack.pop()
    return any(is_diag and enabled for is_diag, enabled in stack)


def check_source():
    source = BACKEND.read_text(encoding="utf-8")
    diagnostics = DIAGNOSTICS.read_text(encoding="utf-8")
    writer = WRITER.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")

    for token in ("ggml_backend_xdna_causal_capture_reserve(",
                  "ggml_backend_xdna_causal_capture_prepare(",
                  "ggml_backend_xdna_causal_capture_event("):
        positions = [i for i in range(len(source)) if source.startswith(token, i)]
        assert positions, f"missing backend hook: {token}"
        assert all(diag_guarded(source, pos) for pos in positions), f"unguarded production hook: {token}"

    for token in ("ggml_backend_xdna_causal_capture_reserve(",
                  "ggml_backend_xdna_causal_capture_prepare(",
                  "ggml_backend_xdna_causal_capture_event("):
        assert token in diagnostics or token in source
    assert "GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_DIR" in diagnostics
    assert "GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_BEGIN" in diagnostics
    assert "GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_END" in diagnostics
    assert "REPLAY_OR_OTHER_CAPTURE_CONTROL_PRESENT" in diagnostics
    assert "max_window_size = 256" in WRITER.with_suffix(".h").read_text(encoding="utf-8")
    assert "fs::exists(absolute, ec)" in writer, "capture session must reject an existing directory"
    assert "safe_relative_path" in writer and "component == \"..\"" in writer
    assert "output.flush()" in writer and "output.close()" in writer
    assert "hash_file(temporary, size, persisted)" in writer
    assert '"event", "PREPARED"' in writer
    assert '"correlation_id"' in writer and '"m9216_ordinal"' in writer
    assert '"src0"' in writer and '"src1"' in writer and '"dst"' in writer

    batched = source.rindex("static bool ggml_backend_xdna_run_batched_shape(")
    batched_end = source.index("\nstatic ", batched + 10)
    batched_body = source[batched:batched_end]
    assert batched_body.index('sync_host_written_bo(*state.bo_q8') < batched_body.index("causal_capture_prepare(")
    assert batched_body.index("causal_capture_prepare(") < batched_body.index("auto run = (*state.kernel)")
    assert "state.rows != 4096" in batched_body

    m9216 = source.rindex("static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216(")
    m9216_end = source.index("\nstatic ", m9216 + 10)
    m9216_body = source[m9216:m9216_end]
    assert m9216_body.index('sync_host_written_bo(bo_q8, "K=2560 M=9216 q8 activation') < m9216_body.index("causal_capture_prepare(")
    assert m9216_body.index("causal_capture_prepare(") < m9216_body.index("auto run = kernel(")
    for event in ("KERNEL_CALL_RETURNED", "WAIT_ENTER", "WAIT_RETURN", "OUTPUT_SYNC_RETURN", "RESULT_COPY_RETURN"):
        assert event in m9216_body and event in batched_body

    assert "ggml_backend_xdna_causal_capture_reserve" in header
    assert "GGML_XDNA_BUILD_DIAGNOSTICS" in header
    assert "ggml_backend_xdna_diagnostic_capture_prepare" not in writer
    assert "replay_native" not in writer and "replay_activation" not in writer


def check_binary(build, expected):
    binary = Path(build) / "bin/ggml-xdna.dll"
    assert binary.is_file(), f"missing XDNA backend DLL: {binary}"
    data = binary.read_bytes()
    present = [marker for marker in MARKERS if marker in data]
    if expected:
        assert present == list(MARKERS), f"diagnostics DLL missing markers: {set(MARKERS) - set(present)}"
    else:
        assert not present, f"production DLL retains causal-capture strings: {present}"


check_source()
if len(sys.argv) == 4:
    check_binary(sys.argv[1], False)
    check_binary(sys.argv[2], True)
    result = subprocess.run([sys.argv[3]], check=True, text=True, capture_output=True)
    assert "XDNA_CAUSAL_PAYLOAD_CAPTURE_SELFTEST=PASS" in result.stdout
elif len(sys.argv) != 1:
    raise SystemExit("usage: test-xdna-causal-payload-capture.py [production-build diagnostics-build selftest-executable]")
print("XDNA_CAUSAL_PAYLOAD_CAPTURE_STATIC_TEST=PASS")
