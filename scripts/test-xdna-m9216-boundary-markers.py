#!/usr/bin/env python3
"""Static source-order and optional binary checks for M9216 boundary markers."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp"
MARKERS = (
    "M9216_BOUNDARY_AFTER_Q8_SYNC",
    "M9216_BOUNDARY_BEFORE_KERNEL_CALL",
    "M9216_BOUNDARY_AFTER_KERNEL_CALL",
    "M9216_BOUNDARY_BEFORE_WAIT",
    "M9216_BOUNDARY_AFTER_WAIT",
    "M9216_BOUNDARY_BEFORE_OUTPUT_SYNC",
    "M9216_BOUNDARY_AFTER_OUTPUT_SYNC",
    "M9216_BOUNDARY_AFTER_RESULT_COPY",
)

source = SOURCE.read_text(encoding="utf-8")
signature = "static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216("
start = source.index(signature, source.index(signature) + 1)
end = source.index("\nstatic ", start + 1)
body = source[start:end]

assert "GGML_XDNA_ERT_WAIT_LIMIT{5000}" in source
# Seven bounded-wait expressions cover the added fused Gate/Up submit path; the M9216 diagnostic and production alternatives are mutually exclusive below.
assert source.count("ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT)") == 7, "all XRT submission waits must be bounded and abort on timeout"
positions = [body.index(marker) for marker in MARKERS]
assert positions == sorted(positions), "M9216 boundary markers are out of order"
lines = body.splitlines()

def is_diagnostics_guarded(source_lines, target_index):
    guards = []
    for line in source_lines[:target_index + 1]:
        directive = line.strip()
        if directive.startswith("#ifndef "):
            guards.append(False)
        elif directive.startswith("#if"):
            guards.append("GGML_XDNA_BUILD_DIAGNOSTICS" in directive and "!" not in directive)
        elif directive.startswith("#elif"):
            if guards:
                guards[-1] = "GGML_XDNA_BUILD_DIAGNOSTICS" in directive and "!" not in directive
        elif directive == "#else":
            if guards:
                guards[-1] = False
        elif directive == "#endif":
            if guards:
                guards.pop()
    return any(guards)

for marker in MARKERS:
    assert body.count(marker) == 1, f"expected one source occurrence: {marker}"
    index = next(i for i, line in enumerate(lines) if marker in line)
    assert is_diagnostics_guarded(lines, index), f"marker is not diagnostics-only: {marker}"

# Negative control: removing the marker's diagnostics condition must be rejected.
marker_index = next(i for i, line in enumerate(lines) if "M9216_BOUNDARY_BEFORE_KERNEL_CALL" in line)
negative_lines = lines.copy()
assert negative_lines[marker_index - 1].strip() == "#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)"
negative_lines[marker_index - 1] = "#if 1"
assert not is_diagnostics_guarded(negative_lines, marker_index), "negative control did not detect an unguarded boundary marker"

assert body.index('sync_host_written_bo(bo_q8, "K=2560 M=9216 q8 activation') < positions[0]
kernel = body.index("auto run = kernel(")
assert positions[1] < kernel < positions[2]
assert body.index("bo_out);", kernel) < positions[2]
diag_wait = body.index("completion_state = timed_wait_diag ? ggml_backend_xdna_wait_run(run,")
prod_wait = body.index("const ert_cmd_state completion_state = ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT);")
diag_wait_if = body.rfind("#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)", 0, diag_wait)
diag_wait_else = body.index("\n#else", diag_wait)
diag_wait_end = body.index("\n#endif", diag_wait_else)
assert diag_wait_if < diag_wait < diag_wait_else < prod_wait < diag_wait_end
assert positions[3] < diag_wait < positions[4]
assert prod_wait < positions[4]
output_sync = body.index("bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE)")
assert positions[5] < output_sync < positions[6]
copy = body.index("std::memcpy(result, ctx->q4_m9216_buf_out")
assert copy < positions[7] < body.index("return true;", copy)


def check_binary(build, expect_present):
    binary = Path(build) / "bin/ggml-xdna.dll"
    assert binary.is_file(), f"missing XDNA backend binary: {binary}"
    data = binary.read_bytes()
    present = [marker for marker in MARKERS if marker.encode("ascii") in data]
    if expect_present:
        assert len(present) == len(MARKERS), f"diagnostics binary missing markers: {set(MARKERS) - set(present)}"
    else:
        assert not present, f"production binary retains markers: {present}"


if len(sys.argv) > 1:
    check_binary(sys.argv[1], False)
if len(sys.argv) > 2:
    check_binary(sys.argv[2], True)
print("XDNA_M9216_BOUNDARY_MARKER_ORDER=PASS")
