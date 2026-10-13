#!/usr/bin/env python3
"""Static and binary guard for XDNA ledger/cache/timing observability."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DIAG = "GGML_XDNA_BUILD_DIAGNOSTICS"
SOURCES = (
    ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp",
    ROOT / "ggml/src/ggml-xdna/ggml-xdna-internal.h",
)
MARKERS = (
    b"ggml_xdna: ledger accepted=",
    b"ggml_xdna: q4_cache enabled=",
    b"ggml_xdna: persistent_weight_bos enabled=",
    b"ggml_xdna: timers_ms",
    b"ggml_xdna: xrt_shape=",
)
STATE = (
    "accepted_q4_nodes", "batched_m9216", "batched_m4096", "batched_m8192",
    "batched_m1024", "fallback_rows", "xrt_runs", "run_waits",
    "run_constructed", "run_completed", "run_destroyed", "run_live",
    "q4_to_device", "q8_to_device", "output_from_device", "timers",
    "shape_timing", "hits", "misses", "hits_m9216", "hits_m4096",
    "hits_m8192", "hits_m1024", "blocks_converted", "native_bytes_converted",
    "transport_bytes_generated", "transport_bytes_copied", "weight_bo_creates",
    "weight_bo_reuses", "weight_bo_failures", "weight_bo_cap_hits",
)


def diag_lines(path):
    stack = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.lstrip()
        if re.match(r"#\s*if\b", stripped):
            stack.append([DIAG in stripped, True])
        elif re.match(r"#\s*else\b", stripped) and stack and stack[-1][0]:
            stack[-1][1] = not stack[-1][1]
        yield number, line, any(is_diag and selected for is_diag, selected in stack)
        if re.match(r"#\s*endif\b", stripped) and stack:
            stack.pop()


for source in SOURCES:
    for number, line, under_diag in diag_lines(source):
        if line.lstrip().startswith("//"):
            continue
        for marker in MARKERS:
            if marker.decode("ascii") in line:
                assert under_diag, f"{source.relative_to(ROOT)}:{number}: summary not diagnostics-only"
        for name in STATE:
            if re.search(rf"\b{re.escape(name)}\b", line):
                assert under_diag or "GGML_XDNA_DIAG_ONLY(" in line, (
                    f"{source.relative_to(ROOT)}:{number}: {name} is not diagnostics-only"
                )


def check_binary(build, expect_present):
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    expected = "ON" if expect_present else "OFF"
    assert f"{DIAG}:BOOL={expected}" in cache, f"expected diagnostics {expected}: {build}"
    checked = 0
    observed = set()
    for name in ("ggml-base.dll", "ggml-xdna.dll", "llama.dll", "llama-cli.exe"):
        binary = build / "bin" / name
        if not binary.exists():
            continue
        checked += 1
        data = binary.read_bytes()
        for marker in MARKERS:
            present = marker in data
            if present:
                observed.add(marker)
            if not expect_present:
                assert not present, f"{marker!r} remains in {binary.name}"
    assert checked, f"no expected production binaries found in {build / 'bin'}"
    if expect_present:
        assert observed == set(MARKERS), f"diagnostics build lost markers: {set(MARKERS) - observed}"

if len(sys.argv) > 1:
    check_binary(Path(sys.argv[1]), False)
if len(sys.argv) > 2:
    check_binary(Path(sys.argv[2]), True)
print(f"XDNA observability isolation: PASS summary_markers={len(MARKERS)} diagnostic_state_fields={len(STATE)}")
