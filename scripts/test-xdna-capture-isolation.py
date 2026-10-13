#!/usr/bin/env python3
"""Host-only static guard for the XDNA diagnostic capture boundary."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
backend = (ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp").read_text(encoding="utf-8")
internal = (ROOT / "ggml/src/ggml-xdna/ggml-xdna-internal.h").read_text(encoding="utf-8")
diagnostics = (ROOT / "ggml/src/ggml-xdna/ggml-xdna-diagnostics.cpp").read_text(encoding="utf-8")
cmake = (ROOT / "ggml/src/ggml-xdna/CMakeLists.txt").read_text(encoding="utf-8")

capture_env = (
    "GGML_XDNA_CAPTURE_EXACT_HISTORY",
    "GGML_XDNA_CAPTURE_HISTORY_DIR",
    "GGML_XDNA_CAPTURE_M9216_PAYLOAD",
)

def diagnostic_if(condition):
    positive = re.search(r"(?<![!\w])defined\s*\(\s*GGML_XDNA_BUILD_DIAGNOSTICS\s*\)", condition)
    negative = re.search(r"!\s*defined\s*\(\s*GGML_XDNA_BUILD_DIAGNOSTICS\s*\)", condition)
    return positive is not None and negative is None

def unguarded_capture_occurrences(source):
    guards = []
    unguarded = []
    for line_number, line in enumerate(source.splitlines(), 1):
        directive = line.strip()
        if directive.startswith(("#if ", "#ifdef ", "#ifndef ")):
            guards.append(diagnostic_if(directive.split(None, 1)[1]))
            continue
        if directive.startswith("#elif "):
            if guards:
                guards[-1] = diagnostic_if(directive.split(None, 1)[1])
            continue
        if directive == "#else":
            if guards:
                guards[-1] = False
            continue
        if directive == "#endif":
            if guards:
                guards.pop()
            continue
        for name in capture_env:
            if name in line and not any(guards):
                unguarded.append((name, line_number))
    return unguarded

# These names remain in the diagnostics-only Q4 control conflict list; guard scope is the invariant.
assert not unguarded_capture_occurrences(backend), unguarded_capture_occurrences(backend)
for name in capture_env:
    assert f'std::getenv("{name}")' in diagnostics, f"capture environment is not read in diagnostics: {name}"
    assert name in diagnostics, f"capture environment missing from diagnostics: {name}"

# Negative control: an actual production getenv must be rejected by this source guard.
leaked_backend = backend + '\nconst char * leaked = std::getenv("GGML_XDNA_CAPTURE_EXACT_HISTORY");\n'
assert unguarded_capture_occurrences(leaked_backend), "negative control did not detect an unguarded capture read"
assert "ggml_xdna_capture::capture_session" not in backend
assert "exact_history_capture_enabled" not in internal
assert "exact_history_capture_dir" not in internal
assert "m9216_diagnostic_ordinal" in internal
assert "GGML_XDNA_BUILD_DIAGNOSTICS" in cmake and " OFF)" in cmake
assert "target_sources(ggml-xdna PRIVATE ggml-xdna-diagnostics.cpp)" in cmake
assert "add_executable(xdna-capture-session-selftest" in cmake
assert "GGML_XDNA_BUILD_DIAGNOSTICS" in backend

# Trace-only controls remain in ordinary source; fault controls are isolated.
for name in ("GGML_XDNA_RUN_STATE_TRACE", "GGML_XDNA_BO_FORENSICS", "GGML_XDNA_ELIGIBILITY_TRACE"):
    assert name in backend, f"unrelated diagnostic control moved or removed: {name}"

if len(sys.argv) > 1:
    build = Path(sys.argv[1])
    ninja = (build / "build.ninja").read_text(encoding="utf-8", errors="replace")
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    assert "GGML_XDNA_BUILD_DIAGNOSTICS:BOOL=OFF" in cache
    assert "ggml-xdna-diagnostics.cpp" not in ninja
    assert "xdna-capture-replay" not in ninja
    assert "xdna-capture-session-selftest" not in ninja
    binary = build / "bin/ggml-xdna.dll"
    assert binary.is_file(), f"missing XDNA production DLL: {binary}"
    binary_data = binary.read_bytes()
    for name in capture_env:
        assert name.encode("ascii") not in binary_data, f"capture control leaked into production DLL: {name}"

print("XDNA capture isolation static check: PASS")
