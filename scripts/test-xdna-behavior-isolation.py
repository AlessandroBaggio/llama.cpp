#!/usr/bin/env python3
"""Static, host-only guard for behavior-changing XDNA diagnostic controls."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = (
    "GGML_XDNA_BQ1_STAGE",
    "GGML_XDNA_DIAG_RECREATE_M9216_BUNDLE_BEFORE_9",
    "GGML_XDNA_DIAG_RECREATE_M9216_KERNEL_BEFORE_9",
    "GGML_XDNA_DIAG_TIMED_WAIT",
    "GGML_XDNA_FORCE_SPLIT_SYNC",
    "GGML_XDNA_STOP_AFTER_RUNTIME_SPLIT",
    "GGML_XDNA_W2IN_BLOCK_PROBE",
    "GGML_XDNA_BH1_REPLAY_F32",
    "GGML_XDNA_BH1_REPLAY_NATIVE_Q4",
    "GGML_XDNA_BH1_REPLAY_Q4_TRANSPORT",
    "GGML_XDNA_BH1_REPLAY_Q8",
)
DIAG = "GGML_XDNA_BUILD_DIAGNOSTICS"


def verify_guarded(path: Path) -> None:
    frames = []  # (is diagnostics conditional, selected branch)
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if re.match(r"#\s*if\b", stripped):
            frames.append((DIAG in stripped, True))
        elif re.match(r"#\s*else\b", stripped) and frames and frames[-1][0]:
            is_diag, selected = frames[-1]
            frames[-1] = (is_diag, not selected)
        elif re.match(r"#\s*endif\b", stripped) and frames:
            frames.pop()
        if any(name in line for name in CONTROLS):
            assert any(is_diag and selected for is_diag, selected in frames), (
                f"{path.relative_to(ROOT)}:{number}: behavior control is not inside "
                f"the diagnostics-only preprocessor branch: {line.strip()}"
            )


backend = ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp"
scheduler = ROOT / "ggml/src/ggml-backend.cpp"
cmake = (ROOT / "ggml/src/ggml-xdna/CMakeLists.txt").read_text(encoding="utf-8")
verify_guarded(backend)
verify_guarded(scheduler)
assert len(CONTROLS) == 11
assert "target_compile_definitions(ggml-base PRIVATE GGML_XDNA_BUILD_DIAGNOSTICS)" in cmake
assert "target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_BUILD_DIAGNOSTICS)" in cmake

if len(sys.argv) > 1:
    build = Path(sys.argv[1])
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    ninja = (build / "build.ninja").read_text(encoding="utf-8", errors="replace")
    assert "GGML_XDNA_BUILD_DIAGNOSTICS:BOOL=OFF" in cache
    assert "GGML_XDNA_BUILD_DIAGNOSTICS" not in ninja
    assert "ggml-xdna-diagnostics.cpp" not in ninja
    assert "xdna-capture-replay" not in ninja
    base_dll = build / "bin" / "ggml-base.dll"
    xdna_dll = build / "bin" / "ggml-xdna.dll"
    assert base_dll.is_file() and xdna_dll.is_file(), "expected production DLLs are missing"
    for binary in (base_dll, xdna_dll):
        data = binary.read_bytes()
        for name in CONTROLS:
            assert name.encode("ascii") not in data, f"{name} remains in default binary {binary.name}"

print(f"XDNA behavior-changing env isolation: PASS controls={len(CONTROLS)}")
