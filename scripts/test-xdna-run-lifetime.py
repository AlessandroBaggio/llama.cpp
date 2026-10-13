#!/usr/bin/env python3
"""Static and modeled checks for XDNA xrt::run ownership ordering."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp").read_text(encoding="utf-8")
DIAGNOSTICS = (ROOT / "ggml/src/ggml-xdna/ggml-xdna-diagnostics.cpp").read_text(encoding="utf-8")


def model(events):
    live = False
    completed = False
    for event in events:
        if event == "submit":
            assert not live
            live, completed = True, False
        elif event == "wait":
            assert live
            completed = True
        elif event in {"wait_timeout", "wait_error"}:
            assert live
        elif event == "abort":
            assert live
            completed = True
        elif event == "release":
            assert live and completed
            live = False
        elif event in {"recreate_context", "destroy_context", "destroy_kernel", "recreate_kernel"}:
            assert not live, f"{event} while associated run is alive"
        else:
            raise AssertionError(f"unknown event: {event}")
    assert not live, "missing run-lifetime release"


def rejects(events):
    try:
        model(events)
    except AssertionError:
        return
    raise AssertionError(f"unsafe lifetime sequence accepted: {events}")


# Production XDNA submissions are local owners, waited before the enclosing
# execution helper returns; the diagnostic recreation owner is an optional.
assert len(re.findall(r"\bstd::optional<xrt::run>\s+run\s*;", DIAGNOSTICS)) == 1
assert SOURCE.count("auto run = kernel(") >= 2
assert SOURCE.count("auto run = (*state.kernel)(") == 1
assert SOURCE.count("state = run.wait(timeout);") == 1
wait_helper = SOURCE[SOURCE.index("static ert_cmd_state ggml_backend_xdna_wait_run("):]
wait_helper = wait_helper[:wait_helper.index("\nstatic ", 1)]
assert "state = run.wait(timeout);" in wait_helper
assert "if (state == ERT_CMD_STATE_TIMEOUT)" in wait_helper
assert wait_helper.count("run.abort()") == 2
assert SOURCE.count("ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT)") == 6
assert SOURCE.count("ggml_backend_xdna_wait_run(run, std::chrono::milliseconds(1000))") == 1
assert "run.reset();" not in SOURCE
assert DIAGNOSTICS.count("run = std::nullopt;") == 2

# Recreation helper checks the actual optional when one is in scope. The only
# live-run exception is the explicit historical diagnostic discriminator.
helper = DIAGNOSTICS[DIAGNOSTICS.index("static void ggml_backend_xdna_replay_recreate_m9216_hw_context("):]
helper = helper[:helper.index("\nstatic ", 1)]
assert "live_run->has_value()" in helper
assert "allow_live_run_control" in helper
call = "ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, false, true, &run, allow_live_run_control);"
assert call in DIAGNOSTICS
assert "const bool allow_live_run_control = post_recreate_execute_wait_no_readback_context_recreate_mode;" in DIAGNOSTICS
release_branch = DIAGNOSTICS.index("run = std::nullopt;", DIAGNOSTICS.index("POST_WAIT_NO_READBACK"))
recreate_call = DIAGNOSTICS.index(call)
assert release_branch < recreate_call

# Normal inference waits inside the local run's scope; that scope ends before
# its helper returns and before later graph dispatch / backend teardown.
for marker in ("static bool ggml_backend_xdna_run_i16_f32_matmul(",
               "static bool ggml_backend_xdna_run_batched_shape(",
               "static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216(",
               "static bool ggml_backend_xdna_run_q4k_q8k_k2560("):
    start = SOURCE.index(marker, SOURCE.index(marker) + 1)
    body = SOURCE[start:]
    body = body[:body.index("\nstatic ", 1)]
    assert "ggml_backend_xdna_wait_run(run," in body

# Negative lifecycle matrix: recreation/destruction with a live run, premature
# release, missing release, and helper invocation before owner release.
rejects(["submit", "wait", "recreate_context"])
rejects(["submit", "wait", "destroy_kernel"])
rejects(["submit", "wait", "destroy_context"])
rejects(["submit", "release", "wait"])
rejects(["submit", "wait"])
rejects(["submit", "wait", "recreate_kernel"])
rejects(["submit", "wait_timeout", "release"])
rejects(["submit", "wait_error", "release"])
model(["submit", "wait", "release", "destroy_kernel", "destroy_context", "recreate_context"])
model(["submit", "wait_timeout", "abort", "release"])

# EN1R remains an explicit opt-in live-run diagnostic control, distinct from
# the release-run path; it is not the production backend lifetime policy.
assert "post_recreate_execute_wait_no_readback_context_recreate_mode;" in DIAGNOSTICS
assert "post_recreate_execute_wait_release_run_context_recreate_mode" in DIAGNOSTICS
print("XDNA_RUN_LIFETIME_STATIC=PASS negative_cases=8 production_run_owners=local diagnostic_control=preserved")
