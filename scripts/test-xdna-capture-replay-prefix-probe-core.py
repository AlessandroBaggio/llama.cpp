#!/usr/bin/env python3
"""Run fixture-independent prefix/probe models and source-order assertions."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("test-xdna-capture-replay-prefix-probe.py")
spec = importlib.util.spec_from_file_location("prefix_probe_models", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load prefix/probe test module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

TESTS = (
    "test_post_recreation_control_guard_model",
    "test_transition_guard_model",
    "test_role_aware_transition_guard_models",
    "test_same_shape_id8_guard_model",
    "test_same_shape_id7_guard_model",
    "test_context_kernel_recreate_to_id8_guard_model",
    "test_context_recreate_lifecycle_model",
    "test_post_id8_context_recreate_guard_model",
    "test_full_control_model",
    "test_second_recreate_no_execution_model",
    "test_program_only_second_recreate_model",
    "test_execute_wait_no_readback_second_recreate_model",
    "test_temp_resource_release_model_and_source_order",
    "test_historical_output_mismatch_models",
    "test_terminal_success_model",
    "test_release_terminal_fallthrough_models",
    "test_run_only_release_model_and_source_order",
)

for name in TESTS:
    getattr(module, name)()
print(f"XDNA_CAPTURE_REPLAY_PREFIX_PROBE_CORE=PASS model_and_source_tests={len(TESTS)}")
