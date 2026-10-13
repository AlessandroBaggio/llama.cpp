#!/usr/bin/env python3
"""Host-only prefix/probe selection tests against the immutable CF1 capture."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[1]
CAPTURE = None


def run(exe, env, args):
    result = subprocess.run([str(exe), str(CAPTURE), *args, "--dry-run"], capture_output=True, text=True, env=env)
    return result


def manifest_sha(capture):
    return hashlib.sha256((capture / "manifest.jsonl").read_bytes()).hexdigest()


def pre_recreate_control_guard(state):
    return (state["history_completed"] == 10 and state["history_outputs_validated"] == 10 and
            not state["replay_failed"] and state["m9216_submissions"] == 7 and
            state["m9216_waits"] == 7 and state["m9216_output_syncs"] == 7 and
            state["context_generation"] == state["kernel_generation"] == state["bo_generation"] == 0 and
            state["context_recreate_count"] == state["kernel_recreate_count"] == state["bo_recreate_count"] == 0)


def test_post_recreation_control_guard_model():
    before = {"history_completed": 10, "history_outputs_validated": 10, "replay_failed": False,
              "m9216_submissions": 7, "m9216_waits": 7, "m9216_output_syncs": 7,
              "context_generation": 0, "kernel_generation": 0, "bo_generation": 0,
              "context_recreate_count": 0, "kernel_recreate_count": 0, "bo_recreate_count": 0,
              "generation_submissions": 7, "generation_waits": 7}
    assert pre_recreate_control_guard(before)  # generation-0 local 7/7 is valid here
    after = dict(before, context_generation=1, kernel_generation=1, bo_generation=1,
                 context_recreate_count=1, kernel_recreate_count=1, bo_recreate_count=1,
                 generation_submissions=0, generation_waits=0, m9216_ordinal=7)
    assert post_recreate_control_guard(after)
    for change in ({"generation_submissions": 1}, {"generation_waits": 1}, {"bo_generation": 2}, {"m9216_submissions": 8}):
        assert not post_recreate_control_guard(dict(after, **change)), change
    assert not pre_recreate_control_guard(dict(before, m9216_submissions=8))
def post_recreate_control_guard(state):
    return (state["m9216_submissions"] == 7 and state["m9216_waits"] == 7 and state["m9216_output_syncs"] == 7 and
            state["m9216_ordinal"] == 7 and state["context_generation"] == state["kernel_generation"] == state["bo_generation"] == 1 and
            state["context_recreate_count"] == state["kernel_recreate_count"] == state["bo_recreate_count"] == 1 and
            state["generation_submissions"] == 0 and state["generation_waits"] == 0)
def transition_guard(state):
    return (state["history_outputs"] == 10 and state["context_generation"] == state["kernel_generation"] == state["bo_generation"] == 1 and
            state["generation1_submissions"] == state["generation1_waits"] == state["generation1_output_syncs"] == 2 and
            state["m4096_submitted"] == state["m4096_waited"] == state["m4096_exact"] == 1 and
            state["id8_submitted"] == state["id8_waited"] == state["id8_exact"] == 1 and state["probe11"] == state["target13"] == 0)


def test_transition_guard_model():
    state = dict(history_outputs=10, context_generation=1, kernel_generation=1, bo_generation=1,
                 generation1_submissions=2, generation1_waits=2, generation1_output_syncs=2,
                 m4096_submitted=1, m4096_waited=1, m4096_exact=1,
                 id8_submitted=1, id8_waited=1, id8_exact=1, probe11=0, target13=0)
    assert transition_guard(state)
    for key, value in (("generation1_submissions", 1), ("generation1_waits", 1), ("generation1_output_syncs", 1),
                       ("m4096_exact", 0), ("id8_exact", 0), ("context_generation", 2), ("probe11", 1), ("target13", 1)):
        assert not transition_guard(dict(state, **{key: value})), key

def role_aware_transition_guard(state, mode):
    identity_and_history = (state["context_generation"] == state["kernel_generation"] == state["bo_generation"] == 1 and
        state["id7_source_id"] == 7 and state["id7_M"] == 4096 and state["next_source_id"] == 8 and state["next_M"] == 9216 and
        state["m9216_submissions"] == state["m9216_waits"] == state["m9216_output_syncs"] == 7 and
        state["same_context"] and state["same_kernel"] and state["no_intervening_submission"] and state["id8_submitted"] == 0)
    if not identity_and_history:
        return False
    if mode == "m4096-program-only-to-id8":
        return (state["generation1_submissions"] == state["generation1_waits"] == 0 and
            state["m4096_submissions"] == state["m4096_waits"] == 0 and state["generation1_ordinal"] == 1)
    m4096_completed = (state["generation1_submissions"] == state["generation1_waits"] == 1 and
        state["m4096_submissions"] == state["m4096_waits"] == 1 and state["generation1_ordinal"] == 2)
    if mode == "m4096-execute-no-readback-to-id8":
        return m4096_completed and all(state[key] == 0 for key in ("output_from_device_syncs", "output_reads", "output_sha_calculations", "output_validations"))
    if mode == "m4096-to-id8":
        return (m4096_completed and state["output_from_device_syncs"] == state["output_reads"] ==
            state["output_sha_calculations"] == state["output_validations"] == state["id7_exact"] == 1)
    return False


def test_role_aware_transition_guard_models():
    common = dict(context_generation=1, kernel_generation=1, bo_generation=1, id7_source_id=7, id7_M=4096,
        next_source_id=8, next_M=9216, m9216_submissions=7, m9216_waits=7, m9216_output_syncs=7,
        same_context=True, same_kernel=True, no_intervening_submission=True, id8_submitted=0,
        generation1_submissions=1, generation1_waits=1, generation1_ordinal=2, m4096_submissions=1, m4096_waits=1,
        output_from_device_syncs=0, output_reads=0, output_sha_calculations=0, output_validations=0, id7_exact=0)
    assert role_aware_transition_guard(common, "m4096-execute-no-readback-to-id8")
    print("NO-READBACK TRANSITION GUARD PASS")
    for key in ("output_from_device_syncs", "output_reads", "output_sha_calculations", "output_validations"):
        assert not role_aware_transition_guard(dict(common, **{key: 1}), "m4096-execute-no-readback-to-id8"), key
    for key, value in (("generation1_submissions", 0), ("generation1_waits", 0), ("m4096_submissions", 0),
                       ("m4096_waits", 0), ("generation1_ordinal", 1), ("next_source_id", 7), ("same_context", False),
                       ("same_kernel", False)):
        assert not role_aware_transition_guard(dict(common, **{key: value}), "m4096-execute-no-readback-to-id8"), key
    ordinary = dict(common, output_from_device_syncs=1, output_reads=1, output_sha_calculations=1, output_validations=1, id7_exact=1)
    assert role_aware_transition_guard(ordinary, "m4096-to-id8")
    print("XDNA_CAPTURE_REPLAY TRANSITION GUARD PASS")
    assert not role_aware_transition_guard(common, "m4096-to-id8")
    program_only = dict(common, generation1_submissions=0, generation1_waits=0, generation1_ordinal=1, m4096_submissions=0, m4096_waits=0)
    assert role_aware_transition_guard(program_only, "m4096-program-only-to-id8")

def same_shape_id8_guard(state):
    return (state["history_completed"] == 10 and state["history_outputs_validated"] == 10 and
            state["context_generation"] == state["kernel_generation"] == state["bo_generation"] == 1 and
            state["context_recreate_count"] == state["kernel_recreate_count"] == state["bo_recreate_count"] == 1 and
            state["first_source_id"] == state["second_source_id"] == 8 and
            state["first_M"] == state["second_M"] == 9216 and
            state["first_completed"] == state["first_validated"] == state["first_exact"] == 1 and
            state["generation1_submissions_before_second"] == state["generation1_waits_before_second"] == 1 and
            state["next_ordinal"] == 2 and state["same_instruction_image"] and
            state["same_context"] and state["same_kernel"] and state["same_bos_and_maps"] and
            state["no_intervening_submission"] and state["other_shape_submissions"] == 0 and
            state["probe11"] == state["target13"] == 0)


def test_same_shape_id8_guard_model():
    state = dict(history_completed=10, history_outputs_validated=10, context_generation=1, kernel_generation=1, bo_generation=1,
        context_recreate_count=1, kernel_recreate_count=1, bo_recreate_count=1, first_source_id=8, second_source_id=8,
        first_M=9216, second_M=9216, first_completed=1, first_validated=1, first_exact=1,
        generation1_submissions_before_second=1, generation1_waits_before_second=1, next_ordinal=2,
        same_instruction_image=True, same_context=True, same_kernel=True, same_bos_and_maps=True,
        no_intervening_submission=True, other_shape_submissions=0, probe11=0, target13=0)
    assert same_shape_id8_guard(state)
    print("SAME-SHAPE ID8 CONTROL GUARD PASS")
    for key, value in (("first_completed", 0), ("first_validated", 0), ("first_exact", 0),
                       ("second_source_id", 7), ("same_instruction_image", False), ("context_generation", 2),
                       ("next_ordinal", 3), ("no_intervening_submission", False),
                       ("generation1_submissions_before_second", 2), ("same_bos_and_maps", False),
                       ("probe11", 1), ("target13", 1)):
        assert not same_shape_id8_guard(dict(state, **{key: value})), key
def same_shape_id7_guard(state):
    return (state["history_completed"] == state["history_outputs_validated"] == 10 and
            state["context_generation"] == state["kernel_generation"] == state["bo_generation"] == 1 and
            state["context_recreate_count"] == state["kernel_recreate_count"] == state["bo_recreate_count"] == 1 and
            state["first_source_id"] == state["second_source_id"] == 7 and state["first_M"] == state["second_M"] == 4096 and
            state["frozen_payload_identity"] and state["first_completed"] == state["first_validated"] == state["first_exact"] == 1 and
            state["id7_submissions_before_second"] == state["id7_waits_before_second"] == state["id7_output_syncs_before_second"] == 1 and
            state["id7_exact_outputs_before_second"] == 1 and state["next_ordinal"] == 2 and
            state["same_context"] and state["same_kernel"] and state["same_bos_and_maps"] and state["no_intervening_submission"] and
            state["m9216_submissions"] == state["m9216_waits"] == state["m9216_output_syncs"] == 7 and state["probe11"] == state["target13"] == 0)


def test_same_shape_id7_guard_model():
    state = dict(history_completed=10, history_outputs_validated=10, context_generation=1, kernel_generation=1, bo_generation=1,
        context_recreate_count=1, kernel_recreate_count=1, bo_recreate_count=1, first_source_id=7, second_source_id=7,
        first_M=4096, second_M=4096, frozen_payload_identity=True, first_completed=1, first_validated=1, first_exact=1,
        id7_submissions_before_second=1, id7_waits_before_second=1, id7_output_syncs_before_second=1, id7_exact_outputs_before_second=1, next_ordinal=2,
        same_context=True, same_kernel=True, same_bos_and_maps=True, no_intervening_submission=True,
        m9216_submissions=7, m9216_waits=7, m9216_output_syncs=7, probe11=0, target13=0)
    assert same_shape_id7_guard(state)
    print("SAME-SHAPE ID7 CONTROL GUARD PASS")
    for key, value in (("history_outputs_validated", 9), ("context_generation", 2), ("second_source_id", 8),
                       ("second_M", 9216), ("frozen_payload_identity", False), ("first_completed", 0),
                       ("first_validated", 0), ("first_exact", 0), ("id7_submissions_before_second", 0),
                       ("id7_waits_before_second", 0), ("id7_output_syncs_before_second", 0),
                       ("id7_exact_outputs_before_second", 0), ("next_ordinal", 3), ("same_context", False),
                       ("same_kernel", False), ("same_bos_and_maps", False), ("no_intervening_submission", False),
                       ("m9216_submissions", 8), ("probe11", 1), ("target13", 1)):
        assert not same_shape_id7_guard(dict(state, **{key: value})), key
def context_kernel_recreate_to_id8_guard(state):
    return (state["history_completed"] == state["history_outputs_validated"] == 10 and
            state["id7_source"] == 7 and state["id7_M"] == 4096 and state["id7_exact"] and
            state["experiment_ordinal"] == 2 and state["generation1_ordinal"] == 1 and state["generation2_ordinal"] == 1 and
            state["context_generation"] == state["kernel_generation"] == 2 and state["bo_generation"] == 1 and
            state["context_recreate_count"] == state["kernel_recreate_count"] == 2 and state["bo_recreate_count"] == 1 and
            state["context_delta"] == state["kernel_delta"] == 1 and state["bo_delta"] == 0 and
            state["id8_source"] == 8 and state["id8_M"] == 9216 and state["id8_frozen_inputs"] and state["id8_exact"] and
            state["context_changed"] and state["kernel_changed"] and state["same_device"] and state["same_xclbin_uuid"] and state["same_kernel_name"] and
            state["same_bos"] and state["same_maps"] and state["same_capacities"] and state["no_intervening_submission"] and
            all(state[key] == 0 for key in ("bo_create_delta", "map_delta", "sync_delta", "submission_delta", "run_wait_delta", "device_reopen_delta", "xclbin_load_register_delta")) and
            state["probe11"] == state["target13"] == 0)


def test_context_kernel_recreate_to_id8_guard_model():
    state = dict(history_completed=10, history_outputs_validated=10, id7_source=7, id7_M=4096, id7_exact=True,
        experiment_ordinal=2, generation1_ordinal=1, generation2_ordinal=1, context_generation=2, kernel_generation=2, bo_generation=1,
        context_recreate_count=2, kernel_recreate_count=2, bo_recreate_count=1, context_delta=1, kernel_delta=1, bo_delta=0,
        id8_source=8, id8_M=9216, id8_frozen_inputs=True, id8_exact=True, context_changed=True, kernel_changed=True,
        same_device=True, same_xclbin_uuid=True, same_kernel_name=True, same_bos=True, same_maps=True, same_capacities=True,
        no_intervening_submission=True, bo_create_delta=0, map_delta=0, sync_delta=0, submission_delta=0, run_wait_delta=0,
        device_reopen_delta=0, xclbin_load_register_delta=0, probe11=0, target13=0)
    assert context_kernel_recreate_to_id8_guard(state)
    for key, value in (("history_outputs_validated", 9), ("id7_exact", False), ("experiment_ordinal", 1),
                       ("generation1_ordinal", 2), ("generation2_ordinal", 0), ("context_generation", 1),
                       ("kernel_generation", 1), ("bo_generation", 2), ("context_recreate_count", 1),
                       ("kernel_recreate_count", 1), ("bo_recreate_count", 2), ("context_delta", 0),
                       ("kernel_delta", 0), ("bo_delta", 1), ("id8_source", 7), ("id8_M", 4096),
                       ("id8_frozen_inputs", False), ("id8_exact", False), ("context_changed", False),
                       ("kernel_changed", False), ("same_device", False), ("same_xclbin_uuid", False),
                       ("same_kernel_name", False), ("same_bos", False), ("same_maps", False),
                       ("same_capacities", False), ("no_intervening_submission", False),
                       ("bo_create_delta", 1), ("map_delta", 1), ("sync_delta", 1), ("submission_delta", 1),
                       ("run_wait_delta", 1), ("device_reopen_delta", 1), ("xclbin_load_register_delta", 1),
                       ("probe11", 1), ("target13", 1)):
        assert not context_kernel_recreate_to_id8_guard(dict(state, **{key: value})), key
def context_recreate_lifecycle_guard(state):
    return (state["old_kernel_valid_before"] and state["old_context_valid_before"] and
        state["old_kernel_released"] and state["old_context_released"] and
        state["new_context_constructed"] and state["new_context_construct_ok"] and
        state["new_kernel_constructed"] and state["new_kernel_construct_ok"] and
        state["post_inspection_reached"] and state["post_inspection_passed"] and
        not state["old_context_getter_after_release"] and not state["old_kernel_getter_after_release"] and
        state["same_device"] and state["same_xclbin"] and state["same_bos"] and state["same_maps"] and
        all(state[key] for key in ("same_instr_bo", "same_q4_bo", "same_q8_bo", "same_output_bo", "same_instr_map", "same_q4_map", "same_q8_map", "same_output_map")) and
        not state["id8_submitted"] and state["bo_construct_delta"] == state["map_delta"] == state["sync_delta"] == 0)


def test_context_recreate_lifecycle_model():
    state = dict(old_kernel_valid_before=True, old_context_valid_before=True, old_kernel_released=True, old_context_released=True,
        new_context_constructed=True, new_context_construct_ok=True, new_kernel_constructed=True, new_kernel_construct_ok=True,
        post_inspection_reached=True, post_inspection_passed=True, old_context_getter_after_release=False,
        old_kernel_getter_after_release=False, same_device=True, same_xclbin=True, same_bos=True, same_maps=True,
        same_instr_bo=True, same_q4_bo=True, same_q8_bo=True, same_output_bo=True, same_instr_map=True, same_q4_map=True, same_q8_map=True, same_output_map=True,
        id8_submitted=False, bo_construct_delta=0, map_delta=0, sync_delta=0)
    assert context_recreate_lifecycle_guard(state)
    failures = (("new_context_constructed", False), ("new_context_construct_ok", False),
        ("new_kernel_constructed", False), ("new_kernel_construct_ok", False),
        ("post_inspection_reached", False), ("post_inspection_passed", False),
        ("old_context_getter_after_release", True), ("old_kernel_getter_after_release", True),
        ("same_device", False), ("same_xclbin", False), ("same_bos", False), ("same_maps", False),
        ("same_instr_bo", False), ("same_q4_bo", False), ("same_q8_bo", False), ("same_output_bo", False),
        ("same_instr_map", False), ("same_q4_map", False), ("same_q8_map", False), ("same_output_map", False),
        ("old_kernel_released", False), ("old_context_released", False), ("id8_submitted", True),
        ("bo_construct_delta", 1), ("map_delta", 1), ("sync_delta", 1))
    for key, value in failures:
        assert not context_recreate_lifecycle_guard(dict(state, **{key: value})), key
def post_id8_context_recreate_guard(state):
    return (state["history_completed"] == state["history_outputs_validated"] == 10 and
        state["id8_source"] == 8 and state["id8_M"] == 9216 and state["id8_exact"] and
        state["experiment_ordinal"] == state["generation1_ordinal"] == 1 and
        state["context_generation"] == state["kernel_generation"] == 2 and state["bo_generation"] == 1 and
        state["context_recreate_delta"] == state["kernel_recreate_delta"] == 1 and state["bo_recreate_delta"] == 0 and
        state["same_device"] and state["same_xclbin"] and state["same_xclbin_uuid"] and state["same_kernel_name"] and
        state["context_changed"] and state["kernel_changed"] and state["same_bos"] and state["same_maps"] and state["same_capacities"] and
        state["bo_construct_delta"] == state["map_delta"] == state["sync_delta"] == 0 and
        state["no_post_recreate_submission"] and state["probe11"] == state["target13"] == 0)


def test_post_id8_context_recreate_guard_model():
    state = dict(history_completed=10, history_outputs_validated=10, id8_source=8, id8_M=9216, id8_exact=True,
        experiment_ordinal=1, generation1_ordinal=1, context_generation=2, kernel_generation=2, bo_generation=1,
        context_recreate_delta=1, kernel_recreate_delta=1, bo_recreate_delta=0, same_device=True, same_xclbin=True, same_xclbin_uuid=True,
        same_kernel_name=True, context_changed=True, kernel_changed=True, same_bos=True, same_maps=True, same_capacities=True,
        bo_construct_delta=0, map_delta=0, sync_delta=0, no_post_recreate_submission=True, probe11=0, target13=0)
    assert post_id8_context_recreate_guard(state)
    for key, value in (("history_outputs_validated", 9), ("id8_exact", False), ("id8_M", 4096),
        ("context_generation", 1), ("kernel_generation", 1), ("bo_generation", 2), ("bo_recreate_delta", 1),
        ("same_device", False), ("same_xclbin_uuid", False), ("same_kernel_name", False),
        ("context_changed", False), ("kernel_changed", False), ("same_bos", False), ("same_maps", False), ("same_capacities", False),
        ("bo_construct_delta", 1), ("map_delta", 1), ("sync_delta", 1),
        ("no_post_recreate_submission", False), ("probe11", 1), ("target13", 1)):
        assert not post_id8_context_recreate_guard(dict(state, **{key: value})), key

def full_control_guard(state):
    return (state["history_completed"] == state["history_outputs_validated"] == 10 and state["history_exact"] and
        state["initial_kind"] == "FULL_RUNTIME_BUNDLE" and state["initial_order"] < state["id8_submit_order"] and
        state["initial_context_generation"] == state["initial_kernel_generation"] == state["initial_bo_generation"] == 1 and
        state["initial_context_recreate_count"] == state["initial_kernel_recreate_count"] == state["initial_bo_recreate_count"] == 1 and
        state["experimental_submissions_before_id8"] == 0 and state["id8_source"] == 8 and state["id8_M"] == 9216 and state["id8_payload_matches"] and
        state["id8_submit_order"] < state["id8_exact_order"] < state["second_order"] < state["terminal_order"] and state["id8_exact"] and
        state["id8_experiment_ordinal"] == state["id8_context_ordinal"] == 1 and
        state["second_kind"] == "CONTEXT_KERNEL_ONLY_AFTER_ID8" and
        state["context_generation"] == state["kernel_generation"] == 2 and state["bo_generation"] == 1 and
        state["context_recreate_count"] == state["kernel_recreate_count"] == 2 and state["bo_recreate_count"] == 1 and
        state["same_device"] and state["same_xclbin"] and state["same_bos"] and state["same_maps"] and state["same_capacities"] and
        state["no_post_recreate_submission"] and state["post_recreate_submissions"] == state["post_recreate_runs"] == state["post_recreate_waits"] == 0)


def test_full_control_model():
    state = dict(history_completed=10, history_outputs_validated=10, history_exact=True, initial_kind="FULL_RUNTIME_BUNDLE",
        initial_order=11, initial_context_generation=1, initial_kernel_generation=1, initial_bo_generation=1,
        initial_context_recreate_count=1, initial_kernel_recreate_count=1, initial_bo_recreate_count=1, experimental_submissions_before_id8=0,
        id8_source=8, id8_M=9216, id8_payload_matches=True, id8_submit_order=12, id8_exact_order=13, id8_exact=True,
        id8_experiment_ordinal=1, id8_context_ordinal=1, second_kind="CONTEXT_KERNEL_ONLY_AFTER_ID8", second_order=14, terminal_order=15,
        context_generation=2, kernel_generation=2, bo_generation=1, context_recreate_count=2, kernel_recreate_count=2, bo_recreate_count=1,
        same_device=True, same_xclbin=True, same_bos=True, same_maps=True, same_capacities=True, no_post_recreate_submission=True,
        post_recreate_submissions=0, post_recreate_runs=0, post_recreate_waits=0)
    assert full_control_guard(state)
    failures = (("history_outputs_validated", 9), ("history_exact", False), ("initial_kind", "NONE"),
        ("initial_order", 13), ("initial_kind", "CONTEXT_KERNEL_ONLY"), ("experimental_submissions_before_id8", 1),
        ("id8_source", 7), ("id8_M", 4096), ("id8_payload_matches", False), ("id8_exact", False),
        ("id8_experiment_ordinal", 2), ("id8_context_ordinal", 2), ("second_kind", "MISSING"),
        ("second_kind", "FULL_RUNTIME_BUNDLE"), ("bo_generation", 2), ("same_bos", False),
        ("same_maps", False), ("same_capacities", False), ("no_post_recreate_submission", False),
        ("post_recreate_submissions", 1), ("post_recreate_runs", 1), ("post_recreate_waits", 1))
    for key, value in failures:
        assert not full_control_guard(dict(state, **{key: value})), (key, value)


def second_recreate_no_execution_guard(state):
    return (state["history_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8] and
        state["history_completed"] == state["history_outputs_validated"] == 10 and state["history_exact"] and
        state["initial_full_bundle"] and state["initial_order"] < state["second_order"] and
        state["initial_context_generation"] == state["initial_kernel_generation"] == state["initial_bo_generation"] == 1 and
        state["initial_context_recreate_count"] == state["initial_kernel_recreate_count"] == state["initial_bo_recreate_count"] == 1 and
        state["generation1_execution_count"] == state["generation1_submissions"] == state["generation1_waits"] == state["generation1_output_syncs"] == 0 and
        state["previous_experimental_source"] == "NONE" and not state["id7_experiment"] and not state["id8_experiment"] and not state["probe11"] and not state["target13"] and
        state["same_device"] and state["same_xclbin"] and state["same_xclbin_uuid"] and
        all(state[key] for key in ("same_instr_bo", "same_q4_bo", "same_q8_bo", "same_output_bo", "same_instr_map", "same_q4_map", "same_q8_map", "same_output_map", "same_capacities")) and
        state["second_helper"] == "CONTEXT_KERNEL_ONLY" and
        state["context_generation"] == state["kernel_generation"] == 2 and state["bo_generation"] == 1 and
        state["context_recreate_count"] == state["kernel_recreate_count"] == 2 and state["bo_recreate_count"] == 1 and
        all(state[key] == 0 for key in ("bo_create_delta", "map_delta", "sync_delta", "between_submissions", "between_runs", "between_waits", "post_second_submissions", "post_second_runs", "post_second_waits")))

def test_second_recreate_no_execution_model():
    state = dict(history_ids=[1, 2, 3, 4, 5, 6, 7, 8, 8, 8], history_completed=10, history_outputs_validated=10, history_exact=True,
        initial_full_bundle=True, initial_order=10, second_order=11, initial_context_generation=1, initial_kernel_generation=1, initial_bo_generation=1,
        initial_context_recreate_count=1, initial_kernel_recreate_count=1, initial_bo_recreate_count=1, generation1_execution_count=0,
        generation1_submissions=0, generation1_waits=0, generation1_output_syncs=0, previous_experimental_source="NONE",
        id7_experiment=False, id8_experiment=False, probe11=False, target13=False, same_device=True, same_xclbin=True, same_xclbin_uuid=True,
        same_instr_bo=True, same_q4_bo=True, same_q8_bo=True, same_output_bo=True, same_instr_map=True, same_q4_map=True, same_q8_map=True,
        same_output_map=True, same_capacities=True, second_helper="CONTEXT_KERNEL_ONLY", context_generation=2, kernel_generation=2, bo_generation=1,
        context_recreate_count=2, kernel_recreate_count=2, bo_recreate_count=1, bo_create_delta=0, map_delta=0, sync_delta=0,
        between_submissions=0, between_runs=0, between_waits=0, post_second_submissions=0, post_second_runs=0, post_second_waits=0)
    assert second_recreate_no_execution_guard(state)
    print("SECOND-RECREATE-NO-EXECUTION MODEL PASS")
    failures = (
        ("history_completed", 9), ("history_exact", False), ("initial_full_bundle", False), ("initial_order", 12),
        ("initial_context_generation", 0), ("initial_kernel_generation", 0), ("initial_bo_generation", 0),
        ("initial_context_recreate_count", 0), ("initial_kernel_recreate_count", 0), ("initial_bo_recreate_count", 0),
        ("generation1_submissions", 1), ("generation1_waits", 1), ("generation1_output_syncs", 1),
        ("id7_experiment", True), ("id8_experiment", True), ("probe11", True), ("target13", True),
        ("same_device", False), ("same_xclbin", False), ("same_xclbin_uuid", False),
        ("same_instr_bo", False), ("same_q4_bo", False), ("same_q8_bo", False), ("same_output_bo", False),
        ("same_instr_map", False), ("same_q4_map", False), ("same_q8_map", False), ("same_output_map", False), ("same_capacities", False),
        ("second_helper", "FULL_RUNTIME_BUNDLE"), ("bo_generation", 2), ("bo_create_delta", 1), ("map_delta", 1), ("sync_delta", 1),
        ("between_submissions", 1), ("between_runs", 1), ("between_waits", 1),
        ("post_second_submissions", 1), ("post_second_runs", 1), ("post_second_waits", 1), ("history_outputs_validated", 9),
        ("history_ids", [1, 2, 3, 4, 5, 6, 7, 8, 8]), ("previous_experimental_source", "ID8"),
        ("second_order", 9), ("context_generation", 1), ("kernel_generation", 1), ("context_recreate_count", 1),
        ("kernel_recreate_count", 1), ("bo_recreate_count", 2), ("generation1_execution_count", 1))
    for key, value in failures:
        assert not second_recreate_no_execution_guard(dict(state, **{key: value})), (key, value)
def program_only_second_recreate_guard(state):
    return (state["history_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8] and
        state["history_completed"] == state["history_outputs_validated"] == 10 and state["history_exact"] and
        state["initial_full_bundle"] and state["initial_order"] < state["program_only_order"] < state["second_order"] and
        state["initial_context_generation"] == state["initial_kernel_generation"] == state["initial_bo_generation"] == 1 and
        state["program_only_source_id"] == 7 and state["program_only_M"] == 4096 and state["frozen_instruction_identity"] and
        state["instruction_bo_creates"] == state["instruction_bo_maps"] == 1 and state["instruction_writes"] == 2 and
        state["instruction_to_device_syncs"] == 2 and state["q4_syncs"] == state["q8_syncs"] == state["output_from_device_syncs"] == 0 and
        state["generation1_kernel_submissions"] == state["generation1_runs"] == state["generation1_waits"] == state["generation1_output_syncs"] == 0 and
        state["second_helper"] == "CONTEXT_KERNEL_ONLY" and state["context_generation"] == state["kernel_generation"] == 2 and
        state["bo_generation"] == 1 and state["context_recreate_count"] == state["kernel_recreate_count"] == 2 and state["bo_recreate_count"] == 1 and
        state["same_device"] and state["same_xclbin"] and state["same_m9216_bos_maps_capacities"] and
        state["post_second_submissions"] == state["post_second_runs"] == state["post_second_waits"] == 0 and
        state["probe11"] == state["target13"] == 0)


def test_program_only_second_recreate_model():
    state = dict(history_ids=[1, 2, 3, 4, 5, 6, 7, 8, 8, 8], history_completed=10, history_outputs_validated=10, history_exact=True,
        initial_full_bundle=True, initial_order=10, program_only_order=11, second_order=12,
        initial_context_generation=1, initial_kernel_generation=1, initial_bo_generation=1,
        program_only_source_id=7, program_only_M=4096, frozen_instruction_identity=True,
        instruction_bo_creates=1, instruction_bo_maps=1, instruction_writes=2, instruction_to_device_syncs=2,
        q4_syncs=0, q8_syncs=0, output_from_device_syncs=0, generation1_kernel_submissions=0, generation1_runs=0,
        generation1_waits=0, generation1_output_syncs=0, second_helper="CONTEXT_KERNEL_ONLY",
        context_generation=2, kernel_generation=2, bo_generation=1, context_recreate_count=2, kernel_recreate_count=2, bo_recreate_count=1,
        same_device=True, same_xclbin=True, same_m9216_bos_maps_capacities=True,
        post_second_submissions=0, post_second_runs=0, post_second_waits=0, probe11=0, target13=0)
    assert program_only_second_recreate_guard(state)
    failures = (("history_completed", 9), ("history_exact", False), ("initial_full_bundle", False), ("program_only_order", 9),
        ("program_only_source_id", 8), ("program_only_M", 9216), ("frozen_instruction_identity", False),
        ("instruction_bo_creates", 0), ("instruction_bo_maps", 0), ("instruction_writes", 1),
        ("instruction_to_device_syncs", 1), ("q4_syncs", 1), ("q8_syncs", 1), ("output_from_device_syncs", 1),
        ("generation1_kernel_submissions", 1), ("generation1_runs", 1), ("generation1_waits", 1), ("generation1_output_syncs", 1),
        ("second_helper", "FULL_RUNTIME_BUNDLE"), ("context_generation", 1), ("kernel_generation", 1), ("bo_generation", 2),
        ("context_recreate_count", 1), ("kernel_recreate_count", 1), ("bo_recreate_count", 2),
        ("same_device", False), ("same_xclbin", False), ("same_m9216_bos_maps_capacities", False),
        ("post_second_submissions", 1), ("post_second_runs", 1), ("post_second_waits", 1), ("probe11", 1), ("target13", 1))
    for key, value in failures:
        assert not program_only_second_recreate_guard(dict(state, **{key: value})), key

def execute_wait_no_readback_second_recreate_guard(state):
    return (state["history_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8] and
        state["history_completed"] == state["history_outputs_validated"] == 10 and state["history_exact"] and
        state["initial_full_bundle"] and state["initial_order"] < state["m4096_order"] < state["second_order"] and
        state["m4096_source_id"] == 7 and state["m4096_M"] == 4096 and state["frozen_instruction_identity"] and
        state["m4096_submissions"] == state["m4096_runs"] == state["m4096_waits"] == 1 and
        state["m4096_output_syncs"] == state["m4096_output_sentinel_syncs"] == state["m4096_output_reads"] == state["m4096_output_validations"] == 0 and
        state["generation1_submissions"] == state["generation1_runs"] == state["generation1_waits"] == 1 and
        state["generation1_output_sentinel_syncs"] == state["generation1_output_syncs"] == state["generation1_output_reads"] == state["generation1_output_validations"] == 0 and
        state["second_helper"] == "CONTEXT_KERNEL_ONLY" and
        state["context_generation"] == state["kernel_generation"] == 2 and state["bo_generation"] == 1 and
        state["context_recreate_count"] == state["kernel_recreate_count"] == 2 and state["bo_recreate_count"] == 1 and
        state["same_device"] and state["same_xclbin"] and state["same_xclbin_uuid"] and
        all(state[key] for key in ("same_instr_bo", "same_q4_bo", "same_q8_bo", "same_output_bo",
            "same_instr_map", "same_q4_map", "same_q8_map", "same_output_map", "same_capacities")) and
        state["temp_resources_separate"] and state["post_second_submissions"] == state["post_second_runs"] == state["post_second_waits"] == 0 and
        state["probe11"] == state["target13"] == 0)


def test_execute_wait_no_readback_second_recreate_model():
    state = dict(history_ids=[1, 2, 3, 4, 5, 6, 7, 8, 8, 8], history_completed=10, history_outputs_validated=10, history_exact=True,
        initial_full_bundle=True, initial_order=10, m4096_order=11, second_order=12, m4096_source_id=7, m4096_M=4096,
        frozen_instruction_identity=True, m4096_submissions=1, m4096_runs=1, m4096_waits=1, m4096_output_syncs=0, m4096_output_sentinel_syncs=0,
        m4096_output_reads=0, m4096_output_validations=0, generation1_submissions=1, generation1_runs=1, generation1_waits=1,
        generation1_output_sentinel_syncs=0, generation1_output_syncs=0, generation1_output_reads=0, generation1_output_validations=0, second_helper="CONTEXT_KERNEL_ONLY",
        context_generation=2, kernel_generation=2, bo_generation=1, context_recreate_count=2, kernel_recreate_count=2, bo_recreate_count=1,
        same_device=True, same_xclbin=True, same_xclbin_uuid=True, same_instr_bo=True, same_q4_bo=True, same_q8_bo=True,
        same_output_bo=True, same_instr_map=True, same_q4_map=True, same_q8_map=True, same_output_map=True, same_capacities=True,
        temp_resources_separate=True, post_second_submissions=0, post_second_runs=0, post_second_waits=0, probe11=0, target13=0)
    assert execute_wait_no_readback_second_recreate_guard(state)
    failures = (("history_completed", 9), ("history_exact", False), ("initial_full_bundle", False), ("m4096_order", 9),
        ("m4096_source_id", 8), ("m4096_M", 9216), ("frozen_instruction_identity", False),
        ("m4096_submissions", 0), ("m4096_runs", 0), ("m4096_waits", 0),
        ("m4096_output_syncs", 1), ("m4096_output_sentinel_syncs", 1), ("m4096_output_reads", 1), ("m4096_output_validations", 1),
        ("generation1_submissions", 0), ("generation1_runs", 0), ("generation1_waits", 0),
        ("generation1_output_sentinel_syncs", 1), ("generation1_output_syncs", 1), ("generation1_output_reads", 1), ("generation1_output_validations", 1),
        ("second_helper", "FULL_RUNTIME_BUNDLE"), ("context_generation", 1), ("kernel_generation", 1),
        ("bo_generation", 2), ("context_recreate_count", 1), ("kernel_recreate_count", 1), ("bo_recreate_count", 2),
        ("same_device", False), ("same_xclbin", False), ("same_xclbin_uuid", False), ("same_instr_bo", False),
        ("same_q4_bo", False), ("same_q8_bo", False), ("same_output_bo", False), ("same_instr_map", False),
        ("same_q4_map", False), ("same_q8_map", False), ("same_output_map", False), ("same_capacities", False),
        ("temp_resources_separate", False), ("post_second_submissions", 1), ("post_second_runs", 1),
        ("post_second_waits", 1), ("probe11", 1), ("target13", 1))
    for key, value in failures:
        assert not execute_wait_no_readback_second_recreate_guard(dict(state, **{key: value})), (key, value)
    print("EXECUTE-WAIT-NO-READBACK SECOND-RECREATE MODEL PASS")


def temp_resource_release_guard(state):
    return (state["history_completed"] == state["history_validated"] == 10 and state["history_sentinel_syncs"] == 10 and
        state["run_alive_before"] and all(state[k] for k in ("instr_bo_alive_before", "q4_bo_alive_before", "q8_bo_alive_before", "out_bo_alive_before", "maps_alive_before")) and
        state["wait_completed"] and not state["readback"] and not state["sentinel_syncs"] and not state["from_device_syncs"] and not state["reads"] and not state["hashes"] and
        not state["run_alive_after"] and all(not state[k] for k in ("instr_bo_alive_after", "q4_bo_alive_after", "q8_bo_alive_after", "out_bo_alive_after", "maps_alive_after")) and
        state["retained_bos_same"] and state["retained_maps_same"] and state["retained_capacities_same"] and
        state["device_same"] and state["xclbin_same"] and state["no_intervening_xrt_ops"] and
        state["submissions"] == state["runs"] == state["waits"] == 1 and state["second_helper"] == "ggml_backend_xdna_replay_recreate_m9216_hw_context" and
        (state["context_generation"], state["kernel_generation"], state["bo_generation"]) == (2, 2, 1) and
        not state["post_compute"] and state["terminal_success"] and not state["terminal_fallthrough"] and state["en1r_unchanged"] and
        not state["already_dead_falsely_released"])


def test_temp_resource_release_model_and_source_order():
    state = dict(history_completed=10, history_validated=10, history_sentinel_syncs=10, run_alive_before=True,
        instr_bo_alive_before=True, q4_bo_alive_before=True, q8_bo_alive_before=True, out_bo_alive_before=True, maps_alive_before=True,
        wait_completed=True, readback=False, sentinel_syncs=0, from_device_syncs=0, reads=0, hashes=0, run_alive_after=False,
        instr_bo_alive_after=False, q4_bo_alive_after=False, q8_bo_alive_after=False, out_bo_alive_after=False, maps_alive_after=False,
        retained_bos_same=True, retained_maps_same=True, retained_capacities_same=True, device_same=True, xclbin_same=True,
        no_intervening_xrt_ops=True, submissions=1, runs=1, waits=1, second_helper="ggml_backend_xdna_replay_recreate_m9216_hw_context",
        context_generation=2, kernel_generation=2, bo_generation=1, post_compute=False, terminal_success=True, terminal_fallthrough=False,
        en1r_unchanged=True, already_dead_falsely_released=False)
    assert temp_resource_release_guard(state)
    for key, value in (("history_completed", 9), ("history_sentinel_syncs", 9), ("wait_completed", False), ("readback", True),
        ("sentinel_syncs", 1), ("from_device_syncs", 1), ("reads", 1), ("hashes", 1), ("run_alive_after", True),
        ("instr_bo_alive_after", True), ("q4_bo_alive_after", True), ("q8_bo_alive_after", True), ("out_bo_alive_after", True),
        ("maps_alive_after", True), ("already_dead_falsely_released", True), ("retained_bos_same", False), ("retained_maps_same", False),
        ("retained_capacities_same", False), ("device_same", False), ("xclbin_same", False), ("no_intervening_xrt_ops", False),
        ("submissions", 2), ("runs", 2), ("waits", 2), ("second_helper", "wrong"), ("bo_generation", 2),
        ("post_compute", True), ("terminal_success", False), ("terminal_fallthrough", True), ("en1r_unchanged", False)):
        assert not temp_resource_release_guard(dict(state, **{key: value})), key
    backend_dir = REPO / "ggml" / "src" / "ggml-xdna"
    source = (backend_dir / "ggml-xdna.cpp").read_text() + "\n" + (backend_dir / "ggml-xdna-diagnostics.cpp").read_text()
    wait = source.index("const auto completion = run->wait()")
    run_before = source.index("BEFORE_COMPLETED_RUN_RELEASE")
    run_release = source.index("run = std::nullopt;", run_before)
    run_after = source.index("AFTER_COMPLETED_RUN_RELEASE", run_release)
    bos_before = source.index("BEFORE_TEMP_M4096_RESOURCE_RELEASE", run_after)
    bo_reset = source.index("transition_bundle.bo_instr.reset()", bos_before)
    bos_after = source.index("AFTER_TEMP_M4096_RESOURCE_RELEASE", bo_reset)
    helper_call = source.index("ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, false, true, &run, allow_live_run_control);", bos_after)
    assert wait < run_before < run_release < run_after < bos_before < bo_reset < bos_after < helper_call
    post_wait = source.index('log_runtime_snapshot("POST_WAIT_NO_READBACK", &*run)')
    assert source.index("if (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode)", post_wait) < run_before
    assert not any(f"ctx->q4_m9216_bo_{name}.reset()" in source[bos_before:helper_call] for name in ("instr", "q4", "q8", "out"))
    assert not any(token in source[run_before:helper_call] for token in ("(*kernel)(", "run.emplace(", "run->wait(", ".sync(", "register_xclbin(", "load_xclbin(", "make_unique<xrt::hw_context>", "make_shared<xrt::hw_context>", "make_unique<xrt::kernel>"))
    release_block = source[bos_before:helper_call]
    assert not any(token in release_block for token in ("(*kernel)(", "run.emplace(", "run->wait(", ".sync(", "register_xclbin(", "load_xclbin(", "make_unique<xrt::hw_context>", "make_shared<xrt::hw_context>", "make_unique<xrt::kernel>"))
    helper = source.split("static void ggml_backend_xdna_replay_recreate_m9216_hw_context(", 1)[1].split("\n}", 1)[0]
    lifecycle = ("RELEASE_OLD_KERNEL kernel_id=", "OLD_KERNEL_RELEASED", "RELEASE_OLD_CONTEXT context_id=", "OLD_CONTEXT_RELEASED",
        "BEFORE_NEW_CONTEXT_CONSTRUCT", "AFTER_NEW_CONTEXT_CONSTRUCT", "BEFORE_NEW_KERNEL_CONSTRUCT", "AFTER_NEW_KERNEL_CONSTRUCT",
        "BEFORE_POST_BOUNDARY_INSPECTION", "AFTER_POST_BOUNDARY_INSPECTION")
    assert [helper.index(item) for item in lifecycle] == sorted(helper.index(item) for item in lifecycle)
    assert "ep1r_retained_capacity_out == ctx->q4_m9216_bo_out->size()" in release_block
    assert "retained_m9216_output_alive=1" in release_block
    assert "m4096-execute-wait-release-temp-context-recreate" in source
    assert "m4096-execute-wait-no-readback-context-recreate" in source  # EN1R remains distinct
    print("POST-WAIT RUNTIME-OBJECT RELEASE MODEL PASS")


def historical_output_validation_model(expected_size, expected_sha, actual_bytes, output_capacity):
    actual_size = len(actual_bytes)
    safe_to_hash = actual_size <= output_capacity
    actual_sha = hashlib.sha256(actual_bytes).hexdigest() if safe_to_hash else None
    matches = safe_to_hash and actual_size == expected_size and actual_sha == expected_sha
    return {
        "mismatch_marker": not matches,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "expected_sha": expected_sha,
        "actual_sha": actual_sha or "NOT_COMPUTED",
        "reason": None if safe_to_hash else "actual_validation_bytes_exceed_output_capacity",
        "fail_closed": not matches,
        "experimental_role_reached": False if not matches else None,
        "outputs_validated_increment": int(matches),
    }


def output_sentinel_sync_required(no_readback_mode, role):
    return not (no_readback_mode and role == "M4096_EXECUTE_AND_WAIT_NO_READBACK")


def test_historical_output_mismatch_models():
    historical_roles = ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2
    assert all(output_sentinel_sync_required(True, role) for role in historical_roles)
    assert not output_sentinel_sync_required(True, "M4096_EXECUTE_AND_WAIT_NO_READBACK")
    assert output_sentinel_sync_required(False, "M4096_EXECUTE_AND_WAIT_NO_READBACK")
    expected = b"A" * 16384
    expected_sha = hashlib.sha256(expected).hexdigest()
    positive = historical_output_validation_model(16384, expected_sha, expected, 16384)
    assert not positive["mismatch_marker"] and not positive["fail_closed"] and positive["outputs_validated_increment"] == 1
    cases = (
        ("correct-size-wrong-sha", 16384, "0" * 64, expected, 16384, True),
        ("wrong-size-sha-calculable", 16384, expected_sha, b"B" * 16380, 16384, True),
        ("wrong-size-sha-unavailable", 16384, expected_sha, b"C" * 16385, 16384, False),
        ("empty-output", 16384, expected_sha, b"", 16384, True),
        ("truncated-output", 16384, expected_sha, b"D" * 8192, 16384, True),
        ("unexpected-larger-output", 16384, expected_sha, b"E" * 16388, 16388, True),
    )
    for name, size, sha, actual, capacity, hashable in cases:
        result = historical_output_validation_model(size, sha, actual, capacity)
        assert result["mismatch_marker"] and result["fail_closed"] and result["experimental_role_reached"] is False, name
        assert result["expected_size"] == size and result["actual_size"] == len(actual), name
        assert result["expected_sha"] == sha, name
        if hashable: assert result["actual_sha"] == hashlib.sha256(actual).hexdigest(), name
        else: assert result["actual_sha"] == "NOT_COMPUTED", name
        assert (result["actual_sha"] != "NOT_COMPUTED") == hashable, name
        assert result["outputs_validated_increment"] == 0, name
        if not hashable: assert result["reason"] == "actual_validation_bytes_exceed_output_capacity", name
    print("HISTORICAL OUTPUT POSITIVE/NEGATIVE MODELS PASS")


def diagnostic_terminal_success(mode, state):
    return (mode in ("double-recreate-no-execution", "m4096-program-only-context-recreate",
                    "id8-context-recreate-only", "m4096-context-recreate-to-id8", "m4096-execute-wait-no-readback-context-recreate",
                    "m4096-execute-wait-release-temp-context-recreate", "m4096-execute-wait-release-run-context-recreate") and
            state["invariants_passed"] and state["cleanup_complete"] and not state["replay_failed"] and state["probe_submitted"] == 0)

def terminal_success_model(state):
    valid = (state["history_completed"] == state["history_outputs_validated"] == 10 and state["history_exact"] and
        state["submissions"] == state["waits"] == state["records"] == 11 and state["outputs_validated"] == 10 and
        state["no_readback"] and state["run_released"] and state["temp_bos_alive"] and state["temp_maps_alive"] and
        state["retained_m9216_same"] and state["helper"] == "context-kernel-only" and
        state["context_ok"] and state["kernel_ok"] and state["inspection_ok"] and
        (state["context_generation"], state["kernel_generation"], state["bo_generation"]) == (2, 2, 1) and
        state["post_recreation_compute"] == 0 and not state["failure_latch"] and state["cleanup_complete"])
    return {"terminal_marker": valid, "backend_result": 0 if valid else 1,
            "driver_result": 0 if valid else 4, "process_result": 0 if valid else 4,
            "generic_failure_after_terminal": False}


def test_terminal_success_model():
    state = dict(history_completed=10, history_outputs_validated=10, history_exact=True, submissions=11, waits=11, records=11,
        outputs_validated=10, no_readback=True, run_released=True, temp_bos_alive=True, temp_maps_alive=True,
        retained_m9216_same=True, helper="context-kernel-only", context_ok=True, kernel_ok=True, inspection_ok=True,
        context_generation=2, kernel_generation=2, bo_generation=1, post_recreation_compute=0,
        failure_latch=False, cleanup_complete=True)
    result = terminal_success_model(state)
    assert result == {"terminal_marker": True, "backend_result": 0, "driver_result": 0,
                      "process_result": 0, "generic_failure_after_terminal": False}
    print("TERMINAL SUCCESS MODEL PASS")
    failures = (("history_completed", 9), ("history_outputs_validated", 9), ("history_exact", False), ("waits", 10),
        ("outputs_validated", 11), ("no_readback", False), ("run_released", False),
        ("temp_bos_alive", False), ("temp_maps_alive", False), ("retained_m9216_same", False),
        ("helper", "wrong"), ("context_ok", False), ("kernel_ok", False), ("inspection_ok", False),
        ("context_generation", 1), ("kernel_generation", 1), ("bo_generation", 2),
        ("post_recreation_compute", 1), ("failure_latch", True), ("cleanup_complete", False))
    for key, value in failures:
        failed = terminal_success_model(dict(state, **{key: value}))
        assert not failed["terminal_marker"] and failed["backend_result"] != 0, key


def normal_probe_terminal_success(state):
    return (state["completed"] == state["outputs_validated"] == state["planned"] and
            state["probe_submitted"] == state["probe_waited"] == 1 and state["probe_output_validated"] == 1)


def generic_replay_telemetry_guard(state, mode):
    expected_outputs = 10 if mode in ("m4096-execute-wait-release-temp-context-recreate", "m4096-execute-wait-no-readback-context-recreate", "m4096-execute-wait-release-run-context-recreate") else state["historical_records"]
    return (state["submitted"] == state["records_total"] and state["waited"] == state["records_total"] and
            state["outputs_validated"] == expected_outputs)


def release_terminal_success(state):
    return (state["history_completed"] == state["history_validated"] == state["history_sentinel_syncs"] == 10 and
        state["history_exact"] and state["full_bundle"] and
        state["experimental_submissions"] == state["experimental_runs"] == state["experimental_waits"] == 1 and
        state["experimental_output_syncs"] == state["experimental_reads"] == state["experimental_validations"] == 0 and
        state["run_alive_before"] and not state["run_alive_after"] and
        all(state[f"{name}_bo_before"] for name in ("instr", "q4", "q8", "out")) and
        all(not state[f"{name}_bo_after"] for name in ("instr", "q4", "q8", "out")) and
        state["maps_before"] and not state["maps_after"] and state["retained_unchanged"] and
        state["release_guard"] and state["intervening_xrt_operations"] == 0 and
        state["new_context"] and state["new_kernel"] and state["inspection_pass"] and
        (state["context_generation"], state["kernel_generation"], state["bo_generation"]) == (2, 2, 1) and
        (state["context_recreates"], state["kernel_recreates"], state["bo_recreates"]) == (2, 2, 1) and
        state["post_recreation_compute"] == 0 and not state["replay_failed"] and
        state["terminal_marker_after_invariants"])


def test_release_terminal_fallthrough_models():
    release_temp_mode = "m4096-execute-wait-release-temp-context-recreate"
    telemetry = dict(submitted=11, waited=11, records_total=11, historical_records=11, outputs_validated=10)
    assert generic_replay_telemetry_guard(telemetry, release_temp_mode)
    no_readback_mode_name = "m4096-execute-wait-no-readback-context-recreate"
    assert generic_replay_telemetry_guard(telemetry, no_readback_mode_name)
    release_run_mode_name = "m4096-execute-wait-release-run-context-recreate"
    assert generic_replay_telemetry_guard(telemetry, release_run_mode_name)
    for key in ("submitted", "waited"):
        assert not generic_replay_telemetry_guard(dict(telemetry, **{key: 10}), release_temp_mode), key
    normal = dict(telemetry, records_total=10, historical_records=10, submitted=10, waited=10, outputs_validated=10)
    assert generic_replay_telemetry_guard(normal, "PREFIX_PROBE")

    state = dict(history_completed=10, history_validated=10, history_sentinel_syncs=10, history_exact=True, full_bundle=True,
        experimental_submissions=1, experimental_runs=1, experimental_waits=1, experimental_output_syncs=0,
        experimental_reads=0, experimental_validations=0, run_alive_before=True, run_alive_after=False,
        instr_bo_before=True, q4_bo_before=True, q8_bo_before=True, out_bo_before=True,
        instr_bo_after=False, q4_bo_after=False, q8_bo_after=False, out_bo_after=False, maps_before=True, maps_after=False,
        retained_unchanged=True, release_guard=True, intervening_xrt_operations=0, new_context=True, new_kernel=True,
        inspection_pass=True, context_generation=2, kernel_generation=2, bo_generation=1,
        context_recreates=2, kernel_recreates=2, bo_recreates=1, post_recreation_compute=0, replay_failed=False,
        terminal_marker_after_invariants=True)
    assert release_terminal_success(state)
    failures = (
        ("history_completed", 9), ("history_validated", 9), ("history_sentinel_syncs", 9), ("history_exact", False),
        ("full_bundle", False), ("experimental_waits", 0), ("experimental_output_syncs", 1),
        ("experimental_reads", 1), ("run_alive_after", True), ("instr_bo_after", True),
        ("q4_bo_after", True), ("q8_bo_after", True), ("out_bo_after", True), ("maps_after", True),
        ("retained_unchanged", False), ("release_guard", False), ("intervening_xrt_operations", 1),
        ("new_context", False), ("new_kernel", False), ("inspection_pass", False),
        ("context_generation", 1), ("kernel_generation", 1), ("bo_generation", 2),
        ("context_recreates", 1), ("kernel_recreates", 1), ("bo_recreates", 2),
        ("post_recreation_compute", 1), ("replay_failed", True), ("terminal_marker_after_invariants", False))
    for key, value in failures:
        assert not release_terminal_success(dict(state, **{key: value})), key
    print("DIAGNOSTIC TERMINAL SUCCESS MODEL PASS")

def resource_release_run_only_release_success(state):
    return (state["history_completed"] == state["history_outputs_validated"] == 10 and state["historical_outputs_validated"] == 10 and
            state["submitted"] == state["waited"] == state["records_total"] == 11 and state["validated_outputs"] == 10 and
            state["run_alive_before"] and not state["run_alive_after"] and all(state[f"{name}_bo_before"] and state[f"{name}_bo_after"] and state[f"{name}_bo_same"] and state[f"{name}_capacity_same"] for name in ("instr", "q4", "q8", "out")) and
            state["maps_before"] and state["maps_after"] and state["maps_same"] and state["retained_m9216_same"] and
            state["m4096_submissions"] == state["m4096_waits"] == 1 and state["m4096_output_syncs"] == state["m4096_output_reads"] == state["m4096_output_validations"] == 0 and
            state["extra_submissions"] == state["extra_waits"] == state["extra_to_device_syncs"] == state["extra_from_device_syncs"] == state["intervening_xrt_operations"] == 0 and
            state["context_generation"] == state["kernel_generation"] == 2 and state["bo_generation"] == 1 and
            state["context_recreates"] == state["kernel_recreates"] == 2 and state["bo_recreates"] == 1 and
            state["post_recreation_compute"] == 0 and not state["replay_failed"] and state["terminal_marker_after_invariants"])

def test_run_only_release_model_and_source_order():
    state = dict(history_completed=10, history_outputs_validated=10, historical_outputs_validated=10, submitted=11, waited=11, records_total=11, validated_outputs=10,
        run_alive_before=True, run_alive_after=False, maps_before=True, maps_after=True, maps_same=True, retained_m9216_same=True,
        m4096_submissions=1, m4096_waits=1, m4096_output_syncs=0, m4096_output_reads=0, m4096_output_validations=0,
        extra_submissions=0, extra_waits=0, extra_to_device_syncs=0, extra_from_device_syncs=0, intervening_xrt_operations=0,
        context_generation=2, kernel_generation=2, bo_generation=1, context_recreates=2, kernel_recreates=2, bo_recreates=1,
        post_recreation_compute=0, replay_failed=False, terminal_marker_after_invariants=True)
    for name in ("instr", "q4", "q8", "out"):
        state.update({f"{name}_bo_before": True, f"{name}_bo_after": True, f"{name}_bo_same": True, f"{name}_capacity_same": True})
    assert resource_release_run_only_release_success(state)
    for key, value in (("run_alive_after", True), ("q4_bo_after", False), ("out_capacity_same", False), ("maps_after", False), ("maps_same", False),
                       ("retained_m9216_same", False), ("m4096_output_reads", 1), ("extra_submissions", 1), ("extra_waits", 1),
                       ("extra_to_device_syncs", 1), ("extra_from_device_syncs", 1), ("intervening_xrt_operations", 1),
                       ("context_generation", 1), ("bo_generation", 2), ("terminal_marker_after_invariants", False)):
        assert not resource_release_run_only_release_success(dict(state, **{key: value})), key
    backend_dir = REPO / "ggml" / "src" / "ggml-xdna"
    source = (backend_dir / "ggml-xdna.cpp").read_text(encoding="utf-8") + "\n" + (backend_dir / "ggml-xdna-diagnostics.cpp").read_text(encoding="utf-8")
    mode = "post_recreate_execute_wait_release_run_context_recreate_mode"
    post_wait = source.index('log_runtime_snapshot("POST_WAIT_NO_READBACK", &*run)')
    start = source.index("else if (" + mode + ")", post_wait)
    release = source.index("run = std::nullopt;", start)
    guard = source.index("RUN_ONLY_RELEASE_GUARD=PASS", release)
    boundary = source.index("TRANSITION_STAGE=SECOND_CONTEXT_RECREATE_AFTER_EXECUTE_WAIT_NO_READBACK", guard)
    recreate = source.index("ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, false, true, &run, allow_live_run_control);", boundary)
    block = source[start:boundary]
    assert start < release < guard < boundary < recreate
    assert block.count("run = std::nullopt;") == 1
    forbidden = ("transition_bundle.bo_instr.reset", "transition_bundle.bo_q4.reset", "transition_bundle.bo_q8.reset", "transition_bundle.bo_out.reset",
                 "transition_bundle.buf_instr = transition_bundle.buf_q4", "transition_bundle.buf_q8 = transition_bundle.buf_out", "transition_bundle.buf_out = nullptr",
                 "sync_host_written_bo", "XCL_BO_SYNC", ".sync(", "run.emplace(", "run->wait(", "(*kernel)(",
                 "register_xclbin(", "load_xclbin(", "make_unique<xrt::hw_context>", "make_shared<xrt::hw_context>", "make_unique<xrt::kernel>",
                 "make_unique<xrt::bo>", "xrt::device(")
    assert not any(token in block for token in forbidden)
    assert "PREFIX_EXECUTE_WAIT_RELEASE_RUN_CONTEXT_RECREATE_CONTROL" in source
    assert "m4096-execute-wait-release-run-context-recreate" in source
    assert "RUN_ONLY_RELEASE_RECREATION_PRESERVATION=PASS" in source
    print("XDNA_RESOURCE_RELEASE RUN-ONLY RELEASE MODEL AND STATIC ORDER PASS")

def main():
    global CAPTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True, help="explicit external immutable CF1 capture directory")
    args = parser.parse_args()
    CAPTURE = args.capture_dir.resolve(strict=True)
    if not CAPTURE.is_dir():
        parser.error("--capture-dir must name a directory")
    test_context_recreate_lifecycle_model()
    test_second_recreate_no_execution_model()
    test_program_only_second_recreate_model()
    test_execute_wait_no_readback_second_recreate_model()
    test_temp_resource_release_model_and_source_order()
    test_historical_output_mismatch_models()
    test_full_control_model()
    test_post_id8_context_recreate_guard_model()
    test_context_kernel_recreate_to_id8_guard_model()
    test_transition_guard_model()
    test_post_recreation_control_guard_model()
    test_role_aware_transition_guard_models()
    test_same_shape_id8_guard_model()
    test_same_shape_id7_guard_model()
    build = Path(os.environ.get("GGML_XDNA_TEST_BUILD_DIR", REPO / "build-xdna"))
    exe = build / "bin" / "xdna-capture-replay.exe"
    if not exe.is_file() or not CAPTURE.is_dir():
        raise RuntimeError("build the replay driver and keep the immutable CF1 capture available")
    before = manifest_sha(CAPTURE)
    metadata_before = hashlib.sha256((CAPTURE / "run_metadata.json").read_bytes()).hexdigest()
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("GGML_XDNA_"):
            env.pop(key)

    valid = ((0, 11), (1, 11), (5, 11), (10, 11), (0, 13), (10, 13), (12, 13))
    for prefix, probe in valid:
        result = run(exe, env, ["--prefix-count", str(prefix), "--probe-id", str(probe)])
        assert result.returncode == 0, (prefix, probe, result.stdout, result.stderr)
        assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in result.stdout, (prefix, probe, result.stdout)
        assert "REPLAY_PLAN_VALID=YES" in result.stdout and "historical_completed=12" in result.stdout
        assert "target_submission_id=13" in result.stdout and "target_node=ffn_gate-4" in result.stdout
        assert "source_records=13" in result.stdout and f"prefix_count={prefix}" in result.stdout
        assert f"probe_id={probe}" in result.stdout and f"executions={prefix + 1}" in result.stdout
        assert "xrt_initialized=NO" in result.stdout
        selection = result.stdout.split("XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS", 1)[1]
        rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|PROBE_RECORD) \|$", selection, re.MULTILINE)
        expected_ids = list(range(1, prefix + 1)) + [probe]
        assert [int(row[1]) for row in rows] == expected_ids, (prefix, probe, rows)
        assert [row[4] for row in rows] == ["PREFIX_RECORD"] * prefix + ["PROBE_RECORD"], (prefix, probe, rows)

    injected = run(exe, env, ["--prefix-count", "8", "--inject-id", "10", "--probe-id", "11"])
    assert injected.returncode == 0, (injected.stdout, injected.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in injected.stdout and "xrt_initialized=NO" in injected.stdout
    assert "prefix_count=8" in injected.stdout and "inject_ids=10" in injected.stdout and "executions=10" in injected.stdout
    selection = injected.stdout.split("XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS", 1)[1]
    rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|PROBE_RECORD) \|$", selection, re.MULTILINE)
    assert [int(row[1]) for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8, 10, 11], rows
    assert [row[4] for row in rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD", "PROBE_RECORD"], rows
    reordered = run(exe, env, ["--prefix-count", "8", "--inject-id", "10", "--inject-id", "9", "--probe-id", "11"])
    assert reordered.returncode == 0, (reordered.stdout, reordered.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in reordered.stdout and "xrt_initialized=NO" in reordered.stdout
    assert "inject_ids=10,9" in reordered.stdout and "executions=11" in reordered.stdout
    selection = reordered.stdout.split("XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS", 1)[1]
    rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|PROBE_RECORD) \|$", selection, re.MULTILINE)
    assert [int(row[1]) for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11], rows
    assert [row[4] for row in rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["PROBE_RECORD"], rows
    duplicate = run(exe, env, ["--prefix-count", "8", "--inject-id", "10", "--inject-id", "10", "--probe-id", "11"])
    assert duplicate.returncode == 0, (duplicate.stdout, duplicate.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in duplicate.stdout and "xrt_initialized=NO" in duplicate.stdout
    assert "inject_ids=10,10" in duplicate.stdout and "executions=11" in duplicate.stdout
    selection = duplicate.stdout.split("XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS", 1)[1]
    rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|PROBE_RECORD) \|$", selection, re.MULTILINE)
    assert [int(row[1]) for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11], rows
    assert [row[4] for row in rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["PROBE_RECORD"], rows
    duplicate9 = run(exe, env, ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--probe-id", "11"])
    assert duplicate9.returncode == 0, (duplicate9.stdout, duplicate9.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in duplicate9.stdout and "xrt_initialized=NO" in duplicate9.stdout
    assert "inject_ids=9,9" in duplicate9.stdout and "executions=11" in duplicate9.stdout
    selection = duplicate9.stdout.split("XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS", 1)[1]
    rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|PROBE_RECORD) \|$", selection, re.MULTILINE)
    assert [int(row[1]) for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11], rows
    assert [row[4] for row in rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["PROBE_RECORD"], rows
    duplicate8 = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11"])
    assert duplicate8.returncode == 0, (duplicate8.stdout, duplicate8.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in duplicate8.stdout and "xrt_initialized=NO" in duplicate8.stdout
    assert "inject_ids=8,8" in duplicate8.stdout and "executions=11" in duplicate8.stdout
    selection = duplicate8.stdout.split("XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS", 1)[1]
    rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|PROBE_RECORD) \|$", selection, re.MULTILINE)
    assert [int(row[1]) for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11], rows
    assert [row[4] for row in rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["PROBE_RECORD"], rows
    recreated = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-bo-bundle-before-probe"])
    assert recreated.returncode == 0, (recreated.stdout, recreated.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in recreated.stdout and "xrt_initialized=NO" in recreated.stdout
    expected_sequence = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_BO_BUNDLE>\n11\nreplay_sequence_end"
    assert expected_sequence in recreated.stdout, recreated.stdout
    kernel_recreated = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-kernel-before-probe"])
    assert kernel_recreated.returncode == 0, (kernel_recreated.stdout, kernel_recreated.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in kernel_recreated.stdout and "xrt_initialized=NO" in kernel_recreated.stdout
    expected_kernel_sequence = expected_sequence.replace("<RECREATE_M9216_BO_BUNDLE>", "<RECREATE_M9216_KERNEL>")
    assert expected_kernel_sequence in kernel_recreated.stdout, kernel_recreated.stdout
    assert "<RECREATE_M9216_BO_BUNDLE>" not in kernel_recreated.stdout
    assert "<RECREATE_M9216_KERNEL>" in kernel_recreated.stdout
    context_recreated = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-hw-context-before-probe"])
    assert context_recreated.returncode == 0, (context_recreated.stdout, context_recreated.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in context_recreated.stdout and "xrt_initialized=NO" in context_recreated.stdout
    expected_context_sequence = expected_sequence.replace("<RECREATE_M9216_BO_BUNDLE>", "<RECREATE_M9216_HW_CONTEXT>")
    assert expected_context_sequence in context_recreated.stdout, context_recreated.stdout
    assert "<RECREATE_M9216_KERNEL>" not in context_recreated.stdout and "<RECREATE_M9216_BO_BUNDLE>" not in context_recreated.stdout
    post_id8_recreate = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "id8-context-recreate-only"])
    assert post_id8_recreate.returncode == 0, (post_id8_recreate.stdout, post_id8_recreate.stderr)
    assert "XDNA_CAPTURE_REPLAY_POST_ID8_CONTEXT_RECREATE_ONLY_DRY_RUN=PASS" in post_id8_recreate.stdout and "xrt_initialized=NO" in post_id8_recreate.stdout
    assert "executions=11" in post_id8_recreate.stdout and "post_recreate_transition_control=id8-context-recreate-only" in post_id8_recreate.stdout
    assert "| 11 | 8 | ffn_gate-2 | 9216 | POST_RECREATE_ID8_FIRST |" in post_id8_recreate.stdout
    expected_post_id8_sequence = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n8\n<POST_RECREATE_ID8_FIRST>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_ID8>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end"
    assert expected_post_id8_sequence in post_id8_recreate.stdout
    no_execution = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "double-recreate-no-execution"])
    assert no_execution.returncode == 0, (no_execution.stdout, no_execution.stderr)
    assert "XDNA_CAPTURE_REPLAY_SECOND_RECREATE_NO_EXECUTION_DRY_RUN=PASS" in no_execution.stdout and "executions=10" in no_execution.stdout and "xrt_initialized=NO" in no_execution.stdout
    assert "TRANSITION_STAGE=SECOND_CONTEXT_RECREATE_NO_EXECUTION generation1_execution_count=0 previous_experimental_source=NONE second_recreation_requested=YES" in no_execution.stdout and "NO_EXECUTION_RECREATE_GUARD=PASS" in no_execution.stdout
    expected_no_execution = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_NO_EXECUTION>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end"
    assert expected_no_execution in no_execution.stdout, no_execution.stdout
    assert all(token not in no_execution.stdout for token in ("POST_RECREATE_ID8_FIRST", "POST_CONTEXT_RECREATE_ID8_M9216", "PROBE_RECORD", "probe_id=", "target_submission_id=13"))

    execute_wait_no_readback_recreate = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
        "--post-recreate-transition-control", "m4096-execute-wait-no-readback-context-recreate"])
    assert execute_wait_no_readback_recreate.returncode == 0, (execute_wait_no_readback_recreate.stdout, execute_wait_no_readback_recreate.stderr)
    out = execute_wait_no_readback_recreate.stdout
    assert "XDNA_CAPTURE_REPLAY_EXECUTE_WAIT_NO_READBACK_SECOND_RECREATE_DRY_RUN=PASS" in out and "executions=11" in out and "xrt_initialized=NO" in out
    assert "post_recreate_transition_control=m4096-execute-wait-no-readback-context-recreate" in out
    assert "| 11 | 7 | z-2 | 4096 | M4096_EXECUTE_AND_WAIT_NO_READBACK |" in out
    assert "probe_id=" not in out and "target_submission_id=13" not in out and "PROBE_RECORD" not in out and "POST_M4096_EXEC_NO_READBACK_ID8_M9216" not in out
    expected_execute_wait_no_readback_recreate = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n7\n\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<EXECUTE_WAIT_NO_READBACK_ID7_M4096>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_EXECUTION>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end"
    assert expected_execute_wait_no_readback_recreate in out, out
    release_mode = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
        "--post-recreate-transition-control", "m4096-execute-wait-release-temp-context-recreate"])
    assert release_mode.returncode == 0, (release_mode.stdout, release_mode.stderr)
    assert "XDNA_CAPTURE_REPLAY_EXECUTE_WAIT_RELEASE_TEMP_SECOND_RECREATE_DRY_RUN=PASS" in release_mode.stdout
    assert "post_recreate_transition_control=m4096-execute-wait-release-temp-context-recreate" in release_mode.stdout
    expected_release = expected_execute_wait_no_readback_recreate.replace("<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_EXECUTION>",
        "<RELEASE_COMPLETED_EXPERIMENT_RUN_IF_STILL_ALIVE>\n<RELEASE_TEMP_M4096_EXECUTION_RESOURCES_IF_STILL_ALIVE>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_EXECUTION_RELEASE>")
    assert expected_release in release_mode.stdout
    old_name = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
        "--post-recreate-transition-control", "m4096-execute-wait-no-readback-release-temp-context-recreate"])
    assert old_name.returncode != 0 and "DRY_RUN=PASS" not in old_name.stdout
    rejected_execute_wait = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11",
        "--post-recreate-transition-control", "m4096-execute-wait-no-readback-context-recreate"])
    assert rejected_execute_wait.returncode != 0 and "DRY_RUN=PASS" not in rejected_execute_wait.stdout
    rejected_execute_wait = run(exe, env, ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9",
        "--post-recreate-transition-control", "m4096-execute-wait-no-readback-context-recreate"])
    assert rejected_execute_wait.returncode != 0 and "DRY_RUN=PASS" not in rejected_execute_wait.stdout
    full_recreated = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe"])
    assert full_recreated.returncode == 0, (full_recreated.stdout, full_recreated.stderr)
    assert "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" in full_recreated.stdout and "xrt_initialized=NO" in full_recreated.stdout
    expected_full_sequence = expected_sequence.replace("<RECREATE_M9216_BO_BUNDLE>", "<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>")
    assert expected_full_sequence in full_recreated.stdout, full_recreated.stdout
    assert "<RECREATE_M9216_HW_CONTEXT>" not in full_recreated.stdout and "<RECREATE_M9216_KERNEL>" not in full_recreated.stdout
    assert "<RECREATE_M9216_BO_BUNDLE>" not in full_recreated.stdout

    control = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-control-id", "8", "--recreate-m9216-context-and-bo-bundle-before-probe"])
    assert control.returncode == 0, (control.stdout, control.stderr)
    assert "XDNA_CAPTURE_REPLAY_POST_RECREATE_CONTROL_DRY_RUN=PASS" in control.stdout and "xrt_initialized=NO" in control.stdout
    assert "post_recreate_control_id=8" in control.stdout and "probe_id=" not in control.stdout and "executions=11" in control.stdout
    assert "ffn_up-3" not in control.stdout and "ffn_gate-4" not in control.stdout
    assert "target_submission_id=13" not in control.stdout and "PROBE_RECORD" not in control.stdout
    control_rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|POST_RECREATE_CONTROL) \|$", control.stdout, re.MULTILINE)
    assert [int(row[1]) for row in control_rows] == list(range(1, 9)) + [8, 8, 8], control_rows
    assert [row[4] for row in control_rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["POST_RECREATE_CONTROL"], control_rows
    expected_control_sequence = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n8\n<POST_RECREATE_CONTROL_ID8>\nreplay_sequence_end"
    assert expected_control_sequence in control.stdout, control.stdout

    transition = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "m4096-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"])
    assert transition.returncode == 0, (transition.stdout, transition.stderr)
    assert "XDNA_CAPTURE_REPLAY_POST_RECREATE_TRANSITION_DRY_RUN=PASS" in transition.stdout and "xrt_initialized=NO" in transition.stdout
    assert "executions=12" in transition.stdout and "post_recreate_transition_control=m4096-to-id8" in transition.stdout
    assert "probe_id=" not in transition.stdout and "ffn_up-3" not in transition.stdout and "ffn_gate-4" not in transition.stdout
    transition_rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|POST_RECREATE_M4096_CONTROL|POST_TRANSITION_ID8_M9216) \|$", transition.stdout, re.MULTILINE)
    assert [int(row[1]) for row in transition_rows] == list(range(1, 9)) + [8, 8, 7, 8], transition_rows
    assert [row[4] for row in transition_rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["POST_RECREATE_M4096_CONTROL", "POST_TRANSITION_ID8_M9216"], transition_rows
    expected_transition_sequence = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<POST_RECREATE_M4096_CONTROL_ID7>\n<POST_TRANSITION_ID8_M9216>\nreplay_sequence_end"
    assert expected_transition_sequence in transition.stdout, transition.stdout

    program_only = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "m4096-program-only-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"])
    assert program_only.returncode == 0, (program_only.stdout, program_only.stderr)
    assert "XDNA_CAPTURE_REPLAY_POST_RECREATE_PROGRAM_ONLY_DRY_RUN=PASS" in program_only.stdout and "xrt_initialized=NO" in program_only.stdout
    assert "executions=12" in program_only.stdout and "post_recreate_transition_control=m4096-program-only-to-id8" in program_only.stdout
    assert "probe_id=" not in program_only.stdout and "ffn_up-3" not in program_only.stdout and "ffn_gate-4" not in program_only.stdout
    program_rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|PROGRAM_M4096_INSTRUCTION_STATE_ONLY|POST_TRANSITION_ID8_M9216) \|$", program_only.stdout, re.MULTILINE)
    assert [int(row[1]) for row in program_rows] == list(range(1, 9)) + [8, 8, 7, 8], program_rows
    assert [row[4] for row in program_rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["PROGRAM_M4096_INSTRUCTION_STATE_ONLY", "POST_TRANSITION_ID8_M9216"], program_rows
    expected_program_only_sequence = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<PROGRAM_M4096_INSTRUCTION_STATE_ONLY>\n<POST_PROGRAM_ID8_M9216>\nreplay_sequence_end"
    assert expected_program_only_sequence in program_only.stdout, program_only.stdout

    no_readback = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "m4096-execute-no-readback-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"])
    assert no_readback.returncode == 0, (no_readback.stdout, no_readback.stderr)
    assert "XDNA_CAPTURE_REPLAY_M4096_NO_READBACK_DRY_RUN=PASS" in no_readback.stdout and "xrt_initialized=NO" in no_readback.stdout
    assert "executions=12" in no_readback.stdout and "probe_id=" not in no_readback.stdout
    no_readback_rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|M4096_EXECUTE_AND_WAIT_NO_READBACK|POST_M4096_EXEC_NO_READBACK_ID8_M9216) \|$", no_readback.stdout, re.MULTILINE)
    assert [int(row[1]) for row in no_readback_rows] == list(range(1, 9)) + [8, 8, 7, 8], no_readback_rows
    assert [row[4] for row in no_readback_rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["M4096_EXECUTE_AND_WAIT_NO_READBACK", "POST_M4096_EXEC_NO_READBACK_ID8_M9216"], no_readback_rows
    expected_no_readback_sequence = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<M4096_EXECUTE_AND_WAIT_NO_READBACK>\n<POST_M4096_EXEC_NO_READBACK_ID8_M9216>\nreplay_sequence_end"
    assert expected_no_readback_sequence in no_readback.stdout, no_readback.stdout

    same_shape = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "id8-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"])
    assert same_shape.returncode == 0, (same_shape.stdout, same_shape.stderr)
    assert "XDNA_CAPTURE_REPLAY_SAME_SHAPE_ID8_DRY_RUN=PASS" in same_shape.stdout and "xrt_initialized=NO" in same_shape.stdout
    assert "post_recreate_transition_control=id8-to-id8" in same_shape.stdout and "executions=12" in same_shape.stdout
    assert "probe_id=" not in same_shape.stdout and "target_submission_id=13" not in same_shape.stdout and "PROBE_RECORD" not in same_shape.stdout
    same_rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|POST_RECREATE_ID8_FIRST|POST_ID8_SAME_SHAPE_ID8_SECOND) \|$", same_shape.stdout, re.MULTILINE)
    assert [int(row[1]) for row in same_rows] == list(range(1, 9)) + [8, 8, 8, 8], same_rows
    assert [row[4] for row in same_rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["POST_RECREATE_ID8_FIRST", "POST_ID8_SAME_SHAPE_ID8_SECOND"], same_rows
    expected_same_shape = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n8\n<POST_RECREATE_ID8_FIRST>\n8\n<POST_ID8_SAME_SHAPE_ID8_SECOND>\nreplay_sequence_end"
    assert expected_same_shape in same_shape.stdout, same_shape.stdout

    same_id7 = run(exe, env, ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "id7-to-id7", "--recreate-m9216-context-and-bo-bundle-before-probe"])
    assert same_id7.returncode == 0, (same_id7.stdout, same_id7.stderr)
    assert "XDNA_CAPTURE_REPLAY_SAME_SHAPE_ID7_DRY_RUN=PASS" in same_id7.stdout and "xrt_initialized=NO" in same_id7.stdout
    assert "post_recreate_transition_control=id7-to-id7" in same_id7.stdout and "executions=12" in same_id7.stdout
    assert "POST_RECREATE_ID7_FIRST source_id=7 M=4096 ordinal=1" in same_id7.stdout
    assert "POST_ID7_SAME_SHAPE_ID7_SECOND source_id=7 M=4096 ordinal=2" in same_id7.stdout
    assert "TRANSITION_STAGE=M4096_TO_SAME_M4096" in same_id7.stdout
    id7_rows = re.findall(r"^\| (\d+) \| (\d+) \| ([^|]+) \| (\d+) \| (PREFIX_RECORD|INJECTED_HISTORY_RECORD|POST_RECREATE_ID7_FIRST|POST_ID7_SAME_SHAPE_ID7_SECOND) \|$", same_id7.stdout, re.MULTILINE)
    assert [int(row[1]) for row in id7_rows] == list(range(1, 9)) + [8, 8, 7, 7], id7_rows
    assert [row[4] for row in id7_rows] == ["PREFIX_RECORD"] * 8 + ["INJECTED_HISTORY_RECORD"] * 2 + ["POST_RECREATE_ID7_FIRST", "POST_ID7_SAME_SHAPE_ID7_SECOND"], id7_rows
    expected_id7_sequence = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<POST_RECREATE_ID7_FIRST>\n<POST_ID7_SAME_SHAPE_ID7_SECOND>\nreplay_sequence_end"
    assert expected_id7_sequence in same_id7.stdout, same_id7.stdout
    assert "PROBE_RECORD" not in same_id7.stdout and "probe_id=" not in same_id7.stdout
    assert "target_submission_id=13" not in same_id7.stdout

    invalid = (
        ("negative prefix", ["--prefix-count", "-1", "--probe-id", "11"]),
        ("prefix equals probe", ["--prefix-count", "11", "--probe-id", "11"]),
        ("prefix exceeds probe", ["--prefix-count", "12", "--probe-id", "11"]),
        ("zero probe", ["--prefix-count", "0", "--probe-id", "0"]),
        ("outside probe", ["--prefix-count", "0", "--probe-id", "14"]),
        ("missing probe option", ["--prefix-count", "0"]),
        ("malformed value", ["--prefix-count", "bad", "--probe-id", "11"]),
        ("duplicate probe", ["--prefix-count", "0", "--probe-id", "11", "--probe-id", "11"]),
        ("duplicate prefix", ["--prefix-count", "0", "--prefix-count", "0", "--probe-id", "11"]),
        ("unknown option", ["--prefix-count", "0", "--probe-id", "11", "--surprise"]),
        ("inject without prefix", ["--inject-id", "10", "--probe-id", "11"]),
        ("wrong injected ID", ["--prefix-count", "8", "--inject-id", "9", "--probe-id", "11"]),
        ("unsupported injected order", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "10", "--probe-id", "11"]),
        ("nonadjacent probe", ["--prefix-count", "8", "--inject-id", "10", "--probe-id", "12"]),
        ("duplicate source before prefix", ["--prefix-count", "8", "--inject-id", "7", "--inject-id", "7", "--probe-id", "11"]),
        ("duplicate source not before probe", ["--prefix-count", "8", "--inject-id", "11", "--inject-id", "11", "--probe-id", "11"]),
        ("bundle recreation without prefix selection", ["--recreate-m9216-bo-bundle-before-probe"]),
        ("bundle recreation wrong prefix", ["--prefix-count", "7", "--inject-id", "8", "--inject-id", "8", "--probe-id", "10", "--recreate-m9216-bo-bundle-before-probe"]),
        ("bundle recreation wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--probe-id", "11", "--recreate-m9216-bo-bundle-before-probe"]),
        ("bundle recreation wrong probe", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "12", "--recreate-m9216-bo-bundle-before-probe"]),
        ("missing injection value", ["--prefix-count", "8", "--inject-id", "--probe-id", "11"]),
        ("kernel recreation without prefix selection", ["--recreate-m9216-kernel-before-probe"]),
        ("kernel recreation wrong prefix", ["--prefix-count", "7", "--inject-id", "8", "--inject-id", "8", "--probe-id", "10", "--recreate-m9216-kernel-before-probe"]),
        ("kernel recreation wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--probe-id", "11", "--recreate-m9216-kernel-before-probe"]),
        ("kernel recreation wrong probe", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "12", "--recreate-m9216-kernel-before-probe"]),
        ("context recreation without prefix/full replay", ["--recreate-m9216-hw-context-before-probe"]),
        ("context recreation wrong prefix", ["--prefix-count", "7", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-hw-context-before-probe"]),
        ("context recreation wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--probe-id", "11", "--recreate-m9216-hw-context-before-probe"]),
        ("context recreation target mode", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "13", "--recreate-m9216-hw-context-before-probe"]),
        ("duplicate context recreation request", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-hw-context-before-probe", "--recreate-m9216-hw-context-before-probe"]),
        ("context plus BO recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-hw-context-before-probe", "--recreate-m9216-bo-bundle-before-probe"]),
        ("context plus kernel recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-hw-context-before-probe", "--recreate-m9216-kernel-before-probe"]),
        ("full bundle recreation without prefix selection", ["--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("full bundle recreation wrong prefix", ["--prefix-count", "7", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("full bundle recreation wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("full bundle recreation wrong probe", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "13", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("duplicate full bundle recreation request", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("full bundle plus BO recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe", "--recreate-m9216-bo-bundle-before-probe"]),
        ("full bundle plus kernel recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe", "--recreate-m9216-kernel-before-probe"]),
        ("full bundle plus context recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe", "--recreate-m9216-hw-context-before-probe"]),
        ("mutually exclusive recreation modes", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--recreate-m9216-bo-bundle-before-probe", "--recreate-m9216-kernel-before-probe"]),
        ("control without recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-control-id", "8"]),
        ("control with probe11", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-control-id", "8", "--probe-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("wrong control ID", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-control-id", "11", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("wrong control injection history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--post-recreate-control-id", "8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("control with target probe", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-control-id", "8", "--recreate-m9216-context-and-bo-bundle-before-probe", "--recreate-m9216-bo-bundle-before-probe"]),
        ("program-only control without recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "m4096-program-only-to-id8"]),
        ("program-only control wrong injection history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--post-recreate-transition-control", "m4096-program-only-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("program-only control with probe", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--post-recreate-transition-control", "m4096-program-only-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("no-readback without recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "m4096-execute-no-readback-to-id8"]),
        ("no-readback with probe", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--post-recreate-transition-control", "m4096-execute-no-readback-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("no-readback wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--post-recreate-transition-control", "m4096-execute-no-readback-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("same-shape without recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "id8-to-id8"]),
        ("same-shape with probe11", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--post-recreate-transition-control", "id8-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("same-shape wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--post-recreate-transition-control", "id8-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("ID7 same-shape wrong prefix", ["--prefix-count", "7", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "id7-to-id7", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("ID7 same-shape missing full recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "id7-to-id7"]),
        ("ID7 same-shape with probe11", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--post-recreate-transition-control", "id7-to-id7", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("ID7 same-shape with target13", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "13", "--post-recreate-transition-control", "id7-to-id7", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("context-kernel transition missing recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "m4096-context-recreate-to-id8"]),
        ("context-kernel transition with probe", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--post-recreate-transition-control", "m4096-context-recreate-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("context-kernel transition wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--post-recreate-transition-control", "m4096-context-recreate-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("context-kernel transition incompatible recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "m4096-context-recreate-to-id8", "--recreate-m9216-context-and-bo-bundle-before-probe", "--recreate-m9216-bo-bundle-before-probe"]),
        ("ID8 context recreate missing bundle contract", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "id8-context-recreate-only", "--probe-id", "11"]),
        ("ID8 context recreate wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--post-recreate-transition-control", "id8-context-recreate-only"]),
        ("double-recreate no-execution wrong prefix", ["--prefix-count", "7", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "double-recreate-no-execution"]),
        ("double-recreate no-execution wrong history", ["--prefix-count", "8", "--inject-id", "9", "--inject-id", "9", "--post-recreate-transition-control", "double-recreate-no-execution"]),
        ("double-recreate no-execution with probe11", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "11", "--post-recreate-transition-control", "double-recreate-no-execution"]),
        ("double-recreate no-execution with target13", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--probe-id", "13", "--post-recreate-transition-control", "double-recreate-no-execution"]),
        ("double-recreate no-execution with competing BO recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "double-recreate-no-execution", "--recreate-m9216-bo-bundle-before-probe"]),
        ("double-recreate no-execution duplicate full recreation", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "double-recreate-no-execution", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
        ("unknown transition control", ["--prefix-count", "8", "--inject-id", "8", "--inject-id", "8", "--post-recreate-transition-control", "bogus", "--recreate-m9216-context-and-bo-bundle-before-probe"]),
    )
    for label, args in invalid:
        result = run(exe, env, args)
        assert result.returncode != 0 and "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS" not in result.stdout, (label, result.stdout, result.stderr)

    backend_dir = REPO / "ggml" / "src" / "ggml-xdna"
    backend_source = (backend_dir / "ggml-xdna.cpp").read_text(encoding="utf-8") + "\n" + (backend_dir / "ggml-xdna-diagnostics.cpp").read_text(encoding="utf-8")
    driver_source = (REPO / "tools" / "xdna-capture-replay.cpp").read_text(encoding="utf-8")
    assert "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" in backend_source and "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3" in backend_source
    assert "old_shape_context_not_reused=YES" in backend_source and "shared_generation1_kernel=YES" in backend_source
    assert "generation1_submissions != 1" in backend_source and "generation1_output_syncs != 1" in backend_source
    assert "transition_m4096_output_validated != 1" in backend_source and "transition_id8_output_validated != 1" in backend_source
    bo_recreate_start = backend_source.index("static void ggml_backend_xdna_replay_recreate_m9216_bo_bundle(")
    bo_recreate_end = backend_source.index("\nstatic void ggml_backend_xdna_replay_recreate_m9216_kernel(", bo_recreate_start)
    bo_recreate_body = backend_source[bo_recreate_start:bo_recreate_end]
    kernel_recreate_start = bo_recreate_end + 1
    kernel_recreate_end = backend_source.index("\nstatic void ggml_backend_xdna_replay_recreate_m9216_hw_context(", kernel_recreate_start)
    kernel_recreate_body = backend_source[kernel_recreate_start:kernel_recreate_end]
    context_recreate_start = kernel_recreate_end + 1
    context_recreate_end = backend_source.index("\nstatic void ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(", context_recreate_start)
    context_recreate_body = backend_source[context_recreate_start:context_recreate_end]
    full_recreate_start = context_recreate_end + 1
    full_recreate_end = backend_source.index("\nstatic void ggml_backend_xdna_replay_create_m4096_transition_bundle(", full_recreate_start)
    full_recreate_body = backend_source[full_recreate_start:full_recreate_end]
    ensure_start = backend_source.index("static bool ggml_backend_xdna_replay_ensure_m9216_bos(")
    ensure_end = backend_source.index("\nstatic const char * ggml_backend_xdna_replay_identity_change", ensure_start)
    ensure_body = backend_source[ensure_start:ensure_end]
    assert "trace->m9216_hw_context_generation == 1 || validate_frozen_instruction" in ensure_body
    replay_body = backend_source.split("static int ggml_backend_xdna_replay_exact_history_impl(", 1)[1].split("\nint ggml_backend_xdna_diagnostic_replay(", 1)[0]
    initial_recreation_predicate = replay_body[replay_body.index("const bool is_recreation_boundary_role"):replay_body.index("if (run_initial_recreation_boundary)")]
    assert "!post_recreate_id8_context_recreate_only_mode || (is_post_recreate_control && recreate_m9216_context_and_bo_bundle)" in initial_recreation_predicate
    initial_full_call = replay_body.index("ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace)")
    initial_boundary = replay_body.index("if (run_initial_recreation_boundary)")
    history_complete_gate = replay_body.index("ctx->replay_prefix_completed != 10", initial_boundary)
    id8_kernel_call = replay_body.index("run.emplace((*kernel)", initial_full_call)
    id8_output_exact = replay_body.index('output_sha != r.at("output_sha256")', id8_kernel_call)
    id8_lifecycle_gate = replay_body.index("ID8 exactness gate failed before lifecycle-only context recreation", id8_output_exact)
    second_context_only_call = replay_body.index("ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, true)", id8_lifecycle_gate)
    terminal_stop = replay_body.index("XDNA_CAPTURE_REPLAY POST-ID8 CONTEXT-RECREATE TERMINAL", second_context_only_call)
    assert history_complete_gate < initial_full_call < id8_kernel_call < id8_output_exact < id8_lifecycle_gate < second_context_only_call < terminal_stop
    assert replay_body.count("ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace)") == 3
    assert not any(token in replay_body[second_context_only_call:] for token in ("auto run = (*kernel)", "run.wait()", "xrt::run"))
    assert "context_generation=2 kernel_generation=2 BO_generation=1 no_post_recreate_submission=YES" in replay_body
    noexec_loop_start = replay_body.index("for (const auto & selected : execution)")
    noexec_start = replay_body.index("if (post_recreate_double_recreate_no_execution_mode) {", noexec_loop_start)
    noexec_end = replay_body.index("if (ctx->replay_kernels_submitted != ctx->replay_records_total", noexec_start)
    noexec_body = replay_body[noexec_start:noexec_end]
    initial_full_recreate = noexec_body.index("ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace)")
    second_recreate = noexec_body.index("ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, true)")
    between_recreates = noexec_body[initial_full_recreate:second_recreate]
    assert initial_full_recreate < second_recreate and "NO_EXECUTION_RECREATE_GUARD=PASS" in between_recreates
    assert all(token in between_recreates for token in ("generation1_submissions != 0", "generation1_wait_completions != 0", "generation1_output_syncs != 0", "experiment_submission_ordinal != 0", "previous_experimental_source=NONE"))
    assert not any(token in between_recreates for token in ("auto run = (*kernel)", "xrt::run", "run.wait()", "bo_out->sync(", "ggml_backend_xdna_sync_host_written_bo(", "std::make_unique<xrt::bo>", ".map<"))
    assert "<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>" in noexec_body and "post_second_recreation_submissions=0" in noexec_body
    assert "context_generation=2 kernel_generation=2 BO_generation=1" in noexec_body and "experiment_submissions=0" in noexec_body
    after_second_recreate = noexec_body[second_recreate:]
    assert not any(token in after_second_recreate for token in ("auto run = (*kernel)", "xrt::run", "run.wait()", "bo_out->sync(", "ggml_backend_xdna_sync_host_written_bo(", "std::make_unique<xrt::bo>", ".map<"))
    id7_second_guard_at = replay_body.index("if (is_same_shape_id7_second) {")
    kernel_submit_at = replay_body.index("run.emplace((*kernel)")
    id7_guard = replay_body[id7_second_guard_at:kernel_submit_at]
    assert id7_second_guard_at < kernel_submit_at and all(token in id7_guard for token in ("id7_exact_outputs == 1", "generation1_execution_ordinal == 2", "same_identity", "frozen_same"))
    assert not any(token in id7_guard for token in ("(*kernel)(", "run.wait(", ".sync("))
    assert all(token in replay_body for token in ("POST_RECREATE_ID7_FIRST", "POST_ID7_SAME_SHAPE_ID7_SECOND", "id7_submissions", "id7_output_syncs", "id7_exact_outputs", "FIRST_ID7_EXACT=1", "SAME-SHAPE ID7-TO-ID7 PASS"))
    program_only_start = backend_source.index("static void ggml_backend_xdna_replay_program_m4096_instruction_only(")
    program_only_end = backend_source.index("\nstatic void ggml_backend_xdna_replay_log_runtime_snapshot(", program_only_start)
    program_only_body = backend_source[program_only_start:program_only_end]
    assert program_only_body.count("std::make_unique<xrt::bo>") == 1 and program_only_body.count("map<void *>()") == 1
    assert program_only_body.count("ggml_backend_xdna_sync_host_written_bo(") == 2
    assert all(token in program_only_body for token in ("0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904", "instruction_file", "instruction_bytes", "instruction_sha256", "instruction_init_sync_count", "instruction_stage_sync_count"))
    assert not any(token in program_only_body for token in ("run.wait(", "(*kernel)(", "bo_q4", "bo_q8", "bo_out", "buf_q4", "buf_q8", "buf_out", "xrt::run"))

    assert all(token in program_only_body for token in ("std::memcpy(bundle->mapped_instr", "sync_host_written_bo(*bundle->bo_instr", "kernel_call_delta=0", "run_creation_delta=0", "BO_construct_delta=1", "BO_map_delta=1", "BO_sync_delta=2"))
    assert not any(token in program_only_body for token in ("(*kernel)(", "xrt::run", "run.wait()", "XCL_BO_SYNC_BO_FROM_DEVICE", "bo_q4", "bo_q8", "bo_out"))
    helper_call = "ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, true)"
    assert helper_call in backend_source
    noexec_at = backend_source.index("TRANSITION_STAGE=PROGRAM_ONLY_M4096_TO_SECOND_CONTEXT_RECREATE")
    program_only_branch_start = backend_source.rfind("if (post_recreate_program_only_context_recreate_mode) {", 0, noexec_at)
    program_only_branch_end = backend_source.index("} else if (post_recreate_double_recreate_no_execution_mode) {", noexec_at)
    program_only_branch = backend_source[program_only_branch_start:program_only_branch_end]
    ordered = ("kind=FULL_RUNTIME_BUNDLE", "ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace)",
        "ggml_backend_xdna_replay_program_m4096_instruction_only", "PROGRAM_ONLY_CONTEXT_RECREATE_GUARD=PASS", helper_call,
        "<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>")
    assert [program_only_branch.index(token) for token in ordered] == sorted(program_only_branch.index(token) for token in ordered)
    assert not any(token in program_only_branch for token in ("auto run = (*kernel)", "(*kernel)(", "xrt::run", "run.wait()", "bo_out->sync(", "XCL_BO_SYNC_BO_FROM_DEVICE"))
    assert "post_recreate_program_only_context_recreate_mode" in backend_source and "frozen_program_only_context_recreate" in backend_source
    assert "PROGRAM_ONLY_RESOURCE_DELTAS" in program_only_branch and "instruction_writes=2" in program_only_branch and "q4_syncs=0 q8_syncs=0" in program_only_branch
    no_readback_after_wait = replay_body.index('if (is_post_recreate_m4096_no_readback) {', replay_body.index('AFTER run.wait'))
    no_readback_output_sync = replay_body.index('bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);', no_readback_after_wait)
    no_readback_guard = replay_body[no_readback_after_wait:no_readback_output_sync]
    assert 'POST_WAIT_NO_READBACK' in no_readback_guard and 'm4096_output_from_device_syncs != 0' in no_readback_guard
    assert '} else {' in no_readback_guard and not any(token in no_readback_guard for token in ('bo_out->sync(', 'capture_hash(buf_out', 'const std::string output_sha'))
    assert replay_body.index('run->wait()') < no_readback_after_wait < no_readback_output_sync
    assert 'ctx->replay_post_recreate_no_readback_mode' in replay_body and 'M4096 execute/wait no-readback to ID8 final telemetry invariant failed' in replay_body
    program_only_stage = replay_body.split("if (is_post_recreate_m4096_program_only) {", 1)[1].split("const size_t instruction_bytes = r.at", 1)[0]
    assert all(token in program_only_stage for token in ("ggml_backend_xdna_replay_program_m4096_instruction_only(", "++ctx->replay_program_only_stages", "continue;", "generation1_submissions", "generation1_wait_completions"))
    assert "generation1_instruction_syncs" not in backend_source and "generation1_instruction_init_syncs + runtime_trace.generation1_instruction_stage_syncs" in backend_source
    forbidden = ("xrt::device(", "xrt::xclbin(", "xrt::hw_context(", "xrt::kernel(", "run.wait(", "(*kernel)(", "xclbin.load(", "register_xclbin(")
    assert not any(token in bo_recreate_body + ensure_body for token in forbidden), forbidden
    assert bo_recreate_body.count(".reset();") == 4 and "ensure_m9216_bos(ctx, trace, true)" in bo_recreate_body
    assert ensure_body.count("std::make_unique<xrt::bo>") == 4
    assert "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" in ensure_body
    kernel_forbidden = ("xrt::device(", "xrt::xclbin(", "xrt::hw_context(", "std::make_unique<xrt::bo>", ".map<", ".sync(", "run.wait(", "(*kernel)(", "register_xclbin(", "xclbin.load(")
    assert not any(token in kernel_recreate_body for token in kernel_forbidden), kernel_forbidden
    assert kernel_recreate_body.count("ctx->q4_m9216_kernel.reset();") == 1
    assert kernel_recreate_body.count("std::make_unique<xrt::kernel>") == 1
    assert kernel_recreate_body.count("ggml_backend_xdna_replay_recreate_m9216_kernel(") == 1
    assert all(token in kernel_recreate_body for token in ("get_memory_group()", "group_id(1)", "group_id(3)", "group_id(4)", "group_id(5)", "->size()", "old_kernel_group_ids", "new_kernel_group_ids", "retained_bo_memory_groups"))
    assert "group_matches" not in kernel_recreate_body and "existing M9216 BO binding disagrees with its kernel group IDs" not in kernel_recreate_body
    old_kernel_groups = (4849665, 4849664, 4849664, 4849664)
    bo_memory_groups = (1, 0, 0, 0)
    assert old_kernel_groups != bo_memory_groups
    kernel_group_relation = lambda old, new: tuple(old) == tuple(new)
    assert kernel_group_relation(old_kernel_groups, old_kernel_groups)  # Numeric BO representation does not enter this check.
    assert not kernel_group_relation(old_kernel_groups, (4849665, 4849664, 4849664, 4849665))
    assert "const bool kernel_groups_same = old_group_instr == new_group_instr" in kernel_recreate_body
    assert all(f"old_group_{suffix} == new_group_{suffix}" in kernel_recreate_body for suffix in ("q4", "q8", "out"))
    assert "SAME_AS_OLD_KERNEL" in kernel_recreate_body and "NEW_KERNEL_GROUP_IDS_CHANGED_WITHIN_SAME_CONTEXT" in kernel_recreate_body
    assert "retained_bo_memory_groups=%u,%u,%u,%u" in kernel_recreate_body
    constructor_at = kernel_recreate_body.index("std::make_unique<xrt::kernel>")
    check_at = kernel_recreate_body.index("const bool checks_passed = resources_same && kernel_groups_same")
    generation_at = kernel_recreate_body.index("trace->m9216_kernel_generation = 1")
    recreate_count_at = kernel_recreate_body.index("trace->m9216_kernel_recreate_count = 1")
    assert check_at < recreate_count_at
    assert constructor_at < check_at < generation_at and "if (checks_passed)" in kernel_recreate_body
    assert "trace->m9216_kernel_recreate_count = 1" in kernel_recreate_body
    assert "uint64_t m9216_kernel_generation = 0, m9216_kernel_recreate_count = 0;" in backend_source
    assert "previous_m9216_kernel_generation == runtime_trace.m9216_kernel_generation" in backend_source
    assert backend_source.count("ggml_backend_xdna_replay_recreate_m9216_kernel(") == 2
    assert backend_source.count("ggml_backend_xdna_replay_recreate_m9216_hw_context(") == 7
    assert context_recreate_body.count("ctx->q4_m9216_kernel.reset();") == 1
    assert context_recreate_body.count("ctx->q4_m9216_hw_context.reset();") == 1
    assert context_recreate_body.count("std::make_shared<xrt::hw_context>") == 1
    assert context_recreate_body.count("std::make_unique<xrt::kernel>") == 1
    assert "std::make_unique<xrt::bo>" not in context_recreate_body
    assert not any(token in context_recreate_body for token in ("xrt::device(", "xrt::xclbin(", "xrt::bo(", "register_xclbin(", "load_xclbin(", ".map<", ".sync(", "run.wait(", "(*kernel)("))
    assert context_recreate_body.index("ctx->q4_m9216_kernel.reset()") < context_recreate_body.index("ctx->q4_m9216_hw_context.reset()") < context_recreate_body.index("std::make_shared<xrt::hw_context>") < context_recreate_body.index("std::make_unique<xrt::kernel>")
    assert ensure_body.count("std::make_unique<xrt::bo>(device") == 4 and "std::make_unique<xrt::bo>(hw_context" not in ensure_body
    assert all(token in context_recreate_body for token in ("get_xclbin_uuid()", "get_name()", "old_kernel_group_ids", "new_kernel_group_ids", "get_memory_group()", "static_connectivity=ARG1_MEM_TOPOLOGY_1_SRAM_ARG3_4_5_MEM_TOPOLOGY_0_HOST"))
    assert "old_group_instr == new_group" not in context_recreate_body and "kernel_group_relation" not in context_recreate_body
    release_kernel = context_recreate_body.index("XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE OLD_KERNEL_RELEASED")
    release_context = context_recreate_body.index("XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE OLD_CONTEXT_RELEASED")
    before_context = context_recreate_body.index("XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE BEFORE_NEW_CONTEXT_CONSTRUCT")
    context_ctor = context_recreate_body.index("std::make_shared<xrt::hw_context>")
    after_context = context_recreate_body.index("XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE AFTER_NEW_CONTEXT_CONSTRUCT")
    before_kernel = context_recreate_body.index("XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE BEFORE_NEW_KERNEL_CONSTRUCT")
    kernel_ctor = context_recreate_body.index("std::make_unique<xrt::kernel>")
    after_kernel = context_recreate_body.index("XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE AFTER_NEW_KERNEL_CONSTRUCT")
    before_inspection = context_recreate_body.index("XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE BEFORE_POST_BOUNDARY_INSPECTION")
    after_inspection = context_recreate_body.index("XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE AFTER_POST_BOUNDARY_INSPECTION")
    context_end = context_recreate_body.index("POST_ID7_CONTEXT_RECREATE_END")
    assert release_kernel < release_context < before_context < context_ctor < after_context < before_kernel < kernel_ctor < after_kernel < before_inspection < after_inspection < context_end
    released_window = context_recreate_body[release_context:after_context]
    assert "get_uuid()" not in released_window and "get_xclbin_uuid()" not in released_window and "group_id(" not in released_window and "get_name()" not in released_window
    assert "get_memory_group()" not in released_window and "->size()" not in released_window
    assert context_recreate_body.count("->get_uuid()") == 2 and context_recreate_body.count("->get_xclbin_uuid()") == 2
    assert context_recreate_body.count("->get_name()") == 2 and context_recreate_body.count("->group_id(") == 8
    assert context_recreate_body.count("->get_memory_group()") == 4 and context_recreate_body.count("->size()") == 8
    assert context_recreate_body.index("xclbin_uuid_value") < release_kernel and context_recreate_body.index("old_kernel_group_ids") < release_kernel
    assert context_recreate_body.count("NEW_CONTEXT_CONSTRUCT_EXCEPTION") == 1 and context_recreate_body.count("NEW_KERNEL_CONSTRUCT_EXCEPTION") == 1
    assert context_recreate_body.count("POST_BOUNDARY_INSPECTION_EXCEPTION") == 1
    assert "throw;" in context_recreate_body and "AFTER_POST_BOUNDARY_INSPECTION result=%s" in context_recreate_body
    assert "std::fflush(stderr)" in context_recreate_body
    for marker in ("OLD_KERNEL_RELEASED", "OLD_CONTEXT_RELEASED", "BEFORE_NEW_CONTEXT_CONSTRUCT", "AFTER_NEW_CONTEXT_CONSTRUCT", "BEFORE_NEW_KERNEL_CONSTRUCT", "AFTER_NEW_KERNEL_CONSTRUCT", "BEFORE_POST_BOUNDARY_INSPECTION", "AFTER_POST_BOUNDARY_INSPECTION"):
        marker_at = context_recreate_body.index(marker)
        marker_line_end = context_recreate_body.index("\n", marker_at)
        assert "std::fflush(stderr);" in context_recreate_body[marker_line_end:context_recreate_body.index("\n", marker_line_end + 1)]
    assert "xrt_delta=hw_context_construct:1,kernel_construct:1,BO_construct:0,BO_map:0,BO_sync:0,kernel_submit:0,run_wait:0,device_reopen:0,xclbin_load_register:0" in context_recreate_body
    id8_guard_at = backend_source.index("if (is_post_context_recreate_id8)")
    id8_guard_throw = backend_source.index("XDNA_RECREATE_GUARD ID8 pre-submit context-generation-2 guard failed", id8_guard_at)
    id8_transition = backend_source.index("TRANSITION_STAGE=M4096_TO_CONTEXT_KERNEL_RECREATE_TO_M9216", id8_guard_at)
    id8_submit = backend_source.index("run.emplace((*kernel)", id8_guard_at)
    assert backend_source.index("POST_ID7_CONTEXT_RECREATE_END") < id8_guard_at < id8_guard_throw < id8_transition < id8_submit
    context_check = context_recreate_body.index("checks_passed = device_same")
    context_generation = context_recreate_body.index("++trace->m9216_hw_context_generation")
    assert context_check < context_generation and "if (!checks_passed) throw" in context_recreate_body
    assert "++trace->m9216_kernel_generation" in context_recreate_body and "++trace->m9216_kernel_recreate_count" in context_recreate_body
    assert "++trace->m9216_bo_bundle_generation" not in context_recreate_body and "trace->m9216_bo_bundle_generation =" not in context_recreate_body
    assert "context_changed && kernel_changed" in context_recreate_body and "operation_counters_unchanged" in context_recreate_body
    inspection_pass = context_recreate_body.index("AFTER_POST_BOUNDARY_INSPECTION result=%s")
    inspection_failure = context_recreate_body.index("if (!checks_passed) throw", inspection_pass)
    generation_commit = context_recreate_body.index("++trace->m9216_hw_context_generation", inspection_failure)
    assert inspection_pass < inspection_failure < generation_commit
    assert all(token in context_recreate_body for token in ("post_id7_bo_instr", "post_id7_map_instr", "post_id7_xclbin_uuid", "post_id7_kernel_name", "context_generation_submission_ordinal = 0"))
    context_forbidden = ("std::make_unique<xrt::bo>", "xrt::bo(", ".map<", ".sync(", "run.wait(", "(*kernel)(", "xrt::device(", "xrt::xclbin(", "register_xclbin(", "load_xclbin(")
    assert not any(token in context_recreate_body for token in context_forbidden), context_forbidden
    context_call = replay_body.index("ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, true)")
    post_output = replay_body.index('log_runtime_snapshot("POST_OUTPUT", run ? &*run : nullptr)')
    id8_input_guard = replay_body.index("const bool frozen_id8")
    output_validation = replay_body.index("output_sha != r.at(\"output_sha256\")")
    assert output_validation < post_output < context_call
    assert id8_input_guard < replay_body.index("run.emplace((*kernel)")
    assert "runtime_trace.experiment_submission_ordinal != 1" in replay_body and "runtime_trace.context_generation_submission_ordinal != 1" in replay_body
    assert "runtime_trace.experiment_submission_ordinal == 2" in replay_body and "runtime_trace.context_generation_submission_ordinal == 1" in replay_body
    assert "context_generation=2 kernel_generation=2 bo_generation=1" in backend_source and "recreate_counts=2/2/1" in backend_source
    assert "POST_ID7_CONTEXT_RECREATE_BEGIN" in context_recreate_body and "POST_ID7_CONTEXT_RECREATE_END" in context_recreate_body
    assert "XDNA_CAPTURE_REPLAY M4096 CONTEXT/KERNEL RECREATE TO ID8 PASS" in replay_body
    assert not any(token in full_recreate_body for token in ("xrt::device(", "xrt::xclbin(", "run.wait(", "(*kernel)(", "xclbin.load(", "register_xclbin("))
    assert full_recreate_body.count("ctx->q4_m9216_kernel.reset();") == 1
    assert full_recreate_body.count("ctx->q4_m9216_hw_context.reset();") == 1
    assert sum(full_recreate_body.count(f"ctx->q4_m9216_bo_{name}.reset();") for name in ("instr", "q4", "q8", "out")) == 4
    assert all(full_recreate_body.count(f"RELEASE_OLD_{name}_BO_{edge}") == 1 for name in ("INSTRUCTION", "Q4", "Q8", "OUTPUT") for edge in ("BEGIN", "END"))
    teardown_order = ("RELEASE_OLD_KERNEL_BEGIN", "ctx->q4_m9216_kernel.reset();", "RELEASE_OLD_KERNEL_END", "INVALIDATE_OLD_MAPS",
                      "RELEASE_OLD_BO_BUNDLE", "RELEASE_OLD_INSTRUCTION_BO_BEGIN", "ctx->q4_m9216_bo_instr.reset();", "RELEASE_OLD_INSTRUCTION_BO_END",
                      "RELEASE_OLD_Q4_BO_BEGIN", "ctx->q4_m9216_bo_q4.reset();", "RELEASE_OLD_Q4_BO_END",
                      "RELEASE_OLD_Q8_BO_BEGIN", "ctx->q4_m9216_bo_q8.reset();", "RELEASE_OLD_Q8_BO_END",
                      "RELEASE_OLD_OUTPUT_BO_BEGIN", "ctx->q4_m9216_bo_out.reset();", "RELEASE_OLD_OUTPUT_BO_END",
                      "OLD_BO_BUNDLE_RELEASED", "RELEASE_OLD_HW_CONTEXT_BEGIN", "ctx->q4_m9216_hw_context.reset();", "RELEASE_OLD_HW_CONTEXT_END",
                      "std::make_shared<xrt::hw_context>")
    teardown_positions = [full_recreate_body.index(token) for token in teardown_order]
    assert teardown_positions == sorted(teardown_positions)
    new_context = full_recreate_body.index("std::make_shared<xrt::hw_context>")
    new_kernel = full_recreate_body.index("std::make_unique<xrt::kernel>")
    new_bundle = full_recreate_body.index("ensure_m9216_bos(ctx, trace, true)")
    assert teardown_positions[-1] < new_kernel < new_bundle
    assert full_recreate_body.count("ensure_m9216_bos(ctx, trace, true)") == 1
    assert full_recreate_body.index("trace->m9216_hw_context_generation = trace->m9216_hw_context_recreate_count = 1") > full_recreate_body.index("const bool checks_passed =")
    assert full_recreate_body.index("trace->m9216_kernel_generation = trace->m9216_kernel_recreate_count = 1") > full_recreate_body.index("const bool checks_passed =")
    assert full_recreate_body.index("trace->m9216_bo_bundle_generation = trace->m9216_bo_bundle_recreate_count = 1") > full_recreate_body.index("const bool checks_passed =")
    assert "if (checks_passed)" in full_recreate_body and "if (!checks_passed) throw" in full_recreate_body
    assert full_recreate_body.count("trace->m9216_hw_context_generation = trace->m9216_hw_context_recreate_count = 1") == 1
    assert full_recreate_body.count("trace->m9216_kernel_generation = trace->m9216_kernel_recreate_count = 1") == 1
    assert full_recreate_body.count("trace->m9216_bo_bundle_generation = trace->m9216_bo_bundle_recreate_count = 1") == 1
    assert full_recreate_body.count("RELEASE_OLD_BO_BUNDLE") == 1 and full_recreate_body.count("OLD_BO_BUNDLE_RELEASED") == 1
    assert full_recreate_body.count("RELEASE_OLD_HW_CONTEXT_BEGIN") == 1 and full_recreate_body.count("RELEASE_OLD_HW_CONTEXT_END") == 1
    for marker in ("RELEASE_OLD_KERNEL_BEGIN", "RELEASE_OLD_KERNEL_END", "INVALIDATE_OLD_MAPS", "RELEASE_OLD_BO_BUNDLE", "OLD_BO_BUNDLE_RELEASED", "RELEASE_OLD_HW_CONTEXT_BEGIN", "RELEASE_OLD_HW_CONTEXT_END") + tuple(f"RELEASE_OLD_{name}_BO_{edge}" for name in ("INSTRUCTION", "Q4", "Q8", "OUTPUT") for edge in ("BEGIN", "END")):
        assert backend_source.count(marker) == 1, marker
    assert new_context < new_kernel < new_bundle

    test_replay = backend_source.split("static int ggml_backend_xdna_replay_exact_history_impl(", 1)[1].split("\nint ggml_backend_xdna_diagnostic_replay(", 1)[0]
    transition_guard_start = test_replay.index("const bool id8_identity = source_id == 8")
    transition_guard_end = test_replay.index('throw std::runtime_error("ID8 generation-1 first-submission/transition boundary invariant failed")', transition_guard_start)
    transition_guard = test_replay[transition_guard_start:transition_guard_end]
    assert all(token in transition_guard for token in ("id7_identity", "if (ctx->replay_post_recreate_no_readback_mode)", "else if (ctx->replay_post_recreate_transition_mode)",
        "generation1_output_from_device_syncs == 0", "m4096_output_from_device_syncs == 0", "m4096_output_reads == 0",
        "m4096_output_sha_calculations == 0", "m4096_output_sha_validations == 0", "ordinary_readback"))
    assert not any(token in transition_guard for token in ("bo_out->sync(", "capture_hash(buf_out", "(*kernel)(", "run.wait()", "make_unique<xrt::"))
    pre_control_guard = test_replay.split("if (is_post_recreate_control && (", 1)[1].split('throw std::runtime_error("post-recreation ID8 control pre-recreation boundary failed")', 1)[0]
    assert "m9216_bo_bundle_generation_submissions" not in pre_control_guard and "m9216_bo_bundle_generation_wait_completions" not in pre_control_guard
    recreate_call = test_replay.index("ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace)")
    post_guard_start = test_replay.index("if (runtime_trace.m9216_hw_context_generation != 1", recreate_call)
    post_guard_end = test_replay.index('throw std::runtime_error("post-recreation control is not first in generation 1")', post_guard_start)
    post_guard = test_replay[post_guard_start:post_guard_end]
    assert all(token in post_guard for token in ("m9216_bo_bundle_generation_submissions != 0", "m9216_bo_bundle_generation_wait_completions != 0", "m9216_submissions != 7", "m9216_wait_completions != 7", "m9216_output_syncs != 7", "m9216_execution_ordinal != 7"))
    pre_telemetry = test_replay.index("POST_RECREATE_CONTROL_BOUNDARY_PRE")
    post_telemetry = test_replay.index("POST_RECREATE_CONTROL_BOUNDARY_POST")
    first_control = test_replay.index("POST-RECREATE CONTROL FIRST SUBMISSION source_id=8")
    id8_load = test_replay.index("ggml_backend_xdna_replay_read_artifact(root, r", post_guard_end)
    assert pre_telemetry < recreate_call < post_guard_start < post_guard_end < post_telemetry < first_control < id8_load
    assert "generation_local_submissions=%llu" in test_replay and "generation_local_waits=%llu" in test_replay
    assert "instruction_image_validation=PASS" in test_replay and "next_m9216_submission_ordinal=%llu" in test_replay
    same_shape_guard_start = test_replay.index("if (is_same_shape_id8_second) {")
    same_shape_guard_end = test_replay.index('throw std::runtime_error("same-shape ID8 second submission requires exact first ID8 and reused generation-1 resources")', same_shape_guard_start)
    same_shape_guard = test_replay[same_shape_guard_start:same_shape_guard_end]
    assert all(token in same_shape_guard for token in ("execution.size() == 12", "execution[10].record == execution[11].record",
        "first_id8_exact", "replay_post_recreate_control_wait_completed == 1", "replay_post_recreate_control_output_validated == 1",
        "generation1_execution_ordinal == 2", "generation1_submissions == 1", "generation1_wait_completions == 1",
        "m9216_bundle_reused", "m9216_hw_context_generation == 1", "m9216_kernel_generation == 1", "m9216_bo_bundle_generation == 1",
        "m9216_submissions == 8", "m9216_wait_completions == 8", "m9216_output_syncs == 8",
        "instruction_sha256", "q4_sha256", "q8_sha256", "first_id8.at(\"M\").get<size_t>() == 9216"))
    assert not any(token in same_shape_guard for token in ("bo_out->sync(", "capture_hash(buf_out", "(*kernel)(", "run.wait()", "make_unique<xrt::"))
    assert "FIRST_ID8_COMPLETED=1 FIRST_ID8_VALIDATED=1 FIRST_ID8_EXACT=1" in test_replay
    assert "SECOND_ID8_OUTPUT_SHA MATCH source_id=8 generation_local_ordinal=2" in test_replay
    assert "TRANSITION_STAGE=M9216_TO_SAME_M9216 previous_source_id=%llu next_source_id=%llu previous_M=%zu next_M=%zu" in test_replay
    assert "generation_local_ordinal=%llu" in test_replay and "POST_SAME_SHAPE_SECOND_ID8" in test_replay
    assert "same_shape_control" in driver_source and "POST_ID8_SAME_SHAPE_ID8_SECOND" in driver_source
    assert 'else if (same_shape_control) name = "id8-to-id8";' in driver_source and "frozen_same_shape" in test_replay
    assert "post_same_shape_second_id8_record" in backend_source and "ctx->replay_target_records != 0" in test_replay and "ctx->replay_probe_submitted != 0" in test_replay
    assert "ctx->replay_target_records != 0" in replay_body and "ctx->replay_probe_submitted != 0" in replay_body

    full_branch = replay_body.split("if (recreate_m9216_context_and_bo_bundle) {", 1)[1].split("} else if (recreate_m9216_hw_context)", 1)[0]
    assert "ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace)" in full_branch
    assert not any(marker in replay_body for marker in ("RELEASE_OLD_INSTRUCTION_BO_BEGIN", "RELEASE_OLD_HW_CONTEXT_BEGIN"))
    assert full_recreate_body.count("trace->m9216_hw_context_generation = trace->m9216_hw_context_recreate_count = 1") == 1
    assert full_recreate_body.count("trace->m9216_kernel_generation = trace->m9216_kernel_recreate_count = 1") == 1
    assert full_recreate_body.count("trace->m9216_bo_bundle_generation = trace->m9216_bo_bundle_recreate_count = 1") == 1
    assert "trace->m9216_bo_bundle_generation_submissions = 0" in full_recreate_body and "trace->m9216_bo_bundle_generation_wait_completions = 0" in full_recreate_body
    assert all(token in full_recreate_body for token in ("history_completed=", "history_outputs_validated=", "cumulative_m9216_ordinal=", "initialization_syncs=", "input_syncs=", "output_sentinel_syncs=", "output_syncs=", "old_bo_memory_groups=", "new_bo_memory_groups=", "old_kernel_group_ids=", "new_kernel_group_ids=", "pointer_reuse_is_not_generation_evidence", "static_connectivity=ARG1_MEM_TOPOLOGY_1_SRAM_ARG3_4_5_MEM_TOPOLOGY_0_HOST"))
    assert "get_memory_group()) ==" not in full_recreate_body and "group_id(1) ==" not in full_recreate_body
    assert "memcpy(" not in full_recreate_body
    assert all(token in ensure_body for token in ("kernel.group_id(1)", "kernel.group_id(3)", "kernel.group_id(4)", "kernel.group_id(5)", "std::make_unique<xrt::bo>", "map<void *>()", "map<float *>()", "sync_host_written_bo(*bi", "recreated M9216 instruction BO SHA mismatch"))
    group_positions = [ensure_body.index(f"kernel.group_id({arg})") for arg in (1, 3, 4, 5)]
    first_bo = ensure_body.index("std::make_unique<xrt::bo>")
    assert max(group_positions) < first_bo
    assert ensure_body.index("std::memcpy(mapped_instr") < ensure_body.index("sync_host_written_bo(*bi") < ensure_body.index("capture_hash(mapped_instr")
    assert "std::memcpy(ctx->q4_m9216_buf_q4" not in ensure_body and "std::memcpy(ctx->q4_m9216_buf_q8" not in ensure_body and "std::memcpy(ctx->q4_m9216_buf_out" not in ensure_body
    assert backend_source.count("ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(") == 4
    assert "recreate_m9216_context_and_bo_bundle" in replay_body and "FULL_RUNTIME_BUNDLE" in replay_body
    assert replay_body.index("ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace)") < replay_body.index("const size_t instruction_bytes = r.at")
    assert "trace->m9216_execution_ordinal = " not in context_recreate_body and "trace->m9216_submissions = " not in context_recreate_body
    assert "trace->m9216_wait_completions = " not in context_recreate_body and "trace->m9216_output_syncs = " not in context_recreate_body
    assert "trace->m9216_bo_bundle_generation = " not in context_recreate_body
    replay_body = backend_source.split("static int ggml_backend_xdna_replay_exact_history_impl(", 1)[1].split("\nint ggml_backend_xdna_diagnostic_replay(", 1)[0]
    assert replay_body.index("ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace)") < replay_body.index("const size_t instruction_bytes = r.at")
    assert "runtime_trace.m9216_hw_context_generation != 0" in replay_body and "runtime_trace.m9216_hw_context_recreate_count != 0" in replay_body

    no_readback_guard_pos = replay_body.index("EXECUTE_WAIT_NO_READBACK_GUARD=PASS")
    no_readback_start = replay_body.rindex("if (is_post_recreate_m4096_no_readback) {", 0, no_readback_guard_pos)
    no_readback_body = replay_body[no_readback_start:replay_body.index("bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE)", no_readback_guard_pos)]
    assert "EXECUTE_WAIT_NO_READBACK_GUARD=PASS" in no_readback_body
    assert "ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, false, true, &run, allow_live_run_control);" in no_readback_body
    assert "bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE)" not in no_readback_body and "ggml_backend_xdna_capture_hash(buf_out" not in no_readback_body
    assert 'const bool skip_output_sentinel_sync = (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) && is_post_recreate_m4096_no_readback;' in replay_body
    assert 'if (!skip_output_sentinel_sync)' in replay_body
    mismatch_at = replay_body.index('XDNA_CAPTURE_REPLAY HISTORICAL OUTPUT MISMATCH')
    mismatch_throw = replay_body.index('throw std::runtime_error("historical output size/SHA mismatch at submission "', mismatch_at)
    mismatch_block = replay_body[mismatch_at:mismatch_throw]
    assert all(token in mismatch_block for token in ('history_ordinal=', 'expected_size=', 'actual_size=', 'expected_sha=', 'actual_sha=', 'output_bo_identity=', 'output_map_identity=', 'output_capacity=', 'expected_output_bytes=', 'actual_validation_bytes=', 'FROM_DEVICE_sync_count=', 'mapped_read_count=', 'context_generation=', 'kernel_generation=', 'BO_generation=', 'BYTE_FINGERPRINT='))
    assert 'std::fflush(stderr);' in mismatch_block and mismatch_at < mismatch_throw
    assert 'output_bytes != r.at("output_bytes").get<size_t>() || output_sha != r.at("output_sha256").get<std::string>()' in replay_body
    assert 'actual_validation_bytes_exceed_output_capacity' in replay_body
    test_run_only_release_model_and_source_order()
    test_release_terminal_fallthrough_models()
    test_terminal_success_model()
    expected_outputs_gate = replay_body.index("ctx->replay_outputs_validated != expected_replay_outputs_validated")
    expected_outputs_assignment = replay_body.index("const uint64_t expected_replay_outputs_validated =")
    assert "(post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) ? 10u : ctx->replay_historical_records" in replay_body
    assert expected_outputs_assignment < expected_outputs_gate
    terminal_branch = replay_body.rindex("else if (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) {", 0, replay_body.index('throw std::runtime_error("execute/wait no-readback second-context-recreate final telemetry invariant failed")'))
    release_guard = replay_body.index("POST_WAIT_RESOURCE_RELEASE_GUARD=PASS")
    helper_call = replay_body.index("ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, false, true, &run, allow_live_run_control);")
    terminal_invariant = replay_body.index('throw std::runtime_error("execute/wait no-readback second-context-recreate final telemetry invariant failed")', terminal_branch)
    terminal_mode_set = replay_body.index("*terminal_mode = post_recreate_execute_wait_release_run_context_recreate_mode", terminal_invariant)
    terminal_return = replay_body.index("return XDNA_REPLAY_DIAGNOSTIC_TERMINAL;", terminal_mode_set)
    generic_probe = replay_body.index("else if (ctx->replay_prefix_probe_mode)", terminal_invariant)
    assert release_guard < helper_call < terminal_branch < terminal_invariant < terminal_mode_set < terminal_return < generic_probe
    ep1r_block = replay_body[terminal_branch:terminal_return]
    assert all(token in ep1r_block for token in ("replay_prefix_completed != 10", "replay_historical_records != 11", "replay_outputs_validated != 10", "generation1_wait_completions != 1", "generation1_output_syncs != 0", "m9216_hw_context_recreate_count != 2", "m9216_kernel_recreate_count != 2", "m9216_bo_bundle_recreate_count != 1", "replay_probe_submitted != 0", "replay_target_records != 0"))
    assert not any(token in ep1r_block for token in ("run.emplace(", "run->wait(", "(*kernel)(", "bo_out->sync("))
    assert '? "m4096-execute-wait-release-run-context-recreate" : (post_recreate_execute_wait_release_temp_context_recreate_mode ? "m4096-execute-wait-release-temp-context-recreate" : "m4096-execute-wait-no-readback-context-recreate")' in ep1r_block
    release_stage_start = replay_body.rindex("if (post_recreate_execute_wait_release_temp_context_recreate_mode) {", 0, release_guard)
    release_block = replay_body[release_stage_start:helper_call]
    assert "if (post_recreate_execute_wait_release_temp_context_recreate_mode)" in release_block
    assert "if (post_recreate_execute_wait_no_readback_context_recreate_mode)" not in release_block
    assert "ctx->replay_probe_submitted != 1" in replay_body
    assert 'throw std::runtime_error("prefix/probe final telemetry invariant failed")' in replay_body
    # Diagnostic terminal controls succeed only after their own telemetry gate; they have no probe by design.
    terminal_modes = ("double-recreate-no-execution", "m4096-program-only-context-recreate",
                      "id8-context-recreate-only", "m4096-context-recreate-to-id8", "m4096-execute-wait-no-readback-context-recreate",
                      "m4096-execute-wait-release-temp-context-recreate", "m4096-execute-wait-release-run-context-recreate")
    terminal_state = {"invariants_passed": True, "cleanup_complete": True, "replay_failed": False, "probe_submitted": 0}
    for mode in terminal_modes:
        assert diagnostic_terminal_success(mode, terminal_state), mode
        assert not diagnostic_terminal_success(mode, dict(terminal_state, invariants_passed=False)), mode
        assert not diagnostic_terminal_success(mode, dict(terminal_state, cleanup_complete=False)), mode
        assert not diagnostic_terminal_success(mode, dict(terminal_state, replay_failed=True)), mode
        assert mode in replay_body
    assert "XDNA_CAPTURE_REPLAY DIAGNOSTIC TERMINAL SUCCESS" not in replay_body
    callback_body = backend_source.split("int ggml_backend_xdna_diagnostic_replay(", 1)[1].split("\n}", 1)[0]
    helper_call_pos = callback_body.index("ggml_backend_xdna_replay_exact_history_impl(backend, plan_path, &terminal_mode)")
    result_gate_pos = callback_body.index("if (result != XDNA_REPLAY_DIAGNOSTIC_TERMINAL) return result;", helper_call_pos)
    terminal_marker_pos = callback_body.index("XDNA_CAPTURE_REPLAY DIAGNOSTIC TERMINAL SUCCESS mode=%s backend_result=PASS replay_failed=0", result_gate_pos)
    callback_return_pos = callback_body.index("return 0;", terminal_marker_pos)
    assert helper_call_pos < result_gate_pos < terminal_marker_pos < callback_return_pos
    assert "replay_prefix_probe_mode" not in callback_body
    assert replay_body.count("return XDNA_REPLAY_DIAGNOSTIC_TERMINAL;") == 5
    assert "catch (...)" in replay_body and 'std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY FAILED CLOSED: unknown error\\n");' in replay_body
    assert replay_body.index("catch (...)") < replay_body.index('XDNA_CAPTURE_REPLAY REPLAY FAILED CLOSED: unknown error')
    probe_state = {"completed": 10, "outputs_validated": 10, "planned": 10,
                   "probe_submitted": 1, "probe_waited": 1, "probe_output_validated": 1}
    assert normal_probe_terminal_success(probe_state)
    assert not normal_probe_terminal_success(dict(probe_state, probe_submitted=0))
    assert not normal_probe_terminal_success(dict(probe_state, probe_waited=0))
    assert not normal_probe_terminal_success(dict(probe_state, probe_output_validated=0))
    assert "ctx->replay_probe_wait_completed != 1" in replay_body
    assert "ctx->replay_probe_output_validated != (ctx->replay_target_records ? 0u : 1u)" in replay_body
    assert 'ctx->replay_probe_submitted != 1' in replay_body
    assert 'throw std::runtime_error("prefix/probe final telemetry invariant failed")' in replay_body
    assert "return result == 0 ? 0 : 4;" in driver_source

    assert manifest_sha(CAPTURE) == before
    assert hashlib.sha256((CAPTURE / "run_metadata.json").read_bytes()).hexdigest() == metadata_before
    print(f"XDNA capture prefix/probe tests: PASS (7 probe selections, {len(invalid)} rejections, second-recreate/no-readback models plus transition/lifecycle guards; dry-run only, no XRT)")


if __name__ == "__main__":
    main()
