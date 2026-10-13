#!/usr/bin/env python3
"""Host-only integration tests for xdna-capture-replay.exe --dry-run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def load_planner():
    path = Path(__file__).with_name("xdna-capture-replay-plan.py")
    spec = importlib.util.spec_from_file_location("capture_replay_plan_for_driver_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load XDNA capture replay planner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    planner = load_planner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--br1-capture-dir", type=Path, help="explicit external CF1 capture directory")
    parser.add_argument("--xclbin-path", type=Path, required=True, help="explicit frozen external XCLBIN used by the synthetic replay fixture")
    args = parser.parse_args()
    args.xclbin_path = args.xclbin_path.resolve(strict=True)
    if not args.xclbin_path.is_file() or hashlib.sha256(args.xclbin_path.read_bytes()).hexdigest() != planner.FROZEN_XCLBIN_SHA256:
        raise RuntimeError("explicit XCLBIN does not match the frozen replay identity")
    if args.br1_capture_dir is not None:
        args.br1_capture_dir = args.br1_capture_dir.resolve(strict=True)
        if not args.br1_capture_dir.is_dir():
            raise RuntimeError("--br1-capture-dir must name a directory")
    build = Path(os.environ.get("GGML_XDNA_TEST_BUILD_DIR", Path(__file__).resolve().parents[1] / "build-xdna"))
    exe = build / "bin" / "xdna-capture-replay.exe"
    if not exe.is_file():
        raise RuntimeError(f"build required first: {exe}")
    cmake = (Path(__file__).resolve().parents[1] / "ggml" / "src" / "ggml-xdna" / "CMakeLists.txt").read_text(encoding="utf-8")
    driver = (Path(__file__).resolve().parents[1] / "tools" / "xdna-capture-replay.cpp").read_text(encoding="utf-8")
    backend_dir = Path(__file__).resolve().parents[1] / "ggml" / "src" / "ggml-xdna"
    production_backend = (backend_dir / "ggml-xdna.cpp").read_text(encoding="utf-8")
    backend = (backend_dir / "ggml-xdna-diagnostics.cpp").read_text(encoding="utf-8")
    diag_target = cmake.split("if(GGML_XDNA_BUILD_DIAGNOSTICS)", 1)[1].rsplit("endif()", 1)[0]
    assert 'option(GGML_XDNA_BUILD_DIAGNOSTICS "Build XDNA replay and capture diagnostics" OFF)' in cmake
    assert "target_sources(ggml-xdna PRIVATE ggml-xdna-diagnostics.cpp)" in diag_target
    assert "target_link_libraries(xdna-capture-replay PRIVATE ggml-base ggml ggml-xdna)" in diag_target
    assert "ggml_backend_xdna_replay_exact_history_impl" not in production_backend
    assert '"ggml_backend_xdna_replay_exact_history"' not in production_backend
    assert "ggml_backend_xdna_diagnostic_replay(backend, temp_plan.string().c_str())" in driver
    assert "ggml_backend_reg_get_proc_address" not in driver
    assert driver.index("if (options.dry_run)") < driver.index("ggml_backend_load_all_from_path") and "xrt::" not in driver
    snapshot = backend.split("static void ggml_backend_xdna_replay_log_runtime_snapshot(", 1)[1].split("\n}", 1)[0]
    assert all(token not in snapshot for token in (".address(", "->address(", ".size(", "->size(", ".map<", ".sync(", ".group_id(", ".wait("))
    replay = backend.split("static int ggml_backend_xdna_replay_exact_history_impl(", 1)[1].split("\nint ggml_backend_xdna_diagnostic_replay(", 1)[0]
    ordered = ("sync_host_written_bo(*bo_instr", "sync_host_written_bo(*bo_q4", "sync_host_written_bo(*bo_q8",
               "sync_host_written_bo(*bo_out", "log_runtime_snapshot(\"PRE_SUBMIT", "run.emplace((*kernel)",
               "run->wait()", "bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE)", "log_runtime_snapshot(\"POST_OUTPUT")
    assert [replay.index(token) for token in ordered] == sorted(replay.index(token) for token in ordered)
    assert "device_bo_addresses=NOT_QUERIED" in snapshot and "SAME_AS_PREVIOUS" in backend
    setup = backend.split("static bool ggml_backend_xdna_replay_ensure_m9216_bos(", 1)[1].split("\n}", 1)[0]
    setup_order = ("kernel.group_id(1)", "kernel.group_id(3)", "kernel.group_id(4)", "kernel.group_id(5)",
                   "auto bi = std::make_unique<xrt::bo>", "auto bq4 = std::make_unique<xrt::bo>",
                   "auto bq8 = std::make_unique<xrt::bo>", "auto bout = std::make_unique<xrt::bo>")
    positions = [setup.index(token) for token in setup_order]
    assert positions == sorted(positions) and setup.count("kernel.group_id(") == 4
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("GGML_XDNA_"):
            env.pop(key)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        valid = root / "valid"
        planner._fixture(valid, args.xclbin_path, shapes=(1024,))
        result = subprocess.run([str(exe), str(valid), "--dry-run"], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        assert "XDNA_CAPTURE_REPLAY_DRY_RUN=PASS" in result.stdout and "target=ffn_gate-4" in result.stdout
        assert "submission_count=2" in result.stdout and "historical=1" in result.stdout
        assert "| 1 | node-1024 | 1024 | 4 | 1556480 | 2920 | 4096 | NO |" in result.stdout
        assert "XDNA_CAPTURE_REPLAY RUNTIME SNAPSHOT" not in result.stderr and "XDNA_CAPTURE_REPLAY PRE-PROBE RUNTIME STATE" not in result.stderr

        cases = (
            ("schema", lambda e, m, r: m.update(schema_version=2)),
            ("wrong xclbin", lambda e, m, r: m.update(xclbin_sha256="0" * 64)),
            ("unsupported shape", lambda e, m, r: e[0].update(M=2048)),
            ("unsafe artifact path", lambda e, m, r: e[0].update(q4_file="../escape.bin")),
            ("bad artifact SHA", lambda e, m, r: e[0].update(q4_sha256="0" * 64)),
            ("wrong size", lambda e, m, r: e[0].update(instruction_bytes=e[0]["instruction_bytes"] + 4)),
            ("nonterminal target", lambda e, m, r: e.append({"event": "PREPARED", "submission_id": 3})),
            ("missing target", lambda e, m, r: e.pop()),
            ("target marked historical", lambda e, m, r: e[-1].update(submitted=True)),
        )
        for label, mutate in cases:
            capture = root / label.replace(" ", "-")
            events = planner._fixture(capture, args.xclbin_path, shapes=(1024,))
            metadata = json.loads((capture / "run_metadata.json").read_text(encoding="utf-8"))
            mutate(events, metadata, [])
            (capture / "manifest.jsonl").write_text(
                "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events), encoding="utf-8")
            (capture / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            result = subprocess.run([str(exe), str(capture), "--dry-run"], capture_output=True, text=True, env=env)
            assert result.returncode != 0 and "XDNA_CAPTURE_REPLAY_DRY_RUN=PASS" not in result.stdout, (label, result.stdout, result.stderr)

        capture = root / "missing-artifact"
        events = planner._fixture(capture, args.xclbin_path, shapes=(1024,))
        (capture / events[0]["q4_file"]).unlink()
        result = subprocess.run([str(exe), str(capture), "--dry-run"], capture_output=True, text=True, env=env)
        assert result.returncode != 0 and "XDNA_CAPTURE_REPLAY_DRY_RUN=PASS" not in result.stdout

        frozen_cf1 = args.br1_capture_dir
        if frozen_cf1 is None:
            print("BR1_REPLAY_TRANSITION_TESTS=NOT_RUN_EXTERNAL_FIXTURE_REQUIRED")
        else:
            if not frozen_cf1.is_dir():
                raise RuntimeError("external BR1 capture directory is unavailable")
            result = subprocess.run([str(exe), str(frozen_cf1), "--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
                                    "--post-recreate-transition-control", "m4096-context-recreate-to-id8",
                                    "--recreate-m9216-context-and-bo-bundle-before-probe", "--dry-run"], capture_output=True, text=True, env=env)
            assert result.returncode == 0, result.stderr
            assert "XDNA_CAPTURE_REPLAY_M4096_CONTEXT_RECREATE_TO_ID8_DRY_RUN=PASS" in result.stdout
            assert "<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n7\n<POST_RECREATE_ID7_FIRST>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_ID7>\n8\n<POST_CONTEXT_RECREATE_ID8_M9216>" in result.stdout
            assert "| 11 | 7 | z-2 | 4096 | POST_RECREATE_ID7_FIRST |" in result.stdout
            assert "| 12 | 8 | ffn_gate-2 | 9216 | POST_CONTEXT_RECREATE_ID8_M9216 |" in result.stdout
            assert "xrt_initialized=NO" in result.stdout
            run_only_release = subprocess.run([str(exe), str(frozen_cf1), "--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
                "--post-recreate-transition-control", "m4096-execute-wait-release-run-context-recreate", "--dry-run"], capture_output=True, text=True, env=env)
            assert run_only_release.returncode == 0, (run_only_release.stdout, run_only_release.stderr)
            assert "XDNA_CAPTURE_REPLAY_RUN_ONLY_RELEASE_SECOND_RECREATE_DRY_RUN=PASS" in run_only_release.stdout
            assert "post_recreate_transition_control=m4096-execute-wait-release-run-context-recreate" in run_only_release.stdout
            assert "executions=11" in run_only_release.stdout and "xrt_initialized=NO" in run_only_release.stdout
            assert "RUN_ONLY_RELEASE_GUARD=PLANNED" in run_only_release.stdout
            expected_run_only = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n7\n\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<EXECUTE_WAIT_NO_READBACK_ID7_M4096>\n<RELEASE_COMPLETED_EXPERIMENT_RUN_ONLY>\n<RETAIN_TEMP_M4096_EXECUTION_RESOURCES_AND_MAP_ALIASES>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_RUN_RELEASE>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end"
            assert expected_run_only in run_only_release.stdout, run_only_release.stdout
            assert "RELEASE_TEMP_M4096_EXECUTION_RESOURCES" not in run_only_release.stdout and "PROBE_RECORD" not in run_only_release.stdout
            no_execution = subprocess.run([str(exe), str(frozen_cf1), "--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
                "--post-recreate-transition-control", "double-recreate-no-execution", "--dry-run"], capture_output=True, text=True, env=env)
            assert no_execution.returncode == 0, (no_execution.stdout, no_execution.stderr)
            assert "XDNA_CAPTURE_REPLAY_SECOND_RECREATE_NO_EXECUTION_DRY_RUN=PASS" in no_execution.stdout
            assert "executions=10" in no_execution.stdout and "xrt_initialized=NO" in no_execution.stdout
            assert "TRANSITION_STAGE=SECOND_CONTEXT_RECREATE_NO_EXECUTION generation1_execution_count=0 previous_experimental_source=NONE second_recreation_requested=YES" in no_execution.stdout
            assert "NO_EXECUTION_RECREATE_GUARD=PASS" in no_execution.stdout
            expected_no_execution = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_NO_EXECUTION>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end"
            assert expected_no_execution in no_execution.stdout, no_execution.stdout
            assert "POST_RECREATE_ID8_FIRST" not in no_execution.stdout and "PROBE_RECORD" not in no_execution.stdout and "probe_id=" not in no_execution.stdout
            assert "target_submission_id=13" not in no_execution.stdout and "POST_CONTEXT_RECREATE_ID8_M9216" not in no_execution.stdout
            program_only = subprocess.run([str(exe), str(frozen_cf1), "--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
                "--post-recreate-transition-control", "m4096-program-only-context-recreate", "--dry-run"], capture_output=True, text=True, env=env)
            assert program_only.returncode == 0, (program_only.stdout, program_only.stderr)
            assert "XDNA_CAPTURE_REPLAY_PROGRAM_ONLY_SECOND_RECREATE_DRY_RUN=PASS" in program_only.stdout
            assert "post_recreate_transition_control=m4096-program-only-context-recreate" in program_only.stdout
            assert "executions=10" in program_only.stdout and "xrt_initialized=NO" in program_only.stdout
            expected_program_only = "replay_sequence_begin\n1\n2\n3\n4\n5\n6\n7\n8\n8\n8\n\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<PROGRAM_ONLY_M4096_TO_SECOND_CONTEXT_RECREATE source_id=7 M=4096>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_PROGRAM_ONLY>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end"
            assert expected_program_only in program_only.stdout, program_only.stdout
            assert "PROGRAM_ONLY_CONTEXT_RECREATE_GUARD=PASS" not in program_only.stdout and "PROBE_RECORD" not in program_only.stdout
            rejected_program_only = subprocess.run([str(exe), str(frozen_cf1), "--prefix-count", "8", "--inject-id", "8", "--inject-id", "8",
                "--probe-id", "11", "--post-recreate-transition-control", "m4096-program-only-context-recreate", "--dry-run"], capture_output=True, text=True, env=env)
            assert rejected_program_only.returncode != 0 and "DRY_RUN=PASS" not in rejected_program_only.stdout
    print("XDNA capture replay synthetic dry-run tests: PASS (valid plan, 10 fail-closed cases; dry-run exits before backend loading/call)")
    if args.br1_capture_dir is not None:
        print("BR1_REPLAY_TRANSITION_TESTS=PASS (frozen transition and no-execution sequences)")


if __name__ == "__main__":
    main()
