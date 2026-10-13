#!/usr/bin/env python3
"""Static causal executor safety checks. Never launches any executor/backend binary."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools/xdna-causal-executor.cpp"
RUNTIME = REPO / "tools/xdna-causal-runtime.cpp"
API = REPO / "tools/xdna-causal-runtime-api.h"
AUTH = REPO / "ggml/src/ggml-xdna/ggml-xdna-execution-authorization.h"
CMAKE = REPO / "ggml/src/ggml-xdna/CMakeLists.txt"
PRODUCTION = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "build-gv1-production-v1"


def function_body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth, i, quote, line_comment, block_comment, escape = 0, opening, None, False, False, False
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if line_comment:
            if ch == "\n": line_comment = False
        elif block_comment:
            if ch == "*" and nxt == "/": block_comment = False; i += 1
        elif quote:
            if escape: escape = False
            elif ch == "\\": escape = True
            elif ch == quote: quote = None
        elif ch == "/" and nxt == "/": line_comment = True; i += 1
        elif ch == "/" and nxt == "*": block_comment = True; i += 1
        elif ch in ("'", '"'): quote = ch
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return source[opening + 1:i]
        i += 1
    raise AssertionError(f"unclosed function body: {signature}")


def main():
    source = SOURCE.read_text(encoding="utf-8")
    runtime = RUNTIME.read_text(encoding="utf-8")
    diagnostics = (REPO / "ggml/src/ggml-xdna/ggml-xdna-diagnostics.cpp").read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")
    auth = AUTH.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")

    host_plan = function_body(source, "static causal_runtime_plan prepare_host_plan(")
    assert not any(call in host_plan for call in ("LoadLibrary", "ggml_backend_", "xrt::", "xrt_device", "xrt_kernel", "xrt::bo"))
    host_gate = function_body(source, "static int run_host_or_authorized(")
    assert "if (!opt.execute)" in host_gate
    assert host_gate.index("if (!opt.execute)") < host_gate.index("return 0;") < host_gate.index("authorize_execution(request)") < host_gate.index("run_authorized_cell(token")
    main_body = function_body(source, "int main(")
    assert "run_host_or_authorized(parse_cli" in main_body and "LoadLibrary" not in main_body

    cli_runtime = function_body(source, "static int run_authorized_cell(")
    assert "const validated_execution_authorization & token" in source[source.index("static int run_authorized_cell("):source.index("static int run_authorized_cell(") + 180]
    assert cli_runtime.index("enter_authorized_runtime(token, plan.identity") < cli_runtime.index("LoadLibraryW(")
    assert "ggml-backend.h" not in source and "ggml_backend_" not in source
    assert "xdna-causal-runtime.dll" in source

    runtime_gate = function_body(runtime, "static int run_backend_cell(")
    assert "validated_execution_authorization" in runtime[runtime.index("static int run_backend_cell("):runtime.index("static int run_backend_cell(") + 220]
    assert runtime_gate.index("enter_authorized_runtime(token, plan.identity") < runtime_gate.index("ggml_backend_load_all_from_path")
    assert runtime_gate.index("ggml_backend_load_all_from_path") < runtime_gate.index("ggml_backend_dev_by_name") < runtime_gate.index("ggml_backend_dev_init") < runtime_gate.index("ggml_backend_xdna_diagnostic_execute_causal_record")
    exported = function_body(runtime, "extern \"C\" __declspec(dllexport) int __cdecl xdna_gn1_run_authorized(")
    assert "validated_execution_authorization * token" in runtime[runtime.index("extern \"C\" __declspec(dllexport) int __cdecl xdna_gn1_run_authorized("):runtime.index("extern \"C\" __declspec(dllexport) int __cdecl xdna_gn1_run_authorized(") + 260]
    assert "run_backend_cell(*token, *plan" in exported
    submit = function_body(diagnostics, "int ggml_backend_xdna_diagnostic_execute_causal_record(")
    assert "validated_execution_authorization & token" in diagnostics[diagnostics.index("int ggml_backend_xdna_diagnostic_execute_causal_record("):diagnostics.index("int ggml_backend_xdna_diagnostic_execute_causal_record(") + 300]
    assert submit.index("authorization_matches(token, identity)") < submit.index("std::make_unique<xrt::bo>") < submit.index("(*kernel)(")

    assert "class validated_execution_authorization" in auth
    token_class = auth.split("class validated_execution_authorization", 1)[1].split("};", 1)[0]
    assert "private:" in token_class and "validated_execution_authorization(std::string" in token_class
    assert "request.execute_requested" in auth and "request.authorize_cell != expected" in auth
    assert "token.selected_ids() == plan.selected_ids" in auth
    assert "validated_execution_authorization *, causal_runtime_plan *" in api

    diagnostics = cmake.split("if(GGML_XDNA_BUILD_DIAGNOSTICS)", 1)[1].rsplit("endif()", 1)[0]
    assert "add_library(xdna-causal-runtime SHARED" in diagnostics
    assert "add_executable(xdna-causal-executor" in diagnostics
    host_target = diagnostics.split("add_executable(xdna-causal-executor", 1)[1].split("add_executable(xdna-causal-execution-gate-selftest", 1)[0]
    assert "target_link_libraries" not in host_target and "ggml-xdna)" not in host_target
    assert "ggml-base" not in host_target and "ggml " not in host_target
    assert "target_link_libraries(xdna-causal-runtime PRIVATE ggml-base ggml ggml-xdna)" in diagnostics
    assert "xdna-causal-execution-gate-selftest" in diagnostics

    assert not (PRODUCTION / "bin/xdna-causal-executor.exe").exists()
    assert not (PRODUCTION / "bin/xdna-causal-runtime.dll").exists()
    for binary in (PRODUCTION / "bin").glob("*"):
        if binary.is_file():
            data = binary.read_bytes()
            assert all(marker not in data for marker in (b"--prefix-ids", b"B_PREFIX_7_12", b"xdna-causal-runtime")), binary.name

    print("HOST_ONLY_FUNCTION_GRAPH_HAS_NO_BACKEND_ENTRY=PASS")
    print("CLI_EXECUTION_GATE=PASS RUNTIME_ENTRY_GATE=PASS GATES_INDEPENDENT=PASS")
    print("STATIC_EXECUTION_GATE_TEST=PASS PRODUCTION_ISOLATION=PASS")


if __name__ == "__main__":
    main()
