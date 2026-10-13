#!/usr/bin/env python3
"""Static contract checks for diagnostics-only M9216 submission correlation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp"
HEADER = ROOT / "ggml/src/ggml-xdna/ggml-xdna-internal.h"
EVENTS = (
    "SUBMISSION_PREPARED",
    "KERNEL_CALL_RETURNED",
    "WAIT_ENTER",
    "WAIT_RETURN",
    "OUTPUT_SYNC_RETURN",
    "RESULT_COPY_RETURN",
)
CORRELATION_MARKERS = (
    b"XDNA_CORR", *(event.encode("ascii") for event in EVENTS),
    b"q4_native_sha256", b"q8_sha256", b"instruction_sha256",
    b"q4_native_ptr", b"activation_f32_ptr", b"q8_host_ptr",
    b"PROCESS_LOCAL_IDENTITY_ONLY",
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
    source = SOURCE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    signature = "static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216("
    start = source.rindex(signature)
    end = source.index("\nstatic ", start + len(signature))
    body = source[start:end]

    assert "m9216_diagnostic_ordinal" in header
    ordinal = header.index("uint64_t m9216_diagnostic_ordinal")
    assert diag_guarded(header, ordinal), "ordinal field leaked into production context"
    assert "const uint64_t capture_index = ++ctx->m9216_diagnostic_ordinal;" in body
    assert diag_guarded(source, source.index("const uint64_t capture_index = ++ctx->m9216_diagnostic_ordinal;"))
    sequence_counter = source.index("ggml_backend_xdna_m9216_submission_sequence{0}")
    assert diag_guarded(source, sequence_counter), "sequence counter leaked into production"
    node_parameter = source.index("const struct ggml_tensor * node, int graph_node_index,")
    assert diag_guarded(source, node_parameter), "tensor metadata parameter leaked into production"
    graph_index_arg = source.index(", i\n#endif\n                );")
    assert diag_guarded(source, graph_index_arg), "graph index inspection leaked into production"

    for token in ("ggml_backend_xdna_m9216_digest(",
                  "ggml_backend_xdna_m9216_correlation_event(",
                  "ggml_backend_xdna_m9216_correlation_prepared(",
                  "XDNA_CORR"):
        positions = [i for i in range(len(source)) if source.startswith(token, i)]
        assert positions, f"missing correlation token: {token}"
        assert all(diag_guarded(source, pos) for pos in positions), f"unguarded correlation token: {token}"

    event_sites = (
        body.index("ggml_backend_xdna_m9216_correlation_prepared("),
        *(body.index(f'"{event}"') for event in EVENTS[1:]),
    )
    assert list(event_sites) == sorted(event_sites), "correlation event order changed"
    assert body.count("ggml_backend_xdna_m9216_correlation_prepared(") == 1
    assert "submission_sequence_id" in body
    for event in EVENTS[1:]:
        pos = body.index(f'"{event}"')
        assert "submission_sequence_id" in body[max(0, pos - 180):pos]
    prepared_call = event_sites[0]
    assert "submission_sequence_id" in body[prepared_call:prepared_call + 220]

    native_hash = body.index("ggml_backend_xdna_m9216_digest(q4_rows, q4_rows_bytes, q4_native_sha)")
    q4_conversion = body.index("ggml_backend_xdna_q4_cache_apply(", native_hash)
    assert native_hash < q4_conversion, "Q4 native digest must precede transport conversion"
    q8_conversion = body.index("quantize_row_q8_K_ref(")
    q8_hash = body.index("ggml_backend_xdna_m9216_digest(\n                ctx->q4_m9216_buf_q8")
    q8_sync = body.index('sync_host_written_bo(bo_q8, "K=2560 M=9216 q8 activation')
    assert q8_conversion < q8_hash < q8_sync, "Q8 digest must follow quantization and precede sync"
    instruction_size = body.index("const size_t instruction_bytes =")
    instruction_hash = body.index("ctx->q4_m9216_instructions.data(), instruction_bytes, instruction_sha")
    kernel_call = body.index("auto run = kernel(")
    assert instruction_size < instruction_hash < kernel_call

    assert "q4_native_sha256=%s" in source
    assert "q4_transport_sha256=NOT_USED" in source
    assert "q8_sha256=%s" in source
    assert "instruction_sha256=%s" in source
    assert "identity_scope=PROCESS_LOCAL_IDENTITY_ONLY" in source
    assert "ggml_backend_xdna_diagnostic_capture_payload" not in source[source.index("static void ggml_backend_xdna_m9216_correlation_prepared"):source.index("#endif", source.index("static void ggml_backend_xdna_m9216_correlation_prepared"))]


def check_binary(path, expected):
    binary = Path(path)
    assert binary.is_file(), f"missing backend DLL: {binary}"
    data = binary.read_bytes()
    present = [marker for marker in CORRELATION_MARKERS if marker in data]
    if expected:
        assert present == list(CORRELATION_MARKERS), f"diagnostics DLL missing markers: {set(CORRELATION_MARKERS) - set(present)}"
    else:
        assert not present, f"production DLL retains correlation markers: {present}"


check_source()
if len(sys.argv) == 3:
    check_binary(sys.argv[1], False)
    check_binary(sys.argv[2], True)
elif len(sys.argv) != 1:
    raise SystemExit("usage: test-xdna-submission-correlation.py [production-build diagnostics-build]")
print("SUBMISSION_CORRELATION_STATIC_TEST=PASS")
