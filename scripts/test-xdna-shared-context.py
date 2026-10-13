#!/usr/bin/env python3
"""Static contract for shared Q4 XRT resource ownership."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "ggml/src/ggml-xdna/ggml-xdna-internal.h").read_text(encoding="utf-8")


def body(source, start, end):
    first = source.index(start)
    last = source.index(end, first + len(start))
    return source[first:last]


ACQUIRE = body(
    SOURCE,
    "static bool ggml_backend_xdna_acquire_q4_shared_kernel(",
    "static bool ggml_backend_xdna_init_kernel(",
)
SHAPE_INIT = body(
    SOURCE,
    "static bool ggml_backend_xdna_init_batched_shape(",
    "static ggml_backend_xdna_q4_shape ggml_backend_xdna_q4_shape_from_rows(",
)
M9216_INIT = body(
    SOURCE,
    "static bool\nggml_backend_xdna_init_q4k_q8k_k2560_m9216(",
    "static ggml_backend_t ggml_backend_xdna_device_init(",
)


def valid_contract(header=HEADER, acquire=ACQUIRE):
    return all((
        "std::mutex state_mutex;" in header,
        "std::mutex resource_mutex;" in header,
        "std::shared_ptr<xrt::xclbin> xclbin;" in header,
        "std::shared_ptr<xrt::hw_context> hw_context;" in header,
        "q4_shared =" in header,
        "std::lock_guard<std::mutex> lock(resources->resource_mutex);" in acquire,
        acquire.count("register_xclbin(") == 1,
        "resources->xclbin_registration_count == 1" in acquire,
        "resources->registered_uuid != resources->xclbin_uuid" in acquire,
        "get_xclbin_uuid().to_string() == resources->xclbin_uuid" in acquire,
        "resources->failure_latched = true;" in acquire,
        "if (resources->failure_latched)" in acquire,
    ))


assert valid_contract(), "shared-resource synchronization/identity/fail-closed contract is incomplete"
assert not valid_contract(header=HEADER.replace("std::mutex resource_mutex;", "")), "negative control missed missing resource mutex"
assert not valid_contract(acquire=ACQUIRE.replace("std::lock_guard<std::mutex> lock(resources->resource_mutex);", "", 1)), "negative control missed unlocked resource acquisition"
assert not valid_contract(acquire=ACQUIRE.replace("resources->registered_uuid != resources->xclbin_uuid", "false", 1)), "negative control missed missing UUID comparison"

PRIMARY_INIT = body(
    SOURCE,
    "static bool ggml_backend_xdna_init_kernel(",
    "// LLAMA-XDNA-NPU1-INTEGRATION-A1 - controlled single-block W2IN invocation.",
)
assert "ctx, \"PRIMARY\"" in PRIMARY_INIT and "ctx->kernel, resolved_kernel_name))" in PRIMARY_INIT
assert "state.xclbin, state.hw_context, state.kernel, resolved_kernel_name" in SHAPE_INIT
assert "ctx->q4_m9216_xclbin, ctx->q4_m9216_hw_context" in M9216_INIT
assert "create_if_missing,\n" not in SHAPE_INIT and "ctx->kernel_name.c_str(), false," in SHAPE_INIT
assert "ctx->kernel_name.c_str(), false," in M9216_INIT
assert "register_xclbin(" not in SHAPE_INIT and "register_xclbin(" not in M9216_INIT
assert "make_unique<xrt::xclbin>" not in SOURCE and "make_unique<xrt::hw_context>" not in SOURCE
assert "GGML_XDNA_XRT_TRACE(ctx, \"M9216\", \"BO_SET_CREATE\"" in SOURCE
assert "\"BO_SET_CREATE\", \"ALLOCATE_AND_MAP\", \"PASS\"" in SOURCE

# Device profile state and backend destruction share the same serialization lock.
assert "std::lock_guard<std::mutex> device_state_lock(device_ctx->q4_shared->state_mutex);" in SOURCE
assert "std::unique_lock<std::mutex> device_state_lock;" in SOURCE
assert "device_ctx->q4_shared->state_mutex);" in SOURCE
print("XDNA_SHARED_Q4_XRT_CONTEXT=PASS one_registration=YES one_context=YES fail_closed=YES negative_controls=3")
