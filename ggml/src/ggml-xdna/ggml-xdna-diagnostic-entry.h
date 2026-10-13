#pragma once

// Private diagnostics tool entrypoint; not installed or a public ABI.
#include <cstddef>
#include "ggml-backend.h"
#include "ggml-xdna-execution-authorization.h"

GGML_BACKEND_API int ggml_backend_xdna_diagnostic_replay(ggml_backend_t backend, const char * plan_path);
GGML_BACKEND_API int ggml_backend_xdna_diagnostic_execute_causal_record(
        const ggml_xdna_causal::validated_execution_authorization & token, const ggml_xdna_causal::plan_identity & identity,
        ggml_backend_t backend, size_t rows, const void * q4_native, size_t q4_native_bytes,
        const void * q8_bytes, size_t q8_bytes_size, void * output, size_t output_bytes,
        const char * cell, const char * selected_prefix_ids, const char * mode, const char * pair, size_t execution_ordinal,
        size_t capture_id, const char * node_name, bool target,
        const char * q4_sha256, const char * q8_sha256);
