#include "xdna-causal-runtime-api.h"
#include "../ggml/src/ggml-xdna/ggml-xdna-diagnostic-entry.h"
#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static void teardown_marker(const char * event) {
    std::fprintf(stderr, "XDNA_CAUSAL_TEARDOWN event=%s\n", event);
    std::fflush(stderr);
}
#endif
static void set_env(const char * name, const std::filesystem::path & path) {
    if (_putenv_s(name, path.string().c_str()) != 0) throw std::runtime_error(std::string("cannot set ") + name);
}

static int run_backend_cell(const ggml_xdna_causal::validated_execution_authorization & token, causal_runtime_plan & plan, const wchar_t * executable_dir) {
    return ggml_xdna_causal::enter_authorized_runtime(token, plan.identity, [&]() -> int {
        if (!executable_dir || plan.records.empty() || plan.records.size() != plan.prefix_count + 1) return 1;
        const size_t target_id = plan.identity.pair == "A" ? 2 : 13;
        for (size_t i = 0; i < plan.records.size(); ++i) {
            const auto & record = plan.records[i];
            if (i < plan.prefix_count) {
                if (record.target || record.capture_id != plan.identity.selected_ids[i] || record.sequence != i + 1) return 1;
            } else if (!record.target || record.capture_id != target_id || record.sequence != i + 1 ||
                       record.m != 9216 || record.k != 2560 || record.node != "ffn_gate-4") return 1;
        }
        if ((plan.identity.mode == "CLEAN" && plan.prefix_count != 0) ||
            (plan.identity.mode == "CAPTURE_PREFIX" && plan.prefix_count != plan.identity.selected_ids.size())) return 1;

        set_env("GGML_XDNA_Q4K_RUNTIME_XCLBIN", plan.xclbin);
        set_env("GGML_XDNA_Q4K_M1024_INSTS", plan.factory_support_paths.at(1024));
        set_env("GGML_XDNA_Q4K_M4096_INSTS", plan.instruction_paths.at(4096));
        set_env("GGML_XDNA_Q4K_M8192_INSTS", plan.factory_support_paths.at(8192));
        set_env("GGML_XDNA_Q4K_M9216_INSTS", plan.instruction_paths.at(9216));

        const std::string backend_dir = std::filesystem::path(executable_dir).string();
        ggml_backend_load_all_from_path(backend_dir.c_str());
        ggml_backend_dev_t device = ggml_backend_dev_by_name("XDNA0");
        if (!device) throw std::runtime_error("XDNA backend unavailable");
        const std::string params = "profile=q4k_q8k_k2560;" + plan.xclbin.string();
        ggml_backend_t backend = ggml_backend_dev_init(device, params.c_str());
        if (!backend) throw std::runtime_error("XDNA backend initialization failed");
        std::fprintf(stderr, "XDNA_CAUSAL_EVENT event=CELL_START cell=%s selected_prefix_ids=%s mode=%s pair=%s\n",
            token.cell().c_str(), plan.selected_csv.c_str(), token.mode().c_str(), token.pair().c_str());
        int result = 0;
        for (auto & record : plan.records) {
            result = ggml_backend_xdna_diagnostic_execute_causal_record(token, plan.identity, backend, record.m, record.q4.data(), record.q4.size(),
                record.q8.data(), record.q8.size(), record.output.data(), record.output.size() * sizeof(float),
                token.cell().c_str(), plan.selected_csv.c_str(), token.mode().c_str(), token.pair().c_str(), record.sequence,
                record.capture_id, record.node.c_str(), record.target, record.q4_sha.c_str(), record.q8_sha.c_str());
            if (result != 0) break;
        }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        teardown_marker("BACKEND_FREE_ENTER");
#endif
        ggml_backend_free(backend);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        teardown_marker("BACKEND_FREE_RETURN");
#endif
        if (result != 0) throw std::runtime_error("submission failed closed; no later record was submitted");
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        teardown_marker("BEFORE_CELL_COMPLETE");
#endif
        std::fprintf(stderr, "XDNA_CAUSAL_EVENT event=CELL_COMPLETE cell=%s selected_prefix_ids=%s mode=%s pair=%s submissions=%zu output_reference_validation=NOT_AVAILABLE\n",
            token.cell().c_str(), plan.selected_csv.c_str(), token.mode().c_str(), token.pair().c_str(), plan.records.size());
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        std::fflush(stderr);
#endif
        return 0;
    });
}

extern "C" __declspec(dllexport) int __cdecl xdna_gn1_run_authorized(
        const ggml_xdna_causal::validated_execution_authorization * token, causal_runtime_plan * plan, const wchar_t * executable_dir) {
    if (!token || !plan) return 1;
    try {
        return run_backend_cell(*token, *plan, executable_dir);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "XDNA_CAUSAL_RUNTIME_FAIL_CLOSED: %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "XDNA_CAUSAL_RUNTIME_FAIL_CLOSED: unknown error\n");
        return 1;
    }
}
