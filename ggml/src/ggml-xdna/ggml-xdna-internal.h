#pragma once

// Private XDNA backend/diagnostic implementation boundary. Not installed and
// not part of ggml-backend.h or a stable ABI. Q4 shares its registered XCLBIN
// and hardware context at device scope; kernels and BOs remain backend-owned.
#include "ggml-backend.h"
#include "ggml-xdna-sha256.h"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <filesystem>
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
#include <chrono>
#endif
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
#include "ggml-xdna-execution-authorization.h"
#include "ggml-xdna-causal-payload-capture.h"
#endif
void ggml_backend_xdna_format_sha256(const unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE], char out[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1]);
bool ggml_backend_xdna_hash_file_sha256(const std::string & path, unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE]);
void ggml_backend_xdna_sync_host_written_bo(xrt::bo & bo, const char * what);

struct ggml_backend_xdna_backend_context;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
struct ggml_backend_xdna_dtor_sentinel {
    const char * marker;
    ~ggml_backend_xdna_dtor_sentinel() noexcept;
};
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
bool ggml_backend_xdna_diagnostic_convert_q4_native(
        ggml_backend_xdna_backend_context * ctx, size_t rows, const void * native, void * transport);
#endif
enum class ggml_backend_xdna_kernel_profile {
    none,
    i16_256,
    q4k_q8k_k2560,
    // W2IN block profile: dedicated slot for the proven real npu1/AIE2
    // W2IN block kernel. It carries a different, smaller ABI than K=2560 and
    // advertises no scheduler ops; it is driven only by the explicit block probe.
    q4k_q8k_w2in_256,
};

// Share one registered Q4 XCLBIN and hardware context across every shape/backend instance.
struct ggml_backend_xdna_q4_shared_resources {
    std::mutex state_mutex;
    std::mutex resource_mutex;
    std::string xclbin_path;
    std::string xclbin_sha256;
    std::string xclbin_uuid;
    std::string registered_uuid;
    std::string failure_message;
    bool failure_latched = false;
    size_t xclbin_registration_count = 0;
    std::shared_ptr<xrt::xclbin> xclbin;
    std::shared_ptr<xrt::hw_context> hw_context;
};

struct ggml_backend_xdna_context {
    std::unique_ptr<xrt::device> xrt_device;
    std::shared_ptr<ggml_backend_xdna_q4_shared_resources> q4_shared =
        std::make_shared<ggml_backend_xdna_q4_shared_resources>();
    std::string description = "AMD XDNA NPU (XRT device 0)";
    std::string device_id;

    // supports_op() is a device-level callback, therefore every live
    // backend instance sharing this device must expose the same profile.
    ggml_backend_xdna_kernel_profile active_profile =
        ggml_backend_xdna_kernel_profile::none;
    size_t active_backend_count = 0;
    bool q4_m9216_available = false;
    bool q4_m1024_available = false;
    bool q4_m4096_available = false;
    bool q4_m8192_available = false;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    bool stop_after_xrt_module_preflight = false;
    bool identity_gate_failed = false;
#endif
};

struct ggml_backend_xdna_q4_cache_entry {
    size_t m = 0;
    size_t native_bytes = 0;
    size_t transport_bytes = 0;
    std::vector<uint8_t> transport;
};

enum class ggml_backend_xdna_q4_shape {
    m9216,
    m4096,
    m8192,
    m1024,
};

struct ggml_backend_xdna_q4_transport_cache {
    bool enabled = false;
    bool env_checked = false;

    std::unordered_map<const void *, ggml_backend_xdna_q4_cache_entry> entries;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    uint64_t hits = 0;
    uint64_t misses = 0;

    uint64_t hits_m9216 = 0;
    uint64_t hits_m4096 = 0;
    uint64_t hits_m8192 = 0;
    uint64_t hits_m1024 = 0;

    uint64_t blocks_converted = 0;
    uint64_t native_bytes_converted = 0;
    uint64_t transport_bytes_generated = 0;
    uint64_t transport_bytes_copied = 0;
#endif
};

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
struct ggml_backend_xdna_timers {
    double q4_conversion_ms = 0.0;
    double q4_cached_memcpy_ms = 0.0;
    double q4_sync_ms = 0.0;
    double q8_quant_ms = 0.0;
    double q8_sync_ms = 0.0;
    double xrt_submit_wait_ms = 0.0;
    double output_ms = 0.0;
};

struct ggml_backend_xdna_shape_timing {
    uint64_t ops = 0;
    double submit_ms = 0.0;
    double wait_ms = 0.0;
};
#endif

struct ggml_backend_xdna_batched_state {
    size_t rows = 0;
    std::string xclbin_path;
    std::vector<uint32_t> instructions;
    std::shared_ptr<xrt::xclbin> xclbin;
    std::shared_ptr<xrt::hw_context> hw_context;
    std::unique_ptr<xrt::kernel> kernel;
    std::unique_ptr<xrt::bo> bo_instr, bo_q4, bo_q8, bo_out;
    uint32_t group_instr = 0, group_q4 = 0, group_q8 = 0, group_out = 0;
    void * buf_instr = nullptr, * buf_q4 = nullptr, * buf_q8 = nullptr;
    float * buf_out = nullptr;
    uint32_t instruction_count = 0;
    bool bos_initialized = false;
    bool available = false;
};

struct ggml_backend_xdna_backend_context {
    ggml_backend_xdna_context * device_context = nullptr;
    uint64_t backend_instance_id = 0;

    ggml_backend_xdna_kernel_profile kernel_profile =
        ggml_backend_xdna_kernel_profile::none;

    bool device_profile_registered = false;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel primary_metadata_after{"MEMBER_DTOR_AFTER_PRIMARY_METADATA"};
#endif
    std::string xclbin_path;
    std::string instructions_path;
    std::vector<uint32_t> instructions;

    std::string kernel_name = "MLIR_AIE";
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel primary_metadata_before{"MEMBER_DTOR_BEFORE_PRIMARY_METADATA"};
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel primary_kernel_context_xclbin_after{"MEMBER_DTOR_AFTER_PRIMARY_KERNEL_CONTEXT_XCLBIN"};
#endif
    std::shared_ptr<xrt::xclbin> xclbin;
    std::shared_ptr<xrt::hw_context> hw_context;
    std::unique_ptr<xrt::kernel> kernel;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel primary_kernel_context_xclbin_before{"MEMBER_DTOR_BEFORE_PRIMARY_KERNEL_CONTEXT_XCLBIN"};
#endif

    // Persistent Q4_K K=2560 BOs remain private to this backend instance.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel primary_q4_bo_owners_after{"MEMBER_DTOR_AFTER_PRIMARY_Q4_BO_OWNERS"};
#endif
    std::unique_ptr<xrt::bo> q4_bo_instr;
    std::unique_ptr<xrt::bo> q4_bo_q4;
    std::unique_ptr<xrt::bo> q4_bo_q8;
    std::unique_ptr<xrt::bo> q4_bo_out;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel primary_q4_bo_owners_before{"MEMBER_DTOR_BEFORE_PRIMARY_Q4_BO_OWNERS"};
#endif

    void * q4_buf_instr = nullptr;
    void * q4_buf_q4 = nullptr;
    void * q4_buf_q8 = nullptr;
    float * q4_buf_out = nullptr;

    uint32_t q4_instruction_count = 0;
    bool q4_bos_initialized = false;

    // Optional full-shape Q4_K x Q8_K K=2560 M=9216 fast path.
    // This is a secondary artifact belonging to the same
    // q4k_q8k_k2560 backend profile. The existing per-row kernel
    // remains the fallback for every other M.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_metadata_after{"MEMBER_DTOR_AFTER_M9216_METADATA"};
#endif
    std::string q4_m9216_xclbin_path;
    std::string q4_m9216_instructions_path;
    std::vector<uint32_t> q4_m9216_instructions;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_metadata_before{"MEMBER_DTOR_BEFORE_M9216_METADATA"};
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_runtime_owners_after{"MEMBER_DTOR_AFTER_M9216_RUNTIME_OWNERS"};
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_kernel_context_xclbin_after{
        "MEMBER_DTOR_AFTER_M9216_KERNEL_CONTEXT_XCLBIN"
    };
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_hw_context_xclbin_after{
        "MEMBER_DTOR_AFTER_M9216_HW_CONTEXT_XCLBIN"
    };
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_xclbin_after{
        "MEMBER_DTOR_AFTER_M9216_XCLBIN"
    };
#endif
#endif
    std::shared_ptr<xrt::xclbin> q4_m9216_xclbin;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_xclbin_before{
        "MEMBER_DTOR_BEFORE_M9216_XCLBIN"
    };
    ggml_backend_xdna_dtor_sentinel m9216_hw_context_after{
        "MEMBER_DTOR_AFTER_M9216_HW_CONTEXT"
    };
#endif
    std::shared_ptr<xrt::hw_context> q4_m9216_hw_context;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_hw_context_before{
        "MEMBER_DTOR_BEFORE_M9216_HW_CONTEXT"
    };
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_hw_context_xclbin_before{
        "MEMBER_DTOR_BEFORE_M9216_HW_CONTEXT_XCLBIN"
    };
    ggml_backend_xdna_dtor_sentinel m9216_kernel_after{
        "MEMBER_DTOR_AFTER_M9216_KERNEL"
    };
#endif
    std::unique_ptr<xrt::kernel> q4_m9216_kernel;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_kernel_before{
        "MEMBER_DTOR_BEFORE_M9216_KERNEL"
    };
    ggml_backend_xdna_dtor_sentinel m9216_kernel_context_xclbin_before{
        "MEMBER_DTOR_BEFORE_M9216_KERNEL_CONTEXT_XCLBIN"
    };
    ggml_backend_xdna_dtor_sentinel m9216_bo_owners_after{
        "MEMBER_DTOR_AFTER_M9216_BO_OWNERS"
    };
#endif

    std::unique_ptr<xrt::bo> q4_m9216_bo_instr;
    std::unique_ptr<xrt::bo> q4_m9216_bo_q4;
    std::unique_ptr<xrt::bo> q4_m9216_bo_q8;
    std::unique_ptr<xrt::bo> q4_m9216_bo_out;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel m9216_bo_owners_before{
        "MEMBER_DTOR_BEFORE_M9216_BO_OWNERS"
    };
    ggml_backend_xdna_dtor_sentinel m9216_runtime_owners_before{"MEMBER_DTOR_BEFORE_M9216_RUNTIME_OWNERS"};
#endif
    uint32_t q4_m9216_group_instr = 0, q4_m9216_group_q4 = 0;
    uint32_t q4_m9216_group_q8 = 0, q4_m9216_group_out = 0;

    void * q4_m9216_buf_instr = nullptr;
    void * q4_m9216_buf_q4 = nullptr;
    void * q4_m9216_buf_q8 = nullptr;
    float * q4_m9216_buf_out = nullptr;

    uint32_t q4_m9216_instruction_count = 0;
    bool q4_m9216_bos_initialized = false;
    bool q4_m9216_available = false;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    uint64_t m9216_diagnostic_ordinal = 0;
    std::chrono::steady_clock::time_point diagnostics_epoch = std::chrono::steady_clock::now();
    uint64_t m9216_next_resource_generation = 0;
    uint64_t m9216_hw_context_generation = 0;
    uint64_t m9216_kernel_generation = 0;
    uint64_t m9216_instruction_bo_generation = 0;
    uint64_t m9216_shared_q4_bo_generation = 0;
    uint64_t m9216_q8_bo_generation = 0;
    uint64_t m9216_output_bo_generation = 0;
    bool q4_control_gate_checked = false;
    bool q4_control_enabled = false;
    bool q4_control_snapshot_valid = false;
    std::unique_ptr<uint8_t[]> q4_control_snapshot;
    size_t q4_control_snapshot_bytes = 0;
    char q4_control_snapshot_sha256[65] = {};
    char q4_control_source_instruction_sha256[65] = {};
    uint64_t q4_control_source_backend_id = 0;
    uint64_t q4_control_source_correlation_id = 0;
    uint64_t q4_control_source_xrt_sequence = 0;
    uint64_t q4_control_source_generations[7] = {};
    const void * q4_control_source_objects[7] = {};
    const void * q4_control_source_xclbin_identity = nullptr;
#endif
    bool exact_history_replay_failed = false;
    uint64_t replay_records_total = 0, replay_historical_records = 0, replay_target_records = 0;
    uint64_t replay_kernels_submitted = 0, replay_waits_completed = 0, replay_outputs_validated = 0;
    bool replay_prefix_probe_mode = false;
    uint64_t replay_prefix_planned = 0, replay_prefix_completed = 0, replay_prefix_outputs_validated = 0;
    uint64_t replay_probe_submitted = 0, replay_probe_wait_completed = 0, replay_probe_output_validated = 0;
    bool replay_post_recreate_control_mode = false;
    uint64_t replay_post_recreate_control_submitted = 0, replay_post_recreate_control_wait_completed = 0;
    uint64_t replay_post_recreate_control_output_validated = 0;
    bool replay_post_recreate_transition_mode = false;
    bool replay_post_recreate_same_shape_id7_mode = false;
    bool replay_post_recreate_context_recreate_id8_mode = false;
    bool replay_post_id8_context_recreate_only_mode = false;
    bool replay_post_recreate_no_readback_mode = false;
    bool replay_post_recreate_program_only_mode = false;
    uint64_t replay_program_only_stages = 0;
    uint64_t replay_transition_m4096_submitted = 0, replay_transition_m4096_wait_completed = 0, replay_transition_m4096_output_validated = 0;
    uint64_t replay_transition_id8_submitted = 0, replay_transition_id8_wait_completed = 0, replay_transition_id8_output_validated = 0;

    // persistent Q4_K weights: per-weight-tensor persistent Q4
    // transport BOs for the M=9216 path. Keyed by the q4 source tensor data
    // pointer, which is stable for the lifetime of an immutable weight tensor.
    // Each BO is created, filled and synced exactly once; every later token
    // reuses it with no repack and no full-weight host memcpy. The shared
    // q4_m9216_bo_q4 above remains the fallback whenever this path is
    // disabled, unavailable or over budget.
    struct q4_m9216_weight_bo_entry {
        std::unique_ptr<xrt::bo> bo;
        size_t mapped_bytes = 0;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        uint64_t diagnostic_generation_id = 0;
#endif
    };

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel persistent_weight_bo_map_after{"MEMBER_DTOR_AFTER_PERSISTENT_WEIGHT_BO_MAP"};
#endif
    std::unordered_map<const void *, q4_m9216_weight_bo_entry> q4_m9216_weight_bos;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel persistent_weight_bo_map_before{"MEMBER_DTOR_BEFORE_PERSISTENT_WEIGHT_BO_MAP"};
#endif
    bool   q4_weight_bo_enabled = true;
    bool   q4_weight_bo_env_checked = false;
    size_t q4_weight_bo_bytes = 0;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    uint64_t weight_bo_creates = 0;
    uint64_t weight_bo_reuses = 0;
    uint64_t weight_bo_failures = 0;
    uint64_t weight_bo_cap_hits = 0;
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel shape_states_after{"MEMBER_DTOR_AFTER_SHAPE_STATES"};
#endif
    ggml_backend_xdna_batched_state q4_m1024;
    ggml_backend_xdna_batched_state q4_m4096;
    ggml_backend_xdna_batched_state q4_m8192;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel shape_states_before{"MEMBER_DTOR_BEFORE_SHAPE_STATES"};
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    uint64_t accepted_q4_nodes = 0;
    uint64_t batched_m9216 = 0, batched_m4096 = 0, batched_m8192 = 0, batched_m1024 = 0;
    uint64_t fallback_rows = 0, xrt_runs = 0, run_waits = 0;
    uint64_t run_constructed = 0, run_completed = 0, run_destroyed = 0, run_live = 0;
    uint64_t q4_to_device = 0, q8_to_device = 0, output_from_device = 0;
    ggml_backend_xdna_timers timers;
    ggml_backend_xdna_shape_timing shape_timing[4];
#endif

    // Persistent Q4_K transport cache (experimental, off by default).
    // See GGML_XDNA_Q4K_TRANSPORT_CACHE. Owned by the backend/model-
    // context lifetime, matching q4_m9216_* and q4_m{1024,4096,8192} BOs.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel q4_cache_after{"MEMBER_DTOR_AFTER_Q4_CACHE"};
#endif
    ggml_backend_xdna_q4_transport_cache q4_cache;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_dtor_sentinel q4_cache_before{"MEMBER_DTOR_BEFORE_Q4_CACHE"};
#endif
    ~ggml_backend_xdna_backend_context();
};
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
std::string ggml_backend_xdna_capture_hash(const void * data, size_t bytes);
bool ggml_backend_xdna_diagnostic_capture_enabled(ggml_backend_xdna_backend_context * ctx);
bool ggml_backend_xdna_diagnostic_capture_can_submit(ggml_backend_xdna_backend_context * ctx);
bool ggml_backend_xdna_diagnostic_capture_session_failed();
uint64_t ggml_backend_xdna_diagnostic_capture_reserve(ggml_backend_xdna_backend_context * ctx);
bool ggml_backend_xdna_diagnostic_capture_prepare(ggml_backend_xdna_backend_context * ctx, uint64_t id, const char * node, int64_t m, int64_t k, int64_t rhs, uint32_t opcode, const void * instr, size_t instr_bytes, const void * q4, size_t q4_bytes, const void * q8, size_t q8_bytes, bool * semantic_target);
bool ggml_backend_xdna_diagnostic_capture_submitted(ggml_backend_xdna_backend_context * ctx, uint64_t id);
bool ggml_backend_xdna_diagnostic_capture_completed(ggml_backend_xdna_backend_context * ctx, uint64_t id, const void * output, size_t output_bytes);
void ggml_backend_xdna_diagnostic_capture_fail(ggml_backend_xdna_backend_context * ctx, uint64_t id, const char * phase, const char * kind, const char * message);
void ggml_backend_xdna_diagnostic_capture_payload(uint64_t index, const char * label, const void * data, size_t bytes);
bool ggml_backend_xdna_diagnostic_capture_payload_enabled();
bool ggml_backend_xdna_diagnostic_capture_validate_profile(ggml_backend_xdna_kernel_profile profile);
void ggml_backend_xdna_diagnostic_capture_fail_if_requested();
bool ggml_backend_xdna_causal_capture_reserve(uint64_t * sequence_id, bool * selected);
bool ggml_backend_xdna_causal_capture_prepare(uint64_t sequence_id, const ggml_xdna_causal_capture::submission_metadata & metadata, const void * q4_native, size_t q4_bytes, const void * q8, size_t q8_bytes);
bool ggml_backend_xdna_causal_capture_event(uint64_t sequence_id, const char * event);
bool ggml_backend_xdna_diagnostic_capture_initialize(ggml_backend_xdna_backend_context * ctx);
void ggml_backend_xdna_diagnostic_capture_mark_initialized(ggml_backend_xdna_backend_context * ctx);
void ggml_backend_xdna_diagnostic_capture_detach(ggml_backend_xdna_backend_context * ctx);
#endif
const char * ggml_backend_xdna_kernel_profile_name(ggml_backend_xdna_kernel_profile profile);
void ggml_backend_xdna_format_sha256(const unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE], char out[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1]);
void ggml_backend_xdna_sync_host_written_bo(xrt::bo & bo, const char * what);


#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
// Private direct-link entrypoint, compiled only when GGML_XDNA_BUILD_DIAGNOSTICS is ON.
GGML_BACKEND_API int ggml_backend_xdna_diagnostic_replay(ggml_backend_t backend, const char * plan_path);
GGML_BACKEND_API int ggml_backend_xdna_diagnostic_execute_causal_record(
        const ggml_xdna_causal::validated_execution_authorization & token, const ggml_xdna_causal::plan_identity & identity,
        ggml_backend_t backend, size_t rows, const void * q4_native, size_t q4_native_bytes,
        const void * q8_bytes, size_t q8_bytes_size, void * output, size_t output_bytes,
        const char * cell, const char * selected_prefix_ids, const char * mode, const char * pair, size_t execution_ordinal,
        size_t capture_id, const char * node_name, bool target,
        const char * q4_sha256, const char * q8_sha256);
#endif
