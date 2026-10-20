#include "ggml-xdna.h"

#include "ggml-backend-impl.h"
#include "ggml-quants.h"
#if defined(GGML_XDNA_RUNTIME_TRACE)
#include "ggml-impl.h"
#endif
#include "ggml-xdna-sha256.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <thread>

#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <atomic>
#include <mutex>

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winver.h>
#endif
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include "ggml-xdna-internal.h"

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
#define GGML_XDNA_DIAG_ONLY(...) __VA_ARGS__
#else
#define GGML_XDNA_DIAG_ONLY(...) ((void) 0);
#endif

// Bound submitted ERT waits; a non-COMPLETED state propagates as failure, never fallback.
static constexpr std::chrono::milliseconds GGML_XDNA_ERT_WAIT_LIMIT{5000};

// XRT leaves timed-out runs submitted; abort synchronously before releasing them.
static ert_cmd_state ggml_backend_xdna_wait_run(
        xrt::run & run, const std::chrono::milliseconds & timeout) {
    ert_cmd_state state;
    try {
        state = run.wait(timeout);
    } catch (...) {
        (void) run.abort();
        throw;
    }
    if (state == ERT_CMD_STATE_TIMEOUT) {
        (void) run.abort();
    }
    return state;
}


#if defined(GGML_XDNA_RUNTIME_TRACE)
struct ggml_backend_xdna_runtime_trace_node {
    uint64_t graph_id;
    int node_index;
    uint64_t scheduler_graph_id;
    uint64_t scheduler_split_id;
    uint64_t scheduler_dispatch_id;
    int global_node_index;
    int local_split_node_index;
    int split_global_node_start;
    int split_global_node_end_exclusive;
    const char * node_name;
    const char * ggml_op;
    int64_t m;
    int64_t k;
    int64_t rhs;
};

static std::atomic<uint64_t> ggml_backend_xdna_runtime_trace_graph_id{1};
static std::atomic<uint64_t> ggml_backend_xdna_runtime_trace_submission_id{1};
static std::mutex ggml_backend_xdna_runtime_trace_mutex;

static bool ggml_backend_xdna_runtime_trace_enabled() noexcept {
    const char * value = std::getenv("GGML_XDNA_RUNTIME_TRACE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static uint64_t ggml_backend_xdna_runtime_trace_allocate_id(std::atomic<uint64_t> & counter) noexcept {
    uint64_t current = counter.load(std::memory_order_relaxed);
    while (current != 0 && current != UINT64_MAX) {
        if (counter.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
            return current;
        }
    }
    return 0;
}

static uint64_t ggml_backend_xdna_runtime_trace_next_graph_id() noexcept {
    return ggml_backend_xdna_runtime_trace_allocate_id(ggml_backend_xdna_runtime_trace_graph_id);
}

static uint64_t ggml_backend_xdna_runtime_trace_next_submission_id() noexcept {
    return ggml_backend_xdna_runtime_trace_allocate_id(ggml_backend_xdna_runtime_trace_submission_id);
}

static void ggml_backend_xdna_runtime_trace_json_string(std::string & out, const char * value) {
    if (value == nullptr) {
        out += "null";
        return;
    }
    static constexpr char hex[] = "0123456789abcdef";
    out.push_back('"');
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p != 0; ++p) {
        switch (*p) {
            case '"':
            case '\\':
                out.push_back('\\');
                out.push_back(static_cast<char>(*p));
                break;
            case '\b': out.push_back('\\'); out.push_back('b'); break;
            case '\f': out.push_back('\\'); out.push_back('f'); break;
            case '\n': out.push_back('\\'); out.push_back('n'); break;
            case '\r': out.push_back('\\'); out.push_back('r'); break;
            case '\t': out.push_back('\\'); out.push_back('t'); break;
            default:
                if (*p < 0x20) {
                    out.push_back('\\'); out.push_back('u'); out.push_back('0'); out.push_back('0');
                    out.push_back(hex[*p >> 4]); out.push_back(hex[*p & 0x0f]);
                } else {
                    out.push_back(static_cast<char>(*p));
                }
        }
    }
    out.push_back('"');
}

static void ggml_backend_xdna_runtime_trace_field(
        std::string & out, const char * name, const char * value) {
    ggml_backend_xdna_runtime_trace_json_string(out, name);
    out.push_back(':');
    ggml_backend_xdna_runtime_trace_json_string(out, value);
}

static void ggml_backend_xdna_runtime_trace_number(
        std::string & out, const char * name, uint64_t value) {
    ggml_backend_xdna_runtime_trace_json_string(out, name);
    out.push_back(':');
    out += std::to_string(value);
}
static void ggml_backend_xdna_runtime_trace_nullable_number(
        std::string & out, const char * name, uint64_t value, bool present) {
    ggml_backend_xdna_runtime_trace_json_string(out, name);
    out.push_back(':');
    out += present ? std::to_string(value) : "null";
}

static void ggml_backend_xdna_scheduler_trace_node_event(
        const ggml_backend_sched_trace_context * context, uint64_t trace_graph_id,
        int trace_node_index, const ggml_tensor * node) noexcept {
    try {
        std::string line = "{\"schema\":\"ggml-xdna-scheduler-trace-v1\",\"event\":\"XDNA_NODE_EXECUTE\"";
        line.push_back(',');
        const bool joined = context != nullptr;
        ggml_backend_xdna_runtime_trace_nullable_number(line, "scheduler_graph_id", joined ? context->scheduler_graph_id : 0, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "scheduler_split_id", joined ? context->scheduler_split_id : 0, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "scheduler_dispatch_id", joined ? context->scheduler_dispatch_id : 0, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "global_node_index", joined ? static_cast<uint64_t>(context->global_node_index_base + trace_node_index) : 0, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "local_split_node_index", joined ? static_cast<uint64_t>(context->local_split_node_index_base + trace_node_index) : 0, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "split_global_node_start", joined ? static_cast<uint64_t>(context->split_global_node_start) : 0, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "split_global_node_end_exclusive", joined ? static_cast<uint64_t>(context->split_global_node_end_exclusive) : 0, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "trace_graph_id", trace_graph_id); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "trace_node_index", static_cast<uint64_t>(trace_node_index)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "node_name", ggml_get_name(node)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "ggml_op", ggml_op_name(node->op)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "backend", "XDNA");
        line.push_back('}');
        ggml_backend_sched_trace_emit_json(line.c_str());
    } catch (...) {
        // Trace allocation or I/O failure must not change XDNA execution semantics.
    }
}

static void ggml_backend_xdna_scheduler_trace_submission_event(
        const ggml_backend_xdna_runtime_trace_node & node, uint64_t submission_id,
        const char * kernel_name, const char * path, int64_t row_index) noexcept {
    try {
        std::string line = "{\"schema\":\"ggml-xdna-scheduler-trace-v1\",\"event\":\"XDNA_SUBMISSION_LINK\"";
        line.push_back(',');
        const bool joined = node.scheduler_graph_id != 0 && node.scheduler_split_id != 0 && node.scheduler_dispatch_id != 0;
        ggml_backend_xdna_runtime_trace_nullable_number(line, "scheduler_graph_id", node.scheduler_graph_id, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "scheduler_split_id", node.scheduler_split_id, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "scheduler_dispatch_id", node.scheduler_dispatch_id, joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "global_node_index", static_cast<uint64_t>(node.global_node_index), joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "local_split_node_index", static_cast<uint64_t>(node.local_split_node_index), joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "split_global_node_start", static_cast<uint64_t>(node.split_global_node_start), joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "split_global_node_end_exclusive", static_cast<uint64_t>(node.split_global_node_end_exclusive), joined); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "trace_graph_id", node.graph_id); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "trace_node_index", static_cast<uint64_t>(node.node_index)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "submission_id", submission_id); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "node_name", node.node_name); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "ggml_op", node.ggml_op); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "backend", "XDNA"); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "M", static_cast<uint64_t>(node.m)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "K", static_cast<uint64_t>(node.k)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "RHS", static_cast<uint64_t>(node.rhs)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "kernel_name", kernel_name); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "path", path); line.push_back(',');
        ggml_backend_xdna_runtime_trace_nullable_number(line, "row_index", row_index < 0 ? 0 : static_cast<uint64_t>(row_index), row_index >= 0);
        line.push_back('}');
        ggml_backend_sched_trace_emit_json(line.c_str());
    } catch (...) {
        // Trace allocation or I/O failure must not change XRT execution semantics.
    }
}


static void ggml_backend_xdna_runtime_trace_event(
        const ggml_backend_xdna_runtime_trace_node & node, uint64_t submission_id,
        const char * kernel_name, const char * path, int64_t row_index,
        const char * event, const char * xrt_state = nullptr, const char * error = nullptr) noexcept {
    if (node.graph_id == 0 || submission_id == 0) {
        return;
    }
    if (std::strcmp(event, "KERNEL_CALL_ENTER") == 0) {
        ggml_backend_xdna_scheduler_trace_submission_event(node, submission_id, kernel_name, path, row_index);
    }
    try {
        std::lock_guard<std::mutex> lock(ggml_backend_xdna_runtime_trace_mutex);
        const uint64_t timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        std::string line;
        line.reserve(640);
        line.push_back('{');
        ggml_backend_xdna_runtime_trace_field(line, "schema", "ggml-xdna-runtime-trace-v1"); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "graph_id", node.graph_id); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "node_index", static_cast<uint64_t>(node.node_index)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "node_name", node.node_name); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "ggml_op", node.ggml_op); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "backend", "XDNA"); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "M", static_cast<uint64_t>(node.m)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "K", static_cast<uint64_t>(node.k)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "RHS", static_cast<uint64_t>(node.rhs)); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "submission_id", submission_id); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "kernel_name", kernel_name); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "path", path); line.push_back(',');
        ggml_backend_xdna_runtime_trace_json_string(line, "row_index"); line.push_back(':');
        line += row_index < 0 ? "null" : std::to_string(row_index); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "event", event); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "xrt_state", xrt_state); line.push_back(',');
        ggml_backend_xdna_runtime_trace_field(line, "error", error); line.push_back(',');
        ggml_backend_xdna_runtime_trace_number(line, "monotonic_timestamp_ns", timestamp);
        line.push_back('}');
        ggml_backend_sched_trace_emit_json(line.c_str());
    } catch (...) {
        // Trace allocation or I/O failure must not change XRT execution semantics.
    }
}

static const char * ggml_backend_xdna_runtime_trace_state_name(ert_cmd_state state) noexcept {
    switch (state) {
        case ERT_CMD_STATE_NEW: return "ERT_CMD_STATE_NEW";
        case ERT_CMD_STATE_QUEUED: return "ERT_CMD_STATE_QUEUED";
        case ERT_CMD_STATE_RUNNING: return "ERT_CMD_STATE_RUNNING";
        case ERT_CMD_STATE_COMPLETED: return "ERT_CMD_STATE_COMPLETED";
        case ERT_CMD_STATE_ERROR: return "ERT_CMD_STATE_ERROR";
        case ERT_CMD_STATE_ABORT: return "ERT_CMD_STATE_ABORT";
        case ERT_CMD_STATE_SUBMITTED: return "ERT_CMD_STATE_SUBMITTED";
        case ERT_CMD_STATE_TIMEOUT: return "ERT_CMD_STATE_TIMEOUT";
        case ERT_CMD_STATE_NORESPONSE: return "ERT_CMD_STATE_NORESPONSE";
        case ERT_CMD_STATE_SKERROR: return "ERT_CMD_STATE_SKERROR";
        case ERT_CMD_STATE_SKCRASHED: return "ERT_CMD_STATE_SKCRASHED";
        case ERT_CMD_STATE_MAX: return "ERT_CMD_STATE_MAX";
        default: return "UNKNOWN";
    }
}

static ert_cmd_state ggml_backend_xdna_runtime_trace_wait(
        const ggml_backend_xdna_runtime_trace_node * node, uint64_t submission_id,
        const char * kernel_name, const char * path, int64_t row_index,
        xrt::run & run, const std::chrono::milliseconds & timeout) {
    if (node != nullptr) ggml_backend_xdna_runtime_trace_event(*node, submission_id, kernel_name, path, row_index, "WAIT_BEGIN");
    ert_cmd_state state;
    try {
        state = ggml_backend_xdna_wait_run(run, timeout);
    } catch (const std::exception & e) {
        if (node != nullptr) ggml_backend_xdna_runtime_trace_event(*node, submission_id, kernel_name, path, row_index, "WAIT_EXCEPTION", nullptr, e.what());
        throw;
    } catch (...) {
        if (node != nullptr) ggml_backend_xdna_runtime_trace_event(*node, submission_id, kernel_name, path, row_index, "WAIT_EXCEPTION", nullptr, "unknown exception");
        throw;
    }
    if (node != nullptr) ggml_backend_xdna_runtime_trace_event(*node, submission_id, kernel_name, path, row_index,
        "WAIT_RETURN_STATE", ggml_backend_xdna_runtime_trace_state_name(state),
        state == ERT_CMD_STATE_COMPLETED ? nullptr : "non_completed_xrt_state");
    return state;
}
#endif



const char * ggml_backend_xdna_kernel_profile_name(
        ggml_backend_xdna_kernel_profile profile) {
    switch (profile) {
        case ggml_backend_xdna_kernel_profile::none:
            return "none";
        case ggml_backend_xdna_kernel_profile::i16_256:
            return "i16_256";
        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560:
            return "q4k_q8k_k2560";

        case ggml_backend_xdna_kernel_profile::q4k_q8k_w2in_256:
            return "q4k_q8k_w2in_256";
    }

    return "unknown";
}




// Persistent host-side Q4_K -> 152-byte transport cache.
//
// This cache exists to test ONE variable: eliminating repeated CPU-side
// native Q4_K -> transport conversion for immutable model weight tensors
// across tokens. It intentionally does NOT change Q4 BO sync/XRT/output
// behavior: a cache hit still ends up producing the exact same bytes in
// the existing mapped Q4 execution BO, followed by the same sync/run/
// wait/output path as a cache miss.
//
// Keyed by the q4 source tensor data pointer (stable for the lifetime of
// an immutable weight tensor within one model/context), validated against
// M, native byte size and transport byte size to reject stale/incorrect
// reuse if a pointer were ever recycled for a different tensor.






// Minimal accumulated per-token-run timers (milliseconds), summed across
// all accepted Q4 nodes for the process lifetime. Diagnostic only.




static bool ggml_backend_xdna_env_flag_enabled(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0;
}


#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static uint64_t ggml_backend_xdna_next_instance_id = 1;
static uint64_t ggml_backend_xdna_next_registration_id = 1;
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static void ggml_backend_xdna_teardown_marker(const char * event) {
    std::fprintf(stderr, "XDNA_CAUSAL_TEARDOWN event=%s\n", event);
    std::fflush(stderr);
}

ggml_backend_xdna_dtor_sentinel::~ggml_backend_xdna_dtor_sentinel() noexcept {
    ggml_backend_xdna_teardown_marker(marker);
}
#endif
ggml_backend_xdna_backend_context::~ggml_backend_xdna_backend_context() {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_teardown_marker("CTX_DTOR_BODY_ENTER");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_teardown_marker("BEFORE_CAPTURE_DETACH");
    ggml_backend_xdna_diagnostic_capture_detach(this);
    ggml_backend_xdna_teardown_marker("AFTER_CAPTURE_DETACH");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_teardown_marker("CTX_DTOR_BODY_EXIT");
#endif
}



#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
static std::string ggml_backend_xdna_diag_utf8(const std::wstring & value) {
    if (value.empty()) return "";
    const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) return "<utf8-error>";
    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(), size, nullptr, nullptr);
    return result;
}

static std::wstring ggml_backend_xdna_diag_module_path(HMODULE module) {
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) return {};
    std::wstring path(buffer.data(), length);
    HANDLE file = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return path;
    std::vector<wchar_t> normalized(32768);
    const DWORD normalized_length = GetFinalPathNameByHandleW(file, normalized.data(),
        static_cast<DWORD>(normalized.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    CloseHandle(file);
    return normalized_length > 0 && normalized_length < normalized.size()
        ? std::wstring(normalized.data(), normalized_length) : path;
}

static bool ggml_backend_xdna_diag_hash_path(const std::wstring & path, char output[65]) {
    std::ifstream file(std::filesystem::path(path), std::ios::binary);
    if (!file) return false;
    ggml_xdna_sha256_t hash;
    ggml_xdna_sha256_init(&hash);
    char buffer[64 * 1024];
    while (file) {
        file.read(buffer, sizeof(buffer));
        const std::streamsize count = file.gcount();
        if (count > 0) ggml_xdna_sha256_update(&hash,
            reinterpret_cast<const unsigned char *>(buffer), static_cast<size_t>(count));
    }
    if (file.bad()) return false;
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    ggml_xdna_sha256_final(&hash, digest);
    static constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(digest); ++i) {
        output[2 * i] = hex[digest[i] >> 4];
        output[2 * i + 1] = hex[digest[i] & 0xf];
    }
    output[64] = '\0';
    return true;
}

static std::string ggml_backend_xdna_diag_file_version(const std::wstring & path) {
    using size_fn = DWORD (WINAPI *)(LPCWSTR, LPDWORD);
    using info_fn = BOOL (WINAPI *)(LPCWSTR, DWORD, DWORD, LPVOID);
    using query_fn = BOOL (WINAPI *)(LPCVOID, LPCWSTR, LPVOID *, PUINT);
    HMODULE version = LoadLibraryExW(L"version.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (version == nullptr) return "UNAVAILABLE";
    const auto get_size = reinterpret_cast<size_fn>(GetProcAddress(version, "GetFileVersionInfoSizeW"));
    const auto get_info = reinterpret_cast<info_fn>(GetProcAddress(version, "GetFileVersionInfoW"));
    const auto query = reinterpret_cast<query_fn>(GetProcAddress(version, "VerQueryValueW"));
    if (get_size == nullptr || get_info == nullptr || query == nullptr) {
        FreeLibrary(version);
        return "UNAVAILABLE";
    }
    DWORD ignored = 0;
    const DWORD size = get_size(path.c_str(), &ignored);
    std::vector<unsigned char> data(size);
    VS_FIXEDFILEINFO * fixed = nullptr;
    UINT fixed_size = 0;
    const bool ok = size != 0 && get_info(path.c_str(), 0, size, data.data()) &&
        query(data.data(), L"\\", reinterpret_cast<LPVOID *>(&fixed), &fixed_size) &&
        fixed != nullptr && fixed_size >= sizeof(VS_FIXEDFILEINFO);
    std::string result = "UNAVAILABLE";
    if (ok) {
        char text[64];
        std::snprintf(text, sizeof(text), "%u.%u.%u.%u",
            HIWORD(fixed->dwFileVersionMS), LOWORD(fixed->dwFileVersionMS),
            HIWORD(fixed->dwFileVersionLS), LOWORD(fixed->dwFileVersionLS));
        result = text;
    }
    FreeLibrary(version);
    return result;
}

static bool ggml_backend_xdna_diag_report_module(const char * stage, const wchar_t * name) {
    const HMODULE module = GetModuleHandleW(name);
    const std::string module_name = ggml_backend_xdna_diag_utf8(name);
    if (module == nullptr) {
        std::fprintf(stderr, "XDNA_PREFLIGHT_XRT_MODULE stage=%s name=%s loaded=NO\n", stage, module_name.c_str());
        return false;
    }
    const std::wstring path = ggml_backend_xdna_diag_module_path(module);
    const std::string path_utf8 = ggml_backend_xdna_diag_utf8(path);
    char sha256[65] = "UNAVAILABLE";
    const bool hashed = !path.empty() && ggml_backend_xdna_diag_hash_path(path, sha256);
    const std::string version = path.empty() ? "UNAVAILABLE" : ggml_backend_xdna_diag_file_version(path);
    std::fprintf(stderr, "XDNA_PREFLIGHT_XRT_MODULE stage=%s name=%s loaded=YES path=\"%s\" sha256=%s file_version=%s\n",
        stage, module_name.c_str(), path_utf8.c_str(), sha256, version.c_str());
    return !path.empty() && hashed && version != "UNAVAILABLE";
}


static std::wstring ggml_backend_xdna_identity_env_wide(const wchar_t * name) {
    const DWORD capacity = GetEnvironmentVariableW(name, nullptr, 0);
    if (capacity == 0) return {};
    std::vector<wchar_t> buffer(capacity);
    const DWORD length = GetEnvironmentVariableW(name, buffer.data(), capacity);
    if (length == 0 || length >= capacity) return {};
    return std::wstring(buffer.data(), length);
}

static std::wstring ggml_backend_xdna_identity_canonical_path(const std::wstring & input) {
    std::wstring path = input;
    if (path.size() >= 2 && path.front() == L'"' && path.back() == L'"') {
        path = path.substr(1, path.size() - 2);
    }
    std::replace(path.begin(), path.end(), L'/', L'\\');
    if (!std::filesystem::path(path).is_absolute()) return {};
    if (path.empty()) return {};
    HANDLE file = CreateFileW(path.c_str(), 0,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetFinalPathNameByHandleW(file, buffer.data(),
        static_cast<DWORD>(buffer.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    CloseHandle(file);
    return length > 0 && length < buffer.size()
        ? std::wstring(buffer.data(), length) : std::wstring();
}

static std::wstring ggml_backend_xdna_identity_path_text(std::wstring path) {
    if (path.size() >= 2 && path.front() == L'"' && path.back() == L'"') {
        path = path.substr(1, path.size() - 2);
    }
    std::replace(path.begin(), path.end(), L'/', L'\\');
    if (path.compare(0, 4, L"\\\\?\\") == 0) path.erase(0, 4);
    while (path.size() > 3 && path.back() == L'\\') path.pop_back();
    return path;
}

static bool ggml_backend_xdna_identity_same_path(const std::wstring & left, const std::wstring & right) {
    return !left.empty() && !right.empty() &&
        CompareStringOrdinal(left.c_str(), static_cast<int>(left.size()),
            right.c_str(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
}

static bool ggml_backend_xdna_identity_path_contains(
        const std::wstring & path_list, const std::wstring & required) {
    const std::wstring wanted = ggml_backend_xdna_identity_path_text(required);
    size_t begin = 0;
    while (begin <= path_list.size()) {
        const size_t end = path_list.find(L';', begin);
        const std::wstring entry = ggml_backend_xdna_identity_path_text(
            path_list.substr(begin, end == std::wstring::npos ? end : end - begin));
        if (ggml_backend_xdna_identity_same_path(entry, wanted)) return true;
        if (end == std::wstring::npos) break;
        begin = end + 1;
    }
    return false;
}

static bool ggml_backend_xdna_identity_directory_exists(const std::wstring & path) {
    if (path.empty()) return false;
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

static bool ggml_backend_xdna_identity_valid_sha256(const char * value) {
    if (value == nullptr || std::strlen(value) != 64) return false;
    for (size_t i = 0; i < 64; ++i) {
        const char c = value[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F'))) return false;
    }
    return true;
}

static bool ggml_backend_xdna_identity_same_sha256(const char * expected, const char * observed) {
    if (!ggml_backend_xdna_identity_valid_sha256(expected) || observed == nullptr) return false;
    for (size_t i = 0; i < 64; ++i) {
        char a = expected[i];
        char b = observed[i];
        if (a >= 'A' && a <= 'F') a = static_cast<char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'F') b = static_cast<char>(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

static bool ggml_backend_xdna_identity_verify_module(
        const char * marker, const wchar_t * module_name,
        const std::wstring & expected_path_raw, const char * expected_sha,
        std::wstring * observed_path_out) {
    const DWORD pid = GetCurrentProcessId();
    const HMODULE module = GetModuleHandleW(module_name);
    const std::string name = ggml_backend_xdna_diag_utf8(module_name);
    const std::wstring expected_path =
        ggml_backend_xdna_identity_canonical_path(expected_path_raw);
    std::wstring observed_path;
    char observed_sha[65] = "UNAVAILABLE";
    bool hashed = false;
    if (module != nullptr) {
        std::vector<wchar_t> buffer(32768);
        const DWORD length = GetModuleFileNameW(module, buffer.data(),
            static_cast<DWORD>(buffer.size()));
        if (length > 0 && length < buffer.size()) {
            observed_path = ggml_backend_xdna_identity_canonical_path(
                std::wstring(buffer.data(), length));
        }
        hashed = !observed_path.empty() &&
            ggml_backend_xdna_diag_hash_path(observed_path, observed_sha);
    }
    if (observed_path_out != nullptr) *observed_path_out = observed_path;

    const bool path_match = ggml_backend_xdna_identity_same_path(expected_path, observed_path);
    const bool sha_valid = ggml_backend_xdna_identity_valid_sha256(expected_sha);
    const bool sha_match = hashed && ggml_backend_xdna_identity_same_sha256(expected_sha, observed_sha);
    const bool ok = module != nullptr && !expected_path_raw.empty() && path_match && sha_valid && sha_match;
    const char * reason = "verified";
    if (expected_path_raw.empty()) reason = "expected_path_missing";
    else if (expected_path.empty()) reason = "expected_path_unresolvable";
    else if (!sha_valid) reason = expected_sha == nullptr ? "expected_sha256_missing" : "expected_sha256_invalid";
    else if (module == nullptr) reason = "module_not_loaded";
    else if (observed_path.empty()) reason = "module_path_query_failed";
    else if (!hashed) reason = "module_sha256_query_failed";
    else if (!path_match) reason = "module_path_mismatch";
    else if (!sha_match) reason = "module_sha256_mismatch";

    const std::string expected_path_utf8 = expected_path.empty()
        ? (expected_path_raw.empty() ? "UNSET" : "UNRESOLVABLE")
        : ggml_backend_xdna_diag_utf8(expected_path);
    const std::string observed_path_utf8 = observed_path.empty()
        ? "UNAVAILABLE" : ggml_backend_xdna_diag_utf8(observed_path);
    const std::string expected_sha_text = expected_sha == nullptr ? "UNSET" :
        (sha_valid ? expected_sha : "INVALID");
    std::fprintf(stderr,
        "%s pid=%lu stage=AFTER_DEVICE_INIT module=%s loaded=%s expected_path=\"%s\" observed_path=\"%s\" expected_sha256=%s observed_sha256=%s path_compare=%s sha_compare=%s compare=%s reason=%s\n",
        marker, static_cast<unsigned long>(pid), name.c_str(), module == nullptr ? "NO" : "YES",
        expected_path_utf8.c_str(), observed_path_utf8.c_str(), expected_sha_text.c_str(),
        observed_sha, path_match ? "PASS" : "FAIL", sha_match ? "PASS" : "FAIL",
        ok ? "PASS" : "FAIL", reason);
    return ok;
}

static bool ggml_backend_xdna_identity_verify_control_identity() {
    const DWORD pid = GetCurrentProcessId();
    std::fprintf(stderr, "XDNA_IDENTITY_IDENTITY_GATE_START pid=%lu stage=AFTER_DEVICE_INIT\n",
        static_cast<unsigned long>(pid));

    const std::wstring system_root = ggml_backend_xdna_identity_env_wide(L"SystemRoot");
    const std::wstring path_list = ggml_backend_xdna_identity_env_wide(L"PATH");
    const std::wstring backend_path = ggml_backend_xdna_identity_env_wide(L"GGML_BACKEND_PATH");
    const std::wstring coreutil_path = ggml_backend_xdna_identity_env_wide(
        L"GGML_XDNA_DIAG_EXPECT_XRT_COREUTIL_PATH");
    const std::wstring core_path = ggml_backend_xdna_identity_env_wide(
        L"GGML_XDNA_DIAG_EXPECT_XRT_CORE_PATH");
    const char * coreutil_sha = std::getenv("GGML_XDNA_DIAG_EXPECT_XRT_COREUTIL_SHA256");
    const char * core_sha = std::getenv("GGML_XDNA_DIAG_EXPECT_XRT_CORE_SHA256");
    const char * plugin_sha = std::getenv("GGML_XDNA_DIAG_EXPECT_PLUGIN_SHA256");

    std::wstring observed_plugin_path;
    const bool coreutil_ok = ggml_backend_xdna_identity_verify_module(
        "XDNA_IDENTITY_XRT_MODULE_IDENTITY", L"xrt_coreutil.dll", coreutil_path,
        coreutil_sha, nullptr);
    const bool core_ok = ggml_backend_xdna_identity_verify_module(
        "XDNA_IDENTITY_XRT_MODULE_IDENTITY", L"xrt_core.dll", core_path,
        core_sha, nullptr);
    const bool plugin_ok = ggml_backend_xdna_identity_verify_module(
        "XDNA_IDENTITY_PLUGIN_IDENTITY", L"ggml-xdna.dll", backend_path,
        plugin_sha, &observed_plugin_path);

    std::wstring system32 = system_root;
    if (!system32.empty() && system32.back() != L'\\' && system32.back() != L'/') {
        system32 += L'\\';
    }
    system32 += L"System32";
    std::vector<wchar_t> windows_directory_buffer(32768);
    const UINT windows_directory_length = GetWindowsDirectoryW(
        windows_directory_buffer.data(), static_cast<UINT>(windows_directory_buffer.size()));
    const std::wstring windows_directory = windows_directory_length > 0 &&
        windows_directory_length < windows_directory_buffer.size()
        ? std::wstring(windows_directory_buffer.data(), windows_directory_length)
        : std::wstring();
    const bool system_root_valid = ggml_backend_xdna_identity_same_path(
        ggml_backend_xdna_identity_path_text(system_root),
        ggml_backend_xdna_identity_path_text(windows_directory));
    const size_t plugin_slash = observed_plugin_path.find_last_of(L"\\/");
    const std::wstring plugin_directory = plugin_slash == std::wstring::npos
        ? std::wstring() : observed_plugin_path.substr(0, plugin_slash);
    const bool system32_exists = ggml_backend_xdna_identity_directory_exists(system32);
    const bool systemroot_in_path = ggml_backend_xdna_identity_path_contains(path_list, system_root);
    const bool system32_in_path = ggml_backend_xdna_identity_path_contains(path_list, system32);
    const bool plugin_directory_in_path =
        ggml_backend_xdna_identity_path_contains(path_list, plugin_directory);
    const bool environment_ok = system_root_valid && system32_exists &&
        systemroot_in_path && system32_in_path && plugin_directory_in_path;
    const bool stop_gate_unset =
        std::getenv("GGML_XDNA_DIAG_STOP_AFTER_XRT_MODULE_PREFLIGHT") == nullptr;
    const bool ok = stop_gate_unset && environment_ok && coreutil_ok && core_ok && plugin_ok;

    std::string reason;
    if (!stop_gate_unset) reason += "stop_gate_must_be_unset;";
    if (!system_root_valid) reason += "systemroot_invalid;";
    if (!system32_exists) reason += "system32_missing;";
    if (!systemroot_in_path) reason += "systemroot_not_in_path;";
    if (!system32_in_path) reason += "system32_not_in_path;";
    if (!plugin_directory_in_path) reason += "plugin_directory_not_in_path;";
    if (!coreutil_ok) reason += "xrt_coreutil_identity_failed;";
    if (!core_ok) reason += "xrt_core_identity_failed;";
    if (!plugin_ok) reason += "plugin_identity_failed;";
    if (ok) reason = "all_identities_verified";
    else if (!reason.empty()) reason.pop_back();
    std::fprintf(stderr,
        "XDNA_IDENTITY_CONTROL_SUBMISSION_ALLOWED=%s pid=%lu stage=AFTER_DEVICE_INIT reason=%s\n",
        ok ? "YES" : "NO", static_cast<unsigned long>(pid), reason.c_str());
    std::fflush(stderr);
    return ok;
}

static void ggml_backend_xdna_identity_log_device_init_failure() {
    const DWORD pid = GetCurrentProcessId();
    std::fprintf(stderr,
        "XDNA_IDENTITY_IDENTITY_GATE_START pid=%lu stage=DEVICE_INIT_FAILED\n"
        "XDNA_IDENTITY_CONTROL_SUBMISSION_ALLOWED=NO pid=%lu stage=DEVICE_INIT_FAILED reason=xrt_device_open_failed\n",
        static_cast<unsigned long>(pid), static_cast<unsigned long>(pid));
    std::fflush(stderr);
}
#endif

static ggml_backend_xdna_context ggml_backend_xdna_create_context() {
    ggml_backend_xdna_context ctx;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
    const char * preflight_value = std::getenv("GGML_XDNA_DIAG_STOP_AFTER_XRT_MODULE_PREFLIGHT");
    const bool preflight_requested = preflight_value != nullptr;
    const char * q4_control_value = std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8");
    const bool identity_control_requested = q4_control_value != nullptr && std::strcmp(q4_control_value, "1") == 0;
    if (identity_control_requested && preflight_requested) {
        ctx.identity_gate_failed = true;
        const unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
        std::fprintf(stderr, "XDNA_IDENTITY_IDENTITY_GATE_START pid=%lu stage=STOP_GATE_CONFLICT\n", pid);
        std::fprintf(stderr, "XDNA_IDENTITY_CONTROL_SUBMISSION_ALLOWED=NO pid=%lu stage=STOP_GATE_CONFLICT reason=stop_gate_must_be_unset\n", pid);
        std::fflush(stderr);
    }
    if (preflight_requested) {
        ctx.stop_after_xrt_module_preflight = true;
        std::fprintf(stderr, "XDNA_PREFLIGHT_PREFLIGHT_START pid=%lu gate=%s submission_allowed=NO\n",
            static_cast<unsigned long>(GetCurrentProcessId()), preflight_value);
        if (std::strcmp(preflight_value, "1") != 0) {
            std::fprintf(stderr, "XDNA_PREFLIGHT_PREFLIGHT_RESULT status=ABORT reason=invalid_gate_value\n");
            return ctx;
        }
        ggml_backend_xdna_diag_report_module("BEFORE_DEVICE_INIT", L"ggml-xdna.dll");
        ggml_backend_xdna_diag_report_module("BEFORE_DEVICE_INIT", L"xrt_coreutil.dll");
        ggml_backend_xdna_diag_report_module("BEFORE_DEVICE_INIT", L"xrt_core.dll");
    }
#endif
    try {
        ctx.xrt_device = std::make_unique<xrt::device>(0);
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to open XRT device 0: %s\n",
            e.what());
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
        if (preflight_requested) {
            ggml_backend_xdna_diag_report_module("DEVICE_INIT_FAILED", L"ggml-xdna.dll");
            ggml_backend_xdna_diag_report_module("DEVICE_INIT_FAILED", L"xrt_coreutil.dll");
            ggml_backend_xdna_diag_report_module("DEVICE_INIT_FAILED", L"xrt_core.dll");
            std::fprintf(stderr, "XDNA_PREFLIGHT_PREFLIGHT_RESULT status=ABORT reason=xrt_device_open_failed\n");
        }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
        if (identity_control_requested && !preflight_requested) {
            ctx.identity_gate_failed = true;
            ggml_backend_xdna_identity_log_device_init_failure();
        }
#endif
        return ctx;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to open XRT device 0: unknown error\n");
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
        if (preflight_requested) {
            ggml_backend_xdna_diag_report_module("DEVICE_INIT_FAILED", L"ggml-xdna.dll");
            ggml_backend_xdna_diag_report_module("DEVICE_INIT_FAILED", L"xrt_coreutil.dll");
            ggml_backend_xdna_diag_report_module("DEVICE_INIT_FAILED", L"xrt_core.dll");
            std::fprintf(stderr, "XDNA_PREFLIGHT_PREFLIGHT_RESULT status=ABORT reason=xrt_device_open_failed\n");
        }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
        if (identity_control_requested && !preflight_requested) {
            ctx.identity_gate_failed = true;
            ggml_backend_xdna_identity_log_device_init_failure();
        }
#endif
        return ctx;
    }

    try {
        const std::string xrt_name =
            ctx.xrt_device->get_info<xrt::info::device::name>();

        if (!xrt_name.empty()) {
            ctx.description = "AMD XDNA NPU (" + xrt_name + ")";
        }
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: XRT device 0 opened, but name query failed: %s\n",
            e.what());
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: XRT device 0 opened, but name query failed\n");
    }

    try {
        ctx.device_id =
            ctx.xrt_device->get_info<xrt::info::device::bdf>();
    } catch (...) {
        // BDF is optional for this milestone.
    }


#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
    if (identity_control_requested && !preflight_requested) {
        try {
            ctx.identity_gate_failed =
                !ggml_backend_xdna_identity_verify_control_identity();
        } catch (const std::exception & e) {
            ctx.identity_gate_failed = true;
            const unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
            std::fprintf(stderr, "XDNA_IDENTITY_CONTROL_SUBMISSION_ALLOWED=NO pid=%lu stage=AFTER_DEVICE_INIT reason=identity_verifier_exception detail=%s\n", pid, e.what());
            std::fflush(stderr);
        } catch (...) {
            ctx.identity_gate_failed = true;
            const unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
            std::fprintf(stderr, "XDNA_IDENTITY_CONTROL_SUBMISSION_ALLOWED=NO pid=%lu stage=AFTER_DEVICE_INIT reason=identity_verifier_unknown_exception\n", pid);
            std::fflush(stderr);
        }
    }
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS) && defined(_WIN32)
    if (preflight_requested) {
        const bool plugin = ggml_backend_xdna_diag_report_module("AFTER_DEVICE_INIT", L"ggml-xdna.dll");
        const bool coreutil = ggml_backend_xdna_diag_report_module("AFTER_DEVICE_INIT", L"xrt_coreutil.dll");
        const bool core = ggml_backend_xdna_diag_report_module("AFTER_DEVICE_INIT", L"xrt_core.dll");
        std::fprintf(stderr, "XDNA_PREFLIGHT_PREFLIGHT_RESULT status=%s xrt_device_open=YES xrt_coreutil_identity=%s xrt_core_identity=%s\n",
            plugin && coreutil && core ? "READY" : "IDENTITY_INCOMPLETE",
            coreutil ? "VERIFIED" : "UNVERIFIED", core ? "VERIFIED" : "UNVERIFIED");
    }
#endif
    return ctx;
}

static ggml_backend_xdna_context * ggml_backend_xdna_get_context(
        ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return nullptr;
    }

    return static_cast<ggml_backend_xdna_context *>(dev->context);
}

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    (void) dev;
    return "XDNA0";
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    const auto * ctx = ggml_backend_xdna_get_context(dev);

    if (ctx == nullptr) {
        return "AMD XDNA NPU";
    }

    return ctx->description.c_str();
}

static void ggml_backend_xdna_device_get_memory(
        ggml_backend_dev_t dev,
        size_t * free,
        size_t * total) {
    (void) dev;

    // XDNA does not expose conventional dedicated VRAM through this
    // registration-only interface.
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    (void) dev;
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xdna_device_get_props(
        ggml_backend_dev_t dev,
        struct ggml_backend_dev_props * props) {
    const auto * ctx = ggml_backend_xdna_get_context(dev);

    props->name         = ggml_backend_xdna_device_get_name(dev);
    props->description  = ggml_backend_xdna_device_get_description(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->type         = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id    =
        ctx != nullptr && !ctx->device_id.empty()
            ? ctx->device_id.c_str()
            : nullptr;
    props->caps         = {};
}

// NPU1 integration forward declarations for the env-gated
// controlled W2IN block invocation (the definition follows the SHA-256
// helpers, so the compile order stays unchanged for everything else).
void ggml_backend_xdna_format_sha256(
        const unsigned char digest[
            GGML_XDNA_SHA256_DIGEST_SIZE],
        char output[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1]);

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static void ggml_backend_xdna_w2in_block_probe(
        ggml_backend_xdna_backend_context * ctx);
#endif

static bool ggml_backend_xdna_parse_init_params(
        const char * params,
        ggml_backend_xdna_kernel_profile * profile,
        std::string * xclbin_path) {
    if (profile == nullptr || xclbin_path == nullptr) {
        return false;
    }

    *profile =
        ggml_backend_xdna_kernel_profile::none;

    xclbin_path->clear();

    if (params == nullptr || params[0] == '\0') {
        return true;
    }

    const std::string value(params);

    static constexpr char i16_prefix[] =
        "profile=i16_256;";

    static constexpr char q4_prefix[] =
        "profile=q4k_q8k_k2560;";

    if (value.rfind(i16_prefix, 0) == 0) {
        *profile =
            ggml_backend_xdna_kernel_profile::i16_256;

        *xclbin_path =
            value.substr(sizeof(i16_prefix) - 1);

        if (xclbin_path->empty()) {
            std::fprintf(
                stderr,
                "ggml_xdna: i16_256 profile requires an XCLBIN path\n");
            return false;
        }

        return true;
    }

    if (value.rfind(q4_prefix, 0) == 0) {
        *profile =
            ggml_backend_xdna_kernel_profile::q4k_q8k_k2560;

        *xclbin_path =
            value.substr(sizeof(q4_prefix) - 1);

        if (xclbin_path->empty()) {
            std::fprintf(
                stderr,
                "ggml_xdna: q4k_q8k_k2560 profile requires an XCLBIN path\n");
            return false;
        }

        return true;
    }

    // Dedicated W2IN block profile (W2IN block profile).
    // Bound to the proven npu1/AIE2 package by the fingerprint tables above;
    // the XCLBIN path is always explicit so the profile can never silently
    // fall back to another package.
    static constexpr char w2in_prefix[] =
        "profile=q4k_q8k_w2in_256;";

    if (value.rfind(w2in_prefix, 0) == 0) {
        *profile =
            ggml_backend_xdna_kernel_profile::q4k_q8k_w2in_256;

        *xclbin_path =
            value.substr(sizeof(w2in_prefix) - 1);

        if (xclbin_path->empty()) {
            std::fprintf(
                stderr,
                "ggml_xdna: q4k_q8k_w2in_256 profile requires an XCLBIN path\n");
            return false;
        }

        return true;
    }

    if (value.rfind("profile=", 0) == 0) {
        std::fprintf(
            stderr,
            "ggml_xdna: unknown kernel profile in backend params: %s\n",
            params);
        return false;
    }

    // Backward-compatible direct/probe mode.
    //
    // A plain path loads the artifact but deliberately carries no
    // advertised kernel profile.
    *xclbin_path = value;
    return true;
}



static constexpr const char * GGML_XDNA_Q4K_K2560_XCLBIN_SHA256 =
    "9a743ff724463aa68d2dd62911c73eeeccad414788097ba90fd0bbff3b682d27";

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static std::string ggml_backend_xdna_xrt_error_code(const char * message) {
    const std::string text = message != nullptr ? message : "";
    const size_t start = text.find("(0x");
    const size_t end = start == std::string::npos ? std::string::npos : text.find(')', start);
    return start == std::string::npos || end == std::string::npos
        ? "UNAVAILABLE" : text.substr(start + 1, end - start - 1);
}

static void ggml_backend_xdna_xrt_context_trace(
        const ggml_backend_xdna_backend_context * ctx,
        const char * shape,
        const char * stage,
        const char * operation,
        const char * result,
        const ggml_backend_xdna_q4_shared_resources * resources,
        const void * kernel,
        const char * error_code,
        const char * message) {
    if (!ggml_backend_xdna_env_flag_enabled("GGML_XDNA_XRT_CONTEXT_TRACE")) {
        return;
    }
    const auto * device_context = ctx != nullptr ? ctx->device_context : nullptr;
    std::fprintf(stderr,
        "XDNA_XRT_CONTEXT backend_id=%llu shape=%s stage=%s operation=%s result=%s device_index=0 device=%p context_count=%zu registrations=%zu xclbin=%p path=%s sha256=%s xclbin_uuid=%s registered_uuid=%s hw_context=%p kernel=%p error_code=%s message=%s\n",
        static_cast<unsigned long long>(ctx != nullptr ? ctx->backend_instance_id : 0),
        shape != nullptr ? shape : "UNKNOWN",
        stage != nullptr ? stage : "UNKNOWN",
        operation != nullptr ? operation : "UNKNOWN",
        result != nullptr ? result : "UNKNOWN",
        device_context != nullptr ? static_cast<void *>(device_context->xrt_device.get()) : nullptr,
        resources != nullptr && resources->hw_context != nullptr ? size_t{1} : size_t{0},
        resources != nullptr ? resources->xclbin_registration_count : 0U,
        resources != nullptr ? static_cast<const void *>(resources->xclbin.get()) : nullptr,
        resources != nullptr && !resources->xclbin_path.empty() ? resources->xclbin_path.c_str() : "-",
        resources != nullptr && !resources->xclbin_sha256.empty() ? resources->xclbin_sha256.c_str() : "-",
        resources != nullptr && !resources->xclbin_uuid.empty() ? resources->xclbin_uuid.c_str() : "-",
        resources != nullptr && !resources->registered_uuid.empty() ? resources->registered_uuid.c_str() : "-",
        resources != nullptr ? static_cast<const void *>(resources->hw_context.get()) : nullptr,
        kernel,
        error_code != nullptr ? error_code : "NONE",
        message != nullptr ? message : "-");
    std::fflush(stderr);
}
#define GGML_XDNA_XRT_TRACE(...) ggml_backend_xdna_xrt_context_trace(__VA_ARGS__)
#else
#define GGML_XDNA_XRT_TRACE(...) ((void) 0)
#endif

static bool ggml_backend_xdna_acquire_q4_shared_kernel(
        ggml_backend_xdna_backend_context * ctx,
        const char * shape,
        const std::string & xclbin_path,
        const char * kernel_prefix,
        bool create_if_missing,
        std::shared_ptr<xrt::xclbin> & xclbin,
        std::shared_ptr<xrt::hw_context> & hw_context,
        std::unique_ptr<xrt::kernel> & kernel,
        std::string & resolved_kernel_name) {
    if (ctx == nullptr || ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr ||
        ctx->device_context->q4_shared == nullptr) {
        return false;
    }

    auto resources = ctx->device_context->q4_shared;
    std::lock_guard<std::mutex> lock(resources->resource_mutex);
    const char * stage = "SHARED_RESOURCE_VALIDATE";
    const char * operation = "ACQUIRE";
    auto fail = [&](const char * error_code, const char * message) {
        resources->failure_latched = true;
        resources->failure_message = message != nullptr ? message : "unknown shared XRT resource failure";
        GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "FAIL", resources.get(), nullptr,
            error_code, resources->failure_message.c_str());
        std::fprintf(stderr, "ggml_xdna: Q4 shared XRT resource failure at %s: %s\n",
            stage, resources->failure_message.c_str());
        return false;
    };

    if (resources->failure_latched) {
        return fail("FAIL_CLOSED", resources->failure_message.c_str());
    }

    try {
        if (resources->hw_context != nullptr) {
            const bool identity_ok = resources->xclbin != nullptr &&
                resources->xclbin_path == xclbin_path &&
                resources->xclbin_sha256 == GGML_XDNA_Q4K_K2560_XCLBIN_SHA256 &&
                resources->xclbin->get_uuid().to_string() == resources->xclbin_uuid &&
                resources->registered_uuid == resources->xclbin_uuid &&
                resources->xclbin_registration_count == 1 &&
                resources->hw_context->get_xclbin_uuid().to_string() == resources->xclbin_uuid;
            if (!identity_ok) {
                return fail("IDENTITY_MISMATCH", "cached Q4 XCLBIN/registration/context identity mismatch");
            }
            stage = "SHARED_CONTEXT_REUSE";
            operation = "REUSE_REGISTERED_CONTEXT";
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "PASS", resources.get(), nullptr,
                "NONE", "one device-scoped context reused");
        } else {
            if (!create_if_missing) {
                return fail("CONTEXT_NOT_INITIALIZED", "shape requested the Q4 context before primary initialization");
            }
            if (resources->xclbin_registration_count != 0 || !resources->registered_uuid.empty()) {
                return fail("REGISTRATION_STATE_INVALID", "registration already attempted without a live shared context");
            }

            stage = "XCLBIN_SHA256_VERIFY";
            operation = "VERIFY_FROZEN_ASSET";
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "BEGIN", resources.get(), nullptr,
                "NONE", "verify canonical XCLBIN before XRT load");
            unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE] = {};
            char actual_sha256[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1] = {};
            if (!ggml_backend_xdna_hash_file_sha256(xclbin_path, digest)) {
                return fail("SHA256_READ_FAILED", "unable to hash canonical XCLBIN");
            }
            ggml_backend_xdna_format_sha256(digest, actual_sha256);
            if (std::strcmp(actual_sha256, GGML_XDNA_Q4K_K2560_XCLBIN_SHA256) != 0) {
                return fail("SHA256_MISMATCH", "canonical XCLBIN SHA-256 changed before shared registration");
            }

            resources->xclbin_path = xclbin_path;
            resources->xclbin_sha256 = actual_sha256;
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "PASS", resources.get(), nullptr,
                "NONE", "canonical XCLBIN SHA-256 matches pinned identity");
            stage = "XCLBIN_LOAD";
            operation = "LOAD_XCLBIN";
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "BEGIN", resources.get(), nullptr,
                "NONE", "construct shared XCLBIN object");
            resources->xclbin = std::make_shared<xrt::xclbin>(xclbin_path);
            resources->xclbin_uuid = resources->xclbin->get_uuid().to_string();
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "PASS", resources.get(), nullptr,
                "NONE", "XCLBIN object loaded");

            auto kernels = resources->xclbin->get_kernels();
            const auto it = std::find_if(kernels.begin(), kernels.end(), [&](xrt::xclbin::kernel candidate) {
                return candidate.get_name().rfind(kernel_prefix, 0) == 0;
            });
            if (it == kernels.end()) {
                stage = "XCLBIN_KERNEL_LOOKUP";
                operation = "FIND_KERNEL";
                return fail("KERNEL_NOT_FOUND", "requested Q4 kernel prefix is absent from canonical XCLBIN");
            }

            stage = "XCLBIN_REGISTER";
            operation = "REGISTER_XCLBIN";
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "BEGIN", resources.get(), nullptr,
                "NONE", "register one canonical XCLBIN on XDNA0");
            const xrt::uuid registered_uuid =
                ctx->device_context->xrt_device->register_xclbin(*resources->xclbin);
            resources->registered_uuid = registered_uuid.to_string();
            ++resources->xclbin_registration_count;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_INSTANCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_BACKEND_INSTANCE REGISTER backend_id=%llu ordinal=%llu primary_xclbin=%p path=%s uuid=%s\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id),
                    static_cast<unsigned long long>(ggml_backend_xdna_next_registration_id++),
                    static_cast<void *>(resources->xclbin.get()), resources->xclbin_path.c_str(),
                    resources->registered_uuid.c_str());
            }
#endif
            if (resources->registered_uuid != resources->xclbin_uuid) {
                stage = "XCLBIN_REGISTER_VERIFY";
                operation = "COMPARE_REGISTERED_UUID";
                return fail("UUID_MISMATCH", "register_xclbin returned a UUID different from the loaded XCLBIN");
            }
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "PASS", resources.get(), nullptr,
                "NONE", "registered UUID matches loaded XCLBIN UUID");

            stage = "HW_CONTEXT_CREATE";
            operation = "CREATE_SHARED_CONTEXT";
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "BEGIN", resources.get(), nullptr,
                "NONE", "create one shared-mode hardware context from registered UUID");
            resources->hw_context = std::make_shared<xrt::hw_context>(
                *ctx->device_context->xrt_device, registered_uuid);
            if (resources->hw_context->get_xclbin_uuid().to_string() != resources->xclbin_uuid) {
                return fail("CONTEXT_UUID_MISMATCH", "created hardware context is bound to a different XCLBIN UUID");
            }
            GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "PASS", resources.get(), nullptr,
                "NONE", "one hardware context created and UUID verified");
        }

        stage = "KERNEL_CREATE";
        operation = "CREATE_BACKEND_KERNEL";
        GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "BEGIN", resources.get(), nullptr,
            "NONE", "create backend-owned kernel on shared context");
        auto kernels = resources->xclbin->get_kernels();
        const auto it = std::find_if(kernels.begin(), kernels.end(), [&](xrt::xclbin::kernel candidate) {
            return candidate.get_name().rfind(kernel_prefix, 0) == 0;
        });
        if (it == kernels.end()) {
            return fail("KERNEL_NOT_FOUND", "requested Q4 kernel prefix is absent from shared XCLBIN");
        }
        resolved_kernel_name = it->get_name();
        auto new_kernel = std::make_unique<xrt::kernel>(*resources->hw_context, resolved_kernel_name);
        xclbin = resources->xclbin;
        hw_context = resources->hw_context;
        kernel = std::move(new_kernel);
        GGML_XDNA_XRT_TRACE(ctx, shape, stage, operation, "PASS", resources.get(), kernel.get(),
            "NONE", "backend-owned kernel created");
        return true;
    } catch (const std::exception & e) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        const std::string error_code = ggml_backend_xdna_xrt_error_code(e.what());
        return fail(error_code.c_str(), e.what());
#else
        return fail("UNAVAILABLE", e.what());
#endif
    } catch (...) {
        return fail("UNKNOWN", "unknown exception while acquiring Q4 shared XRT resources");
    }
}
static bool ggml_backend_xdna_init_kernel(
        ggml_backend_xdna_backend_context * ctx,
        const char * params) {
    if (params == nullptr || params[0] == '\0') {
        // Kernel initialization is optional for now. This preserves the
        // registration/backend-instance milestones when no XCLBIN is supplied.
        return true;
    }

    ctx->xclbin_path = params;

    try {
        std::string resolved_kernel_name;
        if (ctx->kernel_profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
            if (!ggml_backend_xdna_acquire_q4_shared_kernel(
                    ctx, "PRIMARY", ctx->xclbin_path, ctx->kernel_name.c_str(), true,
                    ctx->xclbin, ctx->hw_context, ctx->kernel, resolved_kernel_name)) {
                return false;
            }
        } else {
            ctx->xclbin = std::make_shared<xrt::xclbin>(ctx->xclbin_path);
            auto xkernels = ctx->xclbin->get_kernels();
            auto it = std::find_if(
                xkernels.begin(), xkernels.end(),
                [&](xrt::xclbin::kernel k) {
                    return k.get_name().rfind(ctx->kernel_name, 0) == 0;
                });
            if (it == xkernels.end()) {
                std::fprintf(stderr,
                    "ggml_xdna: kernel '%s' not found in XCLBIN: %s\n",
                    ctx->kernel_name.c_str(), ctx->xclbin_path.c_str());
                return false;
            }
            resolved_kernel_name = it->get_name();
            ctx->device_context->xrt_device->register_xclbin(*ctx->xclbin);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_INSTANCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_BACKEND_INSTANCE REGISTER backend_id=%llu ordinal=%llu primary_xclbin=%p path=%s\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id),
                    static_cast<unsigned long long>(ggml_backend_xdna_next_registration_id++),
                    static_cast<void *>(ctx->xclbin.get()), ctx->xclbin_path.c_str());
            }
#endif
            ctx->hw_context = std::make_shared<xrt::hw_context>(
                *ctx->device_context->xrt_device, ctx->xclbin->get_uuid());
            ctx->kernel = std::make_unique<xrt::kernel>(
                *ctx->hw_context, resolved_kernel_name);
        }

        ctx->kernel_name = resolved_kernel_name;
        std::fprintf(stderr,
            "ggml_xdna: XRT kernel ready: %s\n",
            ctx->kernel_name.c_str());

        // NPU1 integration: controlled one-shot W2IN block
        // Diagnostic-only one-shot W2IN block invocation; it remains off the
        // scheduler path and cannot affect ordinary backend initialization.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        ggml_backend_xdna_w2in_block_probe(ctx);
#endif

        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr,
            "ggml_xdna: failed to initialize XCLBIN '%s': %s\n",
            ctx->xclbin_path.c_str(), e.what());
        return false;
    } catch (...) {
        std::fprintf(stderr,
            "ggml_xdna: failed to initialize XCLBIN '%s': unknown error\n",
            ctx->xclbin_path.c_str());
        return false;
    }
}
// NPU1 integration - controlled single-block W2IN invocation.
//
// The scheduler-advertised Q4 profile of this backend expresses only the
// [2560,M] Q4_K K=2560 geometry and builds its payloads as 10 x 152-byte
// Q4_K transport blocks + 10 x 292-byte Q8_K blocks -> one f32 row. The
// proven real npu1/AIE2 W2IN package has a different, smaller ABI: one
// 256-element block, q8 (292 B) + compact (32 B) -> 32 x i32.
//
// This entry point runs exactly ONE submission of that block through this
// backend's own XCLBIN / hw_context / kernel objects and its own BO policy:
//   opcode 3, instruction BO CACHEABLE at group_id(1),
//   data BOs HOST_ONLY at group_id(3)=q8, group_id(4)=compact,
//   group_id(5)=out, every group id read live from the backend kernel.
// Input-gated W2IN one-shot diagnostic; compile-time isolated below.
// is set, so supports_op(), the scheduler, graph partitioning, tensor shapes
// and every other payload path in this backend stay untouched.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static bool ggml_backend_xdna_w2in_block_probe_read(
        const std::string & path,
        std::vector<uint8_t> & out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: cannot open %s\n", path.c_str());
        return false;
    }

    const std::streamsize size = file.tellg();

    if (size <= 0) {
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: empty file %s\n", path.c_str());
        return false;
    }

    out.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);

    if (!file.read(reinterpret_cast<char *>(out.data()), size)) {
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: cannot read %s\n", path.c_str());
        return false;
    }

    return true;
}
#endif

// MANDATORY pre-launch ordering rule for every HOST_ONLY BO the host wrote.
//
// Established by output-completion validation on this host:
//   * Without the pre-launch output flush: host prefill was visible instead of the
//     device result at +1 us .. +265 ms -> control invalid.
//   * With the same recipe and this flush: the golden block was visible at the FIRST
//     read after run.wait(), +7 us, while sync(FROM_DEVICE) measured 0 us
//     (a cache operation, never a barrier).
// The flush is therefore required, not optional, and it includes output BOs
// that the host prefilled (e.g. with a 0xCC sentinel).
void ggml_backend_xdna_sync_host_written_bo(
        xrt::bo & bo,
        const char * what) {
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::fprintf(
        stderr,
        "ggml_xdna: pre-launch sync(TO_DEVICE) %s\n",
        what);
}

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
// Implementation of the env-gated one-shot W2IN block probe.
//
// W2IN one-shot: this body must run at most ONCE per process.
// The process-wide std::once_flag guard lives in the wrapper below
// (ggml_backend_xdna_w2in_block_probe). The body itself is unchanged by this
// experiment: kernel, XCLBIN, instructions, BO flags, sync rule, opcode,
// scheduler and the K=2560 profiles are all untouched.
static void ggml_backend_xdna_w2in_block_probe_impl(
        ggml_backend_xdna_backend_context * ctx) {
    const char * spec = std::getenv("GGML_XDNA_W2IN_BLOCK_PROBE");

    if (spec == nullptr || spec[0] == '\0') {
        return;
    }

    // Dedicated-profile gate: the block ABI is only valid for the proven
    // npu1/AIE2 W2IN package, so the probe refuses to run on any other
    // profile. K=2560 (per-row and batched) is never touched by this path.
    if (ctx == nullptr ||
        ctx->kernel_profile != ggml_backend_xdna_kernel_profile::q4k_q8k_w2in_256) {
        std::fprintf(
            stderr,
            "ggml_xdna: W2IN block probe: refusing to run on profile '%s'; "
            "the block ABI belongs to profile 'q4k_q8k_w2in_256'\n",
            ctx == nullptr
                ? "(null)"
                : ggml_backend_xdna_kernel_profile_name(ctx->kernel_profile));
        return;
    }

    if (ctx == nullptr || ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr || ctx->kernel == nullptr ||
        ctx->instructions.empty()) {
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: backend kernel not ready\n");
        return;
    }

    std::vector<std::string> paths;
    {
        const std::string s(spec);
        size_t start = 0;

        while (true) {
            const size_t comma = s.find(',', start);
            paths.push_back(s.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }

    if (paths.size() != 4) {
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: need <q8>,<compact>,<out>,<expected>\n");
        return;
    }

    constexpr size_t q8_bytes      = 292;  // native block_q8_K
    constexpr size_t compact_bytes = 32;   // one packed Q4_K nibble region
    constexpr size_t out_bytes     = 192;  // 128-byte block + 64-byte sentinel

    std::vector<uint8_t> q8;
    std::vector<uint8_t> compact;
    std::vector<uint8_t> expected;

    if (!ggml_backend_xdna_w2in_block_probe_read(paths[0], q8) ||
        !ggml_backend_xdna_w2in_block_probe_read(paths[1], compact) ||
        !ggml_backend_xdna_w2in_block_probe_read(paths[3], expected)) {
        return;
    }

    if (q8.size() != q8_bytes || compact.size() != compact_bytes || expected.size() != 128) {
        std::fprintf(stderr,
                     "ggml_xdna: W2IN block probe: unexpected payload sizes q8=%zu compact=%zu expected=%zu\n",
                     q8.size(), compact.size(), expected.size());
        return;
    }

    try {
        xrt::device & device = *ctx->device_context->xrt_device;
        xrt::kernel & kernel = *ctx->kernel;

        const uint32_t instruction_count = static_cast<uint32_t>(ctx->instructions.size());
        const size_t   instruction_bytes = static_cast<size_t>(instruction_count) * sizeof(uint32_t);

        xrt::bo bo_instr(device, instruction_bytes, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        xrt::bo bo_q8   (device, q8_bytes,          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        xrt::bo bo_comp (device, compact_bytes,     XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        xrt::bo bo_out  (device, out_bytes,         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        void * buf_instr = bo_instr.map<void *>();
        void * buf_q8    = bo_q8.map<void *>();
        void * buf_comp  = bo_comp.map<void *>();
        uint8_t * buf_out = bo_out.map<uint8_t *>();

        std::memcpy(buf_instr, ctx->instructions.data(), instruction_bytes);
        std::memcpy(buf_q8, q8.data(), q8_bytes);
        std::memcpy(buf_comp, compact.data(), compact_bytes);
        std::memset(buf_out, 0xCC, out_bytes);

        std::fprintf(stderr,
                     "ggml_xdna: W2IN block probe: kernel=%s instruction_words=%u "
                     "groups=1:%d,3:%d,4:%d,5:%d\n",
                     ctx->kernel_name.c_str(), instruction_count,
                     kernel.group_id(1), kernel.group_id(3),
                     kernel.group_id(4), kernel.group_id(5));
        std::fprintf(stderr,
                     "ggml_xdna: W2IN block probe: host_policy opcode=3 "
                     "instr_bo=cacheable data_bos=host_only live_group_id=true\n");

        // MANDATORY host ordering rule: flush every host-written HOST_ONLY BO to the
        // device before the launch, including the prefilled output BO.
        ggml_backend_xdna_sync_host_written_bo(bo_instr, "instruction BO (CACHEABLE)");
        ggml_backend_xdna_sync_host_written_bo(bo_q8,    "q8 HOST_ONLY");
        ggml_backend_xdna_sync_host_written_bo(bo_comp,  "compact HOST_ONLY");
        ggml_backend_xdna_sync_host_written_bo(bo_out,   "output HOST_ONLY (prefilled 0xCC, mandatory)");

        constexpr unsigned int opcode = 3;

        auto run = kernel(opcode, bo_instr, instruction_count, bo_q8, bo_comp, bo_out);
        const ert_cmd_state completion_state = ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT);
        if (completion_state != ERT_CMD_STATE_COMPLETED) {
            std::fprintf(stderr, "ggml_xdna: ggml_backend_xdna_w2in_block_probe_impl path=W2IN state=%d expected=%d\n",
                         static_cast<int>(completion_state), static_cast<int>(ERT_CMD_STATE_COMPLETED));
            return;
        }

        // FIRST read, immediately after run.wait(), before any other BO operation.
        size_t first_mismatch = 0;
        for (size_t i = 0; i < 128; ++i) {
            if (buf_out[i] != expected[i]) {
                ++first_mismatch;
            }
        }

        unsigned char first_sha[GGML_XDNA_SHA256_DIGEST_SIZE];
        char first_hex[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        ggml_xdna_sha256_hash(first_sha, buf_out, 128);
        ggml_backend_xdna_format_sha256(first_sha, first_hex);

        // Observation point only (documented as a cache operation, never a wait;
        // it measures 0 us on this host). Not required for visibility once the
        // pre-launch flush above has happened.
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);


        unsigned char got_sha[GGML_XDNA_SHA256_DIGEST_SIZE];
        unsigned char want_sha[GGML_XDNA_SHA256_DIGEST_SIZE];
        char got_hex[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        char want_hex[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        ggml_xdna_sha256_hash(got_sha, buf_out, 128);
        ggml_xdna_sha256_hash(want_sha, expected.data(), expected.size());
        ggml_backend_xdna_format_sha256(got_sha, got_hex);
        ggml_backend_xdna_format_sha256(want_sha, want_hex);

        size_t mismatch = 0;
        for (size_t i = 0; i < 128; ++i) {
            if (buf_out[i] != expected[i]) {
                ++mismatch;
            }
        }

        bool sentinel_ok = true;
        for (size_t i = 128; i < out_bytes; ++i) {
            if (buf_out[i] != 0xCC) {
                sentinel_ok = false;
                break;
            }
        }

        bool all_cc = true;
        for (size_t i = 0; i < 128; ++i) {
            if (buf_out[i] != 0xCC) {
                all_cc = false;
                break;
            }
        }

        std::fprintf(stderr,
                     "ggml_xdna: W2IN block probe: profile=%s abi=q8(292)+compact(32)->out(128) ",
                     "submission=1 launches=1 run_state=%d %s\n",
                     ggml_backend_xdna_kernel_profile_name(ctx->kernel_profile),
                     static_cast<int>(completion_state),
                     static_cast<int>(completion_state) == 4
                         ? "(ERT_CMD_STATE_COMPLETED)"
                         : "(unexpected)");
        std::fprintf(stderr,
                     "ggml_xdna: W2IN block probe: first_read_after_wait sha256=%s mismatch_bytes=%zu golden=%s\n",
                     first_hex, first_mismatch,
                     first_mismatch == 0 ? "true" : "false");
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: first_read_sha256=%s\n", first_hex);
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: after_sync_read_sha256=%s\n", got_hex);
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: expected_sha256=%s\n", want_hex);
        std::fprintf(stderr, "ggml_xdna: W2IN block probe: first_read_golden=%s\n",
                     first_mismatch == 0 ? "true" : "false");
        std::fprintf(stderr,
                     "ggml_xdna: W2IN block probe: golden_match=%s mismatch_bytes=%zu "
                     "sentinel_ok=%s all_0xCC=%s\n",
                     mismatch == 0 ? "true" : "false", mismatch,
                     sentinel_ok ? "true" : "false", all_cc ? "true" : "false");

        std::fprintf(stderr, "ggml_xdna: W2IN block probe: int32=");
        for (size_t i = 0; i < 32; ++i) {
            int32_t value = 0;
            std::memcpy(&value, buf_out + i * sizeof(int32_t), sizeof(value));
            std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", value);
        }
        std::fprintf(stderr, "\n");

        std::ofstream out_file(paths[2], std::ios::binary | std::ios::trunc);
        if (out_file) {
            out_file.write(reinterpret_cast<const char *>(buf_out),
                           static_cast<std::streamsize>(out_bytes));
            out_file.close();
        } else {
            std::fprintf(stderr, "ggml_xdna: W2IN block probe: cannot write %s\n", paths[2].c_str());
        }

        return;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ggml_xdna: W2IN block probe failed: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "ggml_xdna: W2IN block probe failed: unknown error\n");
    }
}

// W2IN one-shot - process-wide one-shot guard.
//
// llama.cpp initializes the XDNA ACCEL backend twice in one process, so the
// env-gated block probe used to run (and submit to the NPU) once per
// initialization. This wrapper makes the probe execute at most once per
// process, using a process-wide std::once_flag: exactly one caller runs the
// implementation, every other caller returns without touching the device.
//
// Scope: the guard ONLY wraps GGML_XDNA_W2IN_BLOCK_PROBE. When that variable
// is unset the guard is not consumed and backend initialization proceeds
// exactly as before (the call site in init_kernel is unchanged, and the rest
// of init_kernel always runs).
static void ggml_backend_xdna_w2in_block_probe(
        ggml_backend_xdna_backend_context * ctx) {
    const char * spec = std::getenv("GGML_XDNA_W2IN_BLOCK_PROBE");

    if (spec == nullptr || spec[0] == '\0') {
        // Not requested: do not consume the one-shot, do not log, and leave
        // normal backend initialization untouched.
        return;
    }

    static std::once_flag once;
    static std::atomic<unsigned> calls{0};

    const unsigned call = calls.fetch_add(1) + 1;

    if (call > 1) {
        std::fprintf(
            stderr,
            "ggml_xdna: W2IN block probe: call #%u suppressed by the process-wide "
            "one-shot guard (probe already executed in this process)\n",
            call);
    }

    std::call_once(once, [ctx]() {
        ggml_backend_xdna_w2in_block_probe_impl(ctx);
    });
}
#endif


static bool ggml_backend_xdna_load_instructions(
        ggml_backend_xdna_backend_context * ctx) {
    if (ctx == nullptr) {
        return false;
    }

    if (!ctx->instructions.empty()) {
        return true;
    }

    if (ctx->xclbin_path.empty()) {
        std::fprintf(
            stderr,
            "ggml_xdna: cannot load instructions without an XCLBIN path\n");
        return false;
    }

    try {
        const std::filesystem::path xclbin_path(ctx->xclbin_path);
        const char * q4_insts =
            ctx->kernel_profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560
                ? std::getenv("GGML_XDNA_Q4K_M1024_INSTS")
                : nullptr;
        const std::filesystem::path instructions_path =
            q4_insts != nullptr && q4_insts[0] != '\0'
                ? std::filesystem::path(q4_insts)
                : xclbin_path.parent_path() / "insts.bin";
        ctx->instructions_path = instructions_path.string();

        std::ifstream file(
            instructions_path,
            std::ios::binary | std::ios::ate);

        if (!file) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to open instruction file: %s\n",
                ctx->instructions_path.c_str());
            return false;
        }

        const std::streamsize size = file.tellg();

        if (size <= 0 ||
            size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: invalid instruction file size: %s\n",
                ctx->instructions_path.c_str());
            return false;
        }

        ctx->instructions.resize(
            static_cast<size_t>(size) / sizeof(uint32_t));

        file.seekg(0, std::ios::beg);

        if (!file.read(
                reinterpret_cast<char *>(ctx->instructions.data()),
                size)) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to read instruction file: %s\n",
                ctx->instructions_path.c_str());

            ctx->instructions.clear();
            return false;
        }

        std::fprintf(
            stderr,
            "ggml_xdna: loaded %zu instruction words from %s\n",
            ctx->instructions.size(),
            ctx->instructions_path.c_str());

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to load instructions: %s\n",
            e.what());
        ctx->instructions.clear();
        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to load instructions: unknown error\n");
        ctx->instructions.clear();
        return false;
    }
}

static const unsigned char *
ggml_backend_xdna_kernel_profile_instruction_sha256(
        ggml_backend_xdna_kernel_profile profile) {
    static constexpr unsigned char i16_256_sha256[
        GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xac, 0xa7, 0xc1, 0xab, 0x8a, 0xe9, 0x0f, 0x83,
        0xc8, 0x24, 0xeb, 0xb5, 0x6e, 0x0e, 0x8d, 0x55,
        0x9b, 0x1b, 0x2f, 0x21, 0x7e, 0xd6, 0x6e, 0x08,
        0x51, 0xdf, 0x37, 0xd3, 0x15, 0xce, 0x1d, 0x14,
    };

    // canonical unified multisequence M=1024 instruction stream.
    // SHA256 b9504aa178b3412d7adbdb4cdc641892acae90940095a5a6b1357021209a65b8.
    static constexpr unsigned char q4k_q8k_k2560_sha256[
        GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xb9, 0x50, 0x4a, 0xa1, 0x78, 0xb3, 0x41, 0x2d,
        0x7a, 0xdb, 0xdb, 0x4c, 0xdc, 0x64, 0x18, 0x92,
        0xac, 0xae, 0x90, 0x94, 0x00, 0x95, 0xa5, 0xa6,
        0xb1, 0x35, 0x70, 0x21, 0x20, 0x9a, 0x65, 0xb8
    };

    // Proven real npu1/AIE2 W2IN block package: instructions
    // 0b2a720da5d9b90b2baa2eb81c6df6bcf7da281d06e2b54fa750d1b207a38ab4
    // (420 B / 105 words, target byte 0x03 = AIE2/npu1).
    static constexpr unsigned char q4k_q8k_w2in_256_sha256[
        GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0x0b, 0x2a, 0x72, 0x0d, 0xa5, 0xd9, 0xb9, 0x0b,
        0x2b, 0xaa, 0x2e, 0xb8, 0x1c, 0x6d, 0xf6, 0xbc,
        0xf7, 0xda, 0x28, 0x1d, 0x06, 0xe2, 0xb5, 0x4f,
        0xa7, 0x50, 0xd1, 0xb2, 0x07, 0xa3, 0x8a, 0xb4,
    };

    switch (profile) {
        case ggml_backend_xdna_kernel_profile::none:
            return nullptr;

        case ggml_backend_xdna_kernel_profile::i16_256:
            return i16_256_sha256;

        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560:
            return q4k_q8k_k2560_sha256;

        case ggml_backend_xdna_kernel_profile::q4k_q8k_w2in_256:
            return q4k_q8k_w2in_256_sha256;
    }

    return nullptr;
}

void ggml_backend_xdna_format_sha256(
        const unsigned char digest[
            GGML_XDNA_SHA256_DIGEST_SIZE],
        char output[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1]) {
    static constexpr char hex[] =
        "0123456789abcdef";

    for (size_t i = 0;
         i < GGML_XDNA_SHA256_DIGEST_SIZE;
         ++i) {
        output[2 * i] =
            hex[(digest[i] >> 4) & 0x0f];

        output[2 * i + 1] =
            hex[digest[i] & 0x0f];
    }

    output[
        GGML_XDNA_SHA256_DIGEST_SIZE * 2] = '\0';
}

bool ggml_backend_xdna_hash_file_sha256(
        const std::string & path,
        unsigned char digest[
            GGML_XDNA_SHA256_DIGEST_SIZE]) {
    try {
        std::ifstream file(
            path,
            std::ios::binary);

        if (!file) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to open file for SHA256: %s\n",
                path.c_str());

            return false;
        }

        ggml_xdna_sha256_t hash;
        ggml_xdna_sha256_init(&hash);

        char buffer[64 * 1024];

        while (file) {
            file.read(
                buffer,
                sizeof(buffer));

            const std::streamsize count =
                file.gcount();

            if (count > 0) {
                ggml_xdna_sha256_update(
                    &hash,
                    reinterpret_cast<
                        const unsigned char *>(buffer),
                    static_cast<size_t>(count));
            }
        }

        if (file.bad()) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed while hashing file: %s\n",
                path.c_str());

            return false;
        }

        ggml_xdna_sha256_final(
            &hash,
            digest);

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to hash file '%s': %s\n",
            path.c_str(),
            e.what());

        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to hash file '%s': "
            "unknown error\n",
            path.c_str());

        return false;
    }
}


static bool
ggml_backend_xdna_kernel_profile_xclbin_sha256_matches(
        ggml_backend_xdna_kernel_profile profile,
        const unsigned char digest[
            GGML_XDNA_SHA256_DIGEST_SIZE]) {
    static constexpr unsigned char
        i16_256_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xd4, 0x24, 0x66, 0xe1, 0x54, 0xed, 0xe8, 0xc2,
        0x02, 0x05, 0x67, 0x8d, 0x9f, 0x29, 0x2a, 0xfc,
        0x8a, 0x05, 0xf2, 0x24, 0x09, 0xd5, 0x8d, 0x29,
        0x89, 0x5a, 0x9f, 0xe0, 0x64, 0x9e, 0x79, 0x33,
    };

    static constexpr unsigned char
        q4k_q8k_k2560_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        // canonical unified multisequence XCLBIN.
        0x9a, 0x74, 0x3f, 0xf7, 0x24, 0x46, 0x3a, 0xa6,
        0x8d, 0x2d, 0xd6, 0x29, 0x11, 0xc7, 0x3e, 0xee,
        0xcc, 0xad, 0x41, 0x47, 0x88, 0x09, 0x7b, 0xa9,
        0x0f, 0xd0, 0xbb, 0xff, 0x3b, 0x68, 0x2d, 0x27
    };

    // Proven real npu1/AIE2 W2IN block package XCLBIN:
    // 1e06d3f236f5a12bee58d9a7c008431c0f3bd38172d3cab99548ea2174ccaa83
    // (8918 B, uuid a304d3f5-…; core ELF 2196 B e_flags=0x2 = AIE2/npu1).
    static constexpr unsigned char
        q4k_q8k_w2in_256_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0x1e, 0x06, 0xd3, 0xf2, 0x36, 0xf5, 0xa1, 0x2b,
        0xee, 0x58, 0xd9, 0xa7, 0xc0, 0x08, 0x43, 0x1c,
        0x0f, 0x3b, 0xd3, 0x81, 0x72, 0xd3, 0xca, 0xb9,
        0x95, 0x48, 0xea, 0x21, 0x74, 0xcc, 0xaa, 0x83,
    };

    switch (profile) {
        case ggml_backend_xdna_kernel_profile::none:
            return false;

        case ggml_backend_xdna_kernel_profile::i16_256:
            return std::memcmp(
                digest,
                i16_256_sha256,
                GGML_XDNA_SHA256_DIGEST_SIZE) == 0;

        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560:
            return std::memcmp(
                digest,
                q4k_q8k_k2560_sha256,
                GGML_XDNA_SHA256_DIGEST_SIZE) == 0;

        case ggml_backend_xdna_kernel_profile::q4k_q8k_w2in_256:
            return std::memcmp(
                digest,
                q4k_q8k_w2in_256_sha256,
                GGML_XDNA_SHA256_DIGEST_SIZE) == 0;
    }

    return false;
}

static bool ggml_backend_xdna_validate_kernel_profile(
        ggml_backend_xdna_backend_context * ctx) {
    if (ctx == nullptr) {
        return false;
    }

    if (ctx->kernel_profile ==
        ggml_backend_xdna_kernel_profile::none) {
        return true;
    }

    const unsigned char * expected_instructions =
        ggml_backend_xdna_kernel_profile_instruction_sha256(
            ctx->kernel_profile);

    if (expected_instructions == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: no instruction fingerprint "
            "registered for profile '%s'\n",
            ggml_backend_xdna_kernel_profile_name(
                ctx->kernel_profile));

        return false;
    }

    if (!ggml_backend_xdna_load_instructions(ctx)) {
        return false;
    }

    unsigned char actual_instructions[
        GGML_XDNA_SHA256_DIGEST_SIZE];

    ggml_xdna_sha256_hash(
        actual_instructions,
        reinterpret_cast<const unsigned char *>(
            ctx->instructions.data()),
        ctx->instructions.size() *
            sizeof(uint32_t));

    if (std::memcmp(
            actual_instructions,
            expected_instructions,
            GGML_XDNA_SHA256_DIGEST_SIZE) != 0) {
        char actual_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        char expected_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        ggml_backend_xdna_format_sha256(
            actual_instructions,
            actual_hex);

        ggml_backend_xdna_format_sha256(
            expected_instructions,
            expected_hex);

        std::fprintf(
            stderr,
            "ggml_xdna: instruction fingerprint mismatch "
            "for profile '%s'\n"
            "ggml_xdna: expected SHA256: %s\n"
            "ggml_xdna: actual   SHA256: %s\n"
            "ggml_xdna: instruction file: %s\n",
            ggml_backend_xdna_kernel_profile_name(
                ctx->kernel_profile),
            expected_hex,
            actual_hex,
            ctx->instructions_path.c_str());

        return false;
    }

    std::fprintf(
        stderr,
        "ggml_xdna: instruction fingerprint "
        "validated for profile '%s'\n",
        ggml_backend_xdna_kernel_profile_name(
            ctx->kernel_profile));

    unsigned char actual_xclbin[
        GGML_XDNA_SHA256_DIGEST_SIZE];

    if (!ggml_backend_xdna_hash_file_sha256(
            ctx->xclbin_path,
            actual_xclbin)) {
        return false;
    }

    if (!ggml_backend_xdna_kernel_profile_xclbin_sha256_matches(
            ctx->kernel_profile,
            actual_xclbin)) {
        char actual_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        ggml_backend_xdna_format_sha256(
            actual_xclbin,
            actual_hex);

        std::fprintf(
            stderr,
            "ggml_xdna: XCLBIN fingerprint mismatch "
            "for profile '%s'\n"
            "ggml_xdna: actual SHA256: %s\n"
            "ggml_xdna: XCLBIN file: %s\n",
            ggml_backend_xdna_kernel_profile_name(
                ctx->kernel_profile),
            actual_hex,
            ctx->xclbin_path.c_str());

        return false;
    }

    std::fprintf(
        stderr,
        "ggml_xdna: XCLBIN fingerprint "
        "validated for profile '%s'\n",
        ggml_backend_xdna_kernel_profile_name(
            ctx->kernel_profile));

    return true;
}

// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    (void) backend;
    return GGML_XDNA_NAME;
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(backend->context);


    std::unique_lock<std::mutex> device_state_lock;
    if (ctx != nullptr && ctx->device_context != nullptr &&
        ctx->device_context->q4_shared != nullptr) {
        device_state_lock = std::unique_lock<std::mutex>(
            ctx->device_context->q4_shared->state_mutex);
    }
    if (ctx != nullptr &&
        ctx->device_context != nullptr &&
        ctx->device_profile_registered) {
        auto * device_ctx =
            ctx->device_context;

        if (device_ctx->active_profile !=
            ctx->kernel_profile) {
            std::fprintf(
                stderr,
                "ggml_xdna: backend/device profile mismatch during free\n");
        }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (std::getenv("GGML_XDNA_INSTANCE_TRACE") != nullptr) std::fprintf(stderr, "XDNA_BACKEND_INSTANCE FREE backend_id=%llu active_backend_count_before=%zu device_ctx=%p xrt_device=%p\n", static_cast<unsigned long long>(ctx->backend_instance_id), device_ctx->active_backend_count, static_cast<void *>(device_ctx), static_cast<void *>(device_ctx->xrt_device.get()));
#endif

        if (device_ctx->active_backend_count == 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: active backend count already zero during free\n");
        }
        else {
            --device_ctx->active_backend_count;
        }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (std::getenv("GGML_XDNA_INSTANCE_TRACE") != nullptr) std::fprintf(stderr, "XDNA_BACKEND_INSTANCE FREE backend_id=%llu active_backend_count_after=%zu\n", static_cast<unsigned long long>(ctx->backend_instance_id), device_ctx->active_backend_count);
#endif
        if (device_ctx->active_backend_count == 0) {
            device_ctx->active_profile =
                ggml_backend_xdna_kernel_profile::none;
            device_ctx->q4_m9216_available = false;
            device_ctx->q4_m1024_available = false;
            device_ctx->q4_m4096_available = false;
            device_ctx->q4_m8192_available = false;
        }

        ctx->device_profile_registered = false;
    }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (ctx != nullptr && ctx->kernel_profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
        std::fprintf(stderr, "ggml_xdna: ledger accepted=%llu batched_m9216=%llu batched_m4096=%llu batched_m8192=%llu batched_m1024=%llu fallback=%llu xrt_runs=%llu run_waits=%llu q4_to_device=%llu q8_to_device=%llu output_from_device=%llu\n",
            static_cast<unsigned long long>(ctx->accepted_q4_nodes), static_cast<unsigned long long>(ctx->batched_m9216), static_cast<unsigned long long>(ctx->batched_m4096), static_cast<unsigned long long>(ctx->batched_m8192), static_cast<unsigned long long>(ctx->batched_m1024), static_cast<unsigned long long>(ctx->fallback_rows), static_cast<unsigned long long>(ctx->xrt_runs), static_cast<unsigned long long>(ctx->run_waits), static_cast<unsigned long long>(ctx->q4_to_device), static_cast<unsigned long long>(ctx->q8_to_device), static_cast<unsigned long long>(ctx->output_from_device));
    }

    if (ctx != nullptr && ctx->kernel_profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
        size_t cache_bytes_total = 0;
        for (const auto & kv : ctx->q4_cache.entries) {
            cache_bytes_total += kv.second.transport.size();
        }

        std::fprintf(stderr,
            "ggml_xdna: q4_cache enabled=%d entries=%zu bytes=%zu hits=%llu misses=%llu hits_m9216=%llu hits_m4096=%llu hits_m8192=%llu hits_m1024=%llu blocks_converted=%llu native_bytes_converted=%llu transport_bytes_generated=%llu transport_bytes_copied=%llu\n",
            ctx->q4_cache.enabled ? 1 : 0,
            ctx->q4_cache.entries.size(),
            cache_bytes_total,
            static_cast<unsigned long long>(ctx->q4_cache.hits),
            static_cast<unsigned long long>(ctx->q4_cache.misses),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m9216),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m4096),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m8192),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m1024),
            static_cast<unsigned long long>(ctx->q4_cache.blocks_converted),
            static_cast<unsigned long long>(ctx->q4_cache.native_bytes_converted),
            static_cast<unsigned long long>(ctx->q4_cache.transport_bytes_generated),
            static_cast<unsigned long long>(ctx->q4_cache.transport_bytes_copied));

        std::fprintf(stderr,
            "ggml_xdna: persistent_weight_bos enabled=%d tensors=%zu bytes=%zu creates=%llu reuses=%llu failures=%llu cap_hits=%llu\n",
            ctx->q4_weight_bo_enabled ? 1 : 0,
            ctx->q4_m9216_weight_bos.size(),
            ctx->q4_weight_bo_bytes,
            static_cast<unsigned long long>(ctx->weight_bo_creates),
            static_cast<unsigned long long>(ctx->weight_bo_reuses),
            static_cast<unsigned long long>(ctx->weight_bo_failures),
            static_cast<unsigned long long>(ctx->weight_bo_cap_hits));

        std::fprintf(stderr,
            "ggml_xdna: timers_ms q4_conversion=%.3f q4_cached_memcpy=%.3f q4_sync=%.3f q8_quant=%.3f q8_sync=%.3f xrt_submit_wait=%.3f output=%.3f\n",
            ctx->timers.q4_conversion_ms,
            ctx->timers.q4_cached_memcpy_ms,
            ctx->timers.q4_sync_ms,
            ctx->timers.q8_quant_ms,
            ctx->timers.q8_sync_ms,
            ctx->timers.xrt_submit_wait_ms,
            ctx->timers.output_ms);
    }

        static const char * shape_names[] = {"M9216", "M4096", "M8192", "M1024"};
        for (size_t i = 0; i < 4; ++i) {
            const auto & t = ctx->shape_timing[i];
            std::fprintf(stderr, "ggml_xdna: xrt_shape=%s ops=%llu submit_ms=%.3f submit_avg_ms=%.6f wait_ms=%.3f wait_avg_ms=%.6f combined_ms=%.3f\n", shape_names[i], static_cast<unsigned long long>(t.ops), t.submit_ms, t.ops ? t.submit_ms / t.ops : 0.0, t.wait_ms, t.ops ? t.wait_ms / t.ops : 0.0, t.submit_ms + t.wait_ms);
        }

#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_teardown_marker("AFTER_FREE_DIAGNOSTICS");
#endif
    // Q4 XCLBIN/context references are device-scoped; kernels and BOs remain local.
    // Other profiles retain their backend-owned XRT resources.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_teardown_marker("BEFORE_CTX_DELETE");
#endif
    delete ctx;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_teardown_marker("AFTER_CTX_DELETE");
    ggml_backend_xdna_teardown_marker("BEFORE_BACKEND_DELETE");
#endif
    delete backend;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_teardown_marker("AFTER_BACKEND_DELETE");
#endif
}

static bool ggml_backend_xdna_run_i16_f32_matmul(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const int16_t * a,
        size_t a_elements,
        const int16_t * b,
        size_t b_elements,
        float * c,
        size_t c_elements);

static bool ggml_backend_xdna_run_q4k_q8k_k2560(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const void * q4_row,
        size_t q4_row_bytes,
        const float * activations,
        size_t activation_elements,
        bool refresh_activation,
        float * result
#if defined(GGML_XDNA_RUNTIME_TRACE)
        , const ggml_backend_xdna_runtime_trace_node * trace_node, int64_t row_index
#endif
        );

static bool ggml_backend_xdna_run_batched_shape(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_batched_state & state,
        const void * q4_rows, size_t q4_rows_bytes, const float * activations,
        size_t activation_elements, float * result, size_t result_bytes,
        const char * node_name, int64_t k, int64_t rhs
#if defined(GGML_XDNA_RUNTIME_TRACE)
        , const ggml_backend_xdna_runtime_trace_node * trace_node
#endif
 #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        , const struct ggml_tensor * node, int graph_node_index, uint64_t causal_sequence_id, bool causal_selected
 #endif
        );

static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216(
        ggml_backend_t backend,
        const void * q4_rows,
        size_t q4_rows_bytes,
        const float * activations,
        size_t activation_elements,
        float * result,
        size_t result_bytes
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        , const struct ggml_tensor * node, int graph_node_index, uint64_t causal_sequence_id, bool causal_selected
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
        , const ggml_backend_xdna_runtime_trace_node * trace_node
#endif
        , const char * node_name, int64_t k, int64_t rhs);
void ggml_backend_xdna_format_sha256(const unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE], char out[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1]);


static bool ggml_backend_xdna_is_view_op(enum ggml_op op) {
    return
        op == GGML_OP_VIEW ||
        op == GGML_OP_RESHAPE ||
        op == GGML_OP_PERMUTE ||
        op == GGML_OP_TRANSPOSE;
}

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static std::atomic<uint64_t> ggml_backend_xdna_m9216_submission_sequence{0};

static const char * ggml_backend_xdna_ert_cmd_state_name(ert_cmd_state state) {
    switch (state) {
        case ERT_CMD_STATE_NEW: return "ERT_CMD_STATE_NEW";
        case ERT_CMD_STATE_QUEUED: return "ERT_CMD_STATE_QUEUED";
        case ERT_CMD_STATE_RUNNING: return "ERT_CMD_STATE_RUNNING";
        case ERT_CMD_STATE_COMPLETED: return "ERT_CMD_STATE_COMPLETED";
        case ERT_CMD_STATE_ERROR: return "ERT_CMD_STATE_ERROR";
        case ERT_CMD_STATE_ABORT: return "ERT_CMD_STATE_ABORT";
        case ERT_CMD_STATE_SUBMITTED: return "ERT_CMD_STATE_SUBMITTED";
        case ERT_CMD_STATE_TIMEOUT: return "ERT_CMD_STATE_TIMEOUT";
        case ERT_CMD_STATE_NORESPONSE: return "ERT_CMD_STATE_NORESPONSE";
        case ERT_CMD_STATE_SKERROR: return "ERT_CMD_STATE_SKERROR";
        case ERT_CMD_STATE_SKCRASHED: return "ERT_CMD_STATE_SKCRASHED";
        case ERT_CMD_STATE_MAX: return "ERT_CMD_STATE_MAX";
        default: return "UNKNOWN";
    }
}

static void ggml_backend_xdna_m9216_digest(
        const void * data, size_t bytes, char out[65]) {
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    ggml_xdna_sha256_hash(digest, static_cast<const unsigned char *>(data), bytes);
    ggml_backend_xdna_format_sha256(digest, out);
}
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
static constexpr const char * GGML_XDNA_Q4K_CANONICAL_XCLBIN_SHA =
    "9a743ff724463aa68d2dd62911c73eeeccad414788097ba90fd0bbff3b682d27";
static constexpr const char * GGML_XDNA_Q4K_CANONICAL_INSTRUCTION_SHAS[] = {
    "b9504aa178b3412d7adbdb4cdc641892acae90940095a5a6b1357021209a65b8",
    "9b89a2a63ee7c0c26273a6d25ac6511582769734b48bb9486b56281e1dd78c03",
    "e093f613fcb230dd12e0674a308d42dde248307e106dbcbc4ff20ba470ceb480",
    "d8e42a47f2b9e063a77edb513bd15ccc992204bd6e3cf28a519a821f97feee97",
};
static constexpr const char * GGML_XDNA_Q4K_INSTRUCTION_ENVS[] = {
    "GGML_XDNA_Q4K_M1024_INSTS", "GGML_XDNA_Q4K_M4096_INSTS",
    "GGML_XDNA_Q4K_M8192_INSTS", "GGML_XDNA_Q4K_M9216_INSTS",
};

static unsigned long ggml_backend_xdna_profile_process_id() {
#if defined(_WIN32)
    return static_cast<unsigned long>(GetCurrentProcessId());
#else
    return static_cast<unsigned long>(getpid());
#endif
}

static const char * ggml_backend_xdna_q4_control_conflicting_env(bool allow_host_preflight_gate = false);
static const char * ggml_backend_xdna_profile_environment_error(bool allow_host_preflight_gate = false) {
    const char * gate = std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8");
    const char * host_gate = std::getenv("GGML_XDNA_DIAG_A7H2_HOST_PREFLIGHT_ONLY");
    if (host_gate != nullptr && std::strcmp(host_gate, "1") != 0) return "host_preflight_gate_invalid";
    if (gate == nullptr || std::strcmp(gate, "1") != 0) return "control_gate_not_1";
    const char * runtime_xclbin = std::getenv("GGML_XDNA_Q4K_RUNTIME_XCLBIN");
    if (runtime_xclbin == nullptr || runtime_xclbin[0] == '\0') return "runtime_xclbin_unset";
    const std::string expected_params = std::string("profile=q4k_q8k_k2560;") + runtime_xclbin;
    const char * params = std::getenv("GGML_XDNA_BACKEND_PARAMS");
    if (params == nullptr || std::strcmp(params, expected_params.c_str()) != 0) return "backend_params_mismatch";
    for (size_t i = 0; i < 4; ++i) {
        const char * instructions = std::getenv(GGML_XDNA_Q4K_INSTRUCTION_ENVS[i]);
        if (instructions == nullptr || instructions[0] == '\0') return GGML_XDNA_Q4K_INSTRUCTION_ENVS[i];
    }
    if (std::getenv("GGML_XDNA_DIAG_STOP_AFTER_XRT_MODULE_PREFLIGHT") != nullptr) return "stop_gate_must_be_unset";
    if (std::getenv("GGML_XDNA_Q4K_PERSISTENT_WEIGHTS") == nullptr ||
        std::strcmp(std::getenv("GGML_XDNA_Q4K_PERSISTENT_WEIGHTS"), "0") != 0) return "persistent_weights_must_be_0";
    if (std::getenv("GGML_XDNA_Q4K_TRANSPORT_CACHE") == nullptr ||
        std::strcmp(std::getenv("GGML_XDNA_Q4K_TRANSPORT_CACHE"), "0") != 0) return "transport_cache_must_be_0";
    return ggml_backend_xdna_q4_control_conflicting_env(allow_host_preflight_gate);
}

static bool ggml_backend_xdna_profile_preflight(
        const char * resolved_params,
        ggml_backend_xdna_kernel_profile profile,
        const std::string & parsed_xclbin_path,
        bool allow_host_preflight_gate,
        const char ** failure_reason_out) {
    const unsigned long pid = ggml_backend_xdna_profile_process_id();
    std::fprintf(stderr, "XDNA_PROFILE_PROFILE_PREFLIGHT_START pid=%lu\n", pid);
    const char * backend_params = std::getenv("GGML_XDNA_BACKEND_PARAMS");
    const char * runtime_xclbin = std::getenv("GGML_XDNA_Q4K_RUNTIME_XCLBIN");
    const std::string expected_params = runtime_xclbin != nullptr
        ? std::string("profile=q4k_q8k_k2560;") + runtime_xclbin : std::string();
    const bool profile_ok = backend_params != nullptr && !expected_params.empty() &&
        std::strcmp(backend_params, expected_params.c_str()) == 0 &&
        resolved_params != nullptr && std::strcmp(resolved_params, expected_params.c_str()) == 0 &&
        profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560 &&
        parsed_xclbin_path == runtime_xclbin;
    const char * profile_reason = "validated";
    if (backend_params == nullptr) profile_reason = "backend_params_unset";
    else if (runtime_xclbin == nullptr || runtime_xclbin[0] == '\0') profile_reason = "runtime_xclbin_unset";
    else if (std::strcmp(backend_params, expected_params.c_str()) != 0) profile_reason = "backend_params_mismatch";
    else if (resolved_params == nullptr || std::strcmp(resolved_params, expected_params.c_str()) != 0) profile_reason = "resolved_params_mismatch";
    else if (profile != ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) profile_reason = "profile_not_q4k_q8k_k2560";
    else if (parsed_xclbin_path != runtime_xclbin) profile_reason = "parsed_xclbin_path_mismatch";
    std::fprintf(stderr, "XDNA_PROFILE_PROFILE_SELECTED_CORRECTLY=%s pid=%lu reason=%s\n",
        profile_ok ? "YES" : "NO", pid, profile_reason);

    bool artifacts_ok = profile_ok;
    const char * artifact_reason = profile_ok ? "validated" : "profile_precondition_failed";
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    if (artifacts_ok && !ggml_backend_xdna_hash_file_sha256(runtime_xclbin, digest)) {
        artifacts_ok = false;
        artifact_reason = "xclbin_unreadable";
    }
    if (artifacts_ok) {
        char actual[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        ggml_backend_xdna_format_sha256(digest, actual);
        if (std::strcmp(actual, GGML_XDNA_Q4K_CANONICAL_XCLBIN_SHA) != 0) {
            artifacts_ok = false;
            artifact_reason = "xclbin_sha256_mismatch";
        }
    }
    for (size_t i = 0; artifacts_ok && i < 4; ++i) {
        const char * configured = std::getenv(GGML_XDNA_Q4K_INSTRUCTION_ENVS[i]);
        if (configured == nullptr || configured[0] == '\0') {
            artifacts_ok = false;
            artifact_reason = GGML_XDNA_Q4K_INSTRUCTION_ENVS[i];
            break;
        }
        if (!ggml_backend_xdna_hash_file_sha256(configured, digest)) {
            artifacts_ok = false;
            artifact_reason = "instruction_file_unreadable";
            break;
        }
        char actual[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        ggml_backend_xdna_format_sha256(digest, actual);
        if (std::strcmp(actual, GGML_XDNA_Q4K_CANONICAL_INSTRUCTION_SHAS[i]) != 0) {
            artifacts_ok = false;
            artifact_reason = "instruction_sha256_mismatch";
            break;
        }
    }
    std::fprintf(stderr, "XDNA_PROFILE_PROFILE_ARTIFACTS_VERIFIED=%s pid=%lu reason=%s\n",
        artifacts_ok ? "YES" : "NO", pid, artifact_reason);

    const char * config_reason = ggml_backend_xdna_profile_environment_error(allow_host_preflight_gate);
    const bool config_ok = profile_ok && artifacts_ok && config_reason == nullptr;
    if (config_reason == nullptr && !profile_ok) config_reason = "profile_not_validated";
    if (config_reason == nullptr && !artifacts_ok) config_reason = "artifacts_not_validated";
    std::fprintf(stderr, "XDNA_PROFILE_Q4_CONTROL_CONFIG_VALID=%s pid=%lu reason=%s\n",
        config_ok ? "YES" : "NO", pid, config_ok ? "validated" : config_reason);
    const bool pass = profile_ok && artifacts_ok && config_ok;
    const char * final_reason = pass ? "validated" :
        (!profile_ok ? profile_reason : (!artifacts_ok ? artifact_reason : config_reason));
    if (failure_reason_out != nullptr) *failure_reason_out = pass ? nullptr : final_reason;
    std::fprintf(stderr, "XDNA_PROFILE_PROFILE_PREFLIGHT_RESULT=%s pid=%lu reason=%s\n",
        pass ? "PASS" : "FAIL", pid, final_reason);
    std::fflush(stderr);
    return pass;
}

static void ggml_backend_xdna_log_host_only_stop(
        bool preflight_passed, const char * failure_reason) {
    std::fprintf(stderr,
        "XDNA_HOST_PREFLIGHT_STOP pid=%lu stage=AFTER_PROFILE_PREFLIGHT preflight=%s backend_context_created=NO computational_xrt_setup=NOT_REACHED backend_return=NULL reason=%s",
        ggml_backend_xdna_profile_process_id(), preflight_passed ? "PASS" : "FAIL",
        preflight_passed ? "preflight_only" : "preflight_rejected");
    if (!preflight_passed && failure_reason != nullptr) {
        std::fprintf(stderr, " profile_reason=%s", failure_reason);
    }
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

static const char * ggml_backend_xdna_q4_control_conflicting_env(bool allow_host_preflight_gate) {
    static const char * const names[] = {
        "GGML_XDNA_BH1_REPLAY_F32", "GGML_XDNA_BH1_REPLAY_NATIVE_Q4",
        "GGML_XDNA_BH1_REPLAY_Q4_TRANSPORT", "GGML_XDNA_BH1_REPLAY_Q8",
        "GGML_XDNA_CAPTURE_EXACT_HISTORY", "GGML_XDNA_CAPTURE_M9216_PAYLOAD",
        "GGML_XDNA_CAPTURE_HISTORY_DIR", "GGML_XDNA_BQ1_STAGE",
        "GGML_XDNA_DIAG_RECREATE_M9216_BUNDLE_BEFORE_9",
        "GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_BEGIN", "GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_DIR",
        "GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_END", "GGML_XDNA_BACKEND_PARAMS",
        "GGML_XDNA_DIAG_RECREATE_M9216_KERNEL_BEFORE_9",
        "GGML_XDNA_DIAG_TIMED_WAIT", "GGML_XDNA_W2IN_BLOCK_PROBE",
        "GGML_XDNA_FORCE_SPLIT_SYNC", "GGML_XDNA_STOP_AFTER_RUNTIME_SPLIT",
        "GGML_XDNA_DIAG_STOP_AFTER_XRT_MODULE_PREFLIGHT",
        "GGML_XDNA_DIAG_A7H2_HOST_PREFLIGHT_ONLY",
        "GGML_XDNA_BO_FORENSICS", "GGML_XDNA_INSTANCE_TRACE",
        "GGML_XDNA_RUNTIME_BATCH_TRACE", "GGML_XDNA_RUN_STATE_TRACE",
        "GGML_XDNA_SCHED_PROVENANCE_TRACE", "GGML_XDNA_SPLIT_ORDER_TRACE",
        "GGML_CPU_OP_PROFILE",
        "GGML_SCHED_DEBUG", "GGML_SCHED_DEBUG_REALLOC",
    };
    for (const char * name : names) {
        const char * value = std::getenv(name);
        if (value == nullptr) continue;
        if (std::strcmp(name, "GGML_XDNA_BACKEND_PARAMS") == 0 &&
            std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8") != nullptr &&
            std::strcmp(std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8"), "1") == 0) {
            const char * runtime_xclbin = std::getenv("GGML_XDNA_Q4K_RUNTIME_XCLBIN");
            if (runtime_xclbin != nullptr) {
                const std::string expected_params = std::string("profile=q4k_q8k_k2560;") + runtime_xclbin;
                if (std::strcmp(value, expected_params.c_str()) == 0) continue;
            }
        }
        if (allow_host_preflight_gate &&
            std::strcmp(name, "GGML_XDNA_DIAG_A7H2_HOST_PREFLIGHT_ONLY") == 0 &&
            std::strcmp(value, "1") == 0 &&
            std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8") != nullptr &&
            std::strcmp(std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8"), "1") == 0 &&
            std::getenv("GGML_XDNA_DIAG_STOP_AFTER_XRT_MODULE_PREFLIGHT") == nullptr) continue;
        return name;
    }
    return nullptr;
}
#endif

static uint64_t ggml_backend_xdna_elapsed_ns(
        const ggml_backend_xdna_backend_context * ctx,
        std::chrono::steady_clock::time_point point = std::chrono::steady_clock::now()) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        point - ctx->diagnostics_epoch).count());
}

static uint64_t ggml_backend_xdna_new_resource_generation(
        ggml_backend_xdna_backend_context * ctx) {
    return ++ctx->m9216_next_resource_generation;
}

static const char * ggml_backend_xdna_log_offline(
        ggml_backend_xdna_backend_context * ctx, const char * marker,
        uint64_t ordinal, uint64_t correlation_id, uint64_t xrt_sequence,
        const char * node_name, int node_index) noexcept {
    try {
        const bool offline = ctx->device_context->xrt_device->get_info<xrt::info::device::offline>();
        std::fprintf(stderr,
            "%s backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d value=%d\n",
            marker, static_cast<unsigned long long>(ctx->backend_instance_id),
            static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(correlation_id),
            static_cast<unsigned long long>(xrt_sequence), node_name, node_index, offline ? 1 : 0);
        return offline ? "1" : "0";
    } catch (const std::exception & e) {
        std::fprintf(stderr,
            "%s backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d value=query_error category=std_exception error=%s\n",
            marker, static_cast<unsigned long long>(ctx->backend_instance_id),
            static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(correlation_id),
            static_cast<unsigned long long>(xrt_sequence), node_name, node_index, e.what());
        return "query_error";
    } catch (...) {
        std::fprintf(stderr,
            "%s backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d value=query_error category=unknown\n",
            marker, static_cast<unsigned long long>(ctx->backend_instance_id),
            static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(correlation_id),
            static_cast<unsigned long long>(xrt_sequence), node_name, node_index);
        return "query_error";
    }

}

static void ggml_backend_xdna_log_correlation(
        const ggml_backend_xdna_backend_context * ctx, uint64_t ordinal,
        uint64_t correlation_id, uint64_t xrt_sequence, const char * node_name,
        int node_index, ert_cmd_state pre_wait_state, ert_cmd_state returned_state,
        uint64_t kernel_ns, uint64_t state_query_ns, uint64_t wait_ns,
        uint64_t output_sync_ns, const char * output_sync,
        const char * offline_pre, const char * offline_post,
        uint64_t persistent_q4_generation, const char * q4_sha, const char * q8_sha) {
    std::fprintf(stderr,
        "XDNA_TRACE_CORRELATION backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d M=9216 K=2560 RHS=1 kernel_ns=%llu state_query_ns=%llu wait_ns=%llu output_sync_ns=%llu pre_wait_state=%d:%s returned_state=%d:%s hw_context_gen=%llu kernel_gen=%llu instruction_bo_gen=%llu shared_q4_bo_gen=%llu q8_bo_gen=%llu output_bo_gen=%llu persistent_q4_bo_gen=%llu q4_transport_sha256=%s q8_sha256=%s offline_pre=%s offline_post=%s output_sync=%s\n",
        static_cast<unsigned long long>(ctx->backend_instance_id),
        static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(correlation_id),
        static_cast<unsigned long long>(xrt_sequence), node_name, node_index,
        static_cast<unsigned long long>(kernel_ns), static_cast<unsigned long long>(state_query_ns),
        static_cast<unsigned long long>(wait_ns), static_cast<unsigned long long>(output_sync_ns),
        static_cast<int>(pre_wait_state), ggml_backend_xdna_ert_cmd_state_name(pre_wait_state),
        static_cast<int>(returned_state), ggml_backend_xdna_ert_cmd_state_name(returned_state),
        static_cast<unsigned long long>(ctx->m9216_hw_context_generation),
        static_cast<unsigned long long>(ctx->m9216_kernel_generation),
        static_cast<unsigned long long>(ctx->m9216_instruction_bo_generation),
        static_cast<unsigned long long>(ctx->m9216_shared_q4_bo_generation),
        static_cast<unsigned long long>(ctx->m9216_q8_bo_generation),
        static_cast<unsigned long long>(ctx->m9216_output_bo_generation),
        static_cast<unsigned long long>(persistent_q4_generation), q4_sha, q8_sha,
        offline_pre, offline_post, output_sync);
}

static ggml_xdna_causal_capture::tensor_identity ggml_backend_xdna_causal_tensor_identity(const struct ggml_tensor * tensor) {
    ggml_xdna_causal_capture::tensor_identity identity;
    if (tensor == nullptr) return identity;
    identity.name = ggml_get_name(tensor);
    identity.tensor = tensor;
    identity.data = tensor->data;
    for (size_t i = 0; i < 4; ++i) { identity.ne[i] = tensor->ne[i]; identity.nb[i] = tensor->nb[i]; }
    return identity;
}

static ggml_xdna_causal_capture::submission_metadata ggml_backend_xdna_causal_metadata(
        const struct ggml_tensor * node, int graph_node_index, size_t m, int64_t k,
        std::optional<uint64_t> correlation_id, std::optional<uint64_t> ordinal) {
    ggml_xdna_causal_capture::submission_metadata metadata;
    metadata.shape_m = m;
    metadata.shape_k = k;
    metadata.op = node == nullptr ? "" : ggml_op_name(node->op);
    metadata.graph_node_index = graph_node_index;
    metadata.node_name = node == nullptr ? "" : ggml_get_name(node);
    metadata.node = node;
    metadata.src0 = ggml_backend_xdna_causal_tensor_identity(node == nullptr ? nullptr : node->src[0]);
    metadata.src1 = ggml_backend_xdna_causal_tensor_identity(node == nullptr ? nullptr : node->src[1]);
    metadata.dst = ggml_backend_xdna_causal_tensor_identity(node);
    metadata.correlation_id = correlation_id;
    metadata.m9216_ordinal = ordinal;
    return metadata;
}

static std::string ggml_backend_xdna_m9216_name_hex(
        const char * name, size_t capacity) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string encoded;
    if (name == nullptr) return "-";
    for (size_t i = 0; i < capacity && name[i] != '\0'; ++i) {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        encoded.push_back(digits[c >> 4]);
        encoded.push_back(digits[c & 0x0f]);
    }
    return encoded.empty() ? "-" : encoded;
}

static void ggml_backend_xdna_m9216_correlation_event(
        uint64_t id, const char * event, uint64_t ordinal, int node_index) {
    std::fprintf(stderr,
        "XDNA_CORR id=%llu event=%s ordinal=%llu node_index=%d m=9216 run_state=%s\n",
        static_cast<unsigned long long>(id), event,
        static_cast<unsigned long long>(ordinal), node_index, event);
}
static void ggml_backend_xdna_m9216_correlation_prepared(
        uint64_t id, uint64_t ordinal, int node_index,
        const struct ggml_tensor * node,
        const char * q4_native_sha, const char * q8_sha, const char * instruction_sha,
        const void * q4_native_ptr, const void * activation_f32_ptr, const void * q8_host_ptr,
        const void * kernel, const void * hw_context,
        const void * bo_instr, const void * bo_q4, const void * bo_q8, const void * bo_out) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const std::string node_name = ggml_backend_xdna_m9216_name_hex(node->name, sizeof(node->name));
    const std::string src0_name = ggml_backend_xdna_m9216_name_hex(src0->name, sizeof(src0->name));
    const std::string src1_name = ggml_backend_xdna_m9216_name_hex(src1->name, sizeof(src1->name));
    std::fprintf(stderr,
        "XDNA_CORR id=%llu event=SUBMISSION_PREPARED ordinal=%llu node_index=%d m=9216 op=%s node_name_hex=%s node_ptr=%p src0_name_hex=%s src0_tensor=%p src0_data=%p src0_ne=%lld,%lld,%lld,%lld src0_nb=%zu,%zu,%zu,%zu src1_name_hex=%s src1_tensor=%p src1_data=%p src1_ne=%lld,%lld,%lld,%lld src1_nb=%zu,%zu,%zu,%zu dst_tensor=%p dst_data=%p dst_ne=%lld,%lld,%lld,%lld dst_nb=%zu,%zu,%zu,%zu q4_native_ptr=%p activation_f32_ptr=%p q8_host_ptr=%p q4_native_sha256=%s q4_transport_sha256=NOT_USED q8_sha256=%s instruction_sha256=%s kernel=%p hw_context=%p bo_instr=%p bo_q4=%p bo_q8=%p bo_out=%p identity_scope=PROCESS_LOCAL_IDENTITY_ONLY run_state=SUBMISSION_PREPARED\n",
        static_cast<unsigned long long>(id), static_cast<unsigned long long>(ordinal), node_index,
        ggml_op_name(node->op), node_name.c_str(), static_cast<void *>(const_cast<ggml_tensor *>(node)),
        src0_name.c_str(), static_cast<void *>(const_cast<ggml_tensor *>(src0)), src0->data,
        static_cast<long long>(src0->ne[0]), static_cast<long long>(src0->ne[1]), static_cast<long long>(src0->ne[2]), static_cast<long long>(src0->ne[3]),
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1_name.c_str(), static_cast<void *>(const_cast<ggml_tensor *>(src1)), src1->data,
        static_cast<long long>(src1->ne[0]), static_cast<long long>(src1->ne[1]), static_cast<long long>(src1->ne[2]), static_cast<long long>(src1->ne[3]),
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        static_cast<void *>(const_cast<ggml_tensor *>(node)), node->data,
        static_cast<long long>(node->ne[0]), static_cast<long long>(node->ne[1]), static_cast<long long>(node->ne[2]), static_cast<long long>(node->ne[3]),
        node->nb[0], node->nb[1], node->nb[2], node->nb[3],
        q4_native_ptr, activation_f32_ptr, q8_host_ptr,
        q4_native_sha, q8_sha, instruction_sha,
        const_cast<void *>(kernel), const_cast<void *>(hw_context), const_cast<void *>(bo_instr),
        const_cast<void *>(bo_q4), const_cast<void *>(bo_q8), const_cast<void *>(bo_out));
}


#endif

static enum ggml_status ggml_backend_xdna_graph_compute_single_node(
        ggml_backend_t backend,
        struct ggml_tensor * node
#if defined(GGML_XDNA_RUNTIME_TRACE)
        , const ggml_backend_xdna_runtime_trace_node * runtime_trace_node
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        , int graph_node_index
#endif
) {
    if (backend == nullptr) {
        return GGML_STATUS_FAILED;
    }

    if (node == nullptr ||
        node->op != GGML_OP_MUL_MAT ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled graph path requires one MUL_MAT node\n");
        return GGML_STATUS_FAILED;
    }

    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    // Controlled Q4_K graph path.
    //
    // This intentionally does not change supports_op().
    // It can only be reached when graph_compute() is invoked
    // directly on an XDNA backend initialized with the
    // K=2560 Q4_K x Q8_K artifact.
    if (src0->type == GGML_TYPE_Q4_K) {
        constexpr int64_t k = 2560;

        constexpr size_t q4_native_row_bytes =
            10 * sizeof(block_q4_K);

        constexpr size_t f32_activation_bytes =
            static_cast<size_t>(k) * sizeof(float);

        if (src1->type != GGML_TYPE_F32 ||
            node->type != GGML_TYPE_F32) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires Q4_K x F32 -> F32\n");
            return GGML_STATUS_FAILED;
        }

        const int64_t m = src0->ne[1];

        if (src0->ne[0] != k ||
            m <= 0 ||
            src0->ne[2] != 1 ||
            src0->ne[3] != 1 ||
            src1->ne[0] != k ||
            src1->ne[1] != 1 ||
            src1->ne[2] != 1 ||
            src1->ne[3] != 1 ||
            node->ne[0] != m ||
            node->ne[1] != 1 ||
            node->ne[2] != 1 ||
            node->ne[3] != 1) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires [2560,M] x [2560,1] -> [M,1]\n");
            return GGML_STATUS_FAILED;
        }

        if (!ggml_is_contiguous(src0) ||
            !ggml_is_contiguous(src1) ||
            !ggml_is_contiguous(node)) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires contiguous tensors\n");
            return GGML_STATUS_FAILED;
        }

        if (src0->nb[1] != q4_native_row_bytes ||
            src1->nb[1] != f32_activation_bytes ||
            node->nb[0] != sizeof(float)) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT has unexpected tensor strides\n");
            return GGML_STATUS_FAILED;
        }

        if (src0->data == nullptr ||
            src1->data == nullptr ||
            node->data == nullptr) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires host-accessible tensor data\n");
            return GGML_STATUS_FAILED;
        }

        const size_t rows =
            static_cast<size_t>(m);

        const size_t expected_q4_bytes =
            rows * q4_native_row_bytes;

        const size_t expected_dst_bytes =
            rows * sizeof(float);

        const size_t actual_q4_bytes =
            ggml_nbytes(src0);

        const size_t actual_src1_bytes =
            ggml_nbytes(src1);

        const size_t actual_dst_bytes =
            ggml_nbytes(node);

        if (actual_q4_bytes != expected_q4_bytes ||
            actual_src1_bytes != f32_activation_bytes ||
            actual_dst_bytes != expected_dst_bytes) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT has unexpected tensor byte sizes\n");
            return GGML_STATUS_FAILED;
        }

        auto * ctx =
            static_cast<ggml_backend_xdna_backend_context *>(
                backend->context);

        if (ctx == nullptr ||
            !ggml_backend_xdna_load_instructions(ctx)) {
            return GGML_STATUS_FAILED;
        }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (!ggml_backend_xdna_diagnostic_capture_can_submit(ctx)) return GGML_STATUS_FAILED;
#endif

        const auto * q4_rows =
            static_cast<const uint8_t *>(
                src0->data);

        const auto * activations =
            static_cast<const float *>(
                src1->data);

        auto * result =
            static_cast<float *>(
                node->data);

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        std::fprintf(
            stderr,
            "ggml_xdna: controlled Q4_K K=2560 MUL_MAT graph accepted for M=%lld\n",
            static_cast<long long>(m));
#endif

        GGML_XDNA_DIAG_ONLY(++ctx->accepted_q4_nodes;)
        if (m == 9216 &&
            ctx->q4_m9216_available) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            uint64_t causal_sequence_id = 0;
            bool causal_selected = false;
            if (!ggml_backend_xdna_causal_capture_reserve(&causal_sequence_id, &causal_selected)) return GGML_STATUS_FAILED;
#endif
            if (!ggml_backend_xdna_run_q4k_q8k_k2560_m9216(
                    backend,
                    q4_rows,
                    expected_q4_bytes,
                    activations,
                    static_cast<size_t>(k),
                    result,
                    expected_dst_bytes
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
                    , node, graph_node_index, causal_sequence_id, causal_selected
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
                    , runtime_trace_node
#endif
                    , ggml_get_name(node), k, src1->ne[1])) {
                std::fprintf(stderr, "ggml_xdna: M=9216 multi-row Q4_K K=2560 execution failed\n");
                return GGML_STATUS_FAILED;
            }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K K=2560 MUL_MAT executed via M=9216 multi-row fast path\n");
#endif

            GGML_XDNA_DIAG_ONLY(++ctx->batched_m9216;)
            return GGML_STATUS_SUCCESS;
        }
        if ((m == 1024 && ctx->q4_m1024.available) || (m == 4096 && ctx->q4_m4096.available) || (m == 8192 && ctx->q4_m8192.available)) {
            auto & state = m == 1024 ? ctx->q4_m1024 : (m == 4096 ? ctx->q4_m4096 : ctx->q4_m8192);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            uint64_t causal_sequence_id = 0;
            bool causal_selected = false;
            if (m == 4096 && !ggml_backend_xdna_causal_capture_reserve(&causal_sequence_id, &causal_selected)) return GGML_STATUS_FAILED;
#endif
            if (!ggml_backend_xdna_run_batched_shape(ctx, state, q4_rows, expected_q4_bytes, activations, static_cast<size_t>(k), result, expected_dst_bytes, ggml_get_name(node), k, src1->ne[1]
#if defined(GGML_XDNA_RUNTIME_TRACE)
                    , runtime_trace_node
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
                    , node, graph_node_index, causal_sequence_id, causal_selected
#endif
                    )) return GGML_STATUS_FAILED;
            GGML_XDNA_DIAG_ONLY(if (m == 1024) ++ctx->batched_m1024; else if (m == 4096) ++ctx->batched_m4096; else ++ctx->batched_m8192;)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(stderr, "ggml_xdna: controlled Q4_K K=2560 MUL_MAT executed via M=%lld batched fast path\n", static_cast<long long>(m));
#endif
            return GGML_STATUS_SUCCESS;
        }


        // Fallback primitive: one K=2560 dot product per launch.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (ggml_backend_xdna_diagnostic_capture_enabled(ctx)) {
            const uint64_t id = ggml_backend_xdna_diagnostic_capture_reserve(ctx);
            if (id == 0) return GGML_STATUS_FAILED;
            ggml_backend_xdna_diagnostic_capture_fail(ctx, id, "prepare", "unsupported_shape", "capture refuses uncaptured fallback submissions");
            return GGML_STATUS_FAILED;
        }
#endif
        GGML_XDNA_DIAG_ONLY(++ctx->fallback_rows;)
        // Quantize/upload the shared activation on the first row, then
        // reuse the persistent Q8_K BO for all remaining rows.
        for (size_t row = 0; row < rows; ++row) {
            const void * q4_row =
                q4_rows +
                row * q4_native_row_bytes;

            if (!ggml_backend_xdna_run_q4k_q8k_k2560(
                    backend,
                    ctx->instructions.data(),
                    static_cast<uint32_t>(
                        ctx->instructions.size()),
                    q4_row,
                    q4_native_row_bytes,
                    activations,
                    static_cast<size_t>(k),
                    row == 0,
                    &result[row]
#if defined(GGML_XDNA_RUNTIME_TRACE)
                    , runtime_trace_node, static_cast<int64_t>(row)
#endif
                    )) {
                std::fprintf(
                    stderr,
                    "ggml_xdna: controlled Q4_K K=2560 MUL_MAT execution failed at row %zu\n",
                    row);
                return GGML_STATUS_FAILED;
            }
        }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        std::fprintf(
            stderr,
            "ggml_xdna: controlled Q4_K K=2560 MUL_MAT executed on XDNA for M=%lld\n",
            static_cast<long long>(m));
#endif

        return GGML_STATUS_SUCCESS;
    }

    if (src0->type != GGML_TYPE_I16 ||
        src1->type != GGML_TYPE_I16 ||
        node->type != GGML_TYPE_F32) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires I16 x I16 -> F32\n");
        return GGML_STATUS_FAILED;
    }

    constexpr int64_t size = 256;

    if (src0->ne[0] != size ||
        src0->ne[1] != size ||
        src0->ne[2] != 1 ||
        src0->ne[3] != 1 ||
        src1->ne[0] != size ||
        src1->ne[1] != size ||
        src1->ne[2] != 1 ||
        src1->ne[3] != 1 ||
        node->ne[0] != size ||
        node->ne[1] != size ||
        node->ne[2] != 1 ||
        node->ne[3] != 1) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires 256x256 tensors\n");
        return GGML_STATUS_FAILED;
    }

    if (!ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(node)) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires contiguous tensors\n");
        return GGML_STATUS_FAILED;
    }

    if (src0->data == nullptr ||
        src1->data == nullptr ||
        node->data == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires host-accessible tensor data\n");
        return GGML_STATUS_FAILED;
    }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    std::fprintf(
        stderr,
        "ggml_xdna: controlled 256x256 I16 MUL_MAT graph accepted\n");
#endif

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(backend->context);

    if (ctx == nullptr || !ggml_backend_xdna_load_instructions(ctx)) {
        return GGML_STATUS_FAILED;
    }

    constexpr size_t elements = 256 * 256;

    const auto * src0_data =
        static_cast<const int16_t *>(src0->data);
    const auto * src1_data =
        static_cast<const int16_t *>(src1->data);

    auto * dst_data =
        static_cast<float *>(node->data);


    // GGML MUL_MAT semantics:
    //
    // dst[m, n] = dot(src0[n, :], src1[m, :])
    //
    // The NPU kernel computes C = A * B, therefore:
    // A = src1
    // B = transpose(src0)
    std::vector<int16_t> src0_transposed(elements);


    for (int64_t row = 0; row < size; ++row) {
        for (int64_t col = 0; col < size; ++col) {
            src0_transposed[
                static_cast<size_t>(col) * size + row] =
                src0_data[
                    static_cast<size_t>(row) * size + col];
        }
    }

    if (!ggml_backend_xdna_run_i16_f32_matmul(
            backend,
            ctx->instructions.data(),
            static_cast<uint32_t>(ctx->instructions.size()),
            src1_data,
            elements,
            src0_transposed.data(),
            elements,
            dst_data,
            elements)) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT execution failed\n");
        return GGML_STATUS_FAILED;
    }


#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    std::fprintf(
        stderr,
        "ggml_xdna: controlled 256x256 I16 MUL_MAT executed on XDNA\n");
#endif

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_xdna_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    if (backend == nullptr || cgraph == nullptr) {
        return GGML_STATUS_FAILED;
    }

    const int n_nodes =
        ggml_graph_n_nodes(cgraph);

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (std::getenv("GGML_XDNA_ELIGIBILITY_TRACE") != nullptr) {
        std::fprintf(stderr, "ggml_xdna: graph_compute n_nodes=%d\n", n_nodes);
    }
#endif
    if (n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

#if defined(GGML_XDNA_RUNTIME_TRACE)
    const bool trace_requested = ggml_backend_xdna_runtime_trace_enabled();
    const uint64_t runtime_graph_id = trace_requested
        ? ggml_backend_xdna_runtime_trace_next_graph_id() : 0;
    const bool runtime_trace_enabled = trace_requested && runtime_graph_id != 0;
    ggml_backend_sched_trace_context scheduler_context{};
    const bool scheduler_context_valid = ggml_backend_sched_trace_context_get(&scheduler_context) &&
        scheduler_context.node_count == n_nodes &&
        scheduler_context.global_node_index_base >= scheduler_context.split_global_node_start &&
        scheduler_context.global_node_index_base + n_nodes <= scheduler_context.split_global_node_end_exclusive &&
        scheduler_context.local_split_node_index_base >= 0 &&
        scheduler_context.local_split_node_index_base + n_nodes <=
            scheduler_context.split_global_node_end_exclusive - scheduler_context.split_global_node_start;
    const ggml_backend_sched_trace_context * scheduler_context_ptr = scheduler_context_valid ? &scheduler_context : nullptr;
#endif
    int compute_nodes = 0;
    int view_nodes = 0;

    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node =
            ggml_graph_node(
                cgraph,
                i);

        if (node == nullptr) {
            std::fprintf(
                stderr,
                "ggml_xdna: null graph node at index %d\n",
                i);
            return GGML_STATUS_FAILED;
        }

        // Keep this classification exactly aligned with the GGML
        // scheduler's private ggml_is_view_op() helper.
        if (ggml_backend_xdna_is_view_op(node->op)) {
            ++view_nodes;
            continue;
        }
#if defined(GGML_XDNA_RUNTIME_TRACE)
        ggml_backend_xdna_runtime_trace_node trace_node{};
        const ggml_backend_xdna_runtime_trace_node * trace_node_ptr = nullptr;
        if (runtime_trace_enabled && node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr &&
            node->src[1] != nullptr && node->src[0]->type == GGML_TYPE_Q4_K && node->src[0]->ne[0] == 2560) {
            trace_node.graph_id = runtime_graph_id;
            trace_node.node_index = i;
            trace_node.scheduler_graph_id = scheduler_context_valid ? scheduler_context.scheduler_graph_id : 0;
            trace_node.scheduler_split_id = scheduler_context_valid ? scheduler_context.scheduler_split_id : 0;
            trace_node.scheduler_dispatch_id = scheduler_context_valid ? scheduler_context.scheduler_dispatch_id : 0;
            trace_node.global_node_index = scheduler_context_valid
                ? scheduler_context.global_node_index_base + i : 0;
            trace_node.local_split_node_index = scheduler_context_valid
                ? scheduler_context.local_split_node_index_base + i : 0;
            trace_node.split_global_node_start = scheduler_context_valid ? scheduler_context.split_global_node_start : 0;
            trace_node.split_global_node_end_exclusive = scheduler_context_valid ? scheduler_context.split_global_node_end_exclusive : 0;
            trace_node.node_name = ggml_get_name(node);
            trace_node.ggml_op = ggml_op_name(node->op);
            trace_node.m = node->src[0]->ne[1];
            trace_node.k = node->src[0]->ne[0];
            trace_node.rhs = node->src[1]->ne[1];
            trace_node_ptr = &trace_node;
        }
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
        if (runtime_trace_enabled) {
            ggml_backend_xdna_scheduler_trace_node_event(scheduler_context_ptr, runtime_graph_id, i, node);
        }
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (std::getenv("GGML_XDNA_SPLIT_ORDER_TRACE") != nullptr) {
            std::fprintf(stderr, "XDNA_BACKEND_TRACE XDNA node name=%s src0=%s src0_ptr=%p src1=%s src1_ptr=%p dst_ptr=%p\n", node->name, node->src[0] ? node->src[0]->name : "", node->src[0] ? node->src[0]->data : nullptr, node->src[1] ? node->src[1]->name : "", node->src[1] ? node->src[1]->data : nullptr, node->data);
        }
#endif
        const enum ggml_status status =
            ggml_backend_xdna_graph_compute_single_node(
                backend,
                node
#if defined(GGML_XDNA_RUNTIME_TRACE)
, trace_node_ptr
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
, i
#endif
                );

        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(
                stderr,
                "ggml_xdna: graph node %d failed "
                "(op=%d)\n",
                i,
                static_cast<int>(node->op));
            return status;
        }

        ++compute_nodes;
    }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    std::fprintf(
        stderr,
        "ggml_xdna: graph dispatch complete: "
        "%d compute node(s), %d view node(s) skipped\n",
        compute_nodes,
        view_nodes);
#endif

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_xdna_run_i16_f32_matmul(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const int16_t * a,
        size_t a_elements,
        const int16_t * b,
        size_t b_elements,
        float * c,
        size_t c_elements) {
    if (backend == nullptr ||
        instructions == nullptr ||
        instruction_count == 0 ||
        a == nullptr ||
        a_elements == 0 ||
        b == nullptr ||
        b_elements == 0 ||
        c == nullptr ||
        c_elements == 0) {
        std::fprintf(stderr, "ggml_xdna: invalid i16-f32 matmul arguments\n");
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(backend->context);

    if (ctx == nullptr ||
        ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr ||
        ctx->kernel == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16-f32 matmul requires an initialized XRT kernel\n");
        return false;
    }

    try {
        xrt::device & device =
            *ctx->device_context->xrt_device;

        xrt::kernel & kernel =
            *ctx->kernel;

        xrt::bo bo_instr(
            device,
            static_cast<size_t>(instruction_count) * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE,
            kernel.group_id(1));

        xrt::bo bo_a(
            device,
            a_elements * sizeof(int16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(3));

        xrt::bo bo_b(
            device,
            b_elements * sizeof(int16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(4));

        xrt::bo bo_c(
            device,
            c_elements * sizeof(float),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(5));

        void * buf_instr = bo_instr.map<void *>();
        std::memcpy(
            buf_instr,
            instructions,
            static_cast<size_t>(instruction_count) * sizeof(uint32_t));
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        int16_t * buf_a = bo_a.map<int16_t *>();
        std::memcpy(
            buf_a,
            a,
            a_elements * sizeof(int16_t));
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        int16_t * buf_b = bo_b.map<int16_t *>();
        std::memcpy(
            buf_b,
            b,
            b_elements * sizeof(int16_t));
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        float * buf_c = bo_c.map<float *>();
        std::memset(
            buf_c,
            0,
            c_elements * sizeof(float));
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        constexpr unsigned int opcode = 3;

        auto run = kernel(
            opcode,
            bo_instr,
            instruction_count,
            bo_a,
            bo_b,
            bo_c);

        const ert_cmd_state completion_state = ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT);
        if (completion_state != ERT_CMD_STATE_COMPLETED) {
            std::fprintf(stderr, "ggml_xdna: ggml_backend_xdna_run_i16_f32_matmul M=256 K=256 wait_state=%d expected=%d\n",
                         static_cast<int>(completion_state), static_cast<int>(ERT_CMD_STATE_COMPLETED));
            return false;
        }

        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        std::memcpy(
            c,
            buf_c,
            c_elements * sizeof(float));

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16-f32 matmul execution failed: %s\n",
            e.what());
        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16-f32 matmul execution failed: unknown error\n");
        return false;
    }
}

static bool ggml_backend_xdna_init_batched_shape(
        ggml_backend_xdna_backend_context * ctx,
        ggml_backend_xdna_batched_state & state, size_t rows,
        const char * path, const char * instruction_path,
        const char * expected_xclbin,
        const char * expected_instructions) {
    if (ctx == nullptr || ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr || path == nullptr || path[0] == '\0') return false;
    try {
        unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
        if (!ggml_backend_xdna_hash_file_sha256(path, digest)) return false;
        char actual[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        ggml_backend_xdna_format_sha256(digest, actual);
        if (std::string(actual) != expected_xclbin) {
            std::fprintf(stderr, "ggml_xdna: M=%zu batched XCLBIN fingerprint mismatch: %s\n", rows, actual);
            return false;
        }
        const std::filesystem::path artifact_path(path);
        const std::string inst_path = instruction_path != nullptr && instruction_path[0] != '\0'
            ? instruction_path
            : (artifact_path.parent_path() / "insts.bin").string();
        std::ifstream file(inst_path, std::ios::binary | std::ios::ate);
        const std::streamsize size = file ? file.tellg() : 0;
        if (size <= 0 || size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) return false;
        state.instructions.resize(static_cast<size_t>(size) / sizeof(uint32_t));
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(state.instructions.data()), size)) return false;
        ggml_xdna_sha256_hash(digest, reinterpret_cast<const unsigned char *>(state.instructions.data()), static_cast<size_t>(size));
        ggml_backend_xdna_format_sha256(digest, actual);
        if (std::string(actual) != expected_instructions) {
            std::fprintf(stderr, "ggml_xdna: M=%zu batched instruction fingerprint mismatch: %s\n", rows, actual);
            state.instructions.clear(); return false;
        }
        state.rows = rows; state.xclbin_path = path;
        const std::string shape_name = "M" + std::to_string(rows);
        std::string resolved_kernel_name;
        if (!ggml_backend_xdna_acquire_q4_shared_kernel(
                ctx, shape_name.c_str(), state.xclbin_path, ctx->kernel_name.c_str(), false,
                state.xclbin, state.hw_context, state.kernel, resolved_kernel_name)) {
            return false;
        }
        if (resolved_kernel_name != ctx->kernel_name) {
            std::fprintf(stderr, "ggml_xdna: M=%zu kernel identity differs from primary Q4 kernel\n", rows);
            return false;
        }
        state.available = true;
        std::fprintf(stderr, "ggml_xdna: M=%zu batched fast path ready\n", rows);
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ggml_xdna: M=%zu batched init failed: %s\n", rows, e.what());
        state.available = false; return false;
    } catch (...) { state.available = false; return false; }
}

static ggml_backend_xdna_q4_shape ggml_backend_xdna_q4_shape_from_rows(size_t rows) {
    switch (rows) {
        case 9216: return ggml_backend_xdna_q4_shape::m9216;
        case 4096: return ggml_backend_xdna_q4_shape::m4096;
        case 8192: return ggml_backend_xdna_q4_shape::m8192;
        default:   return ggml_backend_xdna_q4_shape::m1024;
    }
}

// Applies the persistent Q4_K -> 152-byte transport cache when enabled by
// GGML_XDNA_Q4K_TRANSPORT_CACHE or, if disabled/uninitialized, performs the
// exact existing native -> transport conversion directly. Either way,
// `dst_transport` (the mapped Q4 execution BO buffer) ends up holding valid
// transport bytes for `rows` rows; nothing about Q4 BO sync/XRT/output
// changes. `q4_native_rows` is used as the cache key: it is the ggml
// tensor's data pointer, stable for the lifetime of an immutable weight
// tensor within one model/context.
static bool ggml_backend_xdna_q4_cache_apply(
        ggml_backend_xdna_backend_context * ctx,
        ggml_backend_xdna_q4_shape shape,
        const void * q4_native_rows,
        size_t rows,
        void * dst_transport) {
    constexpr size_t block_count = 10;
    constexpr size_t q4_transport_block_bytes = 152;
    constexpr size_t q4_native_row_bytes = block_count * sizeof(block_q4_K);
    constexpr size_t q4_transport_row_bytes = block_count * q4_transport_block_bytes;

    if (ctx == nullptr || q4_native_rows == nullptr || dst_transport == nullptr || rows == 0) {
        return false;
    }

    const size_t native_bytes    = rows * q4_native_row_bytes;
    const size_t transport_bytes = rows * q4_transport_row_bytes;

    auto & cache = ctx->q4_cache;
    if (!cache.env_checked) {
        cache.enabled = ggml_backend_xdna_env_flag_enabled("GGML_XDNA_Q4K_TRANSPORT_CACHE");
        cache.env_checked = true;
        std::fprintf(stderr, "ggml_xdna: Q4 transport cache %s\n", cache.enabled ? "ENABLED" : "disabled");
    }

    auto convert_into = [&](uint8_t * dst) {
        GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
        const auto * native = static_cast<const uint8_t *>(q4_native_rows);
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t * native_row    = native + row * q4_native_row_bytes;
            uint8_t       * transport_row = dst    + row * q4_transport_row_bytes;
            for (size_t block = 0; block < block_count; ++block) {
                block_q4_K b;
                std::memcpy(&b, native_row + block * sizeof(b), sizeof(b));
                uint8_t * out = transport_row + block * q4_transport_block_bytes;
                std::memcpy(out, &b, sizeof(b));
                const float d    = ggml_fp16_to_fp32(b.d);
                const float dmin = ggml_fp16_to_fp32(b.dmin);
                std::memcpy(out + sizeof(b), &d, sizeof(d));
                std::memcpy(out + sizeof(b) + sizeof(d), &dmin, sizeof(dmin));
            }
        }
        GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
        GGML_XDNA_DIAG_ONLY(ctx->timers.q4_conversion_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        GGML_XDNA_DIAG_ONLY(cache.blocks_converted += rows * block_count;)
        GGML_XDNA_DIAG_ONLY(cache.native_bytes_converted += native_bytes;)
        GGML_XDNA_DIAG_ONLY(cache.transport_bytes_generated += transport_bytes;)
    };

    if (!cache.enabled) {
        convert_into(static_cast<uint8_t *>(dst_transport));
        return true;
    }

    auto it = cache.entries.find(q4_native_rows);
    const bool valid_hit =
        it != cache.entries.end() &&
        it->second.m               == rows &&
        it->second.native_bytes    == native_bytes &&
        it->second.transport_bytes == transport_bytes &&
        it->second.transport.size() == transport_bytes;

    ggml_backend_xdna_q4_cache_entry * entry = nullptr;

    if (valid_hit) {
        GGML_XDNA_DIAG_ONLY(++cache.hits;)
        switch (shape) {
            case ggml_backend_xdna_q4_shape::m9216: GGML_XDNA_DIAG_ONLY(++cache.hits_m9216;) break;
            case ggml_backend_xdna_q4_shape::m4096: GGML_XDNA_DIAG_ONLY(++cache.hits_m4096;) break;
            case ggml_backend_xdna_q4_shape::m8192: GGML_XDNA_DIAG_ONLY(++cache.hits_m8192;) break;
            case ggml_backend_xdna_q4_shape::m1024: GGML_XDNA_DIAG_ONLY(++cache.hits_m1024;) break;
        }
        entry = &it->second;
    } else {
        // Miss, or a stale entry whose metadata no longer matches (e.g. a
        // recycled pointer): rebuild safely rather than trust stale bytes.
        GGML_XDNA_DIAG_ONLY(++cache.misses;)
        ggml_backend_xdna_q4_cache_entry fresh;
        fresh.m               = rows;
        fresh.native_bytes    = native_bytes;
        fresh.transport_bytes = transport_bytes;
        fresh.transport.assign(transport_bytes, 0);
        convert_into(fresh.transport.data());
        entry = &(cache.entries[q4_native_rows] = std::move(fresh));
    }

    GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
    std::memcpy(dst_transport, entry->transport.data(), transport_bytes);
    GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
    GGML_XDNA_DIAG_ONLY(ctx->timers.q4_cached_memcpy_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
    GGML_XDNA_DIAG_ONLY(cache.transport_bytes_copied += transport_bytes;)

    return true;
}
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
bool ggml_backend_xdna_diagnostic_convert_q4_native(
        ggml_backend_xdna_backend_context * ctx, size_t rows, const void * native, void * transport) {
    if (rows != 4096 && rows != 9216) return false;
    return ggml_backend_xdna_q4_cache_apply(ctx, ggml_backend_xdna_q4_shape_from_rows(rows), native, rows, transport);
}
#endif

static bool ggml_backend_xdna_run_batched_shape(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_batched_state & state,
        const void * q4_rows, size_t q4_rows_bytes, const float * activations,
        size_t activation_elements, float * result, size_t result_bytes,
        const char * node_name, int64_t k, int64_t rhs
#if defined(GGML_XDNA_RUNTIME_TRACE)
        , const ggml_backend_xdna_runtime_trace_node * trace_node
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        , const struct ggml_tensor * node, int graph_node_index, uint64_t causal_sequence_id, bool causal_selected
#endif
        ) {
    constexpr size_t block_count = 10, q4_transport_block_bytes = 152;
    constexpr size_t q4_native_row_bytes = block_count * sizeof(block_q4_K);
    constexpr size_t q4_transport_row_bytes = block_count * q4_transport_block_bytes;
    constexpr size_t q8_row_bytes = block_count * sizeof(block_q8_K);
    if (!ctx || !state.available || state.kernel == nullptr || q4_rows == nullptr || activations == nullptr || result == nullptr ||
        q4_rows_bytes != state.rows * q4_native_row_bytes || activation_elements != 2560 || result_bytes != state.rows * sizeof(float)) return false;
#if defined(GGML_XDNA_RUNTIME_TRACE)
    uint64_t trace_submission_id = 0;
    const char * trace_kernel_name = ctx->kernel_name.c_str();
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const bool history_capture = ggml_backend_xdna_diagnostic_capture_enabled(ctx);
    uint64_t history_id = 0;
    const char * history_failure_phase = "prepare";
    if (history_capture) {
        if (!ggml_backend_xdna_diagnostic_capture_can_submit(ctx)) return false;
        history_id = ggml_backend_xdna_diagnostic_capture_reserve(ctx);
        if (history_id == 0) return false;
    }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    static uint64_t trace_sequence_id = 0;
#endif
#endif
    try {
        auto & device = *ctx->device_context->xrt_device;
        auto & kernel = *state.kernel;
        const uint32_t instruction_count = static_cast<uint32_t>(state.instructions.size());
        const size_t instruction_bytes = static_cast<size_t>(instruction_count) * sizeof(uint32_t);
        if (!state.bos_initialized) {
            GGML_XDNA_XRT_TRACE(ctx, ("M" + std::to_string(state.rows)).c_str(),
                "BO_SET_CREATE", "ALLOCATE_AND_MAP", "BEGIN", ctx->device_context->q4_shared.get(),
                state.kernel.get(), "NONE", "allocate instruction, Q4, Q8 and output BOs");
            state.bo_instr = std::make_unique<xrt::bo>(device, instruction_count * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
            state.bo_q4 = std::make_unique<xrt::bo>(device, state.rows * q4_transport_row_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
            state.bo_q8 = std::make_unique<xrt::bo>(device, q8_row_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
            state.bo_out = std::make_unique<xrt::bo>(device, state.rows * sizeof(float), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
            state.buf_instr = state.bo_instr->map<void *>(); state.buf_q4 = state.bo_q4->map<void *>();
            state.buf_q8 = state.bo_q8->map<void *>(); state.buf_out = state.bo_out->map<float *>();
            std::memcpy(state.buf_instr, state.instructions.data(), instruction_count * sizeof(uint32_t));
            // K=2560 integration: mandatory host ordering rule.
            ggml_backend_xdna_sync_host_written_bo(*state.bo_instr, "K=2560 batched instruction BO (CACHEABLE)");
            state.instruction_count = instruction_count; state.bos_initialized = true;
            GGML_XDNA_XRT_TRACE(ctx, ("M" + std::to_string(state.rows)).c_str(),
                "BO_SET_CREATE", "ALLOCATE_AND_MAP", "PASS", ctx->device_context->q4_shared.get(),
                state.kernel.get(), "NONE", "backend-owned BO set mapped and instruction BO synced");
        }
        else if (state.instruction_count != instruction_count || std::memcmp(state.buf_instr, state.instructions.data(), instruction_count * sizeof(uint32_t)) != 0) return false;
        if (!ggml_backend_xdna_q4_cache_apply(ctx, ggml_backend_xdna_q4_shape_from_rows(state.rows), q4_rows, state.rows, state.buf_q4)) {
            return false;
        }
        {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
            std::vector<block_q8_K> q8(block_count); std::memset(q8.data(), 0, q8_row_bytes);
            quantize_row_q8_K_ref(activations, q8.data(), 2560); std::memcpy(state.buf_q8, q8.data(), q8_row_bytes);
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(ctx->timers.q8_quant_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        }
        {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
            ggml_backend_xdna_sync_host_written_bo(*state.bo_q4, "K=2560 batched q4 transport (HOST_ONLY)"); GGML_XDNA_DIAG_ONLY(++ctx->q4_to_device;)
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(ctx->timers.q4_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        }
        {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
            ggml_backend_xdna_sync_host_written_bo(*state.bo_q8, "K=2560 batched q8 activation (HOST_ONLY)"); GGML_XDNA_DIAG_ONLY(++ctx->q8_to_device;)
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(ctx->timers.q8_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (causal_selected) {
            if (state.rows != 4096 || node == nullptr) return false;
            const auto metadata = ggml_backend_xdna_causal_metadata(node, graph_node_index, state.rows, k, std::nullopt, std::nullopt);
            if (!ggml_backend_xdna_causal_capture_prepare(causal_sequence_id, metadata, q4_rows, q4_rows_bytes, state.buf_q8, q8_row_bytes)) return false;
        }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (history_capture) {
            bool semantic_target = false;
            if (k != 2560 || rhs != 1 || !node_name ||
                !ggml_backend_xdna_diagnostic_capture_prepare(ctx, history_id, node_name, static_cast<int64_t>(state.rows), k, rhs, 3u,
                    state.buf_instr, instruction_bytes, state.buf_q4, state.rows * q4_transport_row_bytes,
                    state.buf_q8, q8_row_bytes, &semantic_target)) {
                if (!ggml_backend_xdna_diagnostic_capture_session_failed()) ggml_backend_xdna_diagnostic_capture_fail(ctx, history_id, "prepare", "input_validation", "capture input preparation failed");
                return false;
            }
            if (semantic_target) return false;
        }
#endif
        {
            const size_t shape_index = static_cast<size_t>(ggml_backend_xdna_q4_shape_from_rows(state.rows));
            GGML_XDNA_DIAG_ONLY(auto & timing = ctx->shape_timing[shape_index];)
            GGML_XDNA_DIAG_ONLY(const auto submit_begin = std::chrono::steady_clock::now();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            trace_sequence_id++;
#endif
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu BEFORE kernel shape=M%zu node=batched opcode=3 K=2560 rhs=1 instr_bytes=%zu instr_count=%u kernel=%p hw_context=%p group_ids=%u,%u,%u,%u\n",
                    static_cast<unsigned long long>(trace_sequence_id), state.rows, instruction_bytes, instruction_count,
                    static_cast<void *>(state.kernel.get()), static_cast<void *>(state.hw_context.get()),
                    state.kernel->group_id(1), state.kernel->group_id(3), state.kernel->group_id(4), state.kernel->group_id(5));
            }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            history_failure_phase = "submit_record";
            if (history_capture && !ggml_backend_xdna_diagnostic_capture_submitted(ctx, history_id)) return false;
            history_failure_phase = "wait";
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
            trace_submission_id = trace_node == nullptr
                ? 0 : ggml_backend_xdna_runtime_trace_next_submission_id();
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
            auto run = [&]() {
                if (trace_node == nullptr) {
                    return (*state.kernel)(3u, *state.bo_instr, instruction_count, *state.bo_q4, *state.bo_q8, *state.bo_out);
                }
                ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "KERNEL_CALL_ENTER");
                try {
                    return (*state.kernel)(3u, *state.bo_instr, instruction_count, *state.bo_q4, *state.bo_q8, *state.bo_out);
                } catch (const std::exception & e) {
                    ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "KERNEL_CALL_EXCEPTION", nullptr, e.what());
                    throw;
                } catch (...) {
                    ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "KERNEL_CALL_EXCEPTION", nullptr, "unknown exception");
                    throw;
                }
            }();
            if (trace_node != nullptr) ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "KERNEL_CALL_RETURNED");
#else
            auto run = (*state.kernel)(3u, *state.bo_instr, instruction_count, *state.bo_q4, *state.bo_q8, *state.bo_out);
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "KERNEL_CALL_RETURNED");
#endif
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu AFTER kernel run_created shape=M%zu\n", static_cast<unsigned long long>(trace_sequence_id), state.rows);
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu BO instr=%p q4=%p q8=%p out=%p persistent_q4=0\n", static_cast<unsigned long long>(trace_sequence_id), static_cast<void *>(state.bo_instr.get()), static_cast<void *>(state.bo_q4.get()), static_cast<void *>(state.bo_q8.get()), static_cast<void *>(state.bo_out.get()));
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu BEFORE run.wait()\n", static_cast<unsigned long long>(trace_sequence_id));
            }
#endif
            GGML_XDNA_DIAG_ONLY(const auto submit_end = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(timing.submit_ms += std::chrono::duration<double, std::milli>(submit_end - submit_begin).count();)
            GGML_XDNA_DIAG_ONLY(const auto wait_begin = std::chrono::steady_clock::now();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "WAIT_ENTER");
#endif
            GGML_XDNA_DIAG_ONLY(++ctx->xrt_runs; ++ctx->run_waits;)
#if defined(GGML_XDNA_RUNTIME_TRACE)
            const ert_cmd_state completion_state = ggml_backend_xdna_runtime_trace_wait(
                trace_node, trace_submission_id, trace_kernel_name, "batched", -1, run, GGML_XDNA_ERT_WAIT_LIMIT);
#else
            const ert_cmd_state completion_state = ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT);
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "WAIT_RETURN");
#endif
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu AFTER run.wait()\n", static_cast<unsigned long long>(trace_sequence_id));
            }
#endif
            if (completion_state != ERT_CMD_STATE_COMPLETED) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
                std::fprintf(stderr,
                    "ggml_xdna: batched Q4 wait failed M=%zu K=%zu state=%d name=%s expected=COMPLETED\n",
                    state.rows, k, static_cast<int>(completion_state), ggml_backend_xdna_ert_cmd_state_name(completion_state));
                if (history_capture) ggml_backend_xdna_diagnostic_capture_fail(ctx, history_id, "wait", "wait_incomplete", "XRT wait returned non-COMPLETED state");
#else
                std::fprintf(stderr,
                    "ggml_xdna: ggml_backend_xdna_run_batched_shape M=%zu K=%zu wait_state=%d expected=%d\n",
                    state.rows, k, static_cast<int>(completion_state), static_cast<int>(ERT_CMD_STATE_COMPLETED));
#endif
                return false;
            }
            GGML_XDNA_DIAG_ONLY(const auto wait_end = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(timing.wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();)
            GGML_XDNA_DIAG_ONLY(++timing.ops;)
            GGML_XDNA_DIAG_ONLY(ctx->timers.xrt_submit_wait_ms += std::chrono::duration<double, std::milli>(wait_end - submit_begin).count();)
        }
        {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu BEFORE output sync shape=M%zu\n", static_cast<unsigned long long>(trace_sequence_id), state.rows);
            }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            history_failure_phase = "output_sync";
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
            if (trace_node != nullptr) {
                try {
                    state.bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                } catch (const std::exception & e) {
                    ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "OUTPUT_SYNC_EXCEPTION", "ERT_CMD_STATE_COMPLETED", e.what());
                    throw;
                } catch (...) {
                    ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "OUTPUT_SYNC_EXCEPTION", "ERT_CMD_STATE_COMPLETED", "unknown exception");
                    throw;
                }
                ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "OUTPUT_SYNC_RETURNED", "ERT_CMD_STATE_COMPLETED");
            } else
#endif
            state.bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            GGML_XDNA_DIAG_ONLY(++ctx->output_from_device;)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "OUTPUT_SYNC_RETURN");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (history_capture && !ggml_backend_xdna_diagnostic_capture_completed(ctx, history_id, state.buf_out, result_bytes)) return false;
#endif
            std::memcpy(result, state.buf_out, result_bytes);
#if defined(GGML_XDNA_RUNTIME_TRACE)
            if (trace_node != nullptr) ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "batched", -1, "RESULT_COPY_DONE", "ERT_CMD_STATE_COMPLETED");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "RESULT_COPY_RETURN");
#endif
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu AFTER output sync shape=M%zu\n", static_cast<unsigned long long>(trace_sequence_id), state.rows);
            }
#endif
            GGML_XDNA_DIAG_ONLY(ctx->timers.output_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        }
        return true;
    } catch (...) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (history_capture) ggml_backend_xdna_diagnostic_capture_fail(ctx, history_id, history_failure_phase, "submission_or_recording", "captured submission failed");
#endif
        return false;
    }
}
// persistent Q4_K weights
//
// Returns a persistent, already-filled, already-synced Q4 transport BO for one
// M=9216 weight tensor, or nullptr when the persistent path is disabled,
// unavailable or over budget. On nullptr the caller uses the existing shared
// BO plus the existing per-call convert/cache path, so behaviour is unchanged.
//
// The BO is created on first use of that tensor, filled exactly once through the
// existing lossless, deterministic native Q4_K -> 152-byte transport conversion
// (144-byte GGML block + fp32 d + fp32 dmin), synced to the device exactly once,
// and then reused byte-for-byte by every later token.
static xrt::bo * ggml_backend_xdna_m9216_weight_bo(
        ggml_backend_xdna_backend_context * ctx,
        const void * q4_rows,
        size_t transport_bytes) {
    using entry_t =
        ggml_backend_xdna_backend_context::q4_m9216_weight_bo_entry;

    // The M=9216 weight set is ~897 MiB of transport bytes. 2 GiB leaves
    // headroom for the other shapes while refusing unbounded growth.
    constexpr size_t budget_bytes = 2ull * 1024 * 1024 * 1024;

    if (!ctx->q4_weight_bo_env_checked) {
        const char * v = std::getenv("GGML_XDNA_Q4K_PERSISTENT_WEIGHTS");
        ctx->q4_weight_bo_enabled =
            (v == nullptr) || (std::strcmp(v, "0") != 0);
        ctx->q4_weight_bo_env_checked = true;
        std::fprintf(stderr,
            "ggml_xdna: Q4 persistent weight BOs %s\n",
            ctx->q4_weight_bo_enabled ? "ENABLED" : "disabled");
    }

    if (!ctx->q4_weight_bo_enabled) {
        return nullptr;
    }

    auto it = ctx->q4_m9216_weight_bos.find(q4_rows);
    if (it != ctx->q4_m9216_weight_bos.end()) {
        GGML_XDNA_DIAG_ONLY(++ctx->weight_bo_reuses;)
        return it->second.bo.get();
    }

    if (ctx->q4_weight_bo_bytes + transport_bytes > budget_bytes) {
        GGML_XDNA_DIAG_ONLY(++ctx->weight_bo_cap_hits;)
        return nullptr;
    }

    try {
        auto bo = std::make_unique<xrt::bo>(
            *ctx->device_context->xrt_device,
            transport_bytes,
            XRT_BO_FLAGS_HOST_ONLY,
            ctx->q4_m9216_kernel->group_id(3));
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        const uint64_t resource_generation = ggml_backend_xdna_new_resource_generation(ctx);
#endif

        void * mapped = bo->map<void *>();

        if (!ggml_backend_xdna_q4_cache_apply(
                ctx, ggml_backend_xdna_q4_shape::m9216,
                q4_rows, 9216, mapped)) {
            GGML_XDNA_DIAG_ONLY(++ctx->weight_bo_failures;)
            return nullptr;
        }

        ggml_backend_xdna_sync_host_written_bo(
            *bo, "M=9216 persistent weight BO (HOST_ONLY, one-time)");

        entry_t entry;
        entry.mapped_bytes = transport_bytes;
        entry.bo = std::move(bo);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        entry.diagnostic_generation_id = resource_generation;
#endif

        auto & stored = ctx->q4_m9216_weight_bos[q4_rows];
        stored = std::move(entry);

        ctx->q4_weight_bo_bytes += transport_bytes;
        GGML_XDNA_DIAG_ONLY(++ctx->weight_bo_creates;)

        return stored.bo.get();
    } catch (const std::exception & e) {
        GGML_XDNA_DIAG_ONLY(++ctx->weight_bo_failures;)
        std::fprintf(stderr,
            "ggml_xdna: persistent weight BO allocation failed (%.2f MiB): %s\n",
            static_cast<double>(transport_bytes) / (1024.0 * 1024.0),
            e.what());
        return nullptr;
    } catch (...) {
        GGML_XDNA_DIAG_ONLY(++ctx->weight_bo_failures;)
        return nullptr;
    }
}

static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216(
        ggml_backend_t backend,
        const void * q4_rows,
        size_t q4_rows_bytes,
        const float * activations,
        size_t activation_elements,
        float * result,
        size_t result_bytes
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        , const struct ggml_tensor * node, int graph_node_index, uint64_t causal_sequence_id, bool causal_selected
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
        , const ggml_backend_xdna_runtime_trace_node * trace_node
#endif
        , const char * node_name, int64_t k, int64_t rhs) {
    constexpr size_t rows = 9216;
    constexpr int64_t elements = 2560;
    constexpr size_t block_count = 10;
    constexpr size_t q4_transport_block_bytes = 152;

    static_assert(QK_K == 256, "unexpected QK_K");
    static_assert(sizeof(block_q4_K) == 144, "unexpected Q4_K block size");
    static_assert(sizeof(block_q8_K) == 292, "unexpected Q8_K block size");

    constexpr size_t q4_native_row_bytes =
        block_count * sizeof(block_q4_K);

    constexpr size_t q4_transport_row_bytes =
        block_count * q4_transport_block_bytes;

    constexpr size_t q4_transport_bytes =
        rows * q4_transport_row_bytes;

    constexpr size_t q8_row_bytes =
        block_count * sizeof(block_q8_K);

    constexpr size_t output_bytes =
        rows * sizeof(float);

    if (backend == nullptr ||
        q4_rows == nullptr ||
        activations == nullptr ||
        result == nullptr ||
        q4_rows_bytes != rows * q4_native_row_bytes ||
        activation_elements != static_cast<size_t>(elements) ||
        result_bytes != output_bytes) {
        std::fprintf(
            stderr,
            "ggml_xdna: invalid M=9216 multi-row Q4_K x Q8_K arguments\n");
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(
            backend->context);

    if (ctx == nullptr ||
        ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr ||
        !ctx->q4_m9216_available ||
        ctx->q4_m9216_kernel == nullptr ||
        ctx->q4_m9216_instructions.empty()) {
        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row fast path is not initialized\n");
        return false;
    }
#if defined(GGML_XDNA_RUNTIME_TRACE)
    uint64_t trace_submission_id = 0;
    const char * trace_kernel_name = ctx->kernel_name.c_str();
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const char * q4_control_gate_value = std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8");
    const bool q4_control_requested = q4_control_gate_value != nullptr && std::strcmp(q4_control_gate_value, "1") == 0;
    if (q4_control_gate_value != nullptr && !q4_control_requested) {
        std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=invalid_gate_value\n");
        std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=invalid_gate_value\n");
        return false;
    }
    if (!ctx->q4_control_gate_checked) {
        ctx->q4_control_gate_checked = true;
        ctx->q4_control_enabled = q4_control_requested;
        if (q4_control_requested) {
            const char * persistent = std::getenv("GGML_XDNA_Q4K_PERSISTENT_WEIGHTS");
            const char * cache = std::getenv("GGML_XDNA_Q4K_TRANSPORT_CACHE");
            const char * conflict = ggml_backend_xdna_q4_control_conflicting_env();
            const char * profile_env_error = ggml_backend_xdna_profile_environment_error();
            if (persistent == nullptr || std::strcmp(persistent, "0") != 0 ||
                cache == nullptr || std::strcmp(cache, "0") != 0 || conflict != nullptr || profile_env_error != nullptr) {
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=environment_invariant persistent=%s cache=%s conflict=%s profile_env=%s\n",
                    persistent != nullptr ? persistent : "unset", cache != nullptr ? cache : "unset",
                    conflict != nullptr ? conflict : "none", profile_env_error != nullptr ? profile_env_error : "validated");
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=environment_invariant\n");
                return false;
            }
            try {
                ctx->q4_control_snapshot = std::make_unique<uint8_t[]>(q4_transport_bytes);
                ctx->q4_control_snapshot_bytes = q4_transport_bytes;
            } catch (const std::exception & e) {
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=snapshot_preallocation error=%s\n", e.what());
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=snapshot_preallocation\n");
                return false;
            } catch (...) {
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=snapshot_preallocation\n");
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=snapshot_preallocation\n");
                return false;
            }
            std::fprintf(stderr, "XDNA_Q4_CONTROL_SNAPSHOT_PREALLOCATED bytes=%zu before_ordinal=1\n", q4_transport_bytes);
        }
    } else if (ctx->q4_control_enabled != q4_control_requested) {
        std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=gate_changed_during_session\n");
        std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=gate_changed_during_session\n");
        return false;
    }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const uint64_t submission_sequence_id =
        ggml_backend_xdna_m9216_submission_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const bool history_capture = ggml_backend_xdna_diagnostic_capture_enabled(ctx);
    uint64_t history_id = 0;
    const char * history_failure_phase = "prepare";
    if (history_capture) {
        if (!ggml_backend_xdna_diagnostic_capture_can_submit(ctx)) return false;
        history_id = ggml_backend_xdna_diagnostic_capture_reserve(ctx);
        if (history_id == 0) return false;
    }
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const uint64_t capture_index = ++ctx->m9216_diagnostic_ordinal;
    const bool trace_ordinal8_target = capture_index == 8 && k == 2560 && rhs == 1 && std::strcmp(node_name, "ffn_up-3") == 0;
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (ctx->q4_control_enabled) {
        const char * persistent_env = std::getenv("GGML_XDNA_Q4K_PERSISTENT_WEIGHTS");
        const char * cache_env = std::getenv("GGML_XDNA_Q4K_TRANSPORT_CACHE");
        const char * conflicting_env = ggml_backend_xdna_q4_control_conflicting_env();
        const char * profile_env_error = ggml_backend_xdna_profile_environment_error();
        if (persistent_env == nullptr || std::strcmp(persistent_env, "0") != 0 ||
            cache_env == nullptr || std::strcmp(cache_env, "0") != 0 || conflicting_env != nullptr || profile_env_error != nullptr) {
            std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=environment_changed conflict=%s profile_env=%s\n",
                conflicting_env != nullptr ? conflicting_env : "none", profile_env_error != nullptr ? profile_env_error : "validated");
            std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=environment_changed\n");
            return false;
        }
        const bool source_target = capture_index == 7 && rows == 9216 && k == 2560 && rhs == 1 && node_name != nullptr && std::strcmp(node_name, "ffn_gate-3") == 0;
        const bool control_target = capture_index == 8 && rows == 9216 && k == 2560 && rhs == 1 && node_name != nullptr && std::strcmp(node_name, "ffn_up-3") == 0;
        if ((capture_index == 7 && !source_target) || (capture_index == 8 && !control_target) || capture_index > 8) {
            std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=unexpected_ordinal ordinal=%llu node=%s M=%zu K=%lld RHS=%lld\n",
                static_cast<unsigned long long>(capture_index), node_name != nullptr ? node_name : "(null)", rows,
                static_cast<long long>(k), static_cast<long long>(rhs));
            std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=unexpected_ordinal\n");
            return false;
        }
        if (capture_index == 8 && (!ctx->q4_control_snapshot_valid ||
            ctx->q4_control_snapshot == nullptr || ctx->q4_control_snapshot_bytes != q4_transport_bytes)) {
            std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=ordinal7_snapshot_unavailable ordinal=8\n");
            std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=ordinal7_snapshot_unavailable\n");
            return false;
        }
    }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    std::vector<uint8_t> replay_native;
    std::vector<float> replay_activation;
    if (const char * p = std::getenv("GGML_XDNA_BH1_REPLAY_NATIVE_Q4")) {
        replay_native.resize(q4_transport_bytes / 1520 * 1440); std::ifstream in(p, std::ios::binary); if (!in.read(reinterpret_cast<char *>(replay_native.data()), static_cast<std::streamsize>(replay_native.size()))) return false; q4_rows = replay_native.data();
    }
    if (const char * p = std::getenv("GGML_XDNA_BH1_REPLAY_F32")) {
        replay_activation.resize(activation_elements); std::ifstream in(p, std::ios::binary); if (!in.read(reinterpret_cast<char *>(replay_activation.data()), static_cast<std::streamsize>(replay_activation.size() * sizeof(float)))) return false; activations = replay_activation.data();
    }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    char q4_native_sha[65];
    char q8_sha[65] = "NOT_READY";
    char instruction_sha[65] = "NOT_READY";
    char q4_transport_sha[65] = "NOT_CAPTURED";
    ert_cmd_state trace_pre_wait_state = ERT_CMD_STATE_NEW;
    ert_cmd_state trace_wait_returned_state = ERT_CMD_STATE_NEW;
    uint64_t trace_kernel_ns = 0, trace_state_query_ns = 0, trace_wait_ns = 0, trace_output_sync_ns = 0;
    const char * trace_offline_pre = "not_taken";
    const char * trace_offline_post = "not_taken";
    ggml_backend_xdna_m9216_digest(q4_rows, q4_rows_bytes, q4_native_sha);
#endif
    #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    static uint64_t trace_sequence_id = 1000000;
#endif
    try {
        xrt::device & device =
            *ctx->device_context->xrt_device;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        const bool timed_wait_diag = std::getenv("GGML_XDNA_DIAG_TIMED_WAIT") != nullptr;
        const bool recreate_bundle = std::strcmp(node_name, "ffn_gate-4") == 0 && std::getenv("GGML_XDNA_DIAG_RECREATE_M9216_BUNDLE_BEFORE_9") != nullptr;
        if (std::getenv("GGML_XDNA_DIAG_RECREATE_M9216_BUNDLE_BEFORE_9") != nullptr) std::fprintf(stderr, "XDNA_BUFFER_OBJECT node=%s K=%d M=%zu rhs=1 ordinal=%llu recreate=%d\n", node_name, 2560, rows, static_cast<unsigned long long>(capture_index), recreate_bundle ? 1 : 0);
        const char * reset_stage = std::getenv("GGML_XDNA_BQ1_STAGE");
        if (capture_index == 2 && reset_stage != nullptr && std::strcmp(reset_stage, "bo1") == 0) {
            const std::string kernel_name = ctx->q4_m9216_kernel->get_name();
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER BEFORE kernel_reset\n"); ctx->q4_m9216_kernel.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER AFTER kernel_reset\n");
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER BEFORE instr_reset\n"); ctx->q4_m9216_bo_instr.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER AFTER instr_reset\n");
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER BEFORE q4_reset\n"); ctx->q4_m9216_bo_q4.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER AFTER q4_reset\n");
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER BEFORE q8_reset\n"); ctx->q4_m9216_bo_q8.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER AFTER q8_reset\n");
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER BEFORE out_reset\n"); ctx->q4_m9216_bo_out.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER AFTER out_reset\n");
            ctx->q4_m9216_weight_bos.clear(); ctx->q4_m9216_bos_initialized = false;
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER BEFORE hw_context_reset\n"); ctx->q4_m9216_hw_context.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER AFTER hw_context_reset\n");
            device.register_xclbin(*ctx->q4_m9216_xclbin);
            ctx->q4_m9216_hw_context = std::make_shared<xrt::hw_context>(device, ctx->q4_m9216_xclbin->get_uuid());
            ctx->m9216_hw_context_generation = ggml_backend_xdna_new_resource_generation(ctx);
            ctx->q4_m9216_kernel = std::make_unique<xrt::kernel>(*ctx->q4_m9216_hw_context, kernel_name);
            ctx->m9216_kernel_generation = ggml_backend_xdna_new_resource_generation(ctx);
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BUFFER_OBJECT_ORDER RECREATE kernel=%p hw_context=%p groups=%u,%u,%u,%u\n", static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()), ctx->q4_m9216_kernel->group_id(1), ctx->q4_m9216_kernel->group_id(3), ctx->q4_m9216_kernel->group_id(4), ctx->q4_m9216_kernel->group_id(5));
        }
        if (capture_index == 2 && reset_stage != nullptr && std::strcmp(reset_stage, "bo1") != 0 && std::strcmp(reset_stage, "none") != 0) {
            const bool reset_kernel = std::strcmp(reset_stage, "kernel") == 0 || std::strcmp(reset_stage, "context") == 0 || std::strcmp(reset_stage, "xclbin") == 0;
            const bool reset_context = std::strcmp(reset_stage, "context") == 0 || std::strcmp(reset_stage, "xclbin") == 0;
            const bool reset_xclbin = std::strcmp(reset_stage, "xclbin") == 0;
            std::fprintf(stderr, "XDNA_RESOURCE_RESET PARTIAL BEFORE stage=%s m9216_bos=%d weight_bos=%zu\n", reset_stage, ctx->q4_m9216_bos_initialized ? 1 : 0, ctx->q4_m9216_weight_bos.size());
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BEFORE instr_reset\n"); ctx->q4_m9216_bo_instr.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET AFTER instr_reset\n");
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BEFORE q4_reset\n"); ctx->q4_m9216_bo_q4.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET AFTER q4_reset\n");
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BEFORE q8_reset\n"); ctx->q4_m9216_bo_q8.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET AFTER q8_reset\n");
            std::fprintf(stderr, "XDNA_RESOURCE_RESET BEFORE out_reset\n"); ctx->q4_m9216_bo_out.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET AFTER out_reset\n");
            ctx->q4_m9216_weight_bos.clear();
            ctx->q4_m9216_bos_initialized = false;
            if (reset_kernel) {
                const std::string kernel_name = ctx->q4_m9216_kernel->get_name();
                std::fprintf(stderr, "XDNA_RESOURCE_RESET BEFORE kernel_reset\n"); ctx->q4_m9216_kernel.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET AFTER kernel_reset\n");
                if (reset_context) { std::fprintf(stderr, "XDNA_RESOURCE_RESET BEFORE hw_context_reset\n"); ctx->q4_m9216_hw_context.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET AFTER hw_context_reset\n"); }
                if (reset_xclbin) { std::fprintf(stderr, "XDNA_RESOURCE_RESET BEFORE xclbin_reset\n"); ctx->q4_m9216_xclbin.reset(); std::fprintf(stderr, "XDNA_RESOURCE_RESET AFTER xclbin_reset\n"); ctx->q4_m9216_xclbin = std::make_shared<xrt::xclbin>(ctx->q4_m9216_xclbin_path); device.register_xclbin(*ctx->q4_m9216_xclbin); std::fprintf(stderr, "XDNA_RESOURCE_RESET xclbin_recreated uuid=%s\n", ctx->q4_m9216_xclbin->get_uuid().to_string().c_str()); }
                if (reset_context) { ctx->q4_m9216_hw_context = std::make_shared<xrt::hw_context>(device, ctx->q4_m9216_xclbin->get_uuid()); ctx->m9216_hw_context_generation = ggml_backend_xdna_new_resource_generation(ctx); }
                ctx->q4_m9216_kernel = std::make_unique<xrt::kernel>(*ctx->q4_m9216_hw_context, kernel_name);
                ctx->m9216_kernel_generation = ggml_backend_xdna_new_resource_generation(ctx);
                std::fprintf(stderr, "XDNA_RESOURCE_RESET RECREATE stage=%s kernel=%p hw_context=%p groups=%u,%u,%u,%u\n", reset_stage, static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()), ctx->q4_m9216_kernel->group_id(1), ctx->q4_m9216_kernel->group_id(3), ctx->q4_m9216_kernel->group_id(4), ctx->q4_m9216_kernel->group_id(5));
            }
        }
        if (recreate_bundle) {
            const std::string kernel_name = ctx->q4_m9216_kernel->get_name();
            ctx->q4_m9216_kernel.reset();
            ctx->q4_m9216_bo_instr.reset();
            ctx->q4_m9216_bo_q4.reset();
            ctx->q4_m9216_bo_q8.reset();
            ctx->q4_m9216_bo_out.reset();
            ctx->q4_m9216_bos_initialized = false;
            ctx->q4_m9216_hw_context.reset();
            device.register_xclbin(*ctx->q4_m9216_xclbin);
            ctx->q4_m9216_hw_context = std::make_shared<xrt::hw_context>(device, ctx->q4_m9216_xclbin->get_uuid());
            ctx->m9216_hw_context_generation = ggml_backend_xdna_new_resource_generation(ctx);
            ctx->q4_m9216_kernel = std::make_unique<xrt::kernel>(*ctx->q4_m9216_hw_context, kernel_name);
            ctx->m9216_kernel_generation = ggml_backend_xdna_new_resource_generation(ctx);
            std::fprintf(stderr, "XDNA_BUNDLE_RECREATE ordinal=9 kernel=%p hw_context=%p groups=%u,%u,%u,%u\n", static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()), ctx->q4_m9216_kernel->group_id(1), ctx->q4_m9216_kernel->group_id(3), ctx->q4_m9216_kernel->group_id(4), ctx->q4_m9216_kernel->group_id(5));
        }
#endif
        xrt::kernel & kernel =
            *ctx->q4_m9216_kernel;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        ggml_backend_xdna_diagnostic_capture_payload(capture_index, "q4_native.bin", q4_rows, q4_transport_bytes / 1520 * 1440);
        ggml_backend_xdna_diagnostic_capture_payload(capture_index, "activation_f32.bin", activations, activation_elements * sizeof(float));
#endif

        const uint32_t instruction_count =
            static_cast<uint32_t>(
                ctx->q4_m9216_instructions.size());

        const size_t instruction_bytes =
            static_cast<size_t>(instruction_count) *
            sizeof(uint32_t);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        ggml_backend_xdna_m9216_digest(
            ctx->q4_m9216_instructions.data(), instruction_bytes, instruction_sha);
#endif

        if (!ctx->q4_m9216_bos_initialized) {
            GGML_XDNA_XRT_TRACE(ctx, "M9216", "BO_SET_CREATE", "ALLOCATE_AND_MAP", "BEGIN",
                ctx->device_context->q4_shared.get(), ctx->q4_m9216_kernel.get(), "NONE",
                "allocate instruction, Q4, Q8 and output BOs");
            auto bo_instr = std::make_unique<xrt::bo>(
                device,
                instruction_bytes,
                XCL_BO_FLAGS_CACHEABLE,
                kernel.group_id(1));

            auto bo_q4 = std::make_unique<xrt::bo>(
                device,
                q4_transport_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(3));

            auto bo_q8 = std::make_unique<xrt::bo>(
                device,
                q8_row_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(4));

            auto bo_out = std::make_unique<xrt::bo>(
                device,
                output_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(5));

            void * buf_instr =
                bo_instr->map<void *>();

            void * buf_q4 =
                bo_q4->map<void *>();

            void * buf_q8 =
                bo_q8->map<void *>();

            float * buf_out =
                bo_out->map<float *>();

            std::memcpy(
                buf_instr,
                ctx->q4_m9216_instructions.data(),
                instruction_bytes);

            // K=2560 integration: mandatory host ordering rule.
            ggml_backend_xdna_sync_host_written_bo(
                *bo_instr,
                "K=2560 M=9216 instruction BO (CACHEABLE)");

            ctx->q4_m9216_bo_instr =
                std::move(bo_instr);

            ctx->q4_m9216_bo_q4 =
                std::move(bo_q4);

            ctx->q4_m9216_bo_q8 =
                std::move(bo_q8);

            ctx->q4_m9216_bo_out =
                std::move(bo_out);

            ctx->q4_m9216_buf_instr =
                buf_instr;

            ctx->q4_m9216_buf_q4 =
                buf_q4;

            ctx->q4_m9216_buf_q8 =
                buf_q8;

            ctx->q4_m9216_buf_out =
                buf_out;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ctx->m9216_instruction_bo_generation = ggml_backend_xdna_new_resource_generation(ctx);
            ctx->m9216_shared_q4_bo_generation = ggml_backend_xdna_new_resource_generation(ctx);
            ctx->m9216_q8_bo_generation = ggml_backend_xdna_new_resource_generation(ctx);
            ctx->m9216_output_bo_generation = ggml_backend_xdna_new_resource_generation(ctx);
#endif

            ctx->q4_m9216_instruction_count =
                instruction_count;

            ctx->q4_m9216_bos_initialized =
                true;
            GGML_XDNA_XRT_TRACE(ctx, "M9216", "BO_SET_CREATE", "ALLOCATE_AND_MAP", "PASS",
                ctx->device_context->q4_shared.get(), ctx->q4_m9216_kernel.get(), "NONE",
                "backend-owned BO set mapped and instruction BO synced");
        }
        else if (
            ctx->q4_m9216_instruction_count != instruction_count ||
            std::memcmp(
                ctx->q4_m9216_buf_instr,
                ctx->q4_m9216_instructions.data(),
                instruction_bytes) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: persistent M=9216 instruction BO mismatch\n");
            return false;
        }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        ggml_backend_xdna_diagnostic_capture_payload(capture_index, "instructions.bin", ctx->q4_m9216_instructions.data(), instruction_bytes);
#endif

        // persistent Q4_K weights: prefer this tensor's own
        // persistent, already-filled, already-synced BO.  nullptr means the
        // existing shared BO + per-call convert/sync path below is used.
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        const char * replay_q4_path = std::getenv("GGML_XDNA_BH1_REPLAY_Q4_TRANSPORT");
        const char * replay_q8_path = std::getenv("GGML_XDNA_BH1_REPLAY_Q8");
#endif
        xrt::bo * persistent_q4 =
            ggml_backend_xdna_m9216_weight_bo(
                ctx, q4_rows, q4_transport_bytes);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        uint64_t persistent_q4_generation = 0;
        if (persistent_q4 != nullptr) {
            const auto it = ctx->q4_m9216_weight_bos.find(q4_rows);
            if (it != ctx->q4_m9216_weight_bos.end()) persistent_q4_generation = it->second.diagnostic_generation_id;
        }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (recreate_bundle) persistent_q4 = nullptr;
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (replay_q4_path != nullptr) {
            persistent_q4 = nullptr;
            std::ifstream in(replay_q4_path, std::ios::binary);
            if (!in.read(static_cast<char *>(ctx->q4_m9216_buf_q4), static_cast<std::streamsize>(q4_transport_bytes))) return false;
        }
#endif
        xrt::bo & bo_q4 =
            persistent_q4 != nullptr
                ? *persistent_q4
                : *ctx->q4_m9216_bo_q4;

        xrt::bo & bo_q8 =
            *ctx->q4_m9216_bo_q8;

        xrt::bo & bo_out =
            *ctx->q4_m9216_bo_out;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (persistent_q4 == nullptr && replay_q4_path == nullptr) {
#else
        if (persistent_q4 == nullptr) {
#endif
            if (!ggml_backend_xdna_q4_cache_apply(
                    ctx, ggml_backend_xdna_q4_shape::m9216,
                    q4_rows, rows, ctx->q4_m9216_buf_q4)) {
                return false;
            }
        }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (ggml_backend_xdna_diagnostic_capture_payload_enabled()) {
            void * transport = persistent_q4 != nullptr ? persistent_q4->map<void *>() : ctx->q4_m9216_buf_q4;
            ggml_backend_xdna_diagnostic_capture_payload(capture_index, "q4_transport.bin", transport, q4_transport_bytes);
            std::fprintf(stderr, "XDNA_PAYLOAD capture index=%llu q4_mode=%s q4_bo=%p mapped=%p\n", static_cast<unsigned long long>(capture_index), persistent_q4 != nullptr ? "persistent" : "shared", static_cast<void *>(persistent_q4), transport);
        }
#endif
        {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (replay_q8_path != nullptr) {
                std::ifstream in(replay_q8_path, std::ios::binary);
                if (!in.read(static_cast<char *>(ctx->q4_m9216_buf_q8), static_cast<std::streamsize>(q8_row_bytes))) return false;
            } else
#endif
            {
                std::vector<block_q8_K> q8_blocks(block_count);
                std::memset(q8_blocks.data(), 0, q8_row_bytes);
                quantize_row_q8_K_ref(activations, q8_blocks.data(), elements);
                std::memcpy(ctx->q4_m9216_buf_q8, q8_blocks.data(), q8_row_bytes);
            }
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ggml_backend_xdna_diagnostic_capture_payload(capture_index, "q8.bin", ctx->q4_m9216_buf_q8, q8_row_bytes);
#endif
            GGML_XDNA_DIAG_ONLY(ctx->timers.q8_quant_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ggml_backend_xdna_m9216_digest(
                ctx->q4_m9216_buf_q8, q8_row_bytes, q8_sha);
#endif
        }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (ctx->q4_control_enabled && capture_index == 8 && persistent_q4 != nullptr) {
            std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=persistent_q4_in_use\n");
            return false;
        }
#endif
        if (persistent_q4 == nullptr) {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (trace_ordinal8_target) ggml_backend_xdna_m9216_digest(ctx->q4_m9216_buf_q4, q4_transport_bytes, q4_transport_sha);
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (ctx->q4_control_enabled && capture_index == 8) {
                char snapshot_sha_check[65];
                char substituted_sha[65];
                ggml_backend_xdna_m9216_digest(ctx->q4_control_snapshot.get(), q4_transport_bytes, snapshot_sha_check);
                const uint64_t current_generations[7] = {
                    ctx->m9216_hw_context_generation, ctx->m9216_kernel_generation,
                    ctx->m9216_instruction_bo_generation, ctx->m9216_shared_q4_bo_generation,
                    ctx->m9216_q8_bo_generation, ctx->m9216_output_bo_generation, 0,
                };
                const void * current_objects[7] = {
                    ctx->q4_m9216_hw_context.get(), ctx->q4_m9216_kernel.get(), ctx->q4_m9216_bo_instr.get(),
                    ctx->q4_m9216_bo_q4.get(), ctx->q4_m9216_bo_q8.get(), ctx->q4_m9216_bo_out.get(), nullptr,
                };
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ORIGINAL_ORD8 backend_instance_id=%llu correlation_id=%llu xrt_seq=%llu ordinal=8 node=ffn_up-3 bytes=%zu offset=0 original_sha256=%s q8_sha256=%s instruction_sha256=%s xclbin_identity=%p kernel=%p hw_context=%p instruction_bo=%p shared_q4_bo=%p q8_bo=%p output_bo=%p hw_context_gen=%llu kernel_gen=%llu instruction_bo_gen=%llu shared_q4_bo_gen=%llu q8_bo_gen=%llu output_bo_gen=%llu persistent_q4_bo_gen=0\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id + 1),
                    q4_transport_bytes, q4_transport_sha, q8_sha, instruction_sha,
                    static_cast<const void *>(ctx->q4_m9216_xclbin.get()),
                    static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()),
                    static_cast<void *>(ctx->q4_m9216_bo_instr.get()), static_cast<void *>(ctx->q4_m9216_bo_q4.get()),
                    static_cast<void *>(ctx->q4_m9216_bo_q8.get()), static_cast<void *>(ctx->q4_m9216_bo_out.get()),
                    static_cast<unsigned long long>(current_generations[0]), static_cast<unsigned long long>(current_generations[1]),
                    static_cast<unsigned long long>(current_generations[2]), static_cast<unsigned long long>(current_generations[3]),
                    static_cast<unsigned long long>(current_generations[4]), static_cast<unsigned long long>(current_generations[5]));
                const bool invariant_gate = ctx->q4_control_snapshot_valid &&
                    ctx->q4_control_snapshot != nullptr && ctx->q4_control_snapshot_bytes == q4_transport_bytes &&
                    q4_transport_bytes == 14008320 && std::strcmp(snapshot_sha_check, ctx->q4_control_snapshot_sha256) == 0 &&
                    ctx->q4_control_source_backend_id == ctx->backend_instance_id &&
                    ctx->q4_control_source_correlation_id + 1 == submission_sequence_id &&
                    ctx->q4_control_source_xrt_sequence == trace_sequence_id &&
                    ctx->q4_control_source_xclbin_identity == ctx->q4_m9216_xclbin.get() &&
                    current_objects[0] != nullptr && current_objects[1] != nullptr && current_objects[2] != nullptr &&
                    current_objects[3] != nullptr && current_objects[4] != nullptr && current_objects[5] != nullptr &&
                    std::memcmp(current_generations, ctx->q4_control_source_generations, sizeof(current_generations)) == 0 &&
                    std::memcmp(current_objects, ctx->q4_control_source_objects, sizeof(current_objects)) == 0 &&
                    current_generations[0] == 1 && current_generations[1] == 2 && current_generations[2] == 3 &&
                    current_generations[3] == 4 && current_generations[4] == 5 && current_generations[5] == 6 && current_generations[6] == 0 &&
                    std::strcmp(instruction_sha, ctx->q4_control_source_instruction_sha256) == 0 &&
                    std::strcmp(q8_sha, "680d8fe1d964eea065bc4efd261feaa7798143fef06d2930f6f18ef19e473078") == 0 &&
                    std::strcmp(instruction_sha, "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3") == 0 &&
                    std::strcmp(q4_transport_sha, "2c0e3b517d76ad7619fadf6d345f9e04be0dec1356e9216fa922f237c6ef86f5") == 0 &&
                    std::strcmp(q4_transport_sha, ctx->q4_control_snapshot_sha256) != 0 &&
                    persistent_q4 == nullptr && replay_q4_path == nullptr && replay_q8_path == nullptr &&
                    !ctx->q4_cache.enabled && !ctx->q4_weight_bo_enabled &&
                    node_name != nullptr && std::strcmp(node_name, "ffn_up-3") == 0 && rows == 9216 && k == 2560 && rhs == 1;
                if (!invariant_gate) {
                    std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=invariant_gate\n");
                    std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=invariant_gate\n");
                    return false;
                }
                std::memcpy(ctx->q4_m9216_buf_q4, ctx->q4_control_snapshot.get(), q4_transport_bytes);
                ggml_backend_xdna_m9216_digest(ctx->q4_m9216_buf_q4, q4_transport_bytes, substituted_sha);
                const bool digest_equal = std::strcmp(substituted_sha, ctx->q4_control_snapshot_sha256) == 0;
                std::fprintf(stderr, "XDNA_Q4_CONTROL_SUBSTITUTED backend_instance_id=%llu correlation_id=%llu xrt_seq=%llu source_ordinal=7 target_ordinal=8 source_sha256=%s substituted_sha256=%s equality=%s bytes=%zu offset=0 q8_sha256=%s instruction_sha256=%s xclbin_identity=%p kernel=%p hw_context=%p instruction_bo=%p shared_q4_bo=%p q8_bo=%p output_bo=%p hw_context_gen=%llu kernel_gen=%llu instruction_bo_gen=%llu shared_q4_bo_gen=%llu q8_bo_gen=%llu output_bo_gen=%llu persistent_q4_bo_gen=0\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id + 1),
                    ctx->q4_control_snapshot_sha256, substituted_sha, digest_equal ? "PASS" : "FAIL",
                    q4_transport_bytes, q8_sha, instruction_sha,
                    static_cast<const void *>(ctx->q4_m9216_xclbin.get()),
                    static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()),
                    static_cast<void *>(ctx->q4_m9216_bo_instr.get()), static_cast<void *>(ctx->q4_m9216_bo_q4.get()),
                    static_cast<void *>(ctx->q4_m9216_bo_q8.get()), static_cast<void *>(ctx->q4_m9216_bo_out.get()),
                    static_cast<unsigned long long>(current_generations[0]), static_cast<unsigned long long>(current_generations[1]),
                    static_cast<unsigned long long>(current_generations[2]), static_cast<unsigned long long>(current_generations[3]),
                    static_cast<unsigned long long>(current_generations[4]), static_cast<unsigned long long>(current_generations[5]));
                if (!digest_equal) {
                    std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=post_copy_digest_mismatch\n");
                    return false;
                }
                std::memcpy(q4_transport_sha, substituted_sha, sizeof(q4_transport_sha));
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=YES ordinal=8 correlation_id=%llu xrt_seq=%llu q8_sha256=%s shared_q4_bo_generation=%llu\n",
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id + 1),
                    q8_sha, static_cast<unsigned long long>(ctx->m9216_shared_q4_bo_generation));
            }
#endif
            ggml_backend_xdna_sync_host_written_bo(bo_q4, "K=2560 M=9216 q4 transport (HOST_ONLY)");
            GGML_XDNA_DIAG_ONLY(++ctx->q4_to_device;)
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(ctx->timers.q4_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (std::getenv("GGML_XDNA_BO_FORENSICS") != nullptr) {
            std::fprintf(stderr, "XDNA_BUFFER_OBJECT BO backend_id=%llu m9216_index=%llu instr_obj=%p instr_addr=0x%llx instr_size=%zu instr_group=%u q4_obj=%p q4_addr=0x%llx q4_size=%zu q4_group=%u q8_obj=%p q8_addr=0x%llx q8_size=%zu q8_group=%u out_obj=%p out_addr=0x%llx out_size=%zu out_group=%u persistent_q4_count=%zu persistent_q4_bytes=%zu\n",
                static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                static_cast<void *>(ctx->q4_m9216_bo_instr.get()), static_cast<unsigned long long>(ctx->q4_m9216_bo_instr->address()), instruction_bytes, ctx->q4_m9216_kernel->group_id(1),
                static_cast<void *>(&bo_q4), static_cast<unsigned long long>(bo_q4.address()), q4_transport_bytes, ctx->q4_m9216_kernel->group_id(3),
                static_cast<void *>(&bo_q8), static_cast<unsigned long long>(bo_q8.address()), q8_row_bytes, ctx->q4_m9216_kernel->group_id(4),
                static_cast<void *>(&bo_out), static_cast<unsigned long long>(bo_out.address()), output_bytes, ctx->q4_m9216_kernel->group_id(5),
                ctx->q4_m9216_weight_bos.size(), ctx->q4_weight_bo_bytes);
        }
#endif
        {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
            ggml_backend_xdna_sync_host_written_bo(bo_q8, "K=2560 M=9216 q8 activation (HOST_ONLY)");
            GGML_XDNA_DIAG_ONLY(++ctx->q8_to_device;)
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(ctx->timers.q8_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_AFTER_Q8_SYNC\n");
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (causal_selected) {
            if (node == nullptr) return false;
            const auto metadata = ggml_backend_xdna_causal_metadata(node, graph_node_index, rows, k, submission_sequence_id, capture_index);
            if (!ggml_backend_xdna_causal_capture_prepare(causal_sequence_id, metadata, q4_rows, q4_rows_bytes, ctx->q4_m9216_buf_q8, q8_row_bytes)) return false;
        }
        if (history_capture) {
            void * q4_transport = persistent_q4 != nullptr ? persistent_q4->map<void *>() : ctx->q4_m9216_buf_q4;
            bool semantic_target = false;
            if (k != 2560 || rhs != 1 ||
                !ggml_backend_xdna_diagnostic_capture_prepare(ctx, history_id, node_name, rows, k, rhs, 3u,
                    ctx->q4_m9216_buf_instr, instruction_bytes, q4_transport, q4_transport_bytes,
                    ctx->q4_m9216_buf_q8, q8_row_bytes, &semantic_target)) {
                if (!ggml_backend_xdna_diagnostic_capture_session_failed()) ggml_backend_xdna_diagnostic_capture_fail(ctx, history_id, "prepare", "input_validation", "capture input preparation failed");
                return false;
            }
            if (semantic_target) return false;
        }
#endif
        constexpr unsigned int opcode = 3;
        {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ++trace_sequence_id;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (trace_ordinal8_target) {
                std::fprintf(stderr, "XDNA_TRACE_Q4_TRANSPORT_SHA256 backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d M=9216 K=2560 RHS=1 bo_generation=%llu bytes=%zu offset=0 sha256=%s\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                    static_cast<unsigned long long>(ctx->m9216_shared_q4_bo_generation), q4_transport_bytes,
                    q4_transport_sha);
            }
#endif
#endif
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu backend_id=%llu BEFORE kernel shape=M9216 node=m9216 opcode=3 K=2560 rhs=1 instr_bytes=%zu instr_count=%u kernel=%p hw_context=%p group_ids=%u,%u,%u,%u persistent_q4=%p q4_bos_initialized=%d\n",
                    static_cast<unsigned long long>(trace_sequence_id), static_cast<unsigned long long>(ctx->backend_instance_id), instruction_bytes, instruction_count,
                    static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()),
                    ctx->q4_m9216_kernel->group_id(1), ctx->q4_m9216_kernel->group_id(3), ctx->q4_m9216_kernel->group_id(4), ctx->q4_m9216_kernel->group_id(5),
                    static_cast<void *>(persistent_q4), ctx->q4_m9216_bos_initialized ? 1 : 0);
            }
#endif
            GGML_XDNA_DIAG_ONLY(auto & timing = ctx->shape_timing[static_cast<size_t>(ggml_backend_xdna_q4_shape::m9216)];)
            GGML_XDNA_DIAG_ONLY(const auto submit_begin = std::chrono::steady_clock::now();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (capture_index == 9 && std::getenv("GGML_XDNA_DIAG_RECREATE_M9216_KERNEL_BEFORE_9") != nullptr) {
                const std::string kernel_name = ctx->q4_m9216_kernel->get_name();
                ctx->q4_m9216_kernel.reset();
                ctx->q4_m9216_kernel = std::make_unique<xrt::kernel>(*ctx->q4_m9216_hw_context, kernel_name);
                ctx->m9216_kernel_generation = ggml_backend_xdna_new_resource_generation(ctx);
                std::fprintf(stderr, "XDNA_RUN_STATE RECREATE kernel ordinal=9 kernel=%p hw_context=%p group_ids=%u,%u,%u,%u\n", static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()), ctx->q4_m9216_kernel->group_id(1), ctx->q4_m9216_kernel->group_id(3), ctx->q4_m9216_kernel->group_id(4), ctx->q4_m9216_kernel->group_id(5));
            }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            history_failure_phase = "submit_record";
            if (history_capture && !ggml_backend_xdna_diagnostic_capture_submitted(ctx, history_id)) return false;
            history_failure_phase = "wait";
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ggml_backend_xdna_m9216_correlation_prepared(
                submission_sequence_id, capture_index, graph_node_index, node,
                q4_native_sha, q8_sha, instruction_sha,
                q4_rows, activations, ctx->q4_m9216_buf_q8,
                ctx->q4_m9216_kernel.get(), ctx->q4_m9216_hw_context.get(),
                ctx->q4_m9216_bo_instr.get(), &bo_q4, &bo_q8, &bo_out);
            if (trace_ordinal8_target) std::fprintf(stderr,
                "XDNA_TRACE_RESOURCE_GENERATIONS backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d M=9216 K=2560 RHS=1 hw_context_gen=%llu kernel_gen=%llu instruction_bo_gen=%llu shared_q4_bo_gen=%llu q8_bo_gen=%llu output_bo_gen=%llu persistent_q4_bo_gen=%llu persistent_q4=%s command_identity=HOST_CORRELATION_ONLY\n",
                static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                static_cast<unsigned long long>(ctx->m9216_hw_context_generation), static_cast<unsigned long long>(ctx->m9216_kernel_generation),
                static_cast<unsigned long long>(ctx->m9216_instruction_bo_generation), static_cast<unsigned long long>(ctx->m9216_shared_q4_bo_generation),
                static_cast<unsigned long long>(ctx->m9216_q8_bo_generation), static_cast<unsigned long long>(ctx->m9216_output_bo_generation),
                static_cast<unsigned long long>(persistent_q4_generation), persistent_q4 != nullptr ? "used" : "unused");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_BEFORE_KERNEL_CALL\n");
            if (trace_ordinal8_target) trace_offline_pre = ggml_backend_xdna_log_offline(
                ctx, "XDNA_TRACE_DEVICE_OFFLINE_PRE_KERNEL", capture_index, submission_sequence_id,
                trace_sequence_id, node_name, graph_node_index);
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            const uint64_t trace_kernel_start_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
            trace_submission_id = trace_node == nullptr
                ? 0 : ggml_backend_xdna_runtime_trace_next_submission_id();
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
            auto run = [&]() {
                if (trace_node == nullptr) {
                    return kernel(opcode, *ctx->q4_m9216_bo_instr, instruction_count, bo_q4, bo_q8, bo_out);
                }
                ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "KERNEL_CALL_ENTER");
                try {
                    return kernel(opcode, *ctx->q4_m9216_bo_instr, instruction_count, bo_q4, bo_q8, bo_out);
                } catch (const std::exception & e) {
                    ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "KERNEL_CALL_EXCEPTION", nullptr, e.what());
                    throw;
                } catch (...) {
                    ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "KERNEL_CALL_EXCEPTION", nullptr, "unknown exception");
                    throw;
                }
            }();
#else
            auto run = kernel(
                opcode,
                *ctx->q4_m9216_bo_instr,
                instruction_count,
                bo_q4,
                bo_q8,
                bo_out);
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (trace_ordinal8_target) {
                const uint64_t trace_kernel_end_ns = ggml_backend_xdna_elapsed_ns(ctx);
                trace_kernel_ns = trace_kernel_end_ns - trace_kernel_start_ns;
                std::fprintf(stderr, "XDNA_TRACE_TIMING_KERNEL_CALL backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d start_elapsed_ns=%llu end_elapsed_ns=%llu duration_ns=%llu\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                    static_cast<unsigned long long>(trace_kernel_start_ns), static_cast<unsigned long long>(trace_kernel_end_ns),
                    static_cast<unsigned long long>(trace_kernel_ns));
            }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ggml_backend_xdna_m9216_correlation_event(
                submission_sequence_id, "KERNEL_CALL_RETURNED", capture_index, graph_node_index);
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "KERNEL_CALL_RETURNED");
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
            if (trace_node != nullptr) ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "KERNEL_CALL_RETURNED");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_AFTER_KERNEL_CALL\n");
#endif
            GGML_XDNA_DIAG_ONLY(const auto submit_end = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(++ctx->run_constructed;)
            GGML_XDNA_DIAG_ONLY(++ctx->run_live;)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_RUN_STATE_TRACE") != nullptr) std::fprintf(stderr, "XDNA_RUN_STATE RUN constructed ordinal=%llu node=m9216 run=%p kernel=%p hw_context=%p thread=%zu live=%llu constructed=%llu completed=%llu destroyed=%llu native_handle=UNAVAILABLE\n", static_cast<unsigned long long>(capture_index), static_cast<void *>(&run), static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()), std::hash<std::thread::id>{}(std::this_thread::get_id()), static_cast<unsigned long long>(ctx->run_live), static_cast<unsigned long long>(ctx->run_constructed), static_cast<unsigned long long>(ctx->run_completed), static_cast<unsigned long long>(ctx->run_destroyed));
#endif
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu AFTER kernel run_created shape=M9216 BO instr=%p q4=%p q8=%p out=%p\n", static_cast<unsigned long long>(trace_sequence_id), static_cast<void *>(ctx->q4_m9216_bo_instr.get()), static_cast<void *>(persistent_q4), static_cast<void *>(ctx->q4_m9216_bo_q8.get()), static_cast<void *>(ctx->q4_m9216_bo_out.get()));
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu BEFORE run.wait()\n", static_cast<unsigned long long>(trace_sequence_id));
            }
#endif
            GGML_XDNA_DIAG_ONLY(timing.submit_ms += std::chrono::duration<double, std::milli>(submit_end - submit_begin).count();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_RUN_STATE_TRACE") != nullptr) std::fprintf(stderr, "XDNA_RUN_STATE RUN BEFORE_WAIT ordinal=%llu run=%p thread=%zu live=%llu\n", static_cast<unsigned long long>(capture_index), static_cast<void *>(&run), std::hash<std::thread::id>{}(std::this_thread::get_id()), static_cast<unsigned long long>(ctx->run_live));
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            const uint64_t trace_state_start_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
            const ert_cmd_state state_before_wait = run.state();
            const uint64_t trace_state_end_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
            trace_pre_wait_state = state_before_wait;
            if (trace_ordinal8_target) {
                trace_state_query_ns = trace_state_end_ns - trace_state_start_ns;
                std::fprintf(stderr, "XDNA_TRACE_TIMING_STATE_QUERY backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d start_elapsed_ns=%llu end_elapsed_ns=%llu duration_ns=%llu state=%d:%s\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                    static_cast<unsigned long long>(trace_state_start_ns), static_cast<unsigned long long>(trace_state_end_ns),
                    static_cast<unsigned long long>(trace_state_query_ns), static_cast<int>(state_before_wait),
                    ggml_backend_xdna_ert_cmd_state_name(state_before_wait));
            }
            std::fprintf(stderr,
                "XDNA_CORR id=%llu event=STATE_BEFORE_WAIT ordinal=%llu xrt_seq=%llu m=9216 node=%s node_index=%d state=%d name=%s\n",
                static_cast<unsigned long long>(submission_sequence_id),
                static_cast<unsigned long long>(capture_index),
                static_cast<unsigned long long>(trace_sequence_id),
                ggml_get_name(node), graph_node_index, static_cast<int>(state_before_wait),
                ggml_backend_xdna_ert_cmd_state_name(state_before_wait));
#endif
            GGML_XDNA_DIAG_ONLY(const auto wait_begin = std::chrono::steady_clock::now();)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ggml_backend_xdna_m9216_correlation_event(
                submission_sequence_id, "WAIT_ENTER", capture_index, graph_node_index);
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "WAIT_ENTER");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_BEFORE_WAIT\n");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            const uint64_t trace_wait_start_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
            ert_cmd_state completion_state;
            try {
                completion_state = timed_wait_diag ? ggml_backend_xdna_wait_run(run, std::chrono::milliseconds(1000)) : ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT);
            } catch (const std::exception & e) {
                const uint64_t end_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
                trace_wait_ns = end_ns - trace_wait_start_ns;
                if (trace_ordinal8_target) std::fprintf(stderr,
                    "XDNA_TRACE_TIMING_WAIT backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d start_elapsed_ns=%llu end_elapsed_ns=%llu duration_ns=%llu result=exception category=std_exception error=%s\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                    static_cast<unsigned long long>(trace_wait_start_ns), static_cast<unsigned long long>(end_ns),
                    static_cast<unsigned long long>(trace_wait_ns), e.what());
                throw;
            } catch (...) {
                const uint64_t end_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
                trace_wait_ns = end_ns - trace_wait_start_ns;
                if (trace_ordinal8_target) std::fprintf(stderr,
                    "XDNA_TRACE_TIMING_WAIT backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d start_elapsed_ns=%llu end_elapsed_ns=%llu duration_ns=%llu result=exception category=unknown\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                    static_cast<unsigned long long>(trace_wait_start_ns), static_cast<unsigned long long>(end_ns),
                    static_cast<unsigned long long>(trace_wait_ns));
                throw;
            }
            const uint64_t trace_wait_end_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
            trace_wait_returned_state = completion_state;
            if (trace_ordinal8_target && completion_state != ERT_CMD_STATE_COMPLETED) {
                trace_offline_post = ggml_backend_xdna_log_offline(
                    ctx, "XDNA_TRACE_DEVICE_OFFLINE_POST_WAIT", capture_index, submission_sequence_id,
                    trace_sequence_id, node_name, graph_node_index);
            }
            if (trace_ordinal8_target) {
                trace_wait_ns = trace_wait_end_ns - trace_wait_start_ns;
                std::fprintf(stderr, "XDNA_TRACE_TIMING_WAIT backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d start_elapsed_ns=%llu end_elapsed_ns=%llu duration_ns=%llu result=returned state=%d:%s\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                    static_cast<unsigned long long>(trace_wait_start_ns), static_cast<unsigned long long>(trace_wait_end_ns),
                    static_cast<unsigned long long>(trace_wait_ns), static_cast<int>(completion_state),
                    ggml_backend_xdna_ert_cmd_state_name(completion_state));
            }
            std::fprintf(stderr,
                "XDNA_CORR id=%llu event=WAIT_RETURN_STATE ordinal=%llu xrt_seq=%llu m=9216 node=%s node_index=%d state=%d name=%s\n",
                static_cast<unsigned long long>(submission_sequence_id),
                static_cast<unsigned long long>(capture_index),
                static_cast<unsigned long long>(trace_sequence_id),
                ggml_get_name(node), graph_node_index, static_cast<int>(completion_state),
                ggml_backend_xdna_ert_cmd_state_name(completion_state));
            ggml_backend_xdna_m9216_correlation_event(
                submission_sequence_id, "WAIT_RETURN", capture_index, graph_node_index);
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "WAIT_RETURN");
            if (timed_wait_diag) {
                std::fprintf(stderr, "XDNA_TIMED_WAIT ordinal=%llu result=%d timeout=%d\n", static_cast<unsigned long long>(capture_index), static_cast<int>(completion_state), completion_state == ERT_CMD_STATE_TIMEOUT ? 1 : 0);
            }
#else
#if defined(GGML_XDNA_RUNTIME_TRACE)
            const ert_cmd_state completion_state = ggml_backend_xdna_runtime_trace_wait(
                trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, run, GGML_XDNA_ERT_WAIT_LIMIT);
#else
            const ert_cmd_state completion_state = ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT);
#endif
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_AFTER_WAIT\n");
#endif
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu AFTER run.wait()\n", static_cast<unsigned long long>(trace_sequence_id));
            }
#endif
            if (completion_state != ERT_CMD_STATE_COMPLETED) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
                if (trace_ordinal8_target) {
                    std::fprintf(stderr, "XDNA_TRACE_TIMING_OUTPUT_SYNC backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d result=not_reached reason=wait_not_COMPLETED\n",
                        static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                        static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index);
                    ggml_backend_xdna_log_correlation(ctx, capture_index, submission_sequence_id, trace_sequence_id,
                        node_name, graph_node_index, state_before_wait, completion_state, trace_kernel_ns,
                        trace_state_query_ns, trace_wait_ns, 0, "not_reached", trace_offline_pre,
                        trace_offline_post, persistent_q4_generation, q4_transport_sha, q8_sha);
                }
                GGML_XDNA_DIAG_ONLY(--ctx->run_live;)
                GGML_XDNA_DIAG_ONLY(++ctx->run_destroyed;)
                std::fprintf(stderr,
                    "ggml_xdna: M9216 wait failed id=%llu ordinal=%llu M=9216 K=2560 state=%d name=%s expected=COMPLETED\n",
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(capture_index),
                    static_cast<int>(completion_state), ggml_backend_xdna_ert_cmd_state_name(completion_state));
                if (history_capture) ggml_backend_xdna_diagnostic_capture_fail(ctx, history_id, "wait", "wait_incomplete", "XRT wait returned non-COMPLETED state");
#else
                std::fprintf(stderr,
                    "ggml_xdna: ggml_backend_xdna_run_q4k_q8k_k2560_m9216 M=9216 K=2560 wait_state=%d expected=%d\n",
                    static_cast<int>(completion_state), static_cast<int>(ERT_CMD_STATE_COMPLETED));
#endif
                return false;
            }
            GGML_XDNA_DIAG_ONLY(++ctx->xrt_runs; ++ctx->run_waits;)
            GGML_XDNA_DIAG_ONLY(const auto wait_end = std::chrono::steady_clock::now();)
            GGML_XDNA_DIAG_ONLY(++ctx->run_completed;)
            GGML_XDNA_DIAG_ONLY(--ctx->run_live;)
            GGML_XDNA_DIAG_ONLY(++ctx->run_destroyed;)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_RUN_STATE_TRACE") != nullptr) std::fprintf(stderr, "XDNA_RUN_STATE RUN AFTER_WAIT ordinal=%llu run=%p thread=%zu live=%llu completed=%llu destroyed=%llu state=COMPLETED\n", static_cast<unsigned long long>(capture_index), static_cast<void *>(&run), std::hash<std::thread::id>{}(std::this_thread::get_id()), static_cast<unsigned long long>(ctx->run_live), static_cast<unsigned long long>(ctx->run_completed), static_cast<unsigned long long>(ctx->run_destroyed));
#endif
            GGML_XDNA_DIAG_ONLY(timing.wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();)
            GGML_XDNA_DIAG_ONLY(++timing.ops;)
            GGML_XDNA_DIAG_ONLY(ctx->timers.xrt_submit_wait_ms += std::chrono::duration<double, std::milli>(wait_end - submit_begin).count();)
        }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (std::getenv("GGML_XDNA_RUN_STATE_TRACE") != nullptr) std::fprintf(stderr, "XDNA_RUN_STATE RUN SCOPE_EXIT ordinal=%llu thread=%zu live=%llu completed=%llu destroyed=%llu\n", static_cast<unsigned long long>(capture_index), std::hash<std::thread::id>{}(std::this_thread::get_id()), static_cast<unsigned long long>(ctx->run_live), static_cast<unsigned long long>(ctx->run_completed), static_cast<unsigned long long>(ctx->run_destroyed));
#endif
        {
            GGML_XDNA_DIAG_ONLY(const auto t0 = std::chrono::steady_clock::now();)
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu BEFORE output sync shape=M9216\n", static_cast<unsigned long long>(trace_sequence_id));
            }
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            history_failure_phase = "output_sync";
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_BEFORE_OUTPUT_SYNC\n");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            const uint64_t trace_output_sync_start_ns = trace_ordinal8_target ? ggml_backend_xdna_elapsed_ns(ctx) : 0;
#endif
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (ctx->q4_control_enabled && capture_index == 8) {
                try {
                    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                } catch (const std::exception & e) {
                    std::fprintf(stderr, "XDNA_Q4_CONTROL_COMPLETED_STOP ordinal=8 output_sync=FAIL result_copy=NOT_REACHED helper_success=NO error=%s\n", e.what());
                    throw;
                } catch (...) {
                    std::fprintf(stderr, "XDNA_Q4_CONTROL_COMPLETED_STOP ordinal=8 output_sync=FAIL result_copy=NOT_REACHED helper_success=NO error=unknown\n");
                    throw;
                }
            } else
#endif
            {
#if defined(GGML_XDNA_RUNTIME_TRACE)
                if (trace_node != nullptr) {
                    try {
                        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    } catch (const std::exception & e) {
                        ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "OUTPUT_SYNC_EXCEPTION", "ERT_CMD_STATE_COMPLETED", e.what());
                        throw;
                    } catch (...) {
                        ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "OUTPUT_SYNC_EXCEPTION", "ERT_CMD_STATE_COMPLETED", "unknown exception");
                        throw;
                    }
                    ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "OUTPUT_SYNC_RETURNED", "ERT_CMD_STATE_COMPLETED");
                } else
#endif
                bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (trace_ordinal8_target) {
                const uint64_t trace_output_sync_end_ns = ggml_backend_xdna_elapsed_ns(ctx);
                trace_output_sync_ns = trace_output_sync_end_ns - trace_output_sync_start_ns;
                std::fprintf(stderr, "XDNA_TRACE_TIMING_OUTPUT_SYNC backend_id=%llu ordinal=%llu correlation_id=%llu xrt_seq=%llu node=%s node_index=%d start_elapsed_ns=%llu end_elapsed_ns=%llu duration_ns=%llu result=returned\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(capture_index),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id), node_name, graph_node_index,
                    static_cast<unsigned long long>(trace_output_sync_start_ns), static_cast<unsigned long long>(trace_output_sync_end_ns),
                    static_cast<unsigned long long>(trace_output_sync_ns));
                ggml_backend_xdna_log_correlation(ctx, capture_index, submission_sequence_id, trace_sequence_id,
                    node_name, graph_node_index, trace_pre_wait_state, trace_wait_returned_state, trace_kernel_ns,
                    trace_state_query_ns, trace_wait_ns, trace_output_sync_ns, "returned",
                    trace_offline_pre, trace_offline_post, persistent_q4_generation, q4_transport_sha, q8_sha);
            }
#endif
            ggml_backend_xdna_m9216_correlation_event(
                submission_sequence_id, "OUTPUT_SYNC_RETURN", capture_index, graph_node_index);
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "OUTPUT_SYNC_RETURN");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_AFTER_OUTPUT_SYNC\n");
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (ctx->q4_control_enabled && capture_index == 8) {
                ++ctx->output_from_device;
                ctx->timers.output_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "XDNA_Q4_CONTROL_COMPLETED_STOP backend_instance_id=%llu ordinal=8 correlation_id=%llu xrt_seq=%llu wait_state=ERT_CMD_STATE_COMPLETED output_sync=success result_copy=NOT_REACHED helper_success=NO downstream=NO\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<unsigned long long>(submission_sequence_id),
                    static_cast<unsigned long long>(trace_sequence_id));
                return false;
            }
#endif
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (history_capture && !ggml_backend_xdna_diagnostic_capture_completed(ctx, history_id, ctx->q4_m9216_buf_out, output_bytes)) return false;
#endif
            GGML_XDNA_DIAG_ONLY(++ctx->output_from_device;)
            std::memcpy(result, ctx->q4_m9216_buf_out, output_bytes);
#if defined(GGML_XDNA_RUNTIME_TRACE)
            if (trace_node != nullptr) ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "m9216", -1, "RESULT_COPY_DONE", "ERT_CMD_STATE_COMPLETED");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            ggml_backend_xdna_m9216_correlation_event(
                submission_sequence_id, "RESULT_COPY_RETURN", capture_index, graph_node_index);
            (void) ggml_backend_xdna_causal_capture_event(causal_sequence_id, "RESULT_COPY_RETURN");
#endif
            GGML_XDNA_DIAG_ONLY(const auto t1 = std::chrono::steady_clock::now();)
            #if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
            if (std::getenv("GGML_XDNA_XRT_SEQUENCE_TRACE") != nullptr) {
                std::fprintf(stderr, "XDNA_XRT_SEQUENCE seq=%llu AFTER output sync shape=M9216\n", static_cast<unsigned long long>(trace_sequence_id));
            }
#endif
            GGML_XDNA_DIAG_ONLY(ctx->timers.output_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();)
        }

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        std::fprintf(stderr, "ggml_xdna: M9216_BOUNDARY_AFTER_RESULT_COPY\n");
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (ctx->q4_control_enabled && capture_index == 7) {
            const bool capture_preconditions = persistent_q4 == nullptr && replay_q4_path == nullptr &&
                ctx->q4_m9216_buf_q4 != nullptr && ctx->q4_m9216_bo_q4 != nullptr &&
                !ctx->q4_cache.enabled && !ctx->q4_weight_bo_enabled &&
                ctx->q4_control_snapshot != nullptr &&
                ctx->q4_control_snapshot_bytes == q4_transport_bytes &&
                q4_transport_bytes == 14008320 && ctx->q4_m9216_xclbin != nullptr;
            if (!capture_preconditions) {
                std::fprintf(stderr, "XDNA_Q4_CONTROL_CAPTURE backend_instance_id=%llu ordinal=7 bytes=%zu offset=0 sha256=NOT_CAPTURED shared_q4_bo_generation=%llu correlation_id=%llu xrt_seq=%llu capture_success=0 reason=source_invariants\n",
                    static_cast<unsigned long long>(ctx->backend_instance_id), q4_transport_bytes,
                    static_cast<unsigned long long>(ctx->m9216_shared_q4_bo_generation),
                    static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id));
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=source_invariants\n");
                return false;
            }
            std::memcpy(ctx->q4_control_snapshot.get(), ctx->q4_m9216_buf_q4, q4_transport_bytes);
            ggml_backend_xdna_m9216_digest(ctx->q4_control_snapshot.get(), q4_transport_bytes,
                ctx->q4_control_snapshot_sha256);
            ctx->q4_control_source_backend_id = ctx->backend_instance_id;
            ctx->q4_control_source_correlation_id = submission_sequence_id;
            ctx->q4_control_source_xrt_sequence = trace_sequence_id;
            ctx->q4_control_source_generations[0] = ctx->m9216_hw_context_generation;
            ctx->q4_control_source_generations[1] = ctx->m9216_kernel_generation;
            ctx->q4_control_source_generations[2] = ctx->m9216_instruction_bo_generation;
            ctx->q4_control_source_generations[3] = ctx->m9216_shared_q4_bo_generation;
            ctx->q4_control_source_generations[4] = ctx->m9216_q8_bo_generation;
            ctx->q4_control_source_generations[5] = ctx->m9216_output_bo_generation;
            ctx->q4_control_source_generations[6] = 0;
            ctx->q4_control_source_objects[0] = ctx->q4_m9216_hw_context.get();
            ctx->q4_control_source_objects[1] = ctx->q4_m9216_kernel.get();
            ctx->q4_control_source_objects[2] = ctx->q4_m9216_bo_instr.get();
            ctx->q4_control_source_objects[3] = ctx->q4_m9216_bo_q4.get();
            ctx->q4_control_source_objects[4] = ctx->q4_m9216_bo_q8.get();
            ctx->q4_control_source_objects[5] = ctx->q4_m9216_bo_out.get();
            ctx->q4_control_source_objects[6] = nullptr;
            std::memcpy(ctx->q4_control_source_instruction_sha256, instruction_sha, sizeof(instruction_sha));
            ctx->q4_control_source_xclbin_identity = ctx->q4_m9216_xclbin.get();
            ctx->q4_control_snapshot_valid = true;
            std::fprintf(stderr, "XDNA_Q4_CONTROL_CAPTURE backend_instance_id=%llu ordinal=7 node=ffn_gate-3 bytes=%zu offset=0 sha256=%s q8_sha256=%s instruction_sha256=%s xclbin_identity=%p kernel=%p hw_context=%p instruction_bo=%p shared_q4_bo=%p q8_bo=%p output_bo=%p hw_context_gen=%llu kernel_gen=%llu instruction_bo_gen=%llu shared_q4_bo_gen=%llu q8_bo_gen=%llu output_bo_gen=%llu persistent_q4_bo_gen=0 correlation_id=%llu xrt_seq=%llu capture_success=1\n",
                static_cast<unsigned long long>(ctx->backend_instance_id), q4_transport_bytes,
                ctx->q4_control_snapshot_sha256, q8_sha, instruction_sha,
                static_cast<const void *>(ctx->q4_control_source_xclbin_identity),
                static_cast<void *>(ctx->q4_m9216_kernel.get()), static_cast<void *>(ctx->q4_m9216_hw_context.get()),
                static_cast<void *>(ctx->q4_m9216_bo_instr.get()), static_cast<void *>(ctx->q4_m9216_bo_q4.get()),
                static_cast<void *>(ctx->q4_m9216_bo_q8.get()), static_cast<void *>(ctx->q4_m9216_bo_out.get()),
                static_cast<unsigned long long>(ctx->m9216_hw_context_generation), static_cast<unsigned long long>(ctx->m9216_kernel_generation),
                static_cast<unsigned long long>(ctx->m9216_instruction_bo_generation), static_cast<unsigned long long>(ctx->m9216_shared_q4_bo_generation),
                static_cast<unsigned long long>(ctx->m9216_q8_bo_generation), static_cast<unsigned long long>(ctx->m9216_output_bo_generation),
                static_cast<unsigned long long>(submission_sequence_id), static_cast<unsigned long long>(trace_sequence_id));
            if (std::strcmp(q8_sha, "680d8fe1d964eea065bc4efd261feaa7798143fef06d2930f6f18ef19e473078") != 0 ||
                std::strcmp(instruction_sha, "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3") != 0) {
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ONLY_TRANSPORT_BYTES_CHANGED=NO reason=ordinal7_q8_or_instruction_mismatch\n");
                std::fprintf(stderr, "XDNA_Q4_CONTROL_ABORT reason=ordinal7_q8_or_instruction_mismatch\n");
                return false;
            }
        }
#endif
        return true;
    }
    catch (const std::exception & e) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (history_capture) ggml_backend_xdna_diagnostic_capture_fail(ctx, history_id, history_failure_phase, "submission_or_recording", e.what());
#endif
        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row execution failed: %s\n",
            e.what());
        return false;
    }
    catch (...) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (history_capture) ggml_backend_xdna_diagnostic_capture_fail(ctx, history_id, history_failure_phase, "submission_or_recording", "unknown captured submission failure");
#endif
        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row execution failed: unknown error\n");
        return false;
    }
}
static bool ggml_backend_xdna_run_q4k_q8k_k2560(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const void * q4_row,
        size_t q4_row_bytes,
        const float * activations,
        size_t activation_elements,
        bool refresh_activation,
        float * result
#if defined(GGML_XDNA_RUNTIME_TRACE)
        , const ggml_backend_xdna_runtime_trace_node * trace_node, int64_t row_index
#endif
        ) {
    constexpr int64_t elements = 2560;
    constexpr size_t block_count = 10;
    constexpr size_t q4_transport_block_bytes = 152;

    static_assert(QK_K == 256, "unexpected QK_K");
    static_assert(sizeof(block_q4_K) == 144, "unexpected Q4_K block size");
    static_assert(sizeof(block_q8_K) == 292, "unexpected Q8_K block size");

    constexpr size_t q4_native_row_bytes =
        block_count * sizeof(block_q4_K);

    constexpr size_t q4_transport_row_bytes =
        block_count * q4_transport_block_bytes;

    constexpr size_t q8_row_bytes =
        block_count * sizeof(block_q8_K);

    if (backend == nullptr ||
        instructions == nullptr ||
        instruction_count == 0 ||
        q4_row == nullptr ||
        activations == nullptr ||
        result == nullptr ||
        q4_row_bytes != q4_native_row_bytes ||
        activation_elements != static_cast<size_t>(elements)) {
        std::fprintf(
            stderr,
            "ggml_xdna: invalid Q4_K x Q8_K K=2560 arguments\n");
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(
            backend->context);

    if (ctx == nullptr ||
        ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr ||
        ctx->kernel == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: Q4_K x Q8_K K=2560 requires an initialized XRT kernel\n");
        return false;
    }

    const auto * q4_native =
        static_cast<const uint8_t *>(q4_row);

    std::vector<uint8_t> q4_transport(
        q4_transport_row_bytes);

    for (size_t block = 0; block < block_count; ++block) {
        block_q4_K native_block;

        std::memcpy(
            &native_block,
            q4_native + block * sizeof(block_q4_K),
            sizeof(native_block));

        uint8_t * dst =
            q4_transport.data() +
            block * q4_transport_block_bytes;

        std::memcpy(
            dst,
            &native_block,
            sizeof(native_block));

        const float d =
            ggml_fp16_to_fp32(native_block.d);

        const float dmin =
            ggml_fp16_to_fp32(native_block.dmin);

        std::memcpy(
            dst + sizeof(block_q4_K),
            &d,
            sizeof(d));

        std::memcpy(
            dst + sizeof(block_q4_K) + sizeof(d),
            &dmin,
            sizeof(dmin));
    }

    std::vector<block_q8_K> q8_blocks;

    if (refresh_activation) {
        q8_blocks.resize(block_count);

        // quantize_row_q8_K_ref leaves bsums untouched for an
        // all-zero block, so initialize the complete native row.
        std::memset(
            q8_blocks.data(),
            0,
            q8_row_bytes);

        quantize_row_q8_K_ref(
            activations,
            q8_blocks.data(),
            elements);
    }

    try {
        xrt::device & device =
            *ctx->device_context->xrt_device;

        xrt::kernel & kernel =
            *ctx->kernel;

        const size_t instruction_bytes =
            static_cast<size_t>(instruction_count) *
                sizeof(uint32_t);

        if (!ctx->q4_bos_initialized) {
            GGML_XDNA_XRT_TRACE(ctx, "PER_ROW", "BO_SET_CREATE", "ALLOCATE_AND_MAP", "BEGIN",
                ctx->device_context->q4_shared.get(), ctx->kernel.get(), "NONE",
                "allocate instruction, Q4, Q8 and output BOs");
            auto bo_instr = std::make_unique<xrt::bo>(
                device,
                instruction_bytes,
                XCL_BO_FLAGS_CACHEABLE,
                kernel.group_id(1));

            auto bo_q4 = std::make_unique<xrt::bo>(
                device,
                q4_transport_row_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(3));

            auto bo_q8 = std::make_unique<xrt::bo>(
                device,
                q8_row_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(4));

            auto bo_out = std::make_unique<xrt::bo>(
                device,
                sizeof(float),
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(5));

            void * buf_instr =
                bo_instr->map<void *>();

            void * buf_q4 =
                bo_q4->map<void *>();

            void * buf_q8 =
                bo_q8->map<void *>();

            float * buf_out =
                bo_out->map<float *>();

            std::memcpy(
                buf_instr,
                instructions,
                instruction_bytes);

            // K=2560 integration: mandatory host ordering rule.
            ggml_backend_xdna_sync_host_written_bo(
                *bo_instr,
                "K=2560 per-row instruction BO (CACHEABLE)");

            ctx->q4_bo_instr = std::move(bo_instr);
            ctx->q4_bo_q4 = std::move(bo_q4);
            ctx->q4_bo_q8 = std::move(bo_q8);
            ctx->q4_bo_out = std::move(bo_out);

            ctx->q4_buf_instr = buf_instr;
            ctx->q4_buf_q4 = buf_q4;
            ctx->q4_buf_q8 = buf_q8;
            ctx->q4_buf_out = buf_out;

            ctx->q4_instruction_count =
                instruction_count;

            ctx->q4_bos_initialized = true;
            GGML_XDNA_XRT_TRACE(ctx, "PER_ROW", "BO_SET_CREATE", "ALLOCATE_AND_MAP", "PASS",
                ctx->device_context->q4_shared.get(), ctx->kernel.get(), "NONE",
                "backend-owned BO set mapped and instruction BO synced");
        }
        else if (
            ctx->q4_instruction_count != instruction_count ||
            std::memcmp(
                ctx->q4_buf_instr,
                instructions,
                instruction_bytes) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: persistent Q4_K BO instruction mismatch\n");
            return false;
        }

        xrt::bo & bo_instr =
            *ctx->q4_bo_instr;

        xrt::bo & bo_q4 =
            *ctx->q4_bo_q4;

        xrt::bo & bo_q8 =
            *ctx->q4_bo_q8;

        xrt::bo & bo_out =
            *ctx->q4_bo_out;

        void * buf_q4 =
            ctx->q4_buf_q4;

        void * buf_q8 =
            ctx->q4_buf_q8;

        float * buf_out =
            ctx->q4_buf_out;

        std::memcpy(
            buf_q4,
            q4_transport.data(),
            q4_transport_row_bytes);

        // K=2560 integration: mandatory host ordering rule - flush
        // every host-written HOST_ONLY BO before the launch. The output BO is never
        // written by the host on this path (no prefill), so it needs no pre-launch
        // flush; it is only read back after run.wait().
        ggml_backend_xdna_sync_host_written_bo(
            bo_q4,
            "K=2560 per-row q4 transport (HOST_ONLY)");

        if (refresh_activation) {
            std::memcpy(
                buf_q8,
                q8_blocks.data(),
                q8_row_bytes);

            ggml_backend_xdna_sync_host_written_bo(
                bo_q8,
                "K=2560 per-row q8 activation (HOST_ONLY)");
        }


        constexpr unsigned int opcode = 3;

#if defined(GGML_XDNA_RUNTIME_TRACE)
        const uint64_t trace_submission_id = trace_node == nullptr
            ? 0 : ggml_backend_xdna_runtime_trace_next_submission_id();
        const char * trace_kernel_name = ctx->kernel_name.c_str();
#endif
#if defined(GGML_XDNA_RUNTIME_TRACE)
        auto run = [&]() {
            if (trace_node == nullptr) {
                return kernel(opcode, bo_instr, instruction_count, bo_q4, bo_q8, bo_out);
            }
            ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "KERNEL_CALL_ENTER");
            try {
                return kernel(opcode, bo_instr, instruction_count, bo_q4, bo_q8, bo_out);
            } catch (const std::exception & e) {
                ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "KERNEL_CALL_EXCEPTION", nullptr, e.what());
                throw;
            } catch (...) {
                ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "KERNEL_CALL_EXCEPTION", nullptr, "unknown exception");
                throw;
            }
        }();
        if (trace_node != nullptr) ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "KERNEL_CALL_RETURNED");
#else
        auto run = kernel(
            opcode,
            bo_instr,
            instruction_count,
            bo_q4,
            bo_q8,
            bo_out);
#endif

#if defined(GGML_XDNA_RUNTIME_TRACE)
        const ert_cmd_state completion_state = ggml_backend_xdna_runtime_trace_wait(
            trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, run, GGML_XDNA_ERT_WAIT_LIMIT);
#else
        const ert_cmd_state completion_state = ggml_backend_xdna_wait_run(run, GGML_XDNA_ERT_WAIT_LIMIT);
#endif
        if (completion_state != ERT_CMD_STATE_COMPLETED) {
            std::fprintf(stderr, "ggml_xdna: ggml_backend_xdna_run_q4k_q8k_k2560 M=1 K=2560 wait_state=%d expected=%d\n",
                         static_cast<int>(completion_state), static_cast<int>(ERT_CMD_STATE_COMPLETED));
            return false;
        }

#if defined(GGML_XDNA_RUNTIME_TRACE)
        if (trace_node != nullptr) {
            try {
                bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            } catch (const std::exception & e) {
                ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "OUTPUT_SYNC_EXCEPTION", "ERT_CMD_STATE_COMPLETED", e.what());
                throw;
            } catch (...) {
                ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "OUTPUT_SYNC_EXCEPTION", "ERT_CMD_STATE_COMPLETED", "unknown exception");
                throw;
            }
            ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "OUTPUT_SYNC_RETURNED", "ERT_CMD_STATE_COMPLETED");
        } else
#endif
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        *result = *buf_out;
#if defined(GGML_XDNA_RUNTIME_TRACE)
        if (trace_node != nullptr) ggml_backend_xdna_runtime_trace_event(*trace_node, trace_submission_id, trace_kernel_name, "per_row", row_index, "RESULT_COPY_DONE", "ERT_CMD_STATE_COMPLETED");
#endif

        return true;
    }
    catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: Q4_K x Q8_K K=2560 execution failed: %s\n",
            e.what());
        return false;
    }
    catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: Q4_K x Q8_K K=2560 execution failed: unknown error\n");
        return false;
    }
}

static struct ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = {
        0x58, 0x44, 0x4e, 0x41,
        0x2d, 0x47, 0x47, 0x4d,
        0x4c, 0x2d, 0x58, 0x52,
        0x54, 0x2d, 0x30, 0x31
    };

    return &guid;
}

static bool
ggml_backend_xdna_init_q4k_q8k_k2560_m9216(
        ggml_backend_xdna_backend_context * ctx,
        const char * xclbin_path, const char * instruction_path,
        const char * expected_instructions_hex) {
    if (ctx == nullptr ||
        xclbin_path == nullptr ||
        xclbin_path[0] == '\0') {
        return false;
    }

    // canonical unified multisequence XCLBIN and M9216 instruction stream.
    static constexpr unsigned char
        // canonical unified XCLBIN and M9216 instruction fingerprints.
        expected_xclbin_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0x9a, 0x74, 0x3f, 0xf7, 0x24, 0x46, 0x3a, 0xa6,
        0x8d, 0x2d, 0xd6, 0x29, 0x11, 0xc7, 0x3e, 0xee,
        0xcc, 0xad, 0x41, 0x47, 0x88, 0x09, 0x7b, 0xa9,
        0x0f, 0xd0, 0xbb, 0xff, 0x3b, 0x68, 0x2d, 0x27
    };

    static constexpr unsigned char
        expected_instructions_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xd8, 0xe4, 0x2a, 0x47, 0xf2, 0xb9, 0xe0, 0x63,
        0xa7, 0x7e, 0xdb, 0x51, 0x3b, 0xd1, 0x5c, 0xcc,
        0x99, 0x22, 0x04, 0xbd, 0x6e, 0x3c, 0xf2, 0x8a,
        0x51, 0x9a, 0x82, 0x1f, 0x97, 0xfe, 0xee, 0x97
    };

    ctx->q4_m9216_xclbin_path =
        xclbin_path;

    const std::filesystem::path artifact_path(
        ctx->q4_m9216_xclbin_path);

    ctx->q4_m9216_instructions_path =
        instruction_path != nullptr && instruction_path[0] != '\0'
            ? instruction_path
            : (artifact_path.parent_path() / "insts.bin").string();
    unsigned char actual_xclbin[
        GGML_XDNA_SHA256_DIGEST_SIZE];

    if (!ggml_backend_xdna_hash_file_sha256(
            ctx->q4_m9216_xclbin_path,
            actual_xclbin)) {
        return false;
    }

    if (std::memcmp(
            actual_xclbin,
            expected_xclbin_sha256,
            GGML_XDNA_SHA256_DIGEST_SIZE) != 0) {
        char actual_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        ggml_backend_xdna_format_sha256(
            actual_xclbin,
            actual_hex);

        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 XCLBIN fingerprint mismatch\n"
            "ggml_xdna: actual SHA256: %s\n"
            "ggml_xdna: XCLBIN file: %s\n",
            actual_hex,
            ctx->q4_m9216_xclbin_path.c_str());

        return false;
    }

    try {
        std::ifstream file(
            ctx->q4_m9216_instructions_path,
            std::ios::binary |
                std::ios::ate);

        if (!file) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to open M=9216 instructions: %s\n",
                ctx->q4_m9216_instructions_path.c_str());
            return false;
        }

        const std::streamsize size =
            file.tellg();

        if (size <= 0 ||
            size % static_cast<std::streamsize>(
                sizeof(uint32_t)) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: invalid M=9216 instruction file size\n");
            return false;
        }

        ctx->q4_m9216_instructions.resize(
            static_cast<size_t>(size) /
            sizeof(uint32_t));

        file.seekg(
            0,
            std::ios::beg);

        if (!file.read(
                reinterpret_cast<char *>(
                    ctx->q4_m9216_instructions.data()),
                size)) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to read M=9216 instructions\n");

            ctx->q4_m9216_instructions.clear();
            return false;
        }

        unsigned char actual_instructions[
            GGML_XDNA_SHA256_DIGEST_SIZE];

        ggml_xdna_sha256_hash(
            actual_instructions,
            reinterpret_cast<const unsigned char *>(
                ctx->q4_m9216_instructions.data()),
            ctx->q4_m9216_instructions.size() *
                sizeof(uint32_t));

        char actual_hex[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        ggml_backend_xdna_format_sha256(actual_instructions, actual_hex);
        if (expected_instructions_hex == nullptr ||
            std::strcmp(actual_hex, expected_instructions_hex) != 0) {
            std::fprintf(stderr,
                "ggml_xdna: M=9216 instruction fingerprint mismatch\n"
                "ggml_xdna: actual SHA256: %s\n"
                "ggml_xdna: instruction file: %s\n",
                actual_hex, ctx->q4_m9216_instructions_path.c_str());
            ctx->q4_m9216_instructions.clear();
            return false;
        }

        std::string resolved_name;
        if (!ggml_backend_xdna_acquire_q4_shared_kernel(
                ctx, "M9216", ctx->q4_m9216_xclbin_path, ctx->kernel_name.c_str(), false,
                ctx->q4_m9216_xclbin, ctx->q4_m9216_hw_context,
                ctx->q4_m9216_kernel, resolved_name)) {
            return false;
        }
        if (resolved_name != ctx->kernel_name) {
            std::fprintf(stderr, "ggml_xdna: M=9216 kernel identity differs from primary Q4 kernel\n");
            return false;
        }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        ctx->m9216_hw_context_generation = ggml_backend_xdna_new_resource_generation(ctx);
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        ctx->m9216_kernel_generation = ggml_backend_xdna_new_resource_generation(ctx);
#endif

        ctx->q4_m9216_available =
            true;

        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row fast path ready: "
            "%zu instruction words, kernel %s\n",
            ctx->q4_m9216_instructions.size(),
            resolved_name.c_str());

        return true;
    }
    catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to initialize M=9216 fast path: %s\n",
            e.what());

        ctx->q4_m9216_available =
            false;

        return false;
    }
    catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to initialize M=9216 fast path: "
            "unknown error\n");

        ctx->q4_m9216_available =
            false;

        return false;
    }
}
static ggml_backend_t ggml_backend_xdna_device_init(
        ggml_backend_dev_t dev,
        const char * params) {
    ggml_backend_xdna_context * device_ctx =
        ggml_backend_xdna_get_context(dev);

    if (device_ctx == nullptr || device_ctx->xrt_device == nullptr ||
        device_ctx->q4_shared == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> device_state_lock(device_ctx->q4_shared->state_mutex);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const bool host_preflight_host_only =
        std::getenv("GGML_XDNA_DIAG_A7H2_HOST_PREFLIGHT_ONLY") != nullptr;
#endif

    const char * resolved_params = params;

    // Normal llama.cpp backend initialization passes nullptr.
    // Allow an XDNA-specific opt-in without changing generic llama
    // initialization or command-line parameter plumbing.
    //
    // Explicit non-empty backend params always take precedence.
    if (resolved_params == nullptr || resolved_params[0] == '\0') {
        const char * env_params =
            std::getenv("GGML_XDNA_BACKEND_PARAMS");

        if (env_params != nullptr && env_params[0] != '\0') {
            resolved_params = env_params;
        }
    }

    ggml_backend_xdna_kernel_profile requested_profile =
        ggml_backend_xdna_kernel_profile::none;

    std::string xclbin_path;

    if (!ggml_backend_xdna_parse_init_params(
            resolved_params,
            &requested_profile,
            &xclbin_path)) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (host_preflight_host_only) {
            ggml_backend_xdna_log_host_only_stop(false, "backend_params_parse_failed");
        }
#endif
        return nullptr;
    }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    bool profile_preflight_passed = false;
    const char * profile_failure_reason = nullptr;
    const char * q4_control = std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8");
    const bool q4_control_requested = q4_control != nullptr && std::strcmp(q4_control, "1") == 0;
    if (q4_control_requested || host_preflight_host_only) {
        if (host_preflight_host_only) {
            try {
                profile_preflight_passed = ggml_backend_xdna_profile_preflight(
                    resolved_params, requested_profile, xclbin_path, host_preflight_host_only, &profile_failure_reason);
            } catch (const std::exception &) {
                ggml_backend_xdna_log_host_only_stop(false, "profile_preflight_exception");
                return nullptr;
            } catch (...) {
                ggml_backend_xdna_log_host_only_stop(false, "profile_preflight_unknown_exception");
                return nullptr;
            }
        } else {
            profile_preflight_passed = ggml_backend_xdna_profile_preflight(
                resolved_params, requested_profile, xclbin_path, host_preflight_host_only, &profile_failure_reason);
        }
        if (host_preflight_host_only) {
            ggml_backend_xdna_log_host_only_stop(
                profile_preflight_passed, profile_failure_reason);
            return nullptr;
        }
        if (!profile_preflight_passed) return nullptr;
    }
#endif

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (!ggml_backend_xdna_diagnostic_capture_validate_profile(requested_profile)) {
        std::fprintf(stderr, "ggml_xdna: XDNA capture requires the q4k_q8k_k2560 profile; backend init rejected\n");
        return nullptr;
    }
#endif
    if (device_ctx->active_backend_count != 0 && device_ctx->active_profile != requested_profile) {
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        ggml_backend_xdna_diagnostic_capture_fail_if_requested();
#endif
        std::fprintf(
            stderr,
            "ggml_xdna: cannot initialize profile '%s'; "
            "device already has %zu backend instance(s) using profile '%s'\n",
            ggml_backend_xdna_kernel_profile_name(requested_profile),
            device_ctx->active_backend_count,
            ggml_backend_xdna_kernel_profile_name(
                device_ctx->active_profile));
        return nullptr;
    }

    auto * backend_ctx =
        new ggml_backend_xdna_backend_context;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    backend_ctx->backend_instance_id = ggml_backend_xdna_next_instance_id++;
#endif

    backend_ctx->device_context = device_ctx;
    backend_ctx->kernel_profile = requested_profile;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (profile_preflight_passed) {
        // Keep the preflighted control armed so later gate removal fails closed.
        backend_ctx->q4_control_gate_checked = true;
        backend_ctx->q4_control_enabled = true;
    }
#endif
    if (requested_profile !=
        ggml_backend_xdna_kernel_profile::none) {
        const char * runtime_xclbin =
            requested_profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560
                ? std::getenv("GGML_XDNA_Q4K_RUNTIME_XCLBIN")
                : nullptr;
        backend_ctx->xclbin_path =
            runtime_xclbin != nullptr && runtime_xclbin[0] != '\0'
                ? runtime_xclbin
                : xclbin_path;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
        if (!ggml_backend_xdna_diagnostic_capture_initialize(backend_ctx)) {
            std::fprintf(stderr, "ggml_xdna: invalid or non-fresh XDNA capture directory/metadata\n");
            delete backend_ctx;
            return nullptr;
        }
#endif

        if (!ggml_backend_xdna_validate_kernel_profile(
                backend_ctx)) {
            delete backend_ctx;
            return nullptr;
        }
    }

    const char * kernel_params =
        backend_ctx->xclbin_path.empty()
            ? nullptr
            : backend_ctx->xclbin_path.c_str();
    if (!ggml_backend_xdna_init_kernel(
            backend_ctx,
            kernel_params)) {
        delete backend_ctx;
        return nullptr;
    }

    if (requested_profile ==
            ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
        const char * runtime_xclbin = std::getenv("GGML_XDNA_Q4K_RUNTIME_XCLBIN");
        if (runtime_xclbin == nullptr || runtime_xclbin[0] == '\0') {
            std::fprintf(stderr, "ggml_xdna: GGML_XDNA_Q4K_RUNTIME_XCLBIN is required\n");
            delete backend_ctx;
            return nullptr;
        }
        static constexpr const char * frozen_xclbin_sha =
            "9a743ff724463aa68d2dd62911c73eeeccad414788097ba90fd0bbff3b682d27";
        struct shape_spec {
            size_t rows;
            const char * inst_env;
            const char * inst_sha;
            ggml_backend_xdna_batched_state ggml_backend_xdna_backend_context::* state;
        };
        const shape_spec shapes[] = {
            {1024, "GGML_XDNA_Q4K_M1024_INSTS", "b9504aa178b3412d7adbdb4cdc641892acae90940095a5a6b1357021209a65b8", &ggml_backend_xdna_backend_context::q4_m1024},
            {4096, "GGML_XDNA_Q4K_M4096_INSTS", "9b89a2a63ee7c0c26273a6d25ac6511582769734b48bb9486b56281e1dd78c03", &ggml_backend_xdna_backend_context::q4_m4096},
            {8192, "GGML_XDNA_Q4K_M8192_INSTS", "e093f613fcb230dd12e0674a308d42dde248307e106dbcbc4ff20ba470ceb480", &ggml_backend_xdna_backend_context::q4_m8192},
        };
        for (const auto & shape : shapes) {
            const char * insts = std::getenv(shape.inst_env);
            if (insts == nullptr || insts[0] == '\0' ||
                !ggml_backend_xdna_init_batched_shape(
                    backend_ctx, backend_ctx->*shape.state, shape.rows, runtime_xclbin, insts,
                    frozen_xclbin_sha, shape.inst_sha)) {
                std::fprintf(stderr, "ggml_xdna: missing or invalid canonical instructions for M=%zu\n", shape.rows);
                delete backend_ctx;
                return nullptr;
            }
        }
        const char * insts9216 = std::getenv("GGML_XDNA_Q4K_M9216_INSTS");
        if (insts9216 == nullptr || insts9216[0] == '\0' ||
            !ggml_backend_xdna_init_q4k_q8k_k2560_m9216(
                backend_ctx, runtime_xclbin, insts9216,
                "d8e42a47f2b9e063a77edb513bd15ccc992204bd6e3cf28a519a821f97feee97")) {
            std::fprintf(stderr, "ggml_xdna: missing or invalid canonical instructions for M=9216\n");
            delete backend_ctx;
            return nullptr;
        }
    }
    if (requested_profile ==
            ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
        device_ctx->q4_m9216_available = backend_ctx->q4_m9216_available;
        device_ctx->q4_m1024_available = backend_ctx->q4_m1024.available;
        device_ctx->q4_m4096_available = backend_ctx->q4_m4096.available;
        device_ctx->q4_m8192_available = backend_ctx->q4_m8192.available;
    }
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (std::getenv("GGML_XDNA_INSTANCE_TRACE") != nullptr) {
        std::fprintf(stderr, "XDNA_BACKEND_INSTANCE RESOURCES backend_id=%llu primary_xclbin=%p primary_hw=%p primary_kernel=%p M1024_hw=%p M1024_kernel=%p M4096_hw=%p M4096_kernel=%p M8192_hw=%p M8192_kernel=%p M9216_hw=%p M9216_kernel=%p\n", static_cast<unsigned long long>(backend_ctx->backend_instance_id), static_cast<void *>(backend_ctx->xclbin.get()), static_cast<void *>(backend_ctx->hw_context.get()), static_cast<void *>(backend_ctx->kernel.get()), static_cast<void *>(backend_ctx->q4_m1024.hw_context.get()), static_cast<void *>(backend_ctx->q4_m1024.kernel.get()), static_cast<void *>(backend_ctx->q4_m4096.hw_context.get()), static_cast<void *>(backend_ctx->q4_m4096.kernel.get()), static_cast<void *>(backend_ctx->q4_m8192.hw_context.get()), static_cast<void *>(backend_ctx->q4_m8192.kernel.get()), static_cast<void *>(backend_ctx->q4_m9216_hw_context.get()), static_cast<void *>(backend_ctx->q4_m9216_kernel.get()));
    }
#endif
    if (device_ctx->active_backend_count == 0) {
        device_ctx->active_profile =
            requested_profile;
    }

    ++device_ctx->active_backend_count;
    backend_ctx->device_profile_registered = true;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (std::getenv("GGML_XDNA_INSTANCE_TRACE") != nullptr) {
        std::fprintf(stderr, "XDNA_BACKEND_INSTANCE INIT backend_id=%llu backend_ptr=%p device_ctx=%p xrt_device=%p active_backend_count=%zu profile=%s\n", static_cast<unsigned long long>(backend_ctx->backend_instance_id), static_cast<void *>(backend_ctx), static_cast<void *>(device_ctx), static_cast<void *>(device_ctx->xrt_device.get()), device_ctx->active_backend_count, ggml_backend_xdna_kernel_profile_name(requested_profile));
    }
#endif
    std::fprintf(
        stderr,
        "ggml_xdna: backend profile '%s' registered "
        "(%zu active instance(s))\n",
        ggml_backend_xdna_kernel_profile_name(
            requested_profile),
        device_ctx->active_backend_count);

    auto * backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ dev,
        /* .context = */ backend_ctx,
    };
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    ggml_backend_xdna_diagnostic_capture_mark_initialized(backend_ctx);
#endif
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(
        ggml_backend_dev_t dev) {
    (void) dev;
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    (void) dev;
    (void) max_tensor_size;

    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_xdna_device_supports_op(
        ggml_backend_dev_t dev,
        const struct ggml_tensor * op) {
    const ggml_backend_xdna_context * device_ctx =
        ggml_backend_xdna_get_context(dev);

    if (device_ctx == nullptr || device_ctx->q4_shared == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> device_state_lock(device_ctx->q4_shared->state_mutex);
    if (device_ctx->active_backend_count == 0) {
        return false;
    }

    if (op == nullptr ||
        op->op != GGML_OP_MUL_MAT ||
        op->src[0] == nullptr ||
        op->src[1] == nullptr) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const bool trace_eligibility = std::getenv("GGML_XDNA_ELIGIBILITY_TRACE") != nullptr;
    auto trace_candidate = [&](const char * reason) {
        if (!trace_eligibility) {
            return;
        }
        auto tensor_info = [](const ggml_tensor * t) {
            if (t == nullptr) {
                return std::string("null");
            }
            const char * buft = t->buffer != nullptr ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "none";
            const bool host = t->buffer != nullptr && ggml_backend_buft_is_host(ggml_backend_buffer_get_type(t->buffer));
            char text[512];
            std::snprintf(text, sizeof(text), "%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] contig=%d host=%d buft=%s",
                    ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_is_contiguous(t) ? 1 : 0, host ? 1 : 0, buft);
            return std::string(text);
        };
        std::fprintf(stderr, "ggml_xdna: eligibility op=%s name=%s reason=%s src0={%s} src1={%s} dst={%s}\n",
                ggml_op_name(op->op), ggml_get_name(op), reason, tensor_info(src0).c_str(), tensor_info(src1).c_str(), tensor_info(op).c_str());
    };
#else
#define trace_candidate(reason) ((void) 0)
#endif
    switch (device_ctx->active_profile) {
        case ggml_backend_xdna_kernel_profile::none:
            // Plain-path/direct-probe mode intentionally does not
            // advertise operations to the scheduler.
            return false;

        case ggml_backend_xdna_kernel_profile::i16_256: {
            if (src0->type != GGML_TYPE_I16 ||
                src1->type != GGML_TYPE_I16 ||
                op->type != GGML_TYPE_F32) {
                return false;
            }

            constexpr int64_t size = 256;

            if (src0->ne[0] != size ||
                src0->ne[1] != size ||
                src0->ne[2] != 1 ||
                src0->ne[3] != 1 ||
                src1->ne[0] != size ||
                src1->ne[1] != size ||
                src1->ne[2] != 1 ||
                src1->ne[3] != 1 ||
                op->ne[0] != size ||
                op->ne[1] != size ||
                op->ne[2] != 1 ||
                op->ne[3] != 1) {
                return false;
            }

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   ggml_is_contiguous(op);
        }

        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560: {
            constexpr int64_t k = 2560;
            constexpr size_t q4_native_row_bytes = 10 * sizeof(block_q4_K);
            constexpr size_t f32_activation_bytes = static_cast<size_t>(k) * sizeof(float);
            const int64_t m = src0->ne[1];
            const bool artifact_available =
                m == 9216 ? device_ctx->q4_m9216_available :
                m == 1024 ? device_ctx->q4_m1024_available :
                m == 4096 ? device_ctx->q4_m4096_available :
                m == 8192 ? device_ctx->q4_m8192_available : false;
            const char * reason = nullptr;
            if (src0->type != GGML_TYPE_Q4_K) {
                reason = "R02_SRC0_NOT_Q4_K";
            } else if (src1->type != GGML_TYPE_F32) {
                reason = "R03_SRC1_NOT_F32";
            } else if (op->type != GGML_TYPE_F32) {
                reason = "R04_DST_NOT_F32";
            } else if (src0->ne[0] != k) {
                reason = "R05_K_NOT_2560";
            } else if (src1->ne[1] != 1) {
                reason = "R06_RHS_COLS_NOT_1";
            } else if (!artifact_available || (m != 1024 && m != 4096 && m != 8192 && m != 9216)) {
                reason = "R07_M_NOT_WHITELISTED";
            } else if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[0] != k || src1->ne[2] != 1 || src1->ne[3] != 1 || op->ne[0] != m || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
                reason = "R12_OTHER_SHAPE";
            } else if (!ggml_is_contiguous(src0)) {
                reason = "R08_SRC0_LAYOUT_NOT_NATIVE";
            } else if (!ggml_is_contiguous(src1)) {
                reason = "R10_SRC1_NOT_CONTIGUOUS";
            } else if (!ggml_is_contiguous(op)) {
                reason = "R11_DST_NOT_CONTIGUOUS";
            } else if (src0->nb[1] != q4_native_row_bytes) {
                reason = "R08_SRC0_LAYOUT_NOT_NATIVE";
            } else if (src1->nb[1] != f32_activation_bytes) {
                reason = "R10_SRC1_NOT_CONTIGUOUS";
            } else if (op->nb[0] != sizeof(float)) {
                reason = "R11_DST_NOT_CONTIGUOUS";
            } else if (ggml_nbytes(src0) != static_cast<size_t>(m) * q4_native_row_bytes ||
                       ggml_nbytes(src1) != f32_activation_bytes ||
                       ggml_nbytes(op) != static_cast<size_t>(m) * sizeof(float)) {
                reason = "R12_OTHER_SIZE";
            }
            if (reason != nullptr) {
                trace_candidate(reason);
                return false;
            }
            trace_candidate("ACCEPT");
            return true;
        }

        case ggml_backend_xdna_kernel_profile::q4k_q8k_w2in_256:
            // Dedicated W2IN block profile (W2IN block profile).
            // The proven block ABI (q8 292 B + compact 32 B -> 32 x i32) has no
            // scheduler-advertised GEMM geometry; it is driven exclusively by
            // the explicit block probe on this profile, so the scheduler must
            // never route an operation here. The K=2560 profiles are untouched.
            return false;
    }

    return false;
}
#undef trace_candidate

// Permanent XDNA scheduler-offload policy: allow the scheduler to offload
// host-resident operations to XDNA only when they satisfy the same
// constraints enforced by supports_op(). supports_op() is the single source
// of truth for XDNA offload eligibility, so the invariant is
//
//     OFFLOADABLE(op) <=> SUPPORTED(op)
//
// An unsupported operation stays on the CPU/general backend; a supported one
// may be selected by the scheduler. This is the normal accelerator-offload
// policy for host-resident operations, not a CPU bypass.
static bool ggml_backend_xdna_device_offload_op(
        ggml_backend_dev_t dev,
        const struct ggml_tensor * op) {
    const bool supported = ggml_backend_xdna_device_supports_op(dev, op);
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (std::getenv("GGML_XDNA_ELIGIBILITY_TRACE") != nullptr && op != nullptr && op->op == GGML_OP_MUL_MAT) {
        std::fprintf(stderr, "ggml_xdna: offload_decision name=%s result=%d src0_buft=%s src1_buft=%s dst_buft=%s\n",
                ggml_get_name(op), supported ? 1 : 0,
                op->src[0] != nullptr && op->src[0]->buffer != nullptr ? ggml_backend_buft_name(ggml_backend_buffer_get_type(op->src[0]->buffer)) : "none",
                op->src[1] != nullptr && op->src[1]->buffer != nullptr ? ggml_backend_buft_name(ggml_backend_buffer_get_type(op->src[1]->buffer)) : "none",
                op->buffer != nullptr ? ggml_backend_buft_name(ggml_backend_buffer_get_type(op->buffer)) : "none");
    }
#endif
    return supported;
}

static bool ggml_backend_xdna_device_supports_buft(
        ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft) {
    (void) dev;

    return buft != nullptr && ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ ggml_backend_xdna_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    (void) reg;
    return GGML_XDNA_NAME;
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    const auto * device =
        static_cast<ggml_backend_device *>(reg->context);

    if (device == nullptr || device->context == nullptr) {
        return 0;
    }

    const auto * ctx =
        static_cast<ggml_backend_xdna_context *>(device->context);

    if (ctx->xrt_device == nullptr) return 0;
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (ctx->stop_after_xrt_module_preflight) return 0;
#endif
#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    if (ctx->identity_gate_failed) return 0;
#endif
    return 1;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(
        ggml_backend_reg_t reg,
        size_t index) {
    if (index != 0 || ggml_backend_xdna_reg_get_device_count(reg) == 0) {
        return nullptr;
    }

    return static_cast<ggml_backend_device *>(reg->context);
}

static void * ggml_backend_xdna_reg_get_proc_address(
        ggml_backend_reg_t reg,
        const char * name) {
    (void) reg;

#if defined(GGML_XDNA_BUILD_DIAGNOSTICS)
    const char * q4_control_value = std::getenv("GGML_XDNA_DIAG_Q4_ORD7_AT_ORD8");
    if (q4_control_value != nullptr && std::strcmp(q4_control_value, "1") == 0) {
        if (reg == nullptr || reg->context == nullptr) return nullptr;
        const auto * device = static_cast<ggml_backend_device *>(reg->context);
        const auto * ctx = device == nullptr ? nullptr :
            static_cast<ggml_backend_xdna_context *>(device->context);
        if (ctx == nullptr || ctx->identity_gate_failed) return nullptr;
    }
#endif
    if (name != nullptr &&
        std::strcmp(name, "ggml_backend_xdna_run_i16_f32_matmul") == 0) {
        return (void *) ggml_backend_xdna_run_i16_f32_matmul;
    }

    if (name != nullptr &&
        std::strcmp(name, "ggml_backend_xdna_run_q4k_q8k_k2560") == 0) {
        return (void *) ggml_backend_xdna_run_q4k_q8k_k2560;
    }

    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static ggml_backend_xdna_context context =
        ggml_backend_xdna_create_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ nullptr,
    };

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ &reg,
        /* .context = */ &context,
    };

    reg.context = &device;
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
