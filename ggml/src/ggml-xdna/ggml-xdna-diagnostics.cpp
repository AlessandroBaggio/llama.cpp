#include "ggml-xdna-internal.h"
#include "ggml-backend-impl.h"
#include "ggml-xdna-capture-session.h"
#include "ggml-xdna-sha256.h"
#include "../../../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

namespace {
struct capture_backend_state {
    bool enabled = false;
    bool attached = false;
    bool initialized = false;
    uint64_t attach_ordinal = 0;
    std::filesystem::path dir;
    std::string xclbin_sha256;
};

struct capture_record {
    ggml_backend_xdna_backend_context * owner = nullptr;
    uint64_t id = 0;
    int64_t m = 0, k = 0, rhs = 0;
    uint32_t opcode = 3;
    std::string node;
    std::string instr_file, q4_file, q8_file;
    size_t instr_bytes = 0, q4_bytes = 0, q8_bytes = 0;
    std::string instr_sha, q4_sha, q8_sha;
};

std::mutex capture_mutex;
std::unordered_map<ggml_backend_xdna_backend_context *, capture_backend_state> capture_backends;
std::unordered_map<uint64_t, capture_record> capture_records;
ggml_xdna_capture::capture_session capture_session;

bool capture_state(ggml_backend_xdna_backend_context * ctx, capture_backend_state * out) {
    std::lock_guard<std::mutex> lock(capture_mutex);
    const auto it = capture_backends.find(ctx);
    if (it == capture_backends.end()) return false;
    if (out) *out = it->second;
    return true;
}

bool capture_append(ggml_backend_xdna_backend_context * ctx, const ordered_json & event, bool terminal = false, bool allow_failed = false) {
    capture_backend_state state;
    if (!capture_state(ctx, &state) || !state.attached) return false;
    return capture_session.append(event.dump() + "\n", terminal, allow_failed);
}

bool capture_get_record(ggml_backend_xdna_backend_context * ctx, uint64_t id, capture_record * out) {
    std::lock_guard<std::mutex> lock(capture_mutex);
    const auto it = capture_records.find(id);
    if (it == capture_records.end() || it->second.owner != ctx) return false;
    if (out) *out = it->second;
    return true;
}

void capture_erase_record(uint64_t id) {
    std::lock_guard<std::mutex> lock(capture_mutex);
    capture_records.erase(id);
}

std::string capture_filename(uint64_t id, const std::string & node, int64_t m, const char * kind) {
    std::string safe;
    safe.reserve(node.size());
    for (unsigned char c : node) safe.push_back(std::isalnum(c) || c == '-' || c == '_' ? static_cast<char>(c) : '_');
    if (safe.empty()) safe = "unnamed";
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "%06llu_%s_M%lld_", static_cast<unsigned long long>(id), safe.c_str(), static_cast<long long>(m));
    return std::string(prefix) + kind + ".bin";
}

ordered_json capture_event(const capture_record & r, const char * event, bool submitted, bool completed, const std::string & stop_reason, const std::string & output_file = {}, size_t output_bytes = 0, const std::string & output_sha = {}) {
    return ordered_json{{"event", event}, {"submission_id", r.id}, {"node", r.node}, {"M", r.m}, {"K", r.k}, {"rhs", r.rhs}, {"opcode", r.opcode},
        {"instruction_file", r.instr_file}, {"instruction_bytes", r.instr_bytes}, {"instruction_sha256", r.instr_sha},
        {"q4_file", r.q4_file}, {"q4_bytes", r.q4_bytes}, {"q4_sha256", r.q4_sha},
        {"q8_file", r.q8_file}, {"q8_bytes", r.q8_bytes}, {"q8_sha256", r.q8_sha},
        {"submitted", submitted}, {"completed", completed}, {"output_file", output_file}, {"output_bytes", output_bytes},
        {"output_sha256", output_sha}, {"stop_reason", stop_reason}};
}

bool capture_write_atomic(const std::filesystem::path & path, const void * data, size_t bytes) {
    if (std::filesystem::exists(path)) return false;
    const auto temp = path.string() + ".tmp";
    if (std::filesystem::exists(temp)) return false;
    std::ofstream out(temp, std::ios::binary | std::ios::out);
    if (!out) return false;
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    out.flush();
    out.close();
    if (!out) { std::filesystem::remove(temp); return false; }
    std::error_code ec;
    std::filesystem::rename(temp, path, ec);
    if (ec) { std::filesystem::remove(temp); return false; }
    return true;
}

bool capture_mode_requested() {
    const char * mode = std::getenv("GGML_XDNA_CAPTURE_EXACT_HISTORY");
    return mode != nullptr && mode[0] != '\0' && std::strcmp(mode, "0") != 0;
}

struct causal_capture_runtime {
    std::mutex mutex;
    std::atomic<int> mode{0}; // 0=uninitialized, 1=disabled, 2=enabled, -1=configuration or I/O failure
    ggml_xdna_causal_capture::writer output;
};
causal_capture_runtime causal_capture;

bool causal_capture_parse_u64(const char * text, uint64_t * value) {
    if (text == nullptr || text[0] == '\0') return false;
    for (const char * p = text; *p != '\0'; ++p) if (*p < '0' || *p > '9') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed == 0) return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool causal_capture_initialize() {
    int mode = causal_capture.mode.load(std::memory_order_acquire);
    if (mode == 1) return true;
    if (mode == 2) return true;
    if (mode == -1) return false;
    std::lock_guard<std::mutex> lock(causal_capture.mutex);
    mode = causal_capture.mode.load(std::memory_order_relaxed);
    if (mode != 0) return mode > 0;
    const char * dir = std::getenv("GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_DIR");
    const char * begin_text = std::getenv("GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_BEGIN");
    const char * end_text = std::getenv("GGML_XDNA_CAUSAL_PAYLOAD_CAPTURE_END");
    if (dir == nullptr && begin_text == nullptr && end_text == nullptr) {
        causal_capture.mode.store(1, std::memory_order_release);
        return true;
    }
    auto fail = [](const char * reason) {
        std::fprintf(stderr, "XDNA_CAUSAL_CAPTURE_ERROR=%s\n", reason);
        causal_capture.mode.store(-1, std::memory_order_release);
        return false;
    };
    if (dir == nullptr || dir[0] == '\0') return fail("DIR_BEGIN_END_REQUIRED");
    uint64_t begin = 0, end = 0;
    if (!causal_capture_parse_u64(begin_text, &begin) || !causal_capture_parse_u64(end_text, &end) ||
        end < begin || end - begin >= ggml_xdna_causal_capture::max_window_size) return fail("INVALID_BOUNDED_WINDOW");
    const char * forbidden[] = {
        "GGML_XDNA_BH1_REPLAY_NATIVE_Q4", "GGML_XDNA_BH1_REPLAY_F32",
        "GGML_XDNA_BH1_REPLAY_Q4_TRANSPORT", "GGML_XDNA_BH1_REPLAY_Q8",
        "GGML_XDNA_CAPTURE_M9216_PAYLOAD",
    };
    for (const char * name : forbidden) if (std::getenv(name) != nullptr) return fail("REPLAY_OR_OTHER_CAPTURE_CONTROL_PRESENT");
    const char * history = std::getenv("GGML_XDNA_CAPTURE_EXACT_HISTORY");
    if (history != nullptr && history[0] != '\0' && std::strcmp(history, "0") != 0) return fail("CAPTURE_MUST_BE_DISABLED");
    if (!causal_capture.output.open_session(dir, begin, end)) return fail("SESSION_CREATE_OR_VALIDATION_FAILED");
    causal_capture.mode.store(2, std::memory_order_release);
    return true;
}
} // namespace

bool ggml_backend_xdna_causal_capture_reserve(uint64_t * sequence_id, bool * selected) {
    if (sequence_id == nullptr || selected == nullptr || !causal_capture_initialize()) return false;
    const int mode = causal_capture.mode.load(std::memory_order_acquire);
    if (mode == 1) { *sequence_id = 0; *selected = false; return true; }
    if (mode != 2) return false;
    *sequence_id = causal_capture.output.reserve_sequence_id();
    if (*sequence_id == 0) return false;
    *selected = causal_capture.output.selected(*sequence_id);
    return true;
}

bool ggml_backend_xdna_causal_capture_prepare(uint64_t sequence_id, const ggml_xdna_causal_capture::submission_metadata & metadata, const void * q4_native, size_t q4_bytes, const void * q8, size_t q8_bytes) {
    if (sequence_id == 0) return true;
    const bool ok = causal_capture.output.prepare(sequence_id, metadata, q4_native, q4_bytes, q8, q8_bytes);
    if (!ok) std::fprintf(stderr, "XDNA_CAUSAL_CAPTURE_ERROR=PREPARED_RECORD_FAILED id=%llu\n", static_cast<unsigned long long>(sequence_id));
    return ok;
}

bool ggml_backend_xdna_causal_capture_event(uint64_t sequence_id, const char * event) {
    if (sequence_id == 0) return true;
    const bool ok = causal_capture.output.append_event(sequence_id, event);
    if (!ok) std::fprintf(stderr, "XDNA_CAUSAL_CAPTURE_ERROR=OUTCOME_EVENT_FAILED id=%llu event=%s\n", static_cast<unsigned long long>(sequence_id), event == nullptr ? "" : event);
    return ok;
}

std::string ggml_backend_xdna_capture_hash(const void * data, size_t bytes) {
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    char hex[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
    ggml_xdna_sha256_hash(digest, static_cast<const unsigned char *>(data), bytes);
    ggml_backend_xdna_format_sha256(digest, hex);
    return hex;
}

bool ggml_backend_xdna_diagnostic_capture_enabled(ggml_backend_xdna_backend_context * ctx) {
    std::lock_guard<std::mutex> lock(capture_mutex);
    const auto it = capture_backends.find(ctx);
    return it != capture_backends.end() && it->second.enabled && it->second.attached;
}

bool ggml_backend_xdna_diagnostic_capture_can_submit(ggml_backend_xdna_backend_context * ctx) {
    return !ggml_backend_xdna_diagnostic_capture_enabled(ctx) || capture_session.can_submit();
}

bool ggml_backend_xdna_diagnostic_capture_session_failed() {
    return capture_session.failed();
}



uint64_t ggml_backend_xdna_diagnostic_capture_reserve(ggml_backend_xdna_backend_context * ctx) {
    return ggml_backend_xdna_diagnostic_capture_enabled(ctx) ? capture_session.reserve_submission_id() : 0;
}

void ggml_backend_xdna_diagnostic_capture_fail(ggml_backend_xdna_backend_context * ctx, uint64_t id, const char * phase, const char * kind, const char * message) {
    if (!ggml_backend_xdna_diagnostic_capture_enabled(ctx)) return;
    capture_session.fail();
    capture_erase_record(id);
    if (!phase || !kind || !message) return;
    try {
        const ordered_json event{{"event", "ERROR"}, {"submission_id", id}, {"phase", phase}, {"error_kind", kind}, {"message", message}};
        (void) capture_append(ctx, event, false, true);
    } catch (...) { }
}

bool ggml_backend_xdna_diagnostic_capture_prepare(ggml_backend_xdna_backend_context * ctx, uint64_t id, const char * node, int64_t m, int64_t k, int64_t rhs, uint32_t opcode, const void * instr, size_t instr_bytes, const void * q4, size_t q4_bytes, const void * q8, size_t q8_bytes, bool * semantic_target) {
    if (semantic_target) *semantic_target = false;
    if (!ctx || !node || id == 0 || !instr || !q4 || !q8 || !instr_bytes || !q4_bytes || !q8_bytes || !capture_session.can_submit()) return false;
    capture_backend_state state;
    if (!capture_state(ctx, &state) || !state.enabled || !state.attached) return false;
    capture_record r;
    r.owner = ctx; r.id = id; r.node = node; r.m = m; r.k = k; r.rhs = rhs; r.opcode = opcode;
    const bool is_target = r.node == "ffn_gate-4" && k == 2560 && m == 9216 && rhs == 1;
    r.instr_file = capture_filename(id, r.node, m, "instr");
    r.q4_file = capture_filename(id, r.node, m, "q4");
    r.q8_file = capture_filename(id, r.node, m, "q8");
    r.instr_bytes = instr_bytes; r.q4_bytes = q4_bytes; r.q8_bytes = q8_bytes;
    r.instr_sha = ggml_backend_xdna_capture_hash(instr, instr_bytes);
    r.q4_sha = ggml_backend_xdna_capture_hash(q4, q4_bytes);
    r.q8_sha = ggml_backend_xdna_capture_hash(q8, q8_bytes);
    if (!capture_write_atomic(state.dir / r.instr_file, instr, instr_bytes) || !capture_write_atomic(state.dir / r.q4_file, q4, q4_bytes) || !capture_write_atomic(state.dir / r.q8_file, q8, q8_bytes)) {
        ggml_backend_xdna_diagnostic_capture_fail(ctx, id, "prepare", "artifact_write", "input artifact write failed");
        return false;
    }
    if (!capture_append(ctx, capture_event(r, "PREPARED", false, false, is_target ? "semantic_target" : ""), is_target)) {
        ggml_backend_xdna_diagnostic_capture_fail(ctx, id, "prepare", "manifest_write", "PREPARED event write failed");
        return false;
    }
    if (is_target) { if (semantic_target) *semantic_target = true; return true; }
    { std::lock_guard<std::mutex> lock(capture_mutex); capture_records[id] = std::move(r); }
    return true;
}

bool ggml_backend_xdna_diagnostic_capture_submitted(ggml_backend_xdna_backend_context * ctx, uint64_t id) {
    capture_record r;
    if (!capture_get_record(ctx, id, &r)) return false;
    if (capture_append(ctx, capture_event(r, "SUBMITTED", true, false, ""))) return true;
    ggml_backend_xdna_diagnostic_capture_fail(ctx, id, "submit_record", "manifest_write", "SUBMITTED event write failed");
    return false;
}

bool ggml_backend_xdna_diagnostic_capture_completed(ggml_backend_xdna_backend_context * ctx, uint64_t id, const void * output, size_t output_bytes) {
    capture_backend_state state;
    capture_record r;
    if (!output || !output_bytes || !capture_state(ctx, &state) || !capture_get_record(ctx, id, &r)) {
        ggml_backend_xdna_diagnostic_capture_fail(ctx, id, "output_write", "artifact_write", "output artifact write failed");
        return false;
    }
    const std::string output_file = capture_filename(id, r.node, r.m, "out");
    if (!capture_write_atomic(state.dir / output_file, output, output_bytes)) {
        ggml_backend_xdna_diagnostic_capture_fail(ctx, id, "output_write", "artifact_write", "output artifact write failed");
        return false;
    }
    const std::string sha = ggml_backend_xdna_capture_hash(output, output_bytes);
    if (!capture_append(ctx, capture_event(r, "COMPLETED", true, true, "", output_file, output_bytes, sha))) {
        ggml_backend_xdna_diagnostic_capture_fail(ctx, id, "completion_record", "manifest_write", "COMPLETED event write failed");
        return false;
    }
    capture_erase_record(id);
    return true;
}

bool ggml_backend_xdna_diagnostic_capture_payload_enabled() {
    return std::getenv("GGML_XDNA_CAPTURE_M9216_PAYLOAD") != nullptr;
}

void ggml_backend_xdna_diagnostic_capture_payload(uint64_t index, const char * label, const void * data, size_t bytes) {
    if (!ggml_backend_xdna_diagnostic_capture_payload_enabled() || !data) return;
    std::filesystem::create_directories("capture-artifacts");
    char path[256];
    std::snprintf(path, sizeof(path), "capture-artifacts/m9216_%02llu_%s.bin", static_cast<unsigned long long>(index), label);
    std::ofstream out(path, std::ios::binary);
    if (out) out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    char hex[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
    ggml_xdna_sha256_hash(digest, static_cast<const unsigned char *>(data), bytes);
    ggml_backend_xdna_format_sha256(digest, hex);
    std::fprintf(stderr, "XDNA_PAYLOAD capture index=%llu %s bytes=%zu sha256=%s file=%s\n", static_cast<unsigned long long>(index), label, bytes, hex, path);
}

bool ggml_backend_xdna_diagnostic_capture_validate_profile(ggml_backend_xdna_kernel_profile profile) {
    if (!capture_mode_requested()) return true;
    const char * mode = std::getenv("GGML_XDNA_CAPTURE_EXACT_HISTORY");
    if (mode && std::strcmp(mode, "1") == 0 && profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) return true;
    capture_session.fail();
    return false;
}

void ggml_backend_xdna_diagnostic_capture_fail_if_requested() {
    if (capture_mode_requested()) capture_session.fail();
}

bool ggml_backend_xdna_diagnostic_capture_initialize(ggml_backend_xdna_backend_context * ctx) {
    if (!capture_mode_requested()) return true;
    if (!ctx || !ggml_backend_xdna_diagnostic_capture_validate_profile(ctx->kernel_profile)) return false;
    const char * dir = std::getenv("GGML_XDNA_CAPTURE_HISTORY_DIR");
    if (!dir || !dir[0] || ctx->xclbin_path.empty()) { capture_session.fail(); return false; }
    bool attached = false;
    try {
        const auto path = std::filesystem::absolute(dir).lexically_normal();
        const auto xclbin_path = std::filesystem::absolute(ctx->xclbin_path).lexically_normal();
        unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
        char xclbin_sha[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        if (!ggml_backend_xdna_hash_file_sha256(xclbin_path.string(), digest)) { capture_session.fail(); return false; }
        ggml_backend_xdna_format_sha256(digest, xclbin_sha);
        const ordered_json metadata{{"schema_version", 1}, {"HEAD", ggml_commit()}, {"branch", nullptr}, {"capture_mode", "GGML_XDNA_CAPTURE_EXACT_HISTORY=1"},
            {"xclbin_path", xclbin_path.string()}, {"xclbin_sha256", xclbin_sha}, {"profile", ggml_backend_xdna_kernel_profile_name(ctx->kernel_profile)}};
        const std::string metadata_text = metadata.dump(2) + "\n";
        bool created = false; uint64_t ordinal = 0;
        if (!capture_session.attach(path, xclbin_path, ggml_backend_xdna_kernel_profile_name(ctx->kernel_profile), xclbin_sha, metadata_text, &created, &ordinal)) {
            std::fprintf(stderr, "ggml_xdna: XDNA capture session attach rejected backend_id=%llu backend_ctx=%p dir=%s profile=%s xclbin_sha256=%s\n", static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<void *>(ctx), path.string().c_str(), ggml_backend_xdna_kernel_profile_name(ctx->kernel_profile), xclbin_sha);
            return false;
        }
        attached = true;
        capture_backend_state state; state.enabled = true; state.attached = true; state.attach_ordinal = ordinal; state.dir = path; state.xclbin_sha256 = xclbin_sha;
        { std::lock_guard<std::mutex> lock(capture_mutex); capture_backends[ctx] = std::move(state); }
        std::fprintf(stderr, "ggml_xdna: XDNA capture session session_init=%llu backend_id=%llu backend_ctx=%p %s attached=%zu next_submission_id=%llu dir=%s\n", static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<void *>(ctx), created ? "CREATED" : "ATTACHED", capture_session.attached_count(), static_cast<unsigned long long>(capture_session.next_submission_id()), path.string().c_str());
        return true;
    } catch (const std::exception & e) {
        capture_session.fail();
        if (attached) (void) capture_session.detach();
        std::fprintf(stderr, "ggml_xdna: XDNA capture session initialization failed: %s\n", e.what());
        return false;
    } catch (...) {
        capture_session.fail();
        if (attached) (void) capture_session.detach();
        return false;
    }
}

void ggml_backend_xdna_diagnostic_capture_mark_initialized(ggml_backend_xdna_backend_context * ctx) {
    std::lock_guard<std::mutex> lock(capture_mutex);
    const auto it = capture_backends.find(ctx);
    if (it != capture_backends.end()) it->second.initialized = true;
}

void ggml_backend_xdna_diagnostic_capture_detach(ggml_backend_xdna_backend_context * ctx) {
    bool attached_state = false;
    bool initialized = false;
    uint64_t attach_ordinal = 0;
    {
        std::lock_guard<std::mutex> lock(capture_mutex);
        const auto it = capture_backends.find(ctx);
        if (it == capture_backends.end()) return;
        attached_state = it->second.attached;
        initialized = it->second.initialized;
        attach_ordinal = it->second.attach_ordinal;
        capture_backends.erase(it);
        for (auto record = capture_records.begin(); record != capture_records.end();) {
            if (record->second.owner == ctx) record = capture_records.erase(record); else ++record;
        }
    }
    if (!attached_state) return;
    if (!initialized) capture_session.fail();
    const size_t attached = capture_session.detach();
    std::fprintf(stderr, "ggml_xdna: XDNA capture session backend_free session_init=%llu backend_id=%llu backend_ctx=%p attached=%zu next_submission_id=%llu failed=%d terminal=%d init_complete=%d\n", static_cast<unsigned long long>(attach_ordinal), static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<void *>(ctx), attached, static_cast<unsigned long long>(capture_session.next_submission_id()), capture_session.failed() ? 1 : 0, capture_session.terminal() ? 1 : 0, initialized ? 1 : 0);
}



static bool ggml_backend_xdna_replay_read_artifact(
        const std::filesystem::path & root, const nlohmann::json & record,
        const char * file_key, const char * size_key, const char * sha_key,
        std::vector<unsigned char> * bytes, std::string * error) {
    try {
        const std::string name = record.at(file_key).get<std::string>();
        const std::filesystem::path relative(name);
        if (name.empty() || relative.is_absolute() || relative.filename() != relative ||
            name == "." || name == ".." || name.find('/') != std::string::npos || name.find('\\') != std::string::npos) {
            if (error) *error = "unsafe artifact path"; return false;
        }
        const auto path = root / relative;
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(path)) || !std::filesystem::is_regular_file(path)) {
            if (error) *error = "missing/non-regular artifact"; return false;
        }
        const uint64_t declared = record.at(size_key).get<uint64_t>();
        if (declared > SIZE_MAX || std::filesystem::file_size(path) != declared) {
            if (error) *error = "artifact size mismatch"; return false;
        }
        std::vector<unsigned char> data(static_cast<size_t>(declared));
        std::ifstream in(path, std::ios::binary);
        if (!in || (declared != 0 && !in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(declared)))) {
            if (error) *error = "artifact read failed"; return false;
        }
        if (ggml_backend_xdna_capture_hash(data.data(), data.size()) != record.at(sha_key).get<std::string>()) {
            if (error) *error = "artifact SHA mismatch"; return false;
        }
        if (bytes) *bytes = std::move(data);
        return true;
    } catch (const std::exception & e) { if (error) *error = e.what(); return false; }
}

struct ggml_backend_xdna_replay_runtime_trace {
    uint64_t execution_ordinal = 0, m9216_execution_ordinal = 0;
    uint64_t experiment_submission_ordinal = 0, context_generation_submission_ordinal = 0;
    const void * post_id7_device = nullptr, * post_id7_xclbin = nullptr;
    const void * post_id7_context = nullptr, * post_id7_kernel = nullptr;
    const void * post_id7_bo_instr = nullptr, * post_id7_bo_q4 = nullptr, * post_id7_bo_q8 = nullptr, * post_id7_bo_out = nullptr;
    const void * post_id7_map_instr = nullptr, * post_id7_map_q4 = nullptr, * post_id7_map_q8 = nullptr, * post_id7_map_out = nullptr;
    size_t post_id7_instr_capacity = 0, post_id7_q4_capacity = 0, post_id7_q8_capacity = 0, post_id7_out_capacity = 0;
    std::string post_id7_xclbin_uuid, post_id7_kernel_name;
    uint64_t generation1_execution_ordinal = 0, generation1_submissions = 0, generation1_wait_completions = 0;
    uint64_t generation1_output_syncs = 0;
    uint64_t generation1_exact_outputs = 0;
    uint64_t m9216_instruction_syncs = 0;
    uint64_t generation1_instruction_init_syncs = 0, generation1_instruction_stage_syncs = 0;
    uint64_t generation1_q4_syncs = 0, generation1_q8_syncs = 0;
    uint64_t generation1_output_sentinel_syncs = 0, generation1_output_from_device_syncs = 0;
    uint64_t m9216_submissions = 0, m9216_wait_completions = 0;
    uint64_t m9216_wait_calls_returned = 0, m9216_input_syncs = 0;
    uint64_t m9216_initialization_syncs = 0, m9216_output_sentinel_syncs = 0;
    uint64_t m9216_output_syncs = 0, m9216_bundle_reuses = 0;
    uint64_t m9216_bo_bundle_generation = 0, m9216_bo_bundle_recreate_count = 0;
    uint64_t m9216_bo_bundle_generation_submissions = 0, m9216_bo_bundle_generation_wait_completions = 0;
    uint64_t previous_m9216_bo_bundle_generation = 0;
    uint64_t m9216_hw_context_generation = 0, m9216_hw_context_recreate_count = 0;
    uint64_t previous_m9216_hw_context_generation = 0;
    uint64_t m9216_kernel_generation = 0, m9216_kernel_recreate_count = 0;
    uint64_t previous_m9216_kernel_generation = 0;
    uint64_t run_objects_created = 0, run_wait_calls_returned = 0, run_wait_completions = 0;
    uint64_t output_from_device_syncs = 0, output_mapped_read_count = 0;
    bool have_previous_m9216 = false;
    const void * previous_hw_context = nullptr, * previous_kernel = nullptr;
    const void * previous_bo_instr = nullptr, * previous_bo_q4 = nullptr;
    const void * previous_bo_q8 = nullptr, * previous_bo_out = nullptr;
    const void * previous_map_instr = nullptr, * previous_map_q4 = nullptr;
    const void * previous_map_q8 = nullptr, * previous_map_out = nullptr;
};

struct ggml_backend_xdna_replay_transition_bundle {
    xrt::hw_context * hw_context = nullptr;
    xrt::kernel * kernel = nullptr;
    std::unique_ptr<xrt::bo> bo_instr, bo_q4, bo_q8, bo_out;
    void * buf_instr = nullptr, * buf_q4 = nullptr, * buf_q8 = nullptr;
    float * buf_out = nullptr;
    uint32_t group_instr = 0, group_q4 = 0, group_q8 = 0, group_out = 0;
    uint64_t instruction_init_sync_count = 0, instruction_stage_sync_count = 0;
    uint64_t m4096_output_from_device_syncs = 0, m4096_output_reads = 0, m4096_output_sha_calculations = 0, m4096_output_sha_validations = 0;
    uint64_t id7_submissions = 0, id7_waits = 0, id7_output_syncs = 0, id7_validations = 0, id7_exact_outputs = 0;
    bool id7_first_snapshot_valid = false;
    const void * id7_first_hw_context = nullptr, * id7_first_kernel = nullptr;
    const void * id7_first_bo_instr = nullptr, * id7_first_bo_q4 = nullptr, * id7_first_bo_q8 = nullptr, * id7_first_bo_out = nullptr;
    const void * id7_first_map_instr = nullptr, * id7_first_map_q4 = nullptr, * id7_first_map_q8 = nullptr, * id7_first_map_out = nullptr;
    uint64_t generation = 0;
};

struct ggml_backend_xdna_replay_program_only_bundle {
    std::unique_ptr<xrt::bo> bo_instr;
    void * mapped_instr = nullptr;
    uint64_t instruction_init_sync_count = 0, instruction_stage_sync_count = 0;
};

static bool ggml_backend_xdna_replay_ensure_batched_bos(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_batched_state & state) {
    if (!ctx || !ctx->device_context || !ctx->device_context->xrt_device || !state.available || !state.kernel) return false;
    if (state.bos_initialized) return true;
    const size_t instruction_bytes = state.instructions.size() * sizeof(uint32_t);
    if (!instruction_bytes) return false;
    auto & kernel = *state.kernel;
    auto & device = *ctx->device_context->xrt_device;
    state.group_instr = kernel.group_id(1);
    state.bo_instr = std::make_unique<xrt::bo>(device, instruction_bytes, XCL_BO_FLAGS_CACHEABLE, state.group_instr);
    state.group_q4 = kernel.group_id(3);
    state.bo_q4 = std::make_unique<xrt::bo>(device, state.rows * 1520, XRT_BO_FLAGS_HOST_ONLY, state.group_q4);
    state.group_q8 = kernel.group_id(4);
    state.bo_q8 = std::make_unique<xrt::bo>(device, 2920, XRT_BO_FLAGS_HOST_ONLY, state.group_q8);
    state.group_out = kernel.group_id(5);
    state.bo_out = std::make_unique<xrt::bo>(device, state.rows * sizeof(float), XRT_BO_FLAGS_HOST_ONLY, state.group_out);
    state.buf_instr = state.bo_instr->map<void *>(); state.buf_q4 = state.bo_q4->map<void *>();
    state.buf_q8 = state.bo_q8->map<void *>(); state.buf_out = state.bo_out->map<float *>();
    std::memcpy(state.buf_instr, state.instructions.data(), instruction_bytes);
    ggml_backend_xdna_sync_host_written_bo(*state.bo_instr, "XDNA capture replay instruction BO init");
    state.instruction_count = static_cast<uint32_t>(state.instructions.size());
    state.bos_initialized = true;
    return true;
}

static bool ggml_backend_xdna_replay_ensure_m9216_bos(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_replay_runtime_trace * trace, bool validate_frozen_instruction = false) {
    if (!ctx || !ctx->device_context || !ctx->device_context->xrt_device || !ctx->q4_m9216_available || !ctx->q4_m9216_kernel) return false;
    if (ctx->q4_m9216_bos_initialized) return true;
    const size_t instruction_bytes = ctx->q4_m9216_instructions.size() * sizeof(uint32_t);
    if (!instruction_bytes) return false;
    if (validate_frozen_instruction && instruction_bytes != 189108) return false;
    auto & kernel = *ctx->q4_m9216_kernel;
    auto & device = *ctx->device_context->xrt_device;
    ctx->q4_m9216_group_instr = kernel.group_id(1);
    ctx->q4_m9216_group_q4 = kernel.group_id(3);
    ctx->q4_m9216_group_q8 = kernel.group_id(4);
    ctx->q4_m9216_group_out = kernel.group_id(5);
    auto bi = std::make_unique<xrt::bo>(device, instruction_bytes, XCL_BO_FLAGS_CACHEABLE, ctx->q4_m9216_group_instr);
    auto bq4 = std::make_unique<xrt::bo>(device, 9216 * 1520, XRT_BO_FLAGS_HOST_ONLY, ctx->q4_m9216_group_q4);
    auto bq8 = std::make_unique<xrt::bo>(device, 2920, XRT_BO_FLAGS_HOST_ONLY, ctx->q4_m9216_group_q8);
    auto bout = std::make_unique<xrt::bo>(device, 9216 * sizeof(float), XRT_BO_FLAGS_HOST_ONLY, ctx->q4_m9216_group_out);
    void * mapped_instr = bi->map<void *>();
    void * mapped_q4 = bq4->map<void *>();
    void * mapped_q8 = bq8->map<void *>();
    float * mapped_out = bout->map<float *>();
    if (!mapped_instr || !mapped_q4 || !mapped_q8 || !mapped_out) return false;
    std::memcpy(mapped_instr, ctx->q4_m9216_instructions.data(), instruction_bytes);
    ggml_backend_xdna_sync_host_written_bo(*bi, "XDNA capture replay M9216 instruction BO init");
    if (trace) {
        ++trace->m9216_initialization_syncs;
        if (trace->m9216_hw_context_generation == 1 || validate_frozen_instruction) ++trace->generation1_instruction_init_syncs;
    }
    if (validate_frozen_instruction && ggml_backend_xdna_capture_hash(mapped_instr, instruction_bytes) !=
        "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3") {
        throw std::runtime_error("recreated M9216 instruction BO SHA mismatch");
    }
    ctx->q4_m9216_buf_instr = mapped_instr; ctx->q4_m9216_buf_q4 = mapped_q4;
    ctx->q4_m9216_buf_q8 = mapped_q8; ctx->q4_m9216_buf_out = mapped_out;
    ctx->q4_m9216_instruction_count = static_cast<uint32_t>(ctx->q4_m9216_instructions.size());
    ctx->q4_m9216_bo_instr = std::move(bi); ctx->q4_m9216_bo_q4 = std::move(bq4);
    ctx->q4_m9216_bo_q8 = std::move(bq8); ctx->q4_m9216_bo_out = std::move(bout);
    ctx->q4_m9216_bos_initialized = true;
    return true;
}

static const char * ggml_backend_xdna_replay_identity_change(bool have_previous, const void * current, const void * previous) {
    if (!have_previous) return "FIRST";
    return current == previous ? "SAME_AS_PREVIOUS" : "CHANGED";
}

static void ggml_backend_xdna_replay_recreate_m9216_bo_bundle(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_replay_runtime_trace * trace) {
    if (!ctx || !trace || !ctx->device_context || !ctx->device_context->xrt_device ||
        !ctx->q4_m9216_bos_initialized || !ctx->q4_m9216_bo_instr || !ctx->q4_m9216_bo_q4 ||
        !ctx->q4_m9216_bo_q8 || !ctx->q4_m9216_bo_out || trace->m9216_bo_bundle_generation != 0 ||
        trace->m9216_bo_bundle_recreate_count != 0) {
        throw std::runtime_error("M9216 BO-bundle recreation precondition/count violation");
    }

    const void * device_id = ctx->device_context->xrt_device.get();
    const void * xclbin_id = ctx->q4_m9216_xclbin.get();
    const void * hw_context_id = ctx->q4_m9216_hw_context.get();
    const void * kernel_id = ctx->q4_m9216_kernel.get();
    const void * instr_bo_id = ctx->q4_m9216_bo_instr.get();
    const void * q4_bo_id = ctx->q4_m9216_bo_q4.get();
    const void * q8_bo_id = ctx->q4_m9216_bo_q8.get();
    const void * out_bo_id = ctx->q4_m9216_bo_out.get();
    const void * instr_map = ctx->q4_m9216_buf_instr;
    const void * q4_map = ctx->q4_m9216_buf_q4;
    const void * q8_map = ctx->q4_m9216_buf_q8;
    const void * out_map = ctx->q4_m9216_buf_out;
    const uint32_t initial_group_instr = ctx->q4_m9216_group_instr;
    const uint32_t initial_group_q4 = ctx->q4_m9216_group_q4;
    const uint32_t initial_group_q8 = ctx->q4_m9216_group_q8;
    const uint32_t initial_group_out = ctx->q4_m9216_group_out;
    const size_t instr_bytes = ctx->q4_m9216_instructions.size() * sizeof(uint32_t);
    constexpr size_t q4_bytes = 9216 * 1520;
    constexpr size_t q8_bytes = 2920;
    constexpr size_t out_bytes = 9216 * sizeof(float);
    const size_t initial_instr_capacity = ctx->q4_m9216_bo_instr->size();
    const size_t initial_q4_capacity = ctx->q4_m9216_bo_q4->size();
    const size_t initial_q8_capacity = ctx->q4_m9216_bo_q8->size();
    const size_t initial_out_capacity = ctx->q4_m9216_bo_out->size();
    if (initial_instr_capacity != instr_bytes || initial_q4_capacity != q4_bytes ||
        initial_q8_capacity != q8_bytes || initial_out_capacity != out_bytes) {
        throw std::runtime_error("M9216 BO-bundle recreation initial capacity mismatch");
    }

    auto log_bundle = [&](const char * phase) {
        std::fprintf(stderr,
            "%s\n generation=%llu recreate_count=%llu generation_submissions=%llu generation_wait_completions=%llu "
            "device_id=%p xclbin_id=%p hw_context_id=%p kernel_id=%p "
            "instr_bo_id=%p instr_map=%p q4_bo_id=%p q4_map=%p q8_bo_id=%p q8_map=%p out_bo_id=%p out_map=%p "
            "capacities=%zu,%zu,%zu,%zu groups=%u,%u,%u,%u execution_ordinal=%llu m9216_execution_ordinal=%llu "
            "cumulative_m9216_submissions=%llu cumulative_m9216_wait_calls_returned=%llu cumulative_m9216_wait_completions=%llu "
            "cumulative_m9216_initialization_syncs=%llu cumulative_m9216_input_syncs=%llu cumulative_m9216_output_sentinel_syncs=%llu cumulative_m9216_output_syncs=%llu "
            "cumulative_run_objects=%llu cumulative_run_wait_calls_returned=%llu cumulative_run_wait_completions=%llu "
            "replay_kernels_submitted=%llu replay_waits_completed=%llu replay_outputs_validated=%llu\n",
            phase,
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_recreate_count),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation_submissions),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation_wait_completions),
            static_cast<const void *>(ctx->device_context->xrt_device.get()), static_cast<const void *>(ctx->q4_m9216_xclbin.get()),
            static_cast<const void *>(ctx->q4_m9216_hw_context.get()), static_cast<const void *>(ctx->q4_m9216_kernel.get()),
            static_cast<const void *>(ctx->q4_m9216_bo_instr.get()), static_cast<const void *>(ctx->q4_m9216_buf_instr),
            static_cast<const void *>(ctx->q4_m9216_bo_q4.get()), static_cast<const void *>(ctx->q4_m9216_buf_q4),
            static_cast<const void *>(ctx->q4_m9216_bo_q8.get()), static_cast<const void *>(ctx->q4_m9216_buf_q8),
            static_cast<const void *>(ctx->q4_m9216_bo_out.get()), static_cast<const void *>(ctx->q4_m9216_buf_out),
            ctx->q4_m9216_bo_instr->size(), ctx->q4_m9216_bo_q4->size(),
            ctx->q4_m9216_bo_q8->size(), ctx->q4_m9216_bo_out->size(),
            ctx->q4_m9216_group_instr, ctx->q4_m9216_group_q4, ctx->q4_m9216_group_q8, ctx->q4_m9216_group_out,
            static_cast<unsigned long long>(trace->execution_ordinal),
            static_cast<unsigned long long>(trace->m9216_execution_ordinal),
            static_cast<unsigned long long>(trace->m9216_submissions),
            static_cast<unsigned long long>(trace->m9216_wait_calls_returned),
            static_cast<unsigned long long>(trace->m9216_wait_completions),
            static_cast<unsigned long long>(trace->m9216_initialization_syncs),
            static_cast<unsigned long long>(trace->m9216_input_syncs),
            static_cast<unsigned long long>(trace->m9216_output_sentinel_syncs),
            static_cast<unsigned long long>(trace->m9216_output_syncs),
            static_cast<unsigned long long>(trace->run_objects_created),
            static_cast<unsigned long long>(trace->run_wait_calls_returned),
            static_cast<unsigned long long>(trace->run_wait_completions),
            static_cast<unsigned long long>(ctx->replay_kernels_submitted),
            static_cast<unsigned long long>(ctx->replay_waits_completed),
            static_cast<unsigned long long>(ctx->replay_outputs_validated));
    };

    log_bundle("XDNA_CAPTURE_REPLAY M9216 BO BUNDLE RECREATE BEFORE");
    ctx->q4_m9216_buf_instr = nullptr;
    ctx->q4_m9216_buf_q4 = nullptr;
    ctx->q4_m9216_buf_q8 = nullptr;
    ctx->q4_m9216_buf_out = nullptr;
    ctx->q4_m9216_bos_initialized = false;
    ctx->q4_m9216_instruction_count = 0;
    ctx->q4_m9216_bo_instr.reset();
    ctx->q4_m9216_bo_q4.reset();
    ctx->q4_m9216_bo_q8.reset();
    ctx->q4_m9216_bo_out.reset();

    if (!ggml_backend_xdna_replay_ensure_m9216_bos(ctx, trace, true)) {
        throw std::runtime_error("failed to recreate M9216 BO bundle");
    }
    trace->m9216_bo_bundle_generation = 1;
    trace->m9216_bo_bundle_recreate_count = 1;
    trace->m9216_bo_bundle_generation_submissions = 0;
    trace->m9216_bo_bundle_generation_wait_completions = 0;
    log_bundle("XDNA_CAPTURE_REPLAY M9216 BO BUNDLE RECREATE AFTER");
    if (ctx->device_context->xrt_device.get() != device_id || ctx->q4_m9216_xclbin.get() != xclbin_id ||
        ctx->q4_m9216_hw_context.get() != hw_context_id || ctx->q4_m9216_kernel.get() != kernel_id ||
        !ctx->q4_m9216_bo_instr || !ctx->q4_m9216_bo_q4 || !ctx->q4_m9216_bo_q8 || !ctx->q4_m9216_bo_out ||
        !ctx->q4_m9216_buf_instr || !ctx->q4_m9216_buf_q4 || !ctx->q4_m9216_buf_q8 || !ctx->q4_m9216_buf_out ||
        ctx->q4_m9216_bo_instr->size() != initial_instr_capacity || ctx->q4_m9216_bo_q4->size() != initial_q4_capacity ||
        ctx->q4_m9216_bo_q8->size() != initial_q8_capacity || ctx->q4_m9216_bo_out->size() != initial_out_capacity) {
        throw std::runtime_error("M9216 BO recreation changed a forbidden resource, capacity, or left an incomplete bundle");
    }
    std::fprintf(stderr,
        "XDNA_CAPTURE_REPLAY M9216 BO BUNDLE RECREATE RELATIONS hw_context=%s kernel=%s instr_bo_address=%s q4_bo_address=%s "
        "q8_bo_address=%s out_bo_address=%s instr_map_address=%s q4_map_address=%s q8_map_address=%s out_map_address=%s "
        "group_instr=%s group_q4=%s group_q8=%s group_out=%s capacities=SAME generation=CHANGED\n",
        ctx->q4_m9216_hw_context.get() == hw_context_id ? "SAME" : "CHANGED",
        ctx->q4_m9216_kernel.get() == kernel_id ? "SAME" : "CHANGED",
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_instr.get(), instr_bo_id),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_q4.get(), q4_bo_id),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_q8.get(), q8_bo_id),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_out.get(), out_bo_id),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_instr, instr_map),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_q4, q4_map),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_q8, q8_map),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_out, out_map),
        ctx->q4_m9216_group_instr == initial_group_instr ? "SAME_AS_INITIAL" : "CHANGED_FROM_INITIAL",
        ctx->q4_m9216_group_q4 == initial_group_q4 ? "SAME_AS_INITIAL" : "CHANGED_FROM_INITIAL",
        ctx->q4_m9216_group_q8 == initial_group_q8 ? "SAME_AS_INITIAL" : "CHANGED_FROM_INITIAL",
        ctx->q4_m9216_group_out == initial_group_out ? "SAME_AS_INITIAL" : "CHANGED_FROM_INITIAL");
}

static void ggml_backend_xdna_replay_recreate_m9216_kernel(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_replay_runtime_trace * trace) {
    if (!ctx || !trace || !ctx->device_context || !ctx->device_context->xrt_device ||
        !ctx->q4_m9216_xclbin || !ctx->q4_m9216_hw_context || !ctx->q4_m9216_kernel ||
        !ctx->q4_m9216_bos_initialized || !ctx->q4_m9216_bo_instr || !ctx->q4_m9216_bo_q4 ||
        !ctx->q4_m9216_bo_q8 || !ctx->q4_m9216_bo_out || !ctx->q4_m9216_buf_instr ||
        !ctx->q4_m9216_buf_q4 || !ctx->q4_m9216_buf_q8 || !ctx->q4_m9216_buf_out ||
        trace->m9216_kernel_generation != 0 ||
        trace->m9216_kernel_recreate_count != 0) {
        throw std::runtime_error("M9216 kernel recreation precondition/count violation");
    }

    const void * device_id = ctx->device_context->xrt_device.get();
    const void * xclbin_id = ctx->q4_m9216_xclbin.get();
    const void * hw_context_id = ctx->q4_m9216_hw_context.get();
    const void * old_kernel_id = ctx->q4_m9216_kernel.get();
    const void * bo_instr_id = ctx->q4_m9216_bo_instr.get();
    const void * bo_q4_id = ctx->q4_m9216_bo_q4.get();
    const void * bo_q8_id = ctx->q4_m9216_bo_q8.get();
    const void * bo_out_id = ctx->q4_m9216_bo_out.get();
    const void * map_instr = ctx->q4_m9216_buf_instr;
    const void * map_q4 = ctx->q4_m9216_buf_q4;
    const void * map_q8 = ctx->q4_m9216_buf_q8;
    const void * map_out = ctx->q4_m9216_buf_out;
    const size_t instr_capacity = ctx->q4_m9216_bo_instr->size();
    const size_t q4_capacity = ctx->q4_m9216_bo_q4->size();
    const size_t q8_capacity = ctx->q4_m9216_bo_q8->size();
    const size_t out_capacity = ctx->q4_m9216_bo_out->size();
    const uint32_t bo_group_instr = ctx->q4_m9216_bo_instr->get_memory_group();
    const uint32_t bo_group_q4 = ctx->q4_m9216_bo_q4->get_memory_group();
    const uint32_t bo_group_q8 = ctx->q4_m9216_bo_q8->get_memory_group();
    const uint32_t bo_group_out = ctx->q4_m9216_bo_out->get_memory_group();
    const std::string kernel_name = ctx->q4_m9216_kernel->get_name();
    const int old_group_instr = ctx->q4_m9216_kernel->group_id(1);
    const int old_group_q4 = ctx->q4_m9216_kernel->group_id(3);
    const int old_group_q8 = ctx->q4_m9216_kernel->group_id(4);
    const int old_group_out = ctx->q4_m9216_kernel->group_id(5);

    auto log_snapshot = [&](const char * phase, const xrt::kernel * kernel, bool have_new_group_ids,
                            int new_group_instr, int new_group_q4, int new_group_q8, int new_group_out,
                            const char * kernel_group_relation) {
        std::fprintf(stderr,
            "%s\n kernel_generation=%llu kernel_recreate_count=%llu bo_bundle_generation=%llu bo_bundle_recreate_count=%llu device_id=%p xclbin_id=%p hw_context_id=%p kernel_id=%p kernel_name=%s\n",
            phase, static_cast<unsigned long long>(trace->m9216_kernel_generation),
            static_cast<unsigned long long>(trace->m9216_kernel_recreate_count),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_recreate_count),
            static_cast<const void *>(ctx->device_context->xrt_device.get()),
            static_cast<const void *>(ctx->q4_m9216_xclbin.get()),
            static_cast<const void *>(ctx->q4_m9216_hw_context.get()),
            static_cast<const void *>(kernel), kernel_name.c_str());
        std::fprintf(stderr,
            "instr_bo_id=%p instr_map=%p instr_capacity=%zu bo_memory_group=%u "
            "q4_bo_id=%p q4_map=%p q4_capacity=%zu bo_memory_group=%u "
            "q8_bo_id=%p q8_map=%p q8_capacity=%zu bo_memory_group=%u "
            "out_bo_id=%p out_map=%p out_capacity=%zu bo_memory_group=%u\n",
            static_cast<const void *>(ctx->q4_m9216_bo_instr.get()), static_cast<const void *>(ctx->q4_m9216_buf_instr), instr_capacity, bo_group_instr,
            static_cast<const void *>(ctx->q4_m9216_bo_q4.get()), static_cast<const void *>(ctx->q4_m9216_buf_q4), q4_capacity, bo_group_q4,
            static_cast<const void *>(ctx->q4_m9216_bo_q8.get()), static_cast<const void *>(ctx->q4_m9216_buf_q8), q8_capacity, bo_group_q8,
            static_cast<const void *>(ctx->q4_m9216_bo_out.get()), static_cast<const void *>(ctx->q4_m9216_buf_out), out_capacity, bo_group_out);
        std::fprintf(stderr, "old_kernel_group_ids=%d,%d,%d,%d retained_bo_memory_groups=%u,%u,%u,%u\n",
            old_group_instr, old_group_q4, old_group_q8, old_group_out, bo_group_instr, bo_group_q4, bo_group_q8, bo_group_out);
        if (have_new_group_ids) {
            std::fprintf(stderr, "new_kernel_group_ids=%d,%d,%d,%d kernel_group_relation=%s\n",
                new_group_instr, new_group_q4, new_group_q8, new_group_out, kernel_group_relation);
        } else {
            std::fprintf(stderr, "new_kernel_group_ids=NOT_CREATED kernel_group_relation=NOT_TESTED\n");
        }
        std::fprintf(stderr,
            "cumulative_m9216_ordinal=%llu submissions=%llu wait_calls_returned=%llu wait_completions=%llu input_syncs=%llu initialization_syncs=%llu sentinel_syncs=%llu output_syncs=%llu "
            "cumulative_replay_execution=%llu replay_submissions=%llu replay_waits=%llu replay_outputs=%llu run_objects=%llu run_wait_calls_returned=%llu run_wait_completions=%llu\n",
            static_cast<unsigned long long>(trace->m9216_execution_ordinal),
            static_cast<unsigned long long>(trace->m9216_submissions),
            static_cast<unsigned long long>(trace->m9216_wait_calls_returned),
            static_cast<unsigned long long>(trace->m9216_wait_completions),
            static_cast<unsigned long long>(trace->m9216_input_syncs),
            static_cast<unsigned long long>(trace->m9216_initialization_syncs),
            static_cast<unsigned long long>(trace->m9216_output_sentinel_syncs),
            static_cast<unsigned long long>(trace->m9216_output_syncs),
            static_cast<unsigned long long>(trace->execution_ordinal),
            static_cast<unsigned long long>(ctx->replay_kernels_submitted),
            static_cast<unsigned long long>(ctx->replay_waits_completed),
            static_cast<unsigned long long>(ctx->replay_outputs_validated),
            static_cast<unsigned long long>(trace->run_objects_created),
            static_cast<unsigned long long>(trace->run_wait_calls_returned),
            static_cast<unsigned long long>(trace->run_wait_completions));
    };

    if (kernel_name.empty() || kernel_name.rfind("MLIR_AIE", 0) != 0) {
        throw std::runtime_error("M9216 kernel name/selection is not the expected MLIR_AIE binding");
    }
    log_snapshot("XDNA_CAPTURE_REPLAY M9216 KERNEL RECREATE BEFORE", ctx->q4_m9216_kernel.get(), false, 0, 0, 0, 0, "NOT_TESTED");
    ctx->q4_m9216_kernel.reset();
    ctx->q4_m9216_kernel = std::make_unique<xrt::kernel>(*ctx->q4_m9216_hw_context, kernel_name);
    const int new_group_instr = ctx->q4_m9216_kernel->group_id(1);
    const int new_group_q4 = ctx->q4_m9216_kernel->group_id(3);
    const int new_group_q8 = ctx->q4_m9216_kernel->group_id(4);
    const int new_group_out = ctx->q4_m9216_kernel->group_id(5);
    const bool name_same = ctx->q4_m9216_kernel->get_name() == kernel_name;
    // Diagnostic within one live hw_context; this is not an XRT numeric-stability contract.
    const bool kernel_groups_same = old_group_instr == new_group_instr && old_group_q4 == new_group_q4 &&
        old_group_q8 == new_group_q8 && old_group_out == new_group_out;
    const char * kernel_group_relation = kernel_groups_same ? "SAME_AS_OLD_KERNEL" :
        "NEW_KERNEL_GROUP_IDS_CHANGED_WITHIN_SAME_CONTEXT";

    const bool device_same = ctx->device_context->xrt_device.get() == device_id;
    const bool xclbin_same = ctx->q4_m9216_xclbin.get() == xclbin_id;
    const bool hw_context_same = ctx->q4_m9216_hw_context.get() == hw_context_id;
    const bool bo_bundle_same = ctx->q4_m9216_bo_instr.get() == bo_instr_id &&
        ctx->q4_m9216_bo_q4.get() == bo_q4_id && ctx->q4_m9216_bo_q8.get() == bo_q8_id &&
        ctx->q4_m9216_bo_out.get() == bo_out_id;
    const bool maps_same = ctx->q4_m9216_buf_instr == map_instr && ctx->q4_m9216_buf_q4 == map_q4 &&
        ctx->q4_m9216_buf_q8 == map_q8 && ctx->q4_m9216_buf_out == map_out;
    const bool capacities_same = bo_bundle_same && ctx->q4_m9216_bo_instr->size() == instr_capacity &&
        ctx->q4_m9216_bo_q4->size() == q4_capacity && ctx->q4_m9216_bo_q8->size() == q8_capacity &&
        ctx->q4_m9216_bo_out->size() == out_capacity;
    const bool resources_same = device_same && xclbin_same && hw_context_same && name_same && bo_bundle_same &&
        maps_same && capacities_same;
    const bool checks_passed = resources_same && kernel_groups_same;
    if (checks_passed) {
        trace->m9216_kernel_generation = 1;
        trace->m9216_kernel_recreate_count = 1;
    }
    log_snapshot("XDNA_CAPTURE_REPLAY M9216 KERNEL RECREATE AFTER", ctx->q4_m9216_kernel.get(), true,
        new_group_instr, new_group_q4, new_group_q8, new_group_out, kernel_group_relation);
    std::fprintf(stderr,
        "XDNA_CAPTURE_REPLAY M9216 KERNEL RECREATE RELATIONS device=%s xclbin=%s hw_context=%s kernel_wrapper_address=%s "
        "kernel_name=%s BO_bundle=%s maps=%s capacities=%s kernel_group_relation=%s\n",
        device_same ? "SAME" : "CHANGED", xclbin_same ? "SAME" : "CHANGED",
        hw_context_same ? "SAME" : "CHANGED",
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_kernel.get(), old_kernel_id),
        name_same ? "SAME" : "CHANGED", bo_bundle_same ? "SAME" : "CHANGED",
        maps_same ? "SAME" : "CHANGED", capacities_same ? "SAME" : "CHANGED", kernel_group_relation);
    if (!kernel_groups_same) throw std::runtime_error("NEW_KERNEL_GROUP_IDS_CHANGED_WITHIN_SAME_CONTEXT");
    if (!resources_same) {
        throw std::runtime_error("M9216 kernel recreation changed a forbidden resource or invalidated retained BO invariants");
    }
}

// Caller precondition: every run associated with the old context has completed and
// been released. `live_run` is supplied when the caller has a local optional owner;
// the sole live-owner exception is the explicit frozen diagnostic control.
static void ggml_backend_xdna_replay_recreate_m9216_hw_context(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_replay_runtime_trace * trace, bool after_id7_exact = false, bool after_id8_exact = false, bool after_full_recreate_no_execution = false, bool after_id7_wait_no_readback = false, const std::optional<xrt::run> * live_run = nullptr, bool allow_live_run_control = false) {
    if (allow_live_run_control != (live_run != nullptr && live_run->has_value()))
        throw std::runtime_error("M9216 context recreation requires zero live associated runs");
    const bool after_exact_submission = after_id7_exact || after_id8_exact || after_full_recreate_no_execution || after_id7_wait_no_readback;
    if (!ctx || !trace || !ctx->device_context || !ctx->device_context->xrt_device || !ctx->q4_m9216_xclbin ||
        !ctx->q4_m9216_hw_context || !ctx->q4_m9216_kernel || !ctx->q4_m9216_available || !ctx->q4_m9216_bos_initialized ||
        !ctx->q4_m9216_bo_instr || !ctx->q4_m9216_bo_q4 || !ctx->q4_m9216_bo_q8 || !ctx->q4_m9216_bo_out ||
        !ctx->q4_m9216_buf_instr || !ctx->q4_m9216_buf_q4 || !ctx->q4_m9216_buf_q8 || !ctx->q4_m9216_buf_out ||
        trace->m9216_hw_context_generation != (after_exact_submission ? 1u : 0u) || trace->m9216_hw_context_recreate_count != (after_exact_submission ? 1u : 0u) ||
        trace->m9216_kernel_generation != (after_exact_submission ? 1u : 0u) || trace->m9216_kernel_recreate_count != (after_exact_submission ? 1u : 0u) ||
        trace->m9216_bo_bundle_generation != (after_exact_submission ? 1u : 0u) || trace->m9216_bo_bundle_recreate_count != (after_exact_submission ? 1u : 0u) ||
        (after_id7_exact && (!ctx->replay_post_recreate_context_recreate_id8_mode || trace->experiment_submission_ordinal != 1 ||
            trace->context_generation_submission_ordinal != 1 || trace->generation1_submissions != 1 ||
            trace->generation1_wait_completions != 1 || trace->generation1_output_syncs != 1 || trace->generation1_exact_outputs != 1)) ||
        (after_id8_exact && (!ctx->replay_post_id8_context_recreate_only_mode || trace->experiment_submission_ordinal != 1 ||
            trace->context_generation_submission_ordinal != 1 || trace->generation1_submissions != 1 ||
            trace->generation1_wait_completions != 1 || trace->generation1_output_syncs != 1 || trace->generation1_exact_outputs != 1)) ||
        (after_id7_wait_no_readback && (ctx->replay_records_total != 11 || ctx->replay_historical_records != 11 || trace->execution_ordinal != 11 ||
            ctx->replay_kernels_submitted != 11 || ctx->replay_waits_completed != 11 || ctx->replay_outputs_validated != 10 ||
            trace->experiment_submission_ordinal != 1 || trace->context_generation_submission_ordinal != 1 ||
            trace->generation1_execution_ordinal != 1 || trace->generation1_submissions != 1 || trace->generation1_wait_completions != 1 ||
            trace->generation1_output_sentinel_syncs != 0 || trace->generation1_output_syncs != 0 || trace->generation1_output_from_device_syncs != 0 ||
            trace->run_objects_created != 11 || trace->run_wait_calls_returned != 11 || trace->run_wait_completions != 11 ||
            ctx->replay_transition_m4096_submitted != 1 || ctx->replay_transition_m4096_wait_completed != 1 ||
            ctx->replay_transition_m4096_output_validated != 0 || ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)) ||
        (after_full_recreate_no_execution && (trace->execution_ordinal != 10 || ctx->replay_kernels_submitted != 10 ||
            ctx->replay_waits_completed != 10 || ctx->replay_outputs_validated != 10 || trace->experiment_submission_ordinal != 0 ||
            trace->context_generation_submission_ordinal != 0 || trace->generation1_execution_ordinal != 0 || trace->generation1_submissions != 0 ||
            trace->generation1_wait_completions != 0 || trace->generation1_output_syncs != 0 || trace->generation1_exact_outputs != 0 ||
            ctx->replay_post_recreate_control_submitted != 0 || ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)) ||
        (static_cast<unsigned>(after_id7_exact) + static_cast<unsigned>(after_id8_exact) + static_cast<unsigned>(after_full_recreate_no_execution) + static_cast<unsigned>(after_id7_wait_no_readback) > 1)) {
        throw std::runtime_error("M9216 hw_context recreation precondition/count violation");
    }
    if (ctx->replay_prefix_planned != 10 || ctx->replay_prefix_completed != 10 ||
        ctx->replay_prefix_outputs_validated != 10 || ctx->exact_history_replay_failed ||
        trace->m9216_execution_ordinal != (after_id8_exact ? 8u : 7u) || trace->m9216_submissions != (after_id8_exact ? 8u : 7u) ||
        trace->m9216_wait_completions != (after_id8_exact ? 8u : 7u) || trace->m9216_output_syncs != (after_id8_exact ? 8u : 7u) ||
        (after_id7_exact && (trace->execution_ordinal != 11 || ctx->replay_kernels_submitted != 11 ||
            ctx->replay_waits_completed != 11 || ctx->replay_outputs_validated != 11 || ctx->exact_history_replay_failed ||
            trace->experiment_submission_ordinal != 1 || trace->context_generation_submission_ordinal != 1)) ||
        (after_id8_exact && (trace->execution_ordinal != 11 || ctx->replay_kernels_submitted != 11 ||
            ctx->replay_waits_completed != 11 || ctx->replay_outputs_validated != 11 || ctx->exact_history_replay_failed ||
            ctx->replay_post_recreate_control_submitted != 1 || ctx->replay_post_recreate_control_wait_completed != 1 ||
            ctx->replay_post_recreate_control_output_validated != 1 || trace->experiment_submission_ordinal != 1 ||
            trace->context_generation_submission_ordinal != 1))) {
        throw std::runtime_error("M9216 hw_context recreation boundary invariant failed");
    }

    auto & device = *ctx->device_context->xrt_device;
    const void * device_id = ctx->device_context->xrt_device.get();
    const void * xclbin_id = ctx->q4_m9216_xclbin.get();
    const void * old_context_id = ctx->q4_m9216_hw_context.get();
    const void * old_kernel_id = ctx->q4_m9216_kernel.get();
    const void * instr_id = ctx->q4_m9216_bo_instr.get(), * q4_id = ctx->q4_m9216_bo_q4.get();
    const void * q8_id = ctx->q4_m9216_bo_q8.get(), * out_id = ctx->q4_m9216_bo_out.get();
    const void * instr_map = ctx->q4_m9216_buf_instr, * q4_map = ctx->q4_m9216_buf_q4;
    const void * q8_map = ctx->q4_m9216_buf_q8, * out_map = ctx->q4_m9216_buf_out;
    const size_t instr_capacity = ctx->q4_m9216_bo_instr->size(), q4_capacity = ctx->q4_m9216_bo_q4->size();
    const size_t q8_capacity = ctx->q4_m9216_bo_q8->size(), out_capacity = ctx->q4_m9216_bo_out->size();
    const auto xclbin_uuid_value = ctx->q4_m9216_xclbin->get_uuid();
    const std::string xclbin_uuid = xclbin_uuid_value.to_string();
    const std::string kernel_name = ctx->q4_m9216_kernel->get_name();
    const int old_group_instr = ctx->q4_m9216_kernel->group_id(1), old_group_q4 = ctx->q4_m9216_kernel->group_id(3);
    const int old_group_q8 = ctx->q4_m9216_kernel->group_id(4), old_group_out = ctx->q4_m9216_kernel->group_id(5);
    const auto bo_group_instr = ctx->q4_m9216_bo_instr->get_memory_group();
    const auto bo_group_q4 = ctx->q4_m9216_bo_q4->get_memory_group();
    const auto bo_group_q8 = ctx->q4_m9216_bo_q8->get_memory_group();
    const auto bo_group_out = ctx->q4_m9216_bo_out->get_memory_group();
    if (kernel_name.empty() || kernel_name.rfind("MLIR_AIE", 0) != 0 ||
        ctx->q4_m9216_hw_context->get_xclbin_uuid().to_string() != xclbin_uuid) {
        throw std::runtime_error("M9216 context/kernel XCLBIN identity is not the frozen binding");
    }

    auto log_snapshot = [&](const char * phase, const int * new_groups) {
        std::fprintf(stderr,
            "%s\n device_id=%p xclbin_id=%p xclbin_uuid=%s hw_context_id=%p kernel_id=%p kernel_name=%s\n",
            phase, static_cast<const void *>(ctx->device_context->xrt_device.get()),
            static_cast<const void *>(ctx->q4_m9216_xclbin.get()), xclbin_uuid.c_str(),
            static_cast<const void *>(ctx->q4_m9216_hw_context.get()),
            static_cast<const void *>(ctx->q4_m9216_kernel.get()), kernel_name.c_str());
        std::fprintf(stderr,
            "instr_bo_id=%p instr_map=%p instr_capacity=%zu bo_memory_group=%u "
            "q4_bo_id=%p q4_map=%p q4_capacity=%zu bo_memory_group=%u "
            "q8_bo_id=%p q8_map=%p q8_capacity=%zu bo_memory_group=%u "
            "out_bo_id=%p out_map=%p out_capacity=%zu bo_memory_group=%u\n",
            static_cast<const void *>(ctx->q4_m9216_bo_instr.get()), ctx->q4_m9216_buf_instr, instr_capacity,
            static_cast<unsigned>(bo_group_instr), static_cast<const void *>(ctx->q4_m9216_bo_q4.get()),
            ctx->q4_m9216_buf_q4, q4_capacity, static_cast<unsigned>(bo_group_q4),
            static_cast<const void *>(ctx->q4_m9216_bo_q8.get()), ctx->q4_m9216_buf_q8, q8_capacity,
            static_cast<unsigned>(bo_group_q8), static_cast<const void *>(ctx->q4_m9216_bo_out.get()),
            static_cast<const void *>(ctx->q4_m9216_buf_out), out_capacity, static_cast<unsigned>(bo_group_out));
        std::fprintf(stderr, "old_kernel_group_ids=%d,%d,%d,%d",
            old_group_instr, old_group_q4, old_group_q8, old_group_out);
        if (new_groups) std::fprintf(stderr, " new_kernel_group_ids=%d,%d,%d,%d",
            new_groups[0], new_groups[1], new_groups[2], new_groups[3]);
        std::fprintf(stderr,
            "\ncontext_generation=%llu context_recreate_count=%llu kernel_generation=%llu kernel_recreate_count=%llu bo_generation=%llu\n",
            static_cast<unsigned long long>(trace->m9216_hw_context_generation),
            static_cast<unsigned long long>(trace->m9216_hw_context_recreate_count),
            static_cast<unsigned long long>(trace->m9216_kernel_generation),
            static_cast<unsigned long long>(trace->m9216_kernel_recreate_count),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation));
        std::fprintf(stderr,
            "cumulative_m9216_ordinal=%llu submissions=%llu wait_completions=%llu output_syncs=%llu "
            "cumulative_replay_execution=%llu replay_submissions=%llu replay_waits=%llu replay_outputs=%llu experiment_submission_ordinal=%llu context_generation_submission_ordinal=%llu\n",
            static_cast<unsigned long long>(trace->m9216_execution_ordinal),
            static_cast<unsigned long long>(trace->m9216_submissions),
            static_cast<unsigned long long>(trace->m9216_wait_completions),
            static_cast<unsigned long long>(trace->m9216_output_syncs),
            static_cast<unsigned long long>(trace->execution_ordinal),
            static_cast<unsigned long long>(ctx->replay_kernels_submitted),
            static_cast<unsigned long long>(ctx->replay_waits_completed),
            static_cast<unsigned long long>(ctx->replay_outputs_validated),
            static_cast<unsigned long long>(trace->experiment_submission_ordinal),
            static_cast<unsigned long long>(trace->context_generation_submission_ordinal));
    };
    const auto run_objects_before = trace->run_objects_created, run_waits_before = trace->run_wait_calls_returned;
    const auto run_completions_before = trace->run_wait_completions;
    const auto input_syncs_before = trace->m9216_input_syncs, output_syncs_before = trace->m9216_output_syncs;
    const auto instruction_syncs_before = trace->m9216_instruction_syncs;
    const auto generation1_syncs_before = trace->generation1_instruction_init_syncs + trace->generation1_instruction_stage_syncs + trace->generation1_q4_syncs + trace->generation1_q8_syncs + trace->generation1_output_sentinel_syncs + trace->generation1_output_syncs + trace->generation1_output_from_device_syncs;
    const auto sentinel_syncs_before = trace->m9216_output_sentinel_syncs, init_syncs_before = trace->m9216_initialization_syncs;
    log_snapshot(after_id8_exact ? "TRANSITION_STAGE=POST_ID8_CONTEXT_RECREATE_BEGIN" : (after_id7_exact ? "TRANSITION_STAGE=POST_ID7_CONTEXT_RECREATE_BEGIN" : (after_id7_wait_no_readback ? "TRANSITION_STAGE=POST_EXECUTE_WAIT_NO_READBACK_CONTEXT_RECREATE_BEGIN" : "XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE BEFORE")), nullptr);
    std::fflush(stderr);

    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE RELEASE_OLD_KERNEL kernel_id=%p\n", old_kernel_id);
    std::fflush(stderr);
    ctx->q4_m9216_kernel.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE OLD_KERNEL_RELEASED\n");
    std::fflush(stderr);
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE RELEASE_OLD_CONTEXT context_id=%p\n", old_context_id);
    std::fflush(stderr);
    ctx->q4_m9216_hw_context.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE OLD_CONTEXT_RELEASED\n");
    std::fflush(stderr);
    // XCLBIN registration is device-scoped; reuse the UUID value captured before teardown.
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE BEFORE_NEW_CONTEXT_CONSTRUCT\n");
    std::fflush(stderr);
    try {
        ctx->q4_m9216_hw_context = std::make_shared<xrt::hw_context>(device, xclbin_uuid_value);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE NEW_CONTEXT_CONSTRUCT_EXCEPTION what=%s\n", e.what());
        std::fflush(stderr);
        throw;
    }
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE AFTER_NEW_CONTEXT_CONSTRUCT hw_context_id=%p\n", static_cast<const void *>(ctx->q4_m9216_hw_context.get()));
    std::fflush(stderr);
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE BEFORE_NEW_KERNEL_CONSTRUCT\n");
    std::fflush(stderr);
    try {
        ctx->q4_m9216_kernel = std::make_unique<xrt::kernel>(*ctx->q4_m9216_hw_context, kernel_name);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE NEW_KERNEL_CONSTRUCT_EXCEPTION what=%s\n", e.what());
        std::fflush(stderr);
        throw;
    }
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE AFTER_NEW_KERNEL_CONSTRUCT kernel_id=%p\n", static_cast<const void *>(ctx->q4_m9216_kernel.get()));
    std::fflush(stderr);
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE BEFORE_POST_BOUNDARY_INSPECTION\n");
    std::fflush(stderr);
    int new_groups[4] = {};
    bool device_same = false, xclbin_same = false, context_xclbin_same = false, name_same = false;
    bool bos_same = false, maps_same = false, capacities_same = false, context_changed = false, kernel_changed = false;
    bool operation_counters_unchanged = false, checks_passed = false;
    try {
        new_groups[0] = ctx->q4_m9216_kernel->group_id(1); new_groups[1] = ctx->q4_m9216_kernel->group_id(3);
        new_groups[2] = ctx->q4_m9216_kernel->group_id(4); new_groups[3] = ctx->q4_m9216_kernel->group_id(5);
        device_same = ctx->device_context->xrt_device.get() == device_id;
        xclbin_same = ctx->q4_m9216_xclbin.get() == xclbin_id && ctx->q4_m9216_xclbin->get_uuid().to_string() == xclbin_uuid;
        context_xclbin_same = ctx->q4_m9216_hw_context->get_xclbin_uuid().to_string() == xclbin_uuid;
        name_same = ctx->q4_m9216_kernel->get_name() == kernel_name;
        bos_same = ctx->q4_m9216_bo_instr.get() == instr_id && ctx->q4_m9216_bo_q4.get() == q4_id &&
            ctx->q4_m9216_bo_q8.get() == q8_id && ctx->q4_m9216_bo_out.get() == out_id;
        maps_same = ctx->q4_m9216_buf_instr == instr_map && ctx->q4_m9216_buf_q4 == q4_map &&
            ctx->q4_m9216_buf_q8 == q8_map && ctx->q4_m9216_buf_out == out_map;
        capacities_same = bos_same && ctx->q4_m9216_bo_instr->size() == instr_capacity &&
            ctx->q4_m9216_bo_q4->size() == q4_capacity && ctx->q4_m9216_bo_q8->size() == q8_capacity &&
            ctx->q4_m9216_bo_out->size() == out_capacity;
        context_changed = ctx->q4_m9216_hw_context.get() != old_context_id;
        kernel_changed = ctx->q4_m9216_kernel.get() != old_kernel_id;
        operation_counters_unchanged = trace->run_objects_created == run_objects_before &&
            trace->run_wait_calls_returned == run_waits_before && trace->run_wait_completions == run_completions_before &&
            trace->m9216_input_syncs == input_syncs_before && trace->m9216_output_syncs == output_syncs_before &&
            trace->m9216_output_sentinel_syncs == sentinel_syncs_before && trace->m9216_initialization_syncs == init_syncs_before && trace->m9216_instruction_syncs == instruction_syncs_before &&
            trace->generation1_instruction_init_syncs + trace->generation1_instruction_stage_syncs + trace->generation1_q4_syncs + trace->generation1_q8_syncs + trace->generation1_output_sentinel_syncs + trace->generation1_output_syncs + trace->generation1_output_from_device_syncs == generation1_syncs_before;
        checks_passed = device_same && xclbin_same && context_xclbin_same && name_same && context_changed && kernel_changed && bos_same &&
            maps_same && capacities_same && ctx->q4_m9216_bos_initialized && operation_counters_unchanged;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE POST_BOUNDARY_INSPECTION_EXCEPTION what=%s\n", e.what());
        std::fflush(stderr);
        throw;
    }
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE RELATIONS device=%s xclbin=%s context_xclbin=%s kernel_name=%s context_identity_changed=%s kernel_identity_changed=%s "
        "kernel_abi=SAME_XCLBIN_UUID_AND_NAME kernel_wrapper=%s BO_bundle=%s maps=%s capacities=%s static_connectivity=ARG1_MEM_TOPOLOGY_1_SRAM_ARG3_4_5_MEM_TOPOLOGY_0_HOST "
        "xrt_delta=hw_context_construct:1,kernel_construct:1,BO_construct:0,BO_map:0,BO_sync:0,kernel_submit:0,run_wait:0,device_reopen:0,xclbin_load_register:0 "
        "identity_capacity_group_queries=uuid:4,kernel_name:2,group_id:8,bo_memory_group:4,bo_size:8 old_context_id=%p context_lifecycle=DESTROYED_AND_RECONSTRUCTED\n",
        device_same ? "SAME" : "CHANGED", xclbin_same ? "SAME" : "CHANGED",
        context_xclbin_same ? "SAME" : "CHANGED", name_same ? "SAME" : "CHANGED",
        context_changed ? "YES" : "NO", kernel_changed ? "YES" : "NO",
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_kernel.get(), old_kernel_id),
        bos_same ? "SAME" : "CHANGED", maps_same ? "SAME" : "CHANGED", capacities_same ? "SAME" : "CHANGED", old_context_id);
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY XDNA_RECREATE_GUARD RECREATE AFTER_POST_BOUNDARY_INSPECTION result=%s\n", checks_passed ? "PASS" : "FAIL");
    std::fflush(stderr);
    if (!checks_passed) throw std::runtime_error("M9216 hw_context recreation changed a forbidden resource or XCLBIN/kernel binding");
    ++trace->m9216_hw_context_generation; ++trace->m9216_hw_context_recreate_count;
    ++trace->m9216_kernel_generation; ++trace->m9216_kernel_recreate_count;
    if (after_id7_exact || after_id7_wait_no_readback) {
        trace->post_id7_device = device_id; trace->post_id7_xclbin = xclbin_id;
        trace->post_id7_context = old_context_id; trace->post_id7_kernel = old_kernel_id;
        trace->post_id7_bo_instr = instr_id; trace->post_id7_bo_q4 = q4_id;
        trace->post_id7_bo_q8 = q8_id; trace->post_id7_bo_out = out_id;
        trace->post_id7_map_instr = instr_map; trace->post_id7_map_q4 = q4_map;
        trace->post_id7_map_q8 = q8_map; trace->post_id7_map_out = out_map;
        trace->post_id7_instr_capacity = instr_capacity; trace->post_id7_q4_capacity = q4_capacity;
        trace->post_id7_q8_capacity = q8_capacity; trace->post_id7_out_capacity = out_capacity;
        trace->post_id7_xclbin_uuid = xclbin_uuid; trace->post_id7_kernel_name = kernel_name;
        trace->context_generation_submission_ordinal = 0;
    }
    log_snapshot(after_id8_exact ? "TRANSITION_STAGE=POST_ID8_CONTEXT_RECREATE_END" : (after_id7_exact ? "TRANSITION_STAGE=POST_ID7_CONTEXT_RECREATE_END" : (after_id7_wait_no_readback ? "TRANSITION_STAGE=POST_EXECUTE_WAIT_NO_READBACK_CONTEXT_RECREATE_END" : "XDNA_CAPTURE_REPLAY M9216 HW_CONTEXT RECREATE AFTER")), new_groups);
    std::fflush(stderr);
    (void)old_group_instr; (void)old_group_q4; (void)old_group_q8; (void)old_group_out;
}

static void ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_replay_runtime_trace * trace) {
    if (!ctx || !trace || !ctx->device_context || !ctx->device_context->xrt_device || !ctx->q4_m9216_xclbin ||
        !ctx->q4_m9216_hw_context || !ctx->q4_m9216_kernel || !ctx->q4_m9216_available || !ctx->q4_m9216_bos_initialized ||
        !ctx->q4_m9216_bo_instr || !ctx->q4_m9216_bo_q4 || !ctx->q4_m9216_bo_q8 || !ctx->q4_m9216_bo_out ||
        !ctx->q4_m9216_buf_instr || !ctx->q4_m9216_buf_q4 || !ctx->q4_m9216_buf_q8 || !ctx->q4_m9216_buf_out ||
        trace->m9216_hw_context_generation != 0 || trace->m9216_hw_context_recreate_count != 0 ||
        trace->m9216_kernel_generation != 0 || trace->m9216_kernel_recreate_count != 0 ||
        trace->m9216_bo_bundle_generation != 0 || trace->m9216_bo_bundle_recreate_count != 0) {
        throw std::runtime_error("M9216 full runtime bundle recreation precondition/count violation");
    }
    if (ctx->replay_prefix_planned != 10 || ctx->replay_prefix_completed != 10 ||
        ctx->replay_prefix_outputs_validated != 10 || ctx->exact_history_replay_failed ||
        trace->m9216_execution_ordinal != 7 || trace->m9216_submissions != 7 ||
        trace->m9216_wait_completions != 7 || trace->m9216_output_syncs != 7) {
        throw std::runtime_error("M9216 full runtime bundle recreation boundary invariant failed");
    }

    auto & device = *ctx->device_context->xrt_device;
    const void * device_id = ctx->device_context->xrt_device.get();
    const void * xclbin_id = ctx->q4_m9216_xclbin.get();
    const void * old_context_id = ctx->q4_m9216_hw_context.get();
    const void * old_kernel_id = ctx->q4_m9216_kernel.get();
    const void * old_bo_instr = ctx->q4_m9216_bo_instr.get(), * old_bo_q4 = ctx->q4_m9216_bo_q4.get();
    const void * old_bo_q8 = ctx->q4_m9216_bo_q8.get(), * old_bo_out = ctx->q4_m9216_bo_out.get();
    const void * old_map_instr = ctx->q4_m9216_buf_instr, * old_map_q4 = ctx->q4_m9216_buf_q4;
    const void * old_map_q8 = ctx->q4_m9216_buf_q8, * old_map_out = ctx->q4_m9216_buf_out;
    constexpr size_t instruction_capacity = 189108, q4_capacity = 14008320;
    constexpr size_t q8_capacity = 2920, output_capacity = 36864;
    const std::string xclbin_uuid = ctx->q4_m9216_xclbin->get_uuid().to_string();
    const std::string kernel_name = ctx->q4_m9216_kernel->get_name();
    const int old_group_instr = ctx->q4_m9216_kernel->group_id(1), old_group_q4 = ctx->q4_m9216_kernel->group_id(3);
    const int old_group_q8 = ctx->q4_m9216_kernel->group_id(4), old_group_out = ctx->q4_m9216_kernel->group_id(5);
    const unsigned old_bo_group_instr = static_cast<unsigned>(ctx->q4_m9216_bo_instr->get_memory_group());
    const unsigned old_bo_group_q4 = static_cast<unsigned>(ctx->q4_m9216_bo_q4->get_memory_group());
    const unsigned old_bo_group_q8 = static_cast<unsigned>(ctx->q4_m9216_bo_q8->get_memory_group());
    const unsigned old_bo_group_out = static_cast<unsigned>(ctx->q4_m9216_bo_out->get_memory_group());
    const size_t instruction_bytes = ctx->q4_m9216_instructions.size() * sizeof(uint32_t);
    if (kernel_name.empty() || instruction_bytes != instruction_capacity ||
        ctx->q4_m9216_bo_instr->size() != instruction_capacity || ctx->q4_m9216_bo_q4->size() != q4_capacity ||
        ctx->q4_m9216_bo_q8->size() != q8_capacity || ctx->q4_m9216_bo_out->size() != output_capacity ||
        ggml_backend_xdna_capture_hash(ctx->q4_m9216_instructions.data(), instruction_bytes) !=
            "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" ||
        ctx->q4_m9216_hw_context->get_xclbin_uuid().to_string() != xclbin_uuid) {
        throw std::runtime_error("M9216 full runtime bundle frozen input/context invariant failed");
    }

    auto log_snapshot = [&](const char * phase, bool after) {
        std::fprintf(stderr,
            "%s\ndevice_id=%p xclbin_id=%p xclbin_uuid=%s hw_context_id=%p kernel_id=%p kernel_name=%s\n",
            phase, static_cast<const void *>(ctx->device_context->xrt_device.get()),
            static_cast<const void *>(ctx->q4_m9216_xclbin.get()), xclbin_uuid.c_str(),
            static_cast<const void *>(ctx->q4_m9216_hw_context.get()),
            static_cast<const void *>(ctx->q4_m9216_kernel.get()), kernel_name.c_str());
        std::fprintf(stderr, "old_kernel_group_ids=%d,%d,%d,%d\n",
            old_group_instr, old_group_q4, old_group_q8, old_group_out);
        if (after) {
            std::fprintf(stderr, "new_kernel_group_ids=%u,%u,%u,%u\n", ctx->q4_m9216_group_instr,
                ctx->q4_m9216_group_q4, ctx->q4_m9216_group_q8, ctx->q4_m9216_group_out);
        } else {
            std::fprintf(stderr, "new_kernel_group_ids=NOT_CREATED\n");
        }
        std::fprintf(stderr,
            "old_instr_bo_id=%p old_instr_map=%p old_instr_capacity=%zu old_q4_bo_id=%p old_q4_map=%p old_q4_capacity=%zu "
            "old_q8_bo_id=%p old_q8_map=%p old_q8_capacity=%zu old_out_bo_id=%p old_out_map=%p old_out_capacity=%zu "
            "old_bo_memory_groups=%u,%u,%u,%u\n", old_bo_instr, old_map_instr, instruction_capacity,
            old_bo_q4, old_map_q4, q4_capacity, old_bo_q8, old_map_q8, q8_capacity, old_bo_out, old_map_out, output_capacity,
            old_bo_group_instr, old_bo_group_q4, old_bo_group_q8, old_bo_group_out);
        if (after) {
            std::fprintf(stderr,
                "new_instr_bo_id=%p new_instr_map=%p new_instr_capacity=%zu new_q4_bo_id=%p new_q4_map=%p new_q4_capacity=%zu "
                "new_q8_bo_id=%p new_q8_map=%p new_q8_capacity=%zu new_out_bo_id=%p new_out_map=%p new_out_capacity=%zu "
                "new_bo_memory_groups=%u,%u,%u,%u\n",
                static_cast<const void *>(ctx->q4_m9216_bo_instr.get()), ctx->q4_m9216_buf_instr, ctx->q4_m9216_bo_instr->size(),
                static_cast<const void *>(ctx->q4_m9216_bo_q4.get()), ctx->q4_m9216_buf_q4, ctx->q4_m9216_bo_q4->size(),
                static_cast<const void *>(ctx->q4_m9216_bo_q8.get()), ctx->q4_m9216_buf_q8, ctx->q4_m9216_bo_q8->size(),
                static_cast<const void *>(ctx->q4_m9216_bo_out.get()), static_cast<const void *>(ctx->q4_m9216_buf_out), ctx->q4_m9216_bo_out->size(),
                static_cast<unsigned>(ctx->q4_m9216_bo_instr->get_memory_group()),
                static_cast<unsigned>(ctx->q4_m9216_bo_q4->get_memory_group()),
                static_cast<unsigned>(ctx->q4_m9216_bo_q8->get_memory_group()),
                static_cast<unsigned>(ctx->q4_m9216_bo_out->get_memory_group()));
        } else {
            std::fprintf(stderr, "new_BO_bundle=NOT_CREATED new_maps=NOT_CREATED\n");
        }
        std::fprintf(stderr,
            "context_generation=%llu context_recreate_count=%llu kernel_generation=%llu kernel_recreate_count=%llu "
            "bo_generation=%llu bo_recreate_count=%llu bo_generation_submissions=%llu bo_generation_wait_completions=%llu\n",
            static_cast<unsigned long long>(trace->m9216_hw_context_generation),
            static_cast<unsigned long long>(trace->m9216_hw_context_recreate_count),
            static_cast<unsigned long long>(trace->m9216_kernel_generation),
            static_cast<unsigned long long>(trace->m9216_kernel_recreate_count),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_recreate_count),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation_submissions),
            static_cast<unsigned long long>(trace->m9216_bo_bundle_generation_wait_completions));
        std::fprintf(stderr,
            "cumulative_replay_execution=%llu history_completed=%llu history_outputs_validated=%llu replay_submissions=%llu replay_waits=%llu replay_outputs=%llu\n",
            static_cast<unsigned long long>(trace->execution_ordinal),
            static_cast<unsigned long long>(ctx->replay_prefix_completed),
            static_cast<unsigned long long>(ctx->replay_prefix_outputs_validated),
            static_cast<unsigned long long>(ctx->replay_kernels_submitted),
            static_cast<unsigned long long>(ctx->replay_waits_completed),
            static_cast<unsigned long long>(ctx->replay_outputs_validated));
        std::fprintf(stderr,
            "cumulative_m9216_ordinal=%llu submissions=%llu wait_calls_returned=%llu wait_completions=%llu "
            "initialization_syncs=%llu input_syncs=%llu output_sentinel_syncs=%llu output_syncs=%llu "
            "run_objects=%llu run_wait_calls_returned=%llu run_wait_completions=%llu\n",
            static_cast<unsigned long long>(trace->m9216_execution_ordinal),
            static_cast<unsigned long long>(trace->m9216_submissions),
            static_cast<unsigned long long>(trace->m9216_wait_calls_returned),
            static_cast<unsigned long long>(trace->m9216_wait_completions),
            static_cast<unsigned long long>(trace->m9216_initialization_syncs),
            static_cast<unsigned long long>(trace->m9216_input_syncs),
            static_cast<unsigned long long>(trace->m9216_output_sentinel_syncs),
            static_cast<unsigned long long>(trace->m9216_output_syncs),
            static_cast<unsigned long long>(trace->run_objects_created),
            static_cast<unsigned long long>(trace->run_wait_calls_returned),
            static_cast<unsigned long long>(trace->run_wait_completions));
    };

    log_snapshot("XDNA_CAPTURE_REPLAY M9216 FULL RUNTIME BUNDLE RECREATE BEFORE", false);
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_KERNEL_BEGIN kernel_id=%p\n", old_kernel_id);
    ctx->q4_m9216_kernel.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_KERNEL_END\n");
    ctx->q4_m9216_buf_instr = ctx->q4_m9216_buf_q4 = ctx->q4_m9216_buf_q8 = nullptr;
    ctx->q4_m9216_buf_out = nullptr;
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE INVALIDATE_OLD_MAPS\n");
    ctx->q4_m9216_bos_initialized = false;
    ctx->q4_m9216_instruction_count = 0;
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_BO_BUNDLE\n");
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_INSTRUCTION_BO_BEGIN bo_id=%p\n", old_bo_instr);
    ctx->q4_m9216_bo_instr.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_INSTRUCTION_BO_END\n");
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_Q4_BO_BEGIN bo_id=%p\n", old_bo_q4);
    ctx->q4_m9216_bo_q4.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_Q4_BO_END\n");
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_Q8_BO_BEGIN bo_id=%p\n", old_bo_q8);
    ctx->q4_m9216_bo_q8.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_Q8_BO_END\n");
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_OUTPUT_BO_BEGIN bo_id=%p\n", old_bo_out);
    ctx->q4_m9216_bo_out.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_OUTPUT_BO_END\n");
    ctx->q4_m9216_group_instr = ctx->q4_m9216_group_q4 = ctx->q4_m9216_group_q8 = ctx->q4_m9216_group_out = 0;
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE OLD_BO_BUNDLE_RELEASED\n");
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_HW_CONTEXT_BEGIN context_id=%p\n", old_context_id);
    ctx->q4_m9216_hw_context.reset();
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELEASE_OLD_HW_CONTEXT_END\n");

    ctx->q4_m9216_hw_context = std::make_shared<xrt::hw_context>(device, ctx->q4_m9216_xclbin->get_uuid());
    ctx->q4_m9216_kernel = std::make_unique<xrt::kernel>(*ctx->q4_m9216_hw_context, kernel_name);
    if (!ggml_backend_xdna_replay_ensure_m9216_bos(ctx, trace, true)) {
        throw std::runtime_error("failed to create fresh M9216 BO bundle");
    }
    const bool device_same = ctx->device_context->xrt_device.get() == device_id;
    const bool xclbin_same = ctx->q4_m9216_xclbin.get() == xclbin_id &&
        ctx->q4_m9216_xclbin->get_uuid().to_string() == xclbin_uuid;
    const bool context_uuid_same = ctx->q4_m9216_hw_context &&
        ctx->q4_m9216_hw_context->get_xclbin_uuid().to_string() == xclbin_uuid;
    const bool kernel_name_same = ctx->q4_m9216_kernel && ctx->q4_m9216_kernel->get_name() == kernel_name;
    const bool instruction_sha_same = ctx->q4_m9216_buf_instr && instruction_bytes == instruction_capacity &&
        ggml_backend_xdna_capture_hash(ctx->q4_m9216_buf_instr, instruction_bytes) ==
            "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3";
    const bool capacities_same = ctx->q4_m9216_bo_instr && ctx->q4_m9216_bo_q4 && ctx->q4_m9216_bo_q8 && ctx->q4_m9216_bo_out &&
        ctx->q4_m9216_bo_instr->size() == instruction_capacity && ctx->q4_m9216_bo_q4->size() == q4_capacity &&
        ctx->q4_m9216_bo_q8->size() == q8_capacity && ctx->q4_m9216_bo_out->size() == output_capacity;
    const bool fresh_bundle_ready = ctx->q4_m9216_bos_initialized && ctx->q4_m9216_buf_q4 &&
        ctx->q4_m9216_buf_q8 && ctx->q4_m9216_buf_out && ctx->q4_m9216_hw_context && ctx->q4_m9216_kernel;
    const bool checks_passed = device_same && xclbin_same && context_uuid_same && kernel_name_same &&
        instruction_sha_same && capacities_same && fresh_bundle_ready;
    if (checks_passed) {
        trace->m9216_hw_context_generation = trace->m9216_hw_context_recreate_count = 1;
        trace->m9216_kernel_generation = trace->m9216_kernel_recreate_count = 1;
        trace->m9216_bo_bundle_generation = trace->m9216_bo_bundle_recreate_count = 1;
        trace->m9216_bo_bundle_generation_submissions = 0;
        trace->m9216_bo_bundle_generation_wait_completions = 0;
    }
    log_snapshot("XDNA_CAPTURE_REPLAY M9216 FULL RUNTIME BUNDLE RECREATE AFTER", true);
    std::fprintf(stderr,
        "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE RELATIONS device=%s xclbin=%s xclbin_uuid=%s context_uuid=%s kernel_name=%s "
        "instruction_sha=%s capacities=%s context_generation=0->%llu kernel_generation=0->%llu bo_generation=0->%llu "
        "static_connectivity=ARG1_MEM_TOPOLOGY_1_SRAM_ARG3_4_5_MEM_TOPOLOGY_0_HOST\n",
        device_same ? "SAME" : "CHANGED", xclbin_same ? "SAME" : "CHANGED",
        xclbin_uuid.c_str(), context_uuid_same ? "SAME" : "CHANGED", kernel_name_same ? "SAME" : "CHANGED",
        instruction_sha_same ? "PASS" : "FAIL", capacities_same ? "SAME" : "CHANGED",
        static_cast<unsigned long long>(trace->m9216_hw_context_generation),
        static_cast<unsigned long long>(trace->m9216_kernel_generation),
        static_cast<unsigned long long>(trace->m9216_bo_bundle_generation));
    std::fprintf(stderr,
        "XDNA_CAPTURE_REPLAY M9216 FULL BUNDLE ADDRESS RELATIONS pointer_reuse_is_not_generation_evidence context=%s kernel=%s "
        "bo_instr=%s bo_q4=%s bo_q8=%s bo_out=%s map_instr=%s map_q4=%s map_q8=%s map_out=%s\n",
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_hw_context.get(), old_context_id),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_kernel.get(), old_kernel_id),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_instr.get(), old_bo_instr),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_q4.get(), old_bo_q4),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_q8.get(), old_bo_q8),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_bo_out.get(), old_bo_out),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_instr, old_map_instr),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_q4, old_map_q4),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_q8, old_map_q8),
        ggml_backend_xdna_replay_identity_change(true, ctx->q4_m9216_buf_out, old_map_out));
    if (!checks_passed) throw std::runtime_error("M9216 full runtime bundle recreation invariant failed");
}

static void ggml_backend_xdna_replay_create_m4096_transition_bundle(
        ggml_backend_xdna_backend_context * ctx, const ggml_backend_xdna_batched_state & state,
        ggml_backend_xdna_replay_runtime_trace & trace, ggml_backend_xdna_replay_transition_bundle * bundle) {
    if (!ctx || !bundle || !ctx->device_context || !ctx->device_context->xrt_device ||
        !ctx->q4_m9216_hw_context || !ctx->q4_m9216_kernel || !state.available || state.rows != 4096 ||
        !state.xclbin || !state.hw_context || !state.kernel || !state.bos_initialized ||
        !state.bo_instr || !state.bo_q4 || !state.bo_q8 || !state.bo_out ||
        state.hw_context.get() == ctx->q4_m9216_hw_context.get() ||
        state.xclbin_path != ctx->q4_m9216_xclbin_path) {
        throw std::runtime_error("M4096 transition refuses stale or mismatched shape resources");
    }
    constexpr const char * instruction_sha = "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904";
    const size_t instruction_bytes = state.instructions.size() * sizeof(uint32_t);
    if (!instruction_bytes || ggml_backend_xdna_capture_hash(state.instructions.data(), instruction_bytes) != instruction_sha ||
        trace.m9216_hw_context_generation != 1 || trace.m9216_kernel_generation != 1 || trace.m9216_bo_bundle_generation != 1) {
        throw std::runtime_error("M4096 transition instruction/context generation mismatch");
    }
    bundle->hw_context = ctx->q4_m9216_hw_context.get();
    bundle->kernel = ctx->q4_m9216_kernel.get();
    auto & device = *ctx->device_context->xrt_device;
    bundle->group_instr = bundle->kernel->group_id(1);
    bundle->group_q4 = bundle->kernel->group_id(3);
    bundle->group_q8 = bundle->kernel->group_id(4);
    bundle->group_out = bundle->kernel->group_id(5);
    bundle->bo_instr = std::make_unique<xrt::bo>(device, instruction_bytes, XCL_BO_FLAGS_CACHEABLE, bundle->group_instr);
    bundle->bo_q4 = std::make_unique<xrt::bo>(device, 4096 * 1520, XRT_BO_FLAGS_HOST_ONLY, bundle->group_q4);
    bundle->bo_q8 = std::make_unique<xrt::bo>(device, 2920, XRT_BO_FLAGS_HOST_ONLY, bundle->group_q8);
    bundle->bo_out = std::make_unique<xrt::bo>(device, 4096 * sizeof(float), XRT_BO_FLAGS_HOST_ONLY, bundle->group_out);
    bundle->buf_instr = bundle->bo_instr->map<void *>();
    bundle->buf_q4 = bundle->bo_q4->map<void *>();
    bundle->buf_q8 = bundle->bo_q8->map<void *>();
    bundle->buf_out = bundle->bo_out->map<float *>();
    if (!bundle->buf_instr || !bundle->buf_q4 || !bundle->buf_q8 || !bundle->buf_out) {
        throw std::runtime_error("M4096 generation-1 BO map failed");
    }
    std::memcpy(bundle->buf_instr, state.instructions.data(), instruction_bytes);
    if (ggml_backend_xdna_capture_hash(bundle->buf_instr, instruction_bytes) != instruction_sha) {
        throw std::runtime_error("M4096 generation-1 mapped instruction SHA mismatch");
    }
    ggml_backend_xdna_sync_host_written_bo(*bundle->bo_instr, "XDNA capture replay M4096 transition instruction BO init");
    bundle->instruction_init_sync_count = 1;
    ++trace.generation1_instruction_init_syncs;
    bundle->generation = 1;
    std::fprintf(stderr,
        "XDNA_CAPTURE_REPLAY M4096 TRANSITION RESOURCES old_shape_context_not_reused=YES old_shape_kernel_not_reused=YES "
        "context_generation=1 shared_generation1_kernel=YES kernel_name=%s instruction_sha=%s instruction_bytes=%zu "
        "instr_bo=%p q4_bo=%p q8_bo=%p out_bo=%p instr_map=%p q4_map=%p q8_map=%p out_map=%p "
        "capacities=%zu,%zu,%zu,%zu kernel_group_ids=%u,%u,%u,%u bo_memory_groups=%u,%u,%u,%u instruction_init_sync_count=1\n",
        bundle->kernel->get_name().c_str(), instruction_sha, instruction_bytes,
        static_cast<const void *>(bundle->bo_instr.get()), static_cast<const void *>(bundle->bo_q4.get()),
        static_cast<const void *>(bundle->bo_q8.get()), static_cast<const void *>(bundle->bo_out.get()),
        bundle->buf_instr, bundle->buf_q4, bundle->buf_q8, static_cast<const void *>(bundle->buf_out),
        bundle->bo_instr->size(), bundle->bo_q4->size(), bundle->bo_q8->size(), bundle->bo_out->size(),
        bundle->group_instr, bundle->group_q4, bundle->group_q8, bundle->group_out,
        static_cast<unsigned>(bundle->bo_instr->get_memory_group()), static_cast<unsigned>(bundle->bo_q4->get_memory_group()),
        static_cast<unsigned>(bundle->bo_q8->get_memory_group()), static_cast<unsigned>(bundle->bo_out->get_memory_group()));
}

static void ggml_backend_xdna_replay_program_m4096_instruction_only(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_replay_runtime_trace & trace,
        const ggml_backend_xdna_batched_state & state, const std::filesystem::path & root,
        const nlohmann::json & record, ggml_backend_xdna_replay_program_only_bundle * bundle) {
    constexpr const char * instruction_sha = "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904";
    if (!ctx || !bundle || !ctx->device_context || !ctx->device_context->xrt_device ||
        !ctx->q4_m9216_hw_context || !ctx->q4_m9216_kernel || !state.available || state.rows != 4096 ||
        trace.m9216_hw_context_generation != 1 || trace.m9216_kernel_generation != 1 || trace.m9216_bo_bundle_generation != 1 ||
        trace.generation1_submissions != 0 || trace.generation1_wait_completions != 0 ||
        record.at("submission_id").get<uint64_t>() != 7 || record.at("node").get<std::string>() != "z-2" ||
        record.at("M").get<size_t>() != 4096 || record.at("K").get<size_t>() != 2560 || record.at("rhs").get<size_t>() != 1) {
        throw std::runtime_error("M4096 program-only precondition failed");
    }
    const size_t instruction_bytes = state.instructions.size() * sizeof(uint32_t);
    if (!instruction_bytes || instruction_bytes != record.at("instruction_bytes").get<size_t>() ||
        ggml_backend_xdna_capture_hash(state.instructions.data(), instruction_bytes) != instruction_sha ||
        record.at("instruction_sha256").get<std::string>() != instruction_sha) {
        throw std::runtime_error("M4096 program-only instruction identity mismatch");
    }
    std::vector<unsigned char> frozen_instruction;
    std::string error;
    if (!ggml_backend_xdna_replay_read_artifact(root, record, "instruction_file", "instruction_bytes",
            "instruction_sha256", &frozen_instruction, &error)) throw std::runtime_error(error);
    const uint32_t group = ctx->q4_m9216_kernel->group_id(1);
    bundle->bo_instr = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, instruction_bytes, XCL_BO_FLAGS_CACHEABLE, group);
    bundle->mapped_instr = bundle->bo_instr->map<void *>();
    if (!bundle->mapped_instr) throw std::runtime_error("M4096 program-only instruction BO map failed");
    std::memcpy(bundle->mapped_instr, state.instructions.data(), instruction_bytes);
    if (ggml_backend_xdna_capture_hash(bundle->mapped_instr, instruction_bytes) != instruction_sha)
        throw std::runtime_error("M4096 program-only initialization image SHA mismatch");
    ggml_backend_xdna_sync_host_written_bo(*bundle->bo_instr, "XDNA capture replay M4096 program-only instruction BO init");
    ++bundle->instruction_init_sync_count;
    ++trace.generation1_instruction_init_syncs;
    std::memcpy(bundle->mapped_instr, frozen_instruction.data(), instruction_bytes);
    if (ggml_backend_xdna_capture_hash(bundle->mapped_instr, instruction_bytes) != instruction_sha)
        throw std::runtime_error("M4096 program-only frozen instruction SHA mismatch");
    ggml_backend_xdna_sync_host_written_bo(*bundle->bo_instr, "XDNA capture replay M4096 program-only instruction BO stage");
    ++bundle->instruction_stage_sync_count;
    ++trace.generation1_instruction_stage_syncs;
    if (trace.generation1_submissions != 0 || trace.generation1_wait_completions != 0 ||
        bundle->instruction_init_sync_count != 1 || bundle->instruction_stage_sync_count != 1) {
        throw std::runtime_error("M4096 program-only stage changed execution counters or sync accounting");
    }
    std::fprintf(stderr,
        "TRANSITION_STAGE=M4096_PROGRAM_ONLY_END source_id=7 instruction_bo=%p capacity=%zu instruction_bytes=%zu instruction_words=%zu "
        "mapped_sha=%s instruction_init_sync_count=%llu instruction_stage_sync_count=%llu instruction_total_sync_count=%llu "
        "generation1_submissions=0 generation1_waits=0 BO_construct_delta=1 BO_map_delta=1 BO_sync_delta=2 kernel_group_id_query_delta=1 "
        "kernel_call_delta=0 run_creation_delta=0 wait_delta=0 kernel_submissions_delta=0 waits_delta=0 q4_sync_delta=0 q8_sync_delta=0 "
        "output_sentinel_sync_delta=0 output_sync_delta=0\n",
        static_cast<const void *>(bundle->bo_instr.get()), bundle->bo_instr->size(), instruction_bytes,
        instruction_bytes / sizeof(uint32_t),
        ggml_backend_xdna_capture_hash(bundle->mapped_instr, instruction_bytes).c_str(),
        static_cast<unsigned long long>(bundle->instruction_init_sync_count),
        static_cast<unsigned long long>(bundle->instruction_stage_sync_count),
        static_cast<unsigned long long>(bundle->instruction_init_sync_count + bundle->instruction_stage_sync_count));
}


static void ggml_backend_xdna_replay_log_runtime_snapshot(
        const char * phase, uint64_t execution_ordinal, uint64_t m9216_ordinal_for_submission,
        uint64_t source_id, const char * role, const std::string & node, size_t m,
        const ggml_backend_xdna_backend_context * ctx, const ggml_backend_xdna_replay_runtime_trace & trace,
        const xrt::hw_context * hw_context, const xrt::kernel * kernel,
        const xrt::bo * bo_instr, const xrt::bo * bo_q4, const xrt::bo * bo_q8, const xrt::bo * bo_out,
        const void * map_instr, const void * map_q4, const void * map_q8, const void * map_out,
        size_t instr_capacity, size_t q4_capacity, size_t q8_capacity, size_t out_capacity,
        uint32_t group_instr, uint32_t group_q4, uint32_t group_q8, uint32_t group_out,
        const void * run_wrapper, bool m9216) {
    const bool have_previous = m9216 && trace.have_previous_m9216;
    std::fprintf(stderr,
        "XDNA_CAPTURE_REPLAY RUNTIME SNAPSHOT\nphase=%s execution_ordinal=%llu source_id=%llu role=%s node=%s M=%zu "
        "m9216_execution_ordinal=%llu m9216_ordinal_for_submission=%llu backend_instance_id=%llu backend_context_id=%p "
        "device_context_id=%p xrt_device_id=%p m9216_xclbin_id=%p m9216_bo_set_initialized=%d\n",
        phase, static_cast<unsigned long long>(execution_ordinal), static_cast<unsigned long long>(source_id),
        role, node.c_str(), m, static_cast<unsigned long long>(trace.m9216_execution_ordinal),
        static_cast<unsigned long long>(m9216_ordinal_for_submission),
        static_cast<unsigned long long>(ctx->backend_instance_id), static_cast<const void *>(ctx),
        static_cast<const void *>(ctx->device_context), static_cast<const void *>(ctx->device_context->xrt_device.get()),
        static_cast<const void *>(ctx->q4_m9216_xclbin.get()), ctx->q4_m9216_bos_initialized ? 1 : 0);
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY GENERATION TELEMETRY generation1_ordinal=%llu submissions=%llu waits=%llu output_syncs=%llu m9216_instruction_syncs=%llu instruction_init_sync_count=%llu instruction_stage_sync_count=%llu instruction_total_sync_count=%llu q4_sync_count=%llu q8_sync_count=%llu output_sentinel_sync_count=%llu output_from_device_sync_count=%llu\n",
        static_cast<unsigned long long>(trace.generation1_execution_ordinal), static_cast<unsigned long long>(trace.generation1_submissions),
        static_cast<unsigned long long>(trace.generation1_wait_completions), static_cast<unsigned long long>(trace.generation1_output_syncs),
        static_cast<unsigned long long>(trace.m9216_instruction_syncs),
        static_cast<unsigned long long>(trace.generation1_instruction_init_syncs),
        static_cast<unsigned long long>(trace.generation1_instruction_stage_syncs),
        static_cast<unsigned long long>(trace.generation1_instruction_init_syncs + trace.generation1_instruction_stage_syncs),
        static_cast<unsigned long long>(trace.generation1_q4_syncs), static_cast<unsigned long long>(trace.generation1_q8_syncs),
        static_cast<unsigned long long>(trace.generation1_output_sentinel_syncs), static_cast<unsigned long long>(trace.generation1_output_from_device_syncs));
    if (m9216 && std::strcmp(role, "POST_RECREATE_CONTROL") == 0 && std::strcmp(phase, "PRE_SUBMIT") == 0) {
        const bool first_submission = trace.m9216_hw_context_generation == 1 && trace.m9216_kernel_generation == 1 &&
            trace.m9216_bo_bundle_generation == 1 && trace.m9216_bo_bundle_generation_submissions == 0 &&
            trace.m9216_bo_bundle_generation_wait_completions == 0;
        std::fprintf(stderr, "POST_RECREATE_CONTROL snapshot source_id=%llu generation_first_submission=%s context_generation=%llu kernel_generation=%llu bo_generation=%llu fresh_generation_submissions=%llu fresh_generation_waits=%llu\n",
            static_cast<unsigned long long>(source_id), first_submission ? "YES" : "NO",
            static_cast<unsigned long long>(trace.m9216_hw_context_generation),
            static_cast<unsigned long long>(trace.m9216_kernel_generation),
            static_cast<unsigned long long>(trace.m9216_bo_bundle_generation),
            static_cast<unsigned long long>(trace.m9216_bo_bundle_generation_submissions),
            static_cast<unsigned long long>(trace.m9216_bo_bundle_generation_wait_completions));
    }
    std::fprintf(stderr,
        "hw_context_id=%p hw_context_reuse=%s kernel_id=%p kernel_reuse=%s group_ids=%u,%u,%u,%u\n",
        static_cast<const void *>(hw_context),
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, hw_context, trace.previous_hw_context) : "N/A",
        static_cast<const void *>(kernel),
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, kernel, trace.previous_kernel) : "N/A",
        group_instr, group_q4, group_q8, group_out);
    std::fprintf(stderr,
        "instr_bo_id=%p instr_bo_address=wrapper:%p instr_bo_size=%zu instr_map=%p instr_reuse=%s\n",
        static_cast<const void *>(bo_instr), static_cast<const void *>(bo_instr), instr_capacity, map_instr,
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, bo_instr, trace.previous_bo_instr) : "N/A");
    std::fprintf(stderr,
        "q4_bo_id=%p q4_bo_address=wrapper:%p q4_bo_size=%zu q4_map=%p q4_reuse=%s\n",
        static_cast<const void *>(bo_q4), static_cast<const void *>(bo_q4), q4_capacity, map_q4,
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, bo_q4, trace.previous_bo_q4) : "N/A");
    std::fprintf(stderr,
        "q8_bo_id=%p q8_bo_address=wrapper:%p q8_bo_size=%zu q8_map=%p q8_reuse=%s\n",
        static_cast<const void *>(bo_q8), static_cast<const void *>(bo_q8), q8_capacity, map_q8,
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, bo_q8, trace.previous_bo_q8) : "N/A");
    std::fprintf(stderr,
        "out_bo_id=%p out_bo_address=wrapper:%p out_bo_size=%zu out_map=%p out_reuse=%s device_bo_addresses=NOT_QUERIED\n",
        static_cast<const void *>(bo_out), static_cast<const void *>(bo_out), out_capacity, map_out,
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, bo_out, trace.previous_bo_out) : "N/A");
    std::fprintf(stderr,
        "map_reuse_instr=%s q4=%s q8=%s out=%s run_wrapper_address=%p "
        "run_lifetime=one_submission_loop_scope native_run_handle=UNAVAILABLE\n",
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, map_instr, trace.previous_map_instr) : "N/A",
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, map_q4, trace.previous_map_q4) : "N/A",
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, map_q8, trace.previous_map_q8) : "N/A",
        m9216 ? ggml_backend_xdna_replay_identity_change(have_previous, map_out, trace.previous_map_out) : "N/A",
        run_wrapper);
    std::fprintf(stderr,
        "m9216_resource_submissions=%llu m9216_wait_calls_returned=%llu m9216_wait_completions=%llu "
        "m9216_input_syncs=%llu m9216_initialization_syncs=%llu m9216_output_sentinel_syncs=%llu "
        "m9216_output_syncs=%llu m9216_resource_reuses=%llu "
        "run_objects_created=%llu run_wait_calls_returned=%llu run_wait_completions=%llu "
        "m9216_bo_bundle_generation=%llu m9216_bo_bundle_recreate_count=%llu "
        "m9216_bo_bundle_generation_submissions=%llu m9216_bo_bundle_generation_wait_completions=%llu "
        "m9216_kernel_generation=%llu m9216_kernel_recreate_count=%llu\n",
        static_cast<unsigned long long>(trace.m9216_submissions),
        static_cast<unsigned long long>(trace.m9216_wait_calls_returned),
        static_cast<unsigned long long>(trace.m9216_wait_completions),
        static_cast<unsigned long long>(trace.m9216_input_syncs),
        static_cast<unsigned long long>(trace.m9216_initialization_syncs),
        static_cast<unsigned long long>(trace.m9216_output_sentinel_syncs),
        static_cast<unsigned long long>(trace.m9216_output_syncs),
        static_cast<unsigned long long>(trace.m9216_bundle_reuses),
        static_cast<unsigned long long>(trace.run_objects_created),
        static_cast<unsigned long long>(trace.run_wait_calls_returned),
        static_cast<unsigned long long>(trace.run_wait_completions),
        static_cast<unsigned long long>(trace.m9216_bo_bundle_generation),
        static_cast<unsigned long long>(trace.m9216_bo_bundle_recreate_count),
        static_cast<unsigned long long>(trace.m9216_bo_bundle_generation_submissions),
        static_cast<unsigned long long>(trace.m9216_bo_bundle_generation_wait_completions),
        static_cast<unsigned long long>(trace.m9216_kernel_generation),
        static_cast<unsigned long long>(trace.m9216_kernel_recreate_count));
}

static constexpr int XDNA_REPLAY_DIAGNOSTIC_TERMINAL = 2;

// Return this only after mode-specific invariants; the callback emits success after this scope unwinds.
static int ggml_backend_xdna_replay_exact_history_impl(ggml_backend_t backend, const char * plan_path, const char ** terminal_mode) {
    if (!terminal_mode || !backend || !plan_path) return 1;
    *terminal_mode = nullptr;
    auto * ctx = static_cast<ggml_backend_xdna_backend_context *>(backend->context);
    if (!ctx || ctx->exact_history_replay_failed || ggml_backend_xdna_diagnostic_capture_enabled(ctx) ||
        (std::getenv("GGML_XDNA_CAPTURE_EXACT_HISTORY") != nullptr &&
         std::strcmp(std::getenv("GGML_XDNA_CAPTURE_EXACT_HISTORY"), "0") != 0)) return 1;
    ggml_backend_xdna_replay_runtime_trace runtime_trace;
    try {
        ctx->replay_prefix_probe_mode = false;
        ctx->replay_prefix_planned = ctx->replay_prefix_completed = ctx->replay_prefix_outputs_validated = 0;
        ctx->replay_probe_submitted = ctx->replay_probe_wait_completed = ctx->replay_probe_output_validated = 0;
        ctx->replay_post_recreate_control_mode = false;
        ctx->replay_post_id8_context_recreate_only_mode = false;
        ctx->replay_post_recreate_control_submitted = ctx->replay_post_recreate_control_wait_completed = 0;
        ctx->replay_post_recreate_control_output_validated = 0;
        ctx->replay_post_recreate_transition_mode = false;
        ctx->replay_post_recreate_program_only_mode = false;
        ctx->replay_program_only_stages = 0;
        ctx->replay_transition_m4096_submitted = ctx->replay_transition_m4096_wait_completed = ctx->replay_transition_m4096_output_validated = 0;
        ctx->replay_transition_id8_submitted = ctx->replay_transition_id8_wait_completed = ctx->replay_transition_id8_output_validated = 0;
        const auto plan = nlohmann::json::parse(std::ifstream(plan_path));
        if (plan.at("schema_version").get<int>() != 1 || plan.at("profile").get<std::string>() != "q4k_q8k_k2560" ||
            plan.at("xclbin_sha256").get<std::string>() != "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83") throw std::runtime_error("plan identity mismatch");
        const auto root = std::filesystem::canonical(plan.at("source_capture_dir").get<std::string>());
        const auto & records = plan.at("records");
        if (!records.is_array() || records.empty()) throw std::runtime_error("empty replay plan");
        size_t targets = 0;
        for (size_t i = 0; i < records.size(); ++i) {
            const auto & r = records[i];
            const size_t m = r.at("M").get<size_t>();
            const bool target = r.at("semantic_target").get<bool>();
            if (r.at("submission_id").get<uint64_t>() != i + 1 || r.at("K").get<size_t>() != 2560 ||
                r.at("rhs").get<size_t>() != 1 || r.at("opcode").get<size_t>() != 3 ||
                (m != 1024 && m != 4096 && m != 8192 && m != 9216) ||
                r.at("q4_bytes").get<size_t>() != m * 1520 || r.at("q8_bytes").get<size_t>() != 2920 ||
                r.at("instruction_bytes").get<size_t>() == 0 || r.at("instruction_bytes").get<size_t>() % 4 != 0 ||
                (m == 9216 && r.at("instruction_bytes").get<size_t>() != 189108)) throw std::runtime_error("record shape/size/order mismatch");
            if (target) {
                ++targets;
                if (i + 1 != records.size() || r.at("node").get<std::string>() != "ffn_gate-4" || m != 9216 ||
                    !r.at("output_file").is_null() || !r.at("output_bytes").is_null() || !r.at("output_sha256").is_null()) throw std::runtime_error("terminal target mismatch");
            } else if (i + 1 == records.size() || r.at("output_bytes").get<size_t>() != m * sizeof(float)) {
                throw std::runtime_error("historical output contract mismatch");
            }
            std::string error;
            if (!ggml_backend_xdna_replay_read_artifact(root, r, "instruction_file", "instruction_bytes", "instruction_sha256", nullptr, &error) ||
                !ggml_backend_xdna_replay_read_artifact(root, r, "q4_file", "q4_bytes", "q4_sha256", nullptr, &error) ||
                !ggml_backend_xdna_replay_read_artifact(root, r, "q8_file", "q8_bytes", "q8_sha256", nullptr, &error) ||
                (!target && !ggml_backend_xdna_replay_read_artifact(root, r, "output_file", "output_bytes", "output_sha256", nullptr, &error))) throw std::runtime_error(error);
        }
        if (targets != 1 || !records.back().at("semantic_target").get<bool>()) throw std::runtime_error("plan target count/order mismatch");
        enum class replay_role { full_record, prefix_record, injected_history_record, probe_record, post_recreate_control_record, post_recreate_m4096_control_record, post_recreate_m4096_program_only_record, post_recreate_m4096_no_readback_record, post_transition_id8_record, post_no_readback_id8_record, post_same_shape_second_id8_record, post_recreate_id7_first_record, post_same_shape_second_id7_record, post_context_recreate_id8_record };
        struct selected_record { const nlohmann::json * record; replay_role role; };
        std::vector<selected_record> execution;
        const bool has_execution_metadata = plan.contains("execution_mode") || plan.contains("execution") ||
            plan.contains("prefix_count") || plan.contains("inject_id") || plan.contains("inject_ids") || plan.contains("probe_id") ||
            plan.contains("post_recreate_control_id") || plan.contains("post_recreate_transition_control") || plan.contains("recreate_m9216_bo_bundle_before_probe") ||
            plan.contains("recreate_m9216_kernel_before_probe") || plan.contains("recreate_m9216_hw_context_before_probe") ||
            plan.contains("recreate_m9216_context_and_bo_bundle_before_probe");
        size_t prefix_count = 0;
        const bool recreate_m9216_bo_bundle = plan.contains("recreate_m9216_bo_bundle_before_probe") &&
            plan.at("recreate_m9216_bo_bundle_before_probe").get<bool>();
        const bool recreate_m9216_kernel = plan.contains("recreate_m9216_kernel_before_probe") &&
            plan.at("recreate_m9216_kernel_before_probe").get<bool>();
        const bool recreate_m9216_hw_context = plan.contains("recreate_m9216_hw_context_before_probe") &&
            plan.at("recreate_m9216_hw_context_before_probe").get<bool>();
        const bool recreate_m9216_context_and_bo_bundle = plan.contains("recreate_m9216_context_and_bo_bundle_before_probe") &&
            plan.at("recreate_m9216_context_and_bo_bundle_before_probe").get<bool>();
        const unsigned recreation_modes = static_cast<unsigned>(recreate_m9216_bo_bundle) +
            static_cast<unsigned>(recreate_m9216_kernel) + static_cast<unsigned>(recreate_m9216_hw_context) +
            static_cast<unsigned>(recreate_m9216_context_and_bo_bundle);
        if (recreation_modes > 1) throw std::runtime_error("M9216 recreation diagnostics are mutually exclusive");
        std::vector<uint64_t> inject_ids;
        uint64_t probe_id = 0;
        uint64_t post_recreate_control_id = 0;
        bool post_recreate_control_mode = false;
        bool post_recreate_transition_mode = false;
        bool post_recreate_program_only_mode = false;
        bool post_recreate_no_readback_mode = false;
        bool post_recreate_same_shape_mode = false;
        bool post_recreate_same_shape_id7_mode = false;
        bool post_recreate_context_recreate_id8_mode = false;
        bool post_recreate_id8_context_recreate_only_mode = false;
        bool post_recreate_double_recreate_no_execution_mode = false;
        bool post_recreate_program_only_context_recreate_mode = false;
        bool post_recreate_execute_wait_no_readback_context_recreate_mode = false;
        bool post_recreate_execute_wait_release_temp_context_recreate_mode = false;
        bool post_recreate_execute_wait_release_run_context_recreate_mode = false;
        if (has_execution_metadata) {
            const std::string execution_mode = plan.at("execution_mode").get<std::string>();
            post_recreate_control_mode = execution_mode == "PREFIX_POST_RECREATE_CONTROL";
            post_recreate_transition_mode = execution_mode == "PREFIX_POST_RECREATE_TRANSITION_CONTROL";
            post_recreate_program_only_mode = execution_mode == "PREFIX_POST_RECREATE_PROGRAM_ONLY_CONTROL";
            post_recreate_no_readback_mode = execution_mode == "PREFIX_POST_RECREATE_NO_READBACK_CONTROL";
            post_recreate_same_shape_mode = execution_mode == "PREFIX_POST_RECREATE_SAME_SHAPE_CONTROL";
            post_recreate_context_recreate_id8_mode = execution_mode == "PREFIX_POST_RECREATE_CONTEXT_RECREATE_CONTROL" && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-context-recreate-to-id8";
            post_recreate_id8_context_recreate_only_mode = execution_mode == "PREFIX_POST_ID8_CONTEXT_RECREATE_ONLY" && plan.at("post_recreate_transition_control").get<std::string>() == "id8-context-recreate-only";
            post_recreate_double_recreate_no_execution_mode = execution_mode == "PREFIX_SECOND_RECREATE_NO_EXECUTION_CONTROL" && plan.at("post_recreate_transition_control").get<std::string>() == "double-recreate-no-execution";
            post_recreate_program_only_context_recreate_mode = execution_mode == "PREFIX_PROGRAM_ONLY_SECOND_RECREATE_CONTROL" && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-program-only-context-recreate";
            post_recreate_execute_wait_no_readback_context_recreate_mode = execution_mode == "PREFIX_EXECUTE_WAIT_NO_READBACK_CONTEXT_RECREATE_CONTROL" && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-execute-wait-no-readback-context-recreate";
            post_recreate_execute_wait_release_temp_context_recreate_mode = execution_mode == "PREFIX_EXECUTE_WAIT_RELEASE_TEMP_CONTEXT_RECREATE_CONTROL" && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-execute-wait-release-temp-context-recreate";
            post_recreate_execute_wait_release_run_context_recreate_mode = execution_mode == "PREFIX_EXECUTE_WAIT_RELEASE_RUN_CONTEXT_RECREATE_CONTROL" && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-execute-wait-release-run-context-recreate";
            post_recreate_same_shape_id7_mode = post_recreate_same_shape_mode && plan.at("post_recreate_transition_control").get<std::string>() == "id7-to-id7";
            post_recreate_same_shape_mode = post_recreate_same_shape_mode && !post_recreate_same_shape_id7_mode;
            const bool transition_mode = post_recreate_transition_mode || post_recreate_program_only_mode || post_recreate_no_readback_mode || post_recreate_same_shape_mode || post_recreate_same_shape_id7_mode || post_recreate_context_recreate_id8_mode || post_recreate_id8_context_recreate_only_mode || post_recreate_double_recreate_no_execution_mode || post_recreate_program_only_context_recreate_mode || post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode;
            if ((!post_recreate_control_mode && !transition_mode && execution_mode != "PREFIX_PROBE") ||
                post_recreate_control_mode != plan.contains("post_recreate_control_id") ||
                transition_mode != plan.contains("post_recreate_transition_control") ||
                (post_recreate_transition_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-to-id8") ||
                (post_recreate_program_only_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-program-only-to-id8") ||
                (post_recreate_no_readback_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-execute-no-readback-to-id8") ||
                (post_recreate_same_shape_mode && plan.at("post_recreate_transition_control").get<std::string>() != "id8-to-id8") ||
                (post_recreate_same_shape_id7_mode && plan.at("post_recreate_transition_control").get<std::string>() != "id7-to-id7") ||
                (post_recreate_context_recreate_id8_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-context-recreate-to-id8") ||
                (post_recreate_id8_context_recreate_only_mode && plan.at("post_recreate_transition_control").get<std::string>() != "id8-context-recreate-only") ||
                (post_recreate_double_recreate_no_execution_mode && plan.at("post_recreate_transition_control").get<std::string>() != "double-recreate-no-execution") ||
                (post_recreate_execute_wait_no_readback_context_recreate_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-execute-wait-no-readback-context-recreate") ||
                (post_recreate_execute_wait_release_run_context_recreate_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-execute-wait-release-run-context-recreate") ||
                (post_recreate_execute_wait_release_temp_context_recreate_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-execute-wait-release-temp-context-recreate") ||
                (post_recreate_program_only_context_recreate_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-program-only-context-recreate")) {
                throw std::runtime_error("unknown/inconsistent execution mode");
            }
            prefix_count = plan.at("prefix_count").get<size_t>();
            if (plan.contains("inject_id") && plan.contains("inject_ids")) throw std::runtime_error("ambiguous injected IDs");
            if (plan.contains("inject_ids")) inject_ids = plan.at("inject_ids").get<std::vector<uint64_t>>();
            else if (plan.contains("inject_id")) inject_ids.push_back(plan.at("inject_id").get<uint64_t>());
            if (post_recreate_control_mode || transition_mode) {
                if (plan.contains("probe_id")) throw std::runtime_error("post-recreation control cannot include probe11");
                if (post_recreate_control_mode) post_recreate_control_id = plan.at("post_recreate_control_id").get<uint64_t>();
                else if (post_recreate_transition_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-to-id8") throw std::runtime_error("invalid transition mode");
                else if (post_recreate_program_only_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-program-only-to-id8") throw std::runtime_error("invalid program-only mode");
                else if (post_recreate_no_readback_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-execute-no-readback-to-id8") throw std::runtime_error("invalid no-readback mode");
                else if (post_recreate_same_shape_mode && plan.at("post_recreate_transition_control").get<std::string>() != "id8-to-id8") throw std::runtime_error("invalid same-shape ID8 mode");
                else if (post_recreate_same_shape_id7_mode && plan.at("post_recreate_transition_control").get<std::string>() != "id7-to-id7") throw std::runtime_error("invalid same-shape ID7 mode");
                else if (post_recreate_context_recreate_id8_mode && plan.at("post_recreate_transition_control").get<std::string>() != "m4096-context-recreate-to-id8") throw std::runtime_error("invalid M4096/context-recreate/ID8 mode");
                else if (post_recreate_id8_context_recreate_only_mode) post_recreate_control_id = 8;
            } else {
                probe_id = plan.at("probe_id").get<uint64_t>();
            }
            const auto & selection = plan.at("execution");
            const size_t history_count = prefix_count + inject_ids.size();
            const bool one_injected = inject_ids.size() == 1 && inject_ids[0] > prefix_count &&
                inject_ids[0] - prefix_count == 2 && probe_id == inject_ids[0] + 1;
            const bool reversed_pair = inject_ids.size() == 2 && inject_ids[0] > prefix_count &&
                inject_ids[0] - prefix_count == 2 && inject_ids[1] == prefix_count + 1 && probe_id == prefix_count + 3;
            const bool duplicate_pair = inject_ids.size() == 2 && inject_ids[0] >= prefix_count &&
                inject_ids[0] < probe_id && inject_ids[1] == inject_ids[0];
            const bool frozen_control = (post_recreate_control_mode || post_recreate_id8_context_recreate_only_mode) && post_recreate_control_id == 8 && prefix_count == 8 &&
                inject_ids == std::vector<uint64_t>{8, 8} && recreate_m9216_context_and_bo_bundle && recreation_modes == 1 &&
                !plan.contains("probe_id") && records.size() == 13 && records[7].at("node").get<std::string>() == "ffn_gate-2" &&
                records[7].at("M").get<size_t>() == 9216 && records[7].at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" &&
                records[7].at("q4_sha256").get<std::string>() == "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" &&
                records[7].at("q8_sha256").get<std::string>() == "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" &&
                records[7].at("output_sha256").get<std::string>() == "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3";
            const bool frozen_double_recreate_no_execution = post_recreate_double_recreate_no_execution_mode && prefix_count == 8 &&
                inject_ids == std::vector<uint64_t>{8, 8} && recreate_m9216_context_and_bo_bundle && recreation_modes == 1 &&
                !plan.contains("probe_id") && !plan.contains("post_recreate_control_id") && records.size() == 13 &&
                records[7].at("submission_id").get<uint64_t>() == 8 && records[7].at("node").get<std::string>() == "ffn_gate-2" &&
                records[7].at("M").get<size_t>() == 9216 && records[7].at("K").get<size_t>() == 2560 && records[7].at("rhs").get<size_t>() == 1 &&
                !records[7].at("semantic_target").get<bool>() &&
                records[7].at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" &&
                records[7].at("q4_sha256").get<std::string>() == "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" &&
                records[7].at("q8_sha256").get<std::string>() == "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" &&
                records[7].at("output_sha256").get<std::string>() == "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3";
            const bool frozen_program_only_context_recreate = post_recreate_program_only_context_recreate_mode && prefix_count == 8 &&
                inject_ids == std::vector<uint64_t>{8, 8} && recreate_m9216_context_and_bo_bundle && recreation_modes == 1 &&
                !plan.contains("probe_id") && !plan.contains("post_recreate_control_id") && records.size() == 13 &&
                records[6].at("submission_id").get<uint64_t>() == 7 && records[6].at("node").get<std::string>() == "z-2" &&
                records[6].at("M").get<size_t>() == 4096 && records[6].at("K").get<size_t>() == 2560 && records[6].at("rhs").get<size_t>() == 1 &&
                !records[6].at("semantic_target").get<bool>() &&
                records[6].at("instruction_sha256").get<std::string>() == "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" &&
                records[6].at("q4_sha256").get<std::string>() == "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" &&
                records[6].at("q8_sha256").get<std::string>() == "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a" &&
                records[6].at("output_sha256").get<std::string>() == "6da6b7d7ebf672bc203bb00620882d159cc00371900b84ae4f730aef9cb79c66" &&
                records[7].at("submission_id").get<uint64_t>() == 8 && records[7].at("node").get<std::string>() == "ffn_gate-2" &&
                records[7].at("M").get<size_t>() == 9216 && records[7].at("K").get<size_t>() == 2560 && records[7].at("rhs").get<size_t>() == 1 &&
                !records[7].at("semantic_target").get<bool>() &&
                records[7].at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" &&
                records[7].at("q4_sha256").get<std::string>() == "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" &&
                records[7].at("q8_sha256").get<std::string>() == "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" &&
                records[7].at("output_sha256").get<std::string>() == "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3";
            const bool frozen_execute_wait_no_readback_context_recreate = (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) && prefix_count == 8 &&
                inject_ids == std::vector<uint64_t>{8, 8} && recreate_m9216_context_and_bo_bundle && recreation_modes == 1 &&
                !plan.contains("probe_id") && !plan.contains("post_recreate_control_id") && records.size() == 13 && selection.size() == 11 &&
                records[6].at("submission_id").get<uint64_t>() == 7 && records[6].at("node").get<std::string>() == "z-2" &&
                records[6].at("M").get<size_t>() == 4096 && records[6].at("K").get<size_t>() == 2560 && records[6].at("rhs").get<size_t>() == 1 &&
                !records[6].at("semantic_target").get<bool>() &&
                records[6].at("instruction_sha256").get<std::string>() == "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" &&
                records[6].at("q4_sha256").get<std::string>() == "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" &&
                records[6].at("q8_sha256").get<std::string>() == "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a";
            const bool frozen_transition = (post_recreate_transition_mode || post_recreate_program_only_mode || post_recreate_no_readback_mode || post_recreate_context_recreate_id8_mode) && prefix_count == 8 &&
                inject_ids == std::vector<uint64_t>{8, 8} && recreate_m9216_context_and_bo_bundle && recreation_modes == 1 &&
                !plan.contains("probe_id") && !plan.contains("post_recreate_control_id") && records.size() == 13 &&
                records[6].at("submission_id").get<uint64_t>() == 7 && records[6].at("node").get<std::string>() == "z-2" &&
                records[6].at("M").get<size_t>() == 4096 && records[6].at("K").get<size_t>() == 2560 && records[6].at("rhs").get<size_t>() == 1 &&
                records[6].at("instruction_sha256").get<std::string>() == "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" &&
                records[6].at("q4_sha256").get<std::string>() == "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" &&
                records[6].at("q8_sha256").get<std::string>() == "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a" &&
                records[6].at("output_sha256").get<std::string>() == "6da6b7d7ebf672bc203bb00620882d159cc00371900b84ae4f730aef9cb79c66" &&
                records[7].at("submission_id").get<uint64_t>() == 8 && records[7].at("node").get<std::string>() == "ffn_gate-2" &&
                records[7].at("M").get<size_t>() == 9216 && records[7].at("K").get<size_t>() == 2560 && records[7].at("rhs").get<size_t>() == 1 &&
                records[7].at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" &&
                records[7].at("q4_sha256").get<std::string>() == "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" &&
                records[7].at("q8_sha256").get<std::string>() == "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" &&
                records[7].at("output_sha256").get<std::string>() == "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3";
            const bool frozen_same_shape = post_recreate_same_shape_mode && prefix_count == 8 && inject_ids == std::vector<uint64_t>{8, 8} &&
                recreate_m9216_context_and_bo_bundle && recreation_modes == 1 && !plan.contains("probe_id") && !plan.contains("post_recreate_control_id") &&
                records.size() == 13 && records[7].at("submission_id").get<uint64_t>() == 8 && records[7].at("node").get<std::string>() == "ffn_gate-2" &&
                records[7].at("M").get<size_t>() == 9216 && records[7].at("K").get<size_t>() == 2560 && records[7].at("rhs").get<size_t>() == 1 &&
                records[7].at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" &&
                records[7].at("q4_sha256").get<std::string>() == "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" &&
                records[7].at("q8_sha256").get<std::string>() == "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" &&
                records[7].at("output_sha256").get<std::string>() == "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3";
            const bool frozen_same_shape_id7 = post_recreate_same_shape_id7_mode && prefix_count == 8 && inject_ids == std::vector<uint64_t>{8, 8} &&
                recreate_m9216_context_and_bo_bundle && recreation_modes == 1 && !plan.contains("probe_id") && !plan.contains("post_recreate_control_id") &&
                records.size() == 13 && records[6].at("submission_id").get<uint64_t>() == 7 && records[6].at("node").get<std::string>() == "z-2" &&
                records[6].at("M").get<size_t>() == 4096 && records[6].at("K").get<size_t>() == 2560 && records[6].at("rhs").get<size_t>() == 1 &&
                records[6].at("instruction_sha256").get<std::string>() == "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" &&
                records[6].at("q4_sha256").get<std::string>() == "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" &&
                records[6].at("q8_sha256").get<std::string>() == "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a" &&
                records[6].at("output_sha256").get<std::string>() == "6da6b7d7ebf672bc203bb00620882d159cc00371900b84ae4f730aef9cb79c66";
            const bool valid_probe = !post_recreate_control_mode && !post_recreate_transition_mode && !post_recreate_program_only_mode && !post_recreate_no_readback_mode && !post_recreate_execute_wait_no_readback_context_recreate_mode && !post_recreate_execute_wait_release_temp_context_recreate_mode && !post_recreate_execute_wait_release_run_context_recreate_mode && !post_recreate_same_shape_mode && !post_recreate_same_shape_id7_mode && !post_recreate_context_recreate_id8_mode && !post_recreate_id8_context_recreate_only_mode && !post_recreate_double_recreate_no_execution_mode && !post_recreate_program_only_context_recreate_mode && probe_id > 0 && probe_id <= records.size() && prefix_count < probe_id &&
                (inject_ids.empty() || one_injected || reversed_pair || duplicate_pair);
            if ((!frozen_control && !frozen_transition && !frozen_execute_wait_no_readback_context_recreate && !frozen_same_shape && !frozen_same_shape_id7 && !frozen_double_recreate_no_execution && !frozen_program_only_context_recreate && !valid_probe) || !selection.is_array() ||
                selection.size() != history_count + ((frozen_double_recreate_no_execution || frozen_program_only_context_recreate) ? 0 : (frozen_transition || frozen_same_shape || frozen_same_shape_id7 || post_recreate_context_recreate_id8_mode ? 2 : 1))) {
                throw std::runtime_error("invalid prefix/probe or post-recreation control selection");
            }
            for (const uint64_t id : inject_ids) {
                if (id == 0 || id >= records.size() || records[static_cast<size_t>(id - 1)].at("semantic_target").get<bool>()) {
                    throw std::runtime_error("injected IDs must reference historical records");
                }
            }
            execution.reserve(selection.size());
            for (size_t i = 0; i < prefix_count; ++i) {
                if (selection[i].at("source_id").get<uint64_t>() != i + 1 ||
                    selection[i].at("role").get<std::string>() != "PREFIX_RECORD" || records[i].at("semantic_target").get<bool>()) {
                    throw std::runtime_error("invalid prefix record selection");
                }
                execution.push_back({&records[i], replay_role::prefix_record});
            }
            for (size_t i = 0; i < inject_ids.size(); ++i) {
                const auto & injected = selection[prefix_count + i];
                const uint64_t id = inject_ids[i];
                if (injected.at("source_id").get<uint64_t>() != id ||
                    injected.at("role").get<std::string>() != "INJECTED_HISTORY_RECORD") throw std::runtime_error("invalid injected history selection");
                execution.push_back({&records[static_cast<size_t>(id - 1)], replay_role::injected_history_record});
            }
            if (post_recreate_control_mode || post_recreate_id8_context_recreate_only_mode) {
                if (selection[history_count].at("source_id").get<uint64_t>() != 8 ||
                    selection[history_count].at("role").get<std::string>() != (post_recreate_id8_context_recreate_only_mode ? "POST_RECREATE_ID8_FIRST" : "POST_RECREATE_CONTROL")) throw std::runtime_error("invalid post-recreation ID8 control selection");
                execution.push_back({&records[7], replay_role::post_recreate_control_record});
            } else if (post_recreate_double_recreate_no_execution_mode || post_recreate_program_only_context_recreate_mode) {
                if (selection.size() != history_count || execution.size() != 10) throw std::runtime_error("second-recreate diagnostic must select only ten historical records");
            } else if (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) {
                if (!frozen_execute_wait_no_readback_context_recreate || selection.size() != history_count + 1 || execution.size() != 10 ||
                    selection[history_count].at("source_id").get<uint64_t>() != 7 ||
                    selection[history_count].at("role").get<std::string>() != "M4096_EXECUTE_AND_WAIT_NO_READBACK")
                    throw std::runtime_error("execute/wait no-readback second-recreate control requires frozen ID7 after history only");
                execution.push_back({&records[6], replay_role::post_recreate_m4096_no_readback_record});
            } else if (post_recreate_transition_mode || post_recreate_program_only_mode || post_recreate_no_readback_mode) {
                const auto & control = selection[history_count];
                const auto & id8 = selection[history_count + 1];
                const std::string control_role = post_recreate_program_only_mode ? "PROGRAM_M4096_INSTRUCTION_STATE_ONLY" :
                    (post_recreate_no_readback_mode ? "M4096_EXECUTE_AND_WAIT_NO_READBACK" : "POST_RECREATE_M4096_CONTROL");
                const std::string id8_role = post_recreate_no_readback_mode ? "POST_M4096_EXEC_NO_READBACK_ID8_M9216" : "POST_TRANSITION_ID8_M9216";
                if (control.at("source_id").get<uint64_t>() != 7 || control.at("role").get<std::string>() != control_role ||
                    id8.at("source_id").get<uint64_t>() != 8 || id8.at("role").get<std::string>() != id8_role) {
                    throw std::runtime_error("invalid post-recreation M4096-to-ID8 transition selection");
                }
                const replay_role m4096_role = post_recreate_program_only_mode ? replay_role::post_recreate_m4096_program_only_record :
                    (post_recreate_no_readback_mode ? replay_role::post_recreate_m4096_no_readback_record : replay_role::post_recreate_m4096_control_record);
                execution.push_back({&records[6], m4096_role});
                execution.push_back({&records[7], post_recreate_no_readback_mode ? replay_role::post_no_readback_id8_record : replay_role::post_transition_id8_record});
            } else if (post_recreate_context_recreate_id8_mode) {
                const auto & first = selection[history_count];
                const auto & second = selection[history_count + 1];
                if (first.at("source_id").get<uint64_t>() != 7 || first.at("role").get<std::string>() != "POST_RECREATE_ID7_FIRST" ||
                    second.at("source_id").get<uint64_t>() != 8 || second.at("role").get<std::string>() != "POST_CONTEXT_RECREATE_ID8_M9216")
                    throw std::runtime_error("invalid M4096/context-recreate/ID8 execution selection");
                execution.push_back({&records[6], replay_role::post_recreate_id7_first_record});
                execution.push_back({&records[7], replay_role::post_context_recreate_id8_record});
            } else if (post_recreate_same_shape_mode) {
                const auto & first = selection[history_count];
                const auto & second = selection[history_count + 1];
                if (first.at("source_id").get<uint64_t>() != 8 || first.at("role").get<std::string>() != "POST_RECREATE_ID8_FIRST" ||
                    second.at("source_id").get<uint64_t>() != 8 || second.at("role").get<std::string>() != "POST_ID8_SAME_SHAPE_ID8_SECOND")
                    throw std::runtime_error("invalid same-shape ID8 execution selection");
                execution.push_back({&records[7], replay_role::post_recreate_control_record});
                execution.push_back({&records[7], replay_role::post_same_shape_second_id8_record});
            } else if (post_recreate_same_shape_id7_mode) {
                const auto & first = selection[history_count];
                const auto & second = selection[history_count + 1];
                if (first.at("source_id").get<uint64_t>() != 7 || first.at("role").get<std::string>() != "POST_RECREATE_ID7_FIRST" ||
                    second.at("source_id").get<uint64_t>() != 7 || second.at("role").get<std::string>() != "POST_ID7_SAME_SHAPE_ID7_SECOND")
                    throw std::runtime_error("invalid same-shape ID7 execution selection");
                execution.push_back({&records[6], replay_role::post_recreate_id7_first_record});
                execution.push_back({&records[6], replay_role::post_same_shape_second_id7_record});
            } else {
                if (selection[history_count].at("source_id").get<uint64_t>() != probe_id ||
                    selection[history_count].at("role").get<std::string>() != "PROBE_RECORD") throw std::runtime_error("invalid probe selection");
                execution.push_back({&records[static_cast<size_t>(probe_id - 1)], replay_role::probe_record});
            }
        } else {
            execution.reserve(records.size());
            for (const auto & record : records) execution.push_back({&record, replay_role::full_record});
        }
        if (recreate_m9216_bo_bundle || recreate_m9216_kernel || recreate_m9216_hw_context || recreate_m9216_context_and_bo_bundle) {
            if (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 || inject_ids != std::vector<uint64_t>{8, 8} ||
                    probe_id != 0 || post_recreate_control_id != 0 || records.size() != 13 || execution.size() != 11 ||
                    execution[10].role != replay_role::post_recreate_m4096_no_readback_record || execution[10].record != &records[6])
                    throw std::runtime_error("execute/wait no-readback second-recreate requires frozen history and ID7 only after full bundle recreation");
                for (const auto & item : execution) if (item.record->at("semantic_target").get<bool>())
                    throw std::runtime_error("execute/wait no-readback second-recreate must not select target13");
            } else if (post_recreate_program_only_context_recreate_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 ||
                    prefix_count != 8 || inject_ids != std::vector<uint64_t>{8, 8} || probe_id != 0 || post_recreate_control_id != 0 ||
                    records.size() != 13 || execution.size() != 10 || execution.back().record != &records[7])
                    throw std::runtime_error("program-only second-recreate control must select frozen history 1..8,8,8");
                for (const auto & item : execution) if (item.record->at("semantic_target").get<bool>())
                    throw std::runtime_error("program-only second-recreate control must not select target13");
            } else if (post_recreate_double_recreate_no_execution_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 || inject_ids != std::vector<uint64_t>{8, 8} ||
                    probe_id != 0 || post_recreate_control_id != 0 || records.size() != 13 || execution.size() != 10 ||
                    execution.back().role != replay_role::injected_history_record || execution.back().record != &records[7])
                    throw std::runtime_error("double-recreate no-execution must select only frozen history 1..8,8,8");
                for (const auto & item : execution) if (item.record->at("semantic_target").get<bool>() || item.record->at("submission_id").get<uint64_t>() == 11)
                    throw std::runtime_error("double-recreate no-execution must not select ID11 or target13");
            } else if (post_recreate_id8_context_recreate_only_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 || inject_ids != std::vector<uint64_t>{8, 8} ||
                    probe_id != 0 || post_recreate_control_id != 8 || records.size() != 13 || execution.size() != 11 ||
                    execution.back().role != replay_role::post_recreate_control_record || execution.back().record != &records[7])
                    throw std::runtime_error("ID8 context-recreate-only requires frozen ID8 as its sole post-recreation execution");
                for (const auto & item : execution) if (item.record->at("submission_id").get<uint64_t>() == 11 || item.record->at("semantic_target").get<bool>())
                    throw std::runtime_error("ID8 context-recreate-only must not select probe11 or target13");
            } else if (post_recreate_context_recreate_id8_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 || inject_ids != std::vector<uint64_t>{8, 8} ||
                    probe_id != 0 || post_recreate_control_id != 0 || records.size() != 13 || execution.size() != 12 ||
                    execution[10].role != replay_role::post_recreate_id7_first_record || execution[10].record != &records[6] ||
                    execution[11].role != replay_role::post_context_recreate_id8_record || execution[11].record != &records[7])
                    throw std::runtime_error("M4096/context-recreate/ID8 requires frozen ID7 and ID8 after the initial full bundle recreation");
                for (const auto & item : execution) if (item.record->at("submission_id").get<uint64_t>() == 11 || item.record->at("semantic_target").get<bool>())
                    throw std::runtime_error("M4096/context-recreate/ID8 must not select probe11 or target13");
            } else if (post_recreate_transition_mode || post_recreate_program_only_mode || post_recreate_no_readback_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 ||
                    inject_ids != std::vector<uint64_t>{8, 8} || post_recreate_control_id != 0 || probe_id != 0 ||
                    records.size() != 13 || execution.size() != 12 ||
                    execution[10].role != (post_recreate_program_only_mode ? replay_role::post_recreate_m4096_program_only_record : (post_recreate_no_readback_mode ? replay_role::post_recreate_m4096_no_readback_record : replay_role::post_recreate_m4096_control_record)) || execution[10].record->at("submission_id").get<uint64_t>() != 7 ||
                    execution[11].role != (post_recreate_no_readback_mode ? replay_role::post_no_readback_id8_record : replay_role::post_transition_id8_record) || execution[11].record->at("submission_id").get<uint64_t>() != 8) {
                    throw std::runtime_error("transition control requires frozen ID7 program/control then ID8 after full-bundle recreation");
                }
                for (const auto & item : execution) {
                    if (item.record->at("submission_id").get<uint64_t>() == 11 || item.record->at("semantic_target").get<bool>())
                        throw std::runtime_error("transition control must not select probe11 or target13");
                }
            } else if (post_recreate_same_shape_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 ||
                    inject_ids != std::vector<uint64_t>{8, 8} || probe_id != 0 || post_recreate_control_id != 0 ||
                    records.size() != 13 || execution.size() != 12 ||
                    execution[10].role != replay_role::post_recreate_control_record ||
                    execution[11].role != replay_role::post_same_shape_second_id8_record ||
                    execution[10].record != execution[11].record || execution[10].record != &records[7]) {
                    throw std::runtime_error("same-shape ID8 control requires two identical frozen ID8 records after full recreation");
                }
                for (const auto & item : execution) {
                    if (item.record->at("submission_id").get<uint64_t>() == 11 || item.record->at("semantic_target").get<bool>())
                        throw std::runtime_error("same-shape ID8 control must not select probe11 or target13");
                }
            } else if (post_recreate_same_shape_id7_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 ||
                    inject_ids != std::vector<uint64_t>{8, 8} || probe_id != 0 || post_recreate_control_id != 0 ||
                    records.size() != 13 || execution.size() != 12 ||
                    execution[10].role != replay_role::post_recreate_id7_first_record || execution[11].role != replay_role::post_same_shape_second_id7_record ||
                    execution[10].record != execution[11].record || execution[10].record != &records[6])
                    throw std::runtime_error("same-shape ID7 control requires two frozen ID7 records after full recreation");
                for (const auto & item : execution) {
                    if (item.record->at("submission_id").get<uint64_t>() == 11 || item.record->at("semantic_target").get<bool>())
                        throw std::runtime_error("same-shape ID7 control must not select probe11 or target13");
                }
            } else if (post_recreate_control_mode) {
                if (!recreate_m9216_context_and_bo_bundle || recreation_modes != 1 || prefix_count != 8 ||
                    inject_ids != std::vector<uint64_t>{8, 8} || post_recreate_control_id != 8 || probe_id != 0 ||
                    records.size() != 13 || records[7].at("node").get<std::string>() != "ffn_gate-2" ||
                    records[7].at("M").get<size_t>() != 9216 || execution.size() != 11 ||
                    execution.back().role != replay_role::post_recreate_control_record ||
                    execution.back().record->at("submission_id").get<uint64_t>() != 8) {
                    throw std::runtime_error("full-bundle control requires frozen ID8 as its only post-recreation record");
                }
                for (const auto & item : execution) {
                    if (item.record->at("submission_id").get<uint64_t>() == 11 || item.record->at("semantic_target").get<bool>()) {
                        throw std::runtime_error("post-recreation control plan must not select probe11 or target13");
                    }
                }
            } else if (!has_execution_metadata || prefix_count != 8 || inject_ids != std::vector<uint64_t>{8, 8} || probe_id != 11 ||
                records.size() != 13 || records[7].at("M").get<size_t>() != 9216 ||
                records[10].at("M").get<size_t>() != 9216 || records[10].at("node").get<std::string>() != "ffn_up-3" ||
                records[10].at("semantic_target").get<bool>() || execution.size() != 11) {
                throw std::runtime_error("M9216 recreation requires the frozen CF1 M9216 ID8/ID11 prefix-probe plan");
            }
        }
        ctx->replay_post_recreate_control_mode = post_recreate_control_mode || post_recreate_same_shape_mode || post_recreate_id8_context_recreate_only_mode;
        ctx->replay_post_recreate_transition_mode = post_recreate_transition_mode;
        ctx->replay_post_recreate_program_only_mode = post_recreate_program_only_mode;
        ctx->replay_post_recreate_no_readback_mode = post_recreate_no_readback_mode;
        ctx->replay_post_recreate_same_shape_id7_mode = post_recreate_same_shape_id7_mode;
        ctx->replay_post_recreate_context_recreate_id8_mode = post_recreate_context_recreate_id8_mode;
        ctx->replay_post_id8_context_recreate_only_mode = post_recreate_id8_context_recreate_only_mode;
        size_t selected_historical = 0, selected_targets = 0;
        for (const auto & item : execution) {
            if (item.role == replay_role::post_recreate_m4096_program_only_record) continue;
            if (item.record->at("semantic_target").get<bool>()) ++selected_targets;
            else ++selected_historical;
        }
        ctx->replay_prefix_probe_mode = has_execution_metadata;
        ctx->replay_prefix_planned = has_execution_metadata ? prefix_count + inject_ids.size() : 0;
        ctx->replay_prefix_completed = ctx->replay_prefix_outputs_validated = 0;
        ctx->replay_probe_submitted = ctx->replay_probe_wait_completed = ctx->replay_probe_output_validated = 0;

        // Allocate one reusable BO set for every shape before the first kernel call.
        for (const auto & item : execution) {
            const auto & r = *item.record;
            const size_t m = r.at("M").get<size_t>();
            if (item.role == replay_role::post_recreate_m4096_control_record || item.role == replay_role::post_recreate_m4096_program_only_record || item.role == replay_role::post_recreate_m4096_no_readback_record || item.role == replay_role::post_recreate_id7_first_record || item.role == replay_role::post_same_shape_second_id7_record) continue;
            if (m == 9216) {
                if (!ggml_backend_xdna_replay_ensure_m9216_bos(ctx, &runtime_trace) ||
                    r.at("instruction_bytes").get<size_t>() != ctx->q4_m9216_instructions.size() * sizeof(uint32_t) ||
                    r.at("q4_bytes").get<size_t>() != ctx->q4_m9216_bo_q4->size() ||
                    r.at("q8_bytes").get<size_t>() != ctx->q4_m9216_bo_q8->size() ||
                    m * sizeof(float) != ctx->q4_m9216_bo_out->size()) throw std::runtime_error("M9216 replay BO capacity/resource mismatch");
            } else {
                auto & state = m == 1024 ? ctx->q4_m1024 : (m == 4096 ? ctx->q4_m4096 : ctx->q4_m8192);
                if (!ggml_backend_xdna_replay_ensure_batched_bos(ctx, state) ||
                    r.at("instruction_bytes").get<size_t>() != state.instructions.size() * sizeof(uint32_t) ||
                    r.at("q4_bytes").get<size_t>() != state.bo_q4->size() ||
                    r.at("q8_bytes").get<size_t>() != state.bo_q8->size() ||
                    m * sizeof(float) != state.bo_out->size()) throw std::runtime_error("batched replay BO capacity/resource mismatch");
            }
        }
        ctx->replay_records_total = execution.size() - (post_recreate_program_only_mode ? 1 : 0);
        ctx->replay_target_records = selected_targets;
        ctx->replay_historical_records = selected_historical - static_cast<size_t>(post_recreate_no_readback_mode);
        ctx->replay_kernels_submitted = ctx->replay_waits_completed = ctx->replay_outputs_validated = 0;
        ggml_backend_xdna_replay_transition_bundle transition_bundle;
        const void * resource_release_temp_bo_instr = nullptr, * resource_release_temp_bo_q4 = nullptr, * resource_release_temp_bo_q8 = nullptr, * resource_release_temp_bo_out = nullptr;
        const void * resource_release_temp_map_instr = nullptr, * resource_release_temp_map_q4 = nullptr, * resource_release_temp_map_q8 = nullptr, * resource_release_temp_map_out = nullptr;
        size_t resource_release_temp_capacity_instr = 0, resource_release_temp_capacity_q4 = 0, resource_release_temp_capacity_q8 = 0, resource_release_temp_capacity_out = 0;
        bool resource_release_run_release_guard_passed = false, resource_release_temp_resources_preserved_through_recreate = false;
        const void * ep1r_retained_device = nullptr, * ep1r_retained_xclbin = nullptr, * ep1r_retained_context = nullptr, * ep1r_retained_kernel = nullptr;
        const void * ep1r_retained_bo_instr = nullptr, * ep1r_retained_bo_q4 = nullptr, * ep1r_retained_bo_q8 = nullptr, * ep1r_retained_bo_out = nullptr;
        const void * ep1r_retained_map_instr = nullptr, * ep1r_retained_map_q4 = nullptr, * ep1r_retained_map_q8 = nullptr, * ep1r_retained_map_out = nullptr;
        size_t ep1r_retained_capacity_instr = 0, ep1r_retained_capacity_q4 = 0, ep1r_retained_capacity_q8 = 0, ep1r_retained_capacity_out = 0;
        ggml_backend_xdna_replay_program_only_bundle program_only_bundle;
        uint64_t program_only_init_syncs_before = 0, program_only_stage_syncs_before = 0;
        for (const auto & selected : execution) {
            const auto & r = *selected.record;
            if (ctx->exact_history_replay_failed) throw std::runtime_error("replay failure latch set");
            const size_t m = r.at("M").get<size_t>();
            const bool target = r.at("semantic_target").get<bool>();
            const bool is_prefix = selected.role == replay_role::prefix_record;
            const bool is_injected = selected.role == replay_role::injected_history_record;
            const bool is_preprobe_history = is_prefix || is_injected;
            const bool is_historical_record = selected.role == replay_role::full_record || is_preprobe_history;
            const bool is_probe = selected.role == replay_role::probe_record;
            const bool is_post_recreate_control = selected.role == replay_role::post_recreate_control_record;
            const bool is_post_id8_context_recreate_only = is_post_recreate_control && ctx->replay_post_id8_context_recreate_only_mode;
            const bool is_same_shape_id8_first = is_post_recreate_control && post_recreate_same_shape_mode;
            const bool is_post_context_recreate_id8 = selected.role == replay_role::post_context_recreate_id8_record;
            const bool is_same_shape_id8_second = selected.role == replay_role::post_same_shape_second_id8_record;
            const bool is_post_recreate_id7_first = selected.role == replay_role::post_recreate_id7_first_record;
            const bool is_same_shape_id7_second = selected.role == replay_role::post_same_shape_second_id7_record;
            const bool is_post_recreate_id7_execution = is_post_recreate_id7_first || is_same_shape_id7_second;
            const bool is_post_recreate_m4096_control = selected.role == replay_role::post_recreate_m4096_control_record;
            const bool is_post_recreate_m4096_program_only = selected.role == replay_role::post_recreate_m4096_program_only_record;
            const bool is_post_recreate_m4096_no_readback = selected.role == replay_role::post_recreate_m4096_no_readback_record;
            const bool is_post_recreate_m4096_execution = is_post_recreate_m4096_control || is_post_recreate_m4096_no_readback;
            const bool is_post_transition_id8 = selected.role == replay_role::post_transition_id8_record || selected.role == replay_role::post_no_readback_id8_record;
            const bool is_recreation_boundary_role = is_probe || is_post_recreate_control || is_post_recreate_m4096_execution || is_post_recreate_m4096_program_only || is_post_recreate_id7_first;
            const bool run_initial_recreation_boundary = is_recreation_boundary_role &&
                (recreate_m9216_bo_bundle || recreate_m9216_kernel || recreate_m9216_hw_context || recreate_m9216_context_and_bo_bundle) &&
                (!post_recreate_id8_context_recreate_only_mode || (is_post_recreate_control && recreate_m9216_context_and_bo_bundle));
            if (run_initial_recreation_boundary) {
                if (ctx->replay_prefix_planned != 10 || ctx->replay_prefix_completed != 10 ||
                    ctx->replay_prefix_outputs_validated != 10 || ctx->exact_history_replay_failed ||
                    runtime_trace.m9216_execution_ordinal != 7 || runtime_trace.m9216_submissions != 7 ||
                    runtime_trace.m9216_wait_completions != 7 || runtime_trace.m9216_output_syncs != 7 ||
                    runtime_trace.m9216_bo_bundle_generation != 0 || runtime_trace.m9216_bo_bundle_recreate_count != 0 ||
                    runtime_trace.m9216_kernel_generation != 0 || runtime_trace.m9216_kernel_recreate_count != 0 ||
                    runtime_trace.m9216_hw_context_generation != 0 || runtime_trace.m9216_hw_context_recreate_count != 0) {
                    throw std::runtime_error("M9216 recreation boundary invariant failed");
                }
                if (is_post_recreate_control && (!ctx->replay_post_recreate_control_mode || r.at("submission_id").get<uint64_t>() != 8 ||
                    r.at("node").get<std::string>() != "ffn_gate-2" || m != 9216 || runtime_trace.m9216_submissions != 7 ||
                    runtime_trace.m9216_wait_completions != 7 || runtime_trace.m9216_output_syncs != 7 ||
                    runtime_trace.m9216_hw_context_generation != 0 || runtime_trace.m9216_kernel_generation != 0 ||
                    runtime_trace.m9216_bo_bundle_generation != 0)) {
                    throw std::runtime_error("post-recreation ID8 control pre-recreation boundary failed");
                }
                if (is_post_recreate_control) {
                    std::fprintf(stderr,
                        "POST_RECREATE_CONTROL_BOUNDARY_PRE history_completed=%llu history_outputs_validated=%llu replay_failed=%d "
                        "cumulative_m9216_submissions=%llu cumulative_m9216_wait_completions=%llu cumulative_m9216_output_syncs=%llu "
                        "context_generation=%llu kernel_generation=%llu bo_generation=%llu "
                        "context_recreate_count=%llu kernel_recreate_count=%llu bo_recreate_count=%llu "
                        "generation_local_submissions=%llu generation_local_waits=%llu\n",
                        static_cast<unsigned long long>(ctx->replay_prefix_completed),
                        static_cast<unsigned long long>(ctx->replay_prefix_outputs_validated), ctx->exact_history_replay_failed ? 1 : 0,
                        static_cast<unsigned long long>(runtime_trace.m9216_submissions),
                        static_cast<unsigned long long>(runtime_trace.m9216_wait_completions),
                        static_cast<unsigned long long>(runtime_trace.m9216_output_syncs),
                        static_cast<unsigned long long>(runtime_trace.m9216_hw_context_generation),
                        static_cast<unsigned long long>(runtime_trace.m9216_kernel_generation),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation),
                        static_cast<unsigned long long>(runtime_trace.m9216_hw_context_recreate_count),
                        static_cast<unsigned long long>(runtime_trace.m9216_kernel_recreate_count),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_recreate_count),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation_submissions),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation_wait_completions));
                }
                const char * kind = recreate_m9216_context_and_bo_bundle ? "FULL_RUNTIME_BUNDLE" :
                    (recreate_m9216_hw_context ? "HW_CONTEXT" : (recreate_m9216_kernel ? "KERNEL" : "BO_BUNDLE"));
                std::fprintf(stderr,
                    "XDNA_CAPTURE_REPLAY M9216 RECREATION BOUNDARY kind=%s history_completed=10 history_outputs_validated=10 "
                    "m9216_history_submissions=7 m9216_history_wait_completions=7 m9216_history_output_syncs=7\n", kind);
                if (recreate_m9216_context_and_bo_bundle) {
                    ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace);
                } else if (recreate_m9216_hw_context) {
                    ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace);
                } else if (recreate_m9216_kernel) {
                    ggml_backend_xdna_replay_recreate_m9216_kernel(ctx, &runtime_trace);
                } else {
                    ggml_backend_xdna_replay_recreate_m9216_bo_bundle(ctx, &runtime_trace);
                }
                if (post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) {
                    ep1r_retained_device = ctx->device_context->xrt_device.get();
                    ep1r_retained_xclbin = ctx->q4_m9216_xclbin.get();
                    ep1r_retained_context = ctx->q4_m9216_hw_context.get(); ep1r_retained_kernel = ctx->q4_m9216_kernel.get();
                    ep1r_retained_bo_instr = ctx->q4_m9216_bo_instr.get(); ep1r_retained_bo_q4 = ctx->q4_m9216_bo_q4.get();
                    ep1r_retained_bo_q8 = ctx->q4_m9216_bo_q8.get(); ep1r_retained_bo_out = ctx->q4_m9216_bo_out.get();
                    ep1r_retained_map_instr = ctx->q4_m9216_buf_instr; ep1r_retained_map_q4 = ctx->q4_m9216_buf_q4;
                    ep1r_retained_map_q8 = ctx->q4_m9216_buf_q8; ep1r_retained_map_out = ctx->q4_m9216_buf_out;
                    ep1r_retained_capacity_instr = ctx->q4_m9216_bo_instr->size(); ep1r_retained_capacity_q4 = ctx->q4_m9216_bo_q4->size();
                    ep1r_retained_capacity_q8 = ctx->q4_m9216_bo_q8->size(); ep1r_retained_capacity_out = ctx->q4_m9216_bo_out->size();
                }
                if (is_post_recreate_control) {
                    if (runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_hw_context_recreate_count != 1 ||
                        runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_kernel_recreate_count != 1 ||
                        runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                        runtime_trace.m9216_execution_ordinal != 7 || runtime_trace.m9216_submissions != 7 ||
                        runtime_trace.m9216_wait_completions != 7 || runtime_trace.m9216_output_syncs != 7 ||
                        runtime_trace.m9216_bo_bundle_generation_submissions != 0 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 0) {
                        throw std::runtime_error("post-recreation control is not first in generation 1");
                    }
                    std::fprintf(stderr,
                        "POST_RECREATE_CONTROL_BOUNDARY_POST cumulative_m9216_submissions=%llu cumulative_m9216_wait_completions=%llu cumulative_m9216_output_syncs=%llu "
                        "context_generation=%llu kernel_generation=%llu bo_generation=%llu context_recreate_count=%llu kernel_recreate_count=%llu bo_recreate_count=%llu "
                        "generation_local_submissions=%llu generation_local_waits=%llu next_m9216_submission_ordinal=%llu "
                        "instruction_sha256=3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3 instruction_image_validation=PASS instruction_bo_generation=%llu instruction_initialization_syncs=%llu\n",
                        static_cast<unsigned long long>(runtime_trace.m9216_submissions),
                        static_cast<unsigned long long>(runtime_trace.m9216_wait_completions),
                        static_cast<unsigned long long>(runtime_trace.m9216_output_syncs),
                        static_cast<unsigned long long>(runtime_trace.m9216_hw_context_generation),
                        static_cast<unsigned long long>(runtime_trace.m9216_kernel_generation),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation),
                        static_cast<unsigned long long>(runtime_trace.m9216_hw_context_recreate_count),
                        static_cast<unsigned long long>(runtime_trace.m9216_kernel_recreate_count),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_recreate_count),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation_submissions),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation_wait_completions),
                        static_cast<unsigned long long>(runtime_trace.m9216_execution_ordinal + 1),
                        static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation),
                        static_cast<unsigned long long>(runtime_trace.m9216_initialization_syncs));
                    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-RECREATE CONTROL FIRST SUBMISSION source_id=8 generation_first_submission=YES context_generation=1 kernel_generation=1 bo_generation=1 fresh_generation_submissions=0 fresh_generation_waits=0\n");
                }
                if (is_post_recreate_m4096_execution || is_post_recreate_id7_first) {
                    if ((!ctx->replay_post_recreate_transition_mode && !ctx->replay_post_recreate_no_readback_mode && !(post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) && !ctx->replay_post_recreate_same_shape_id7_mode && !ctx->replay_post_recreate_context_recreate_id8_mode) || runtime_trace.m9216_hw_context_generation != 1 ||
                        runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                        runtime_trace.m9216_submissions != 7 || runtime_trace.m9216_wait_completions != 7 ||
                        runtime_trace.m9216_output_syncs != 7 || runtime_trace.generation1_submissions != 0 ||
                        runtime_trace.generation1_wait_completions != 0) {
                        throw std::runtime_error("post-recreation M4096 transition is not first in generation 1");
                    }
                    auto & m4096_state = ctx->q4_m4096;
                    ggml_backend_xdna_replay_create_m4096_transition_bundle(ctx, m4096_state, runtime_trace, &transition_bundle);
                    std::fprintf(stderr,
                        "POST_RECREATE_M4096_BOUNDARY_POST cumulative_m9216_submissions=%llu cumulative_m9216_waits=%llu "
                        "context_generation=1 kernel_generation=1 bo_generation=1 generation1_submissions=0 generation1_waits=0 "
                        "next_generation1_submission=M4096_ID7 transition_bundle_generation=%llu\n",
                        static_cast<unsigned long long>(runtime_trace.m9216_submissions),
                        static_cast<unsigned long long>(runtime_trace.m9216_wait_completions),
                        static_cast<unsigned long long>(transition_bundle.generation));
                }
                if (is_post_recreate_m4096_program_only) {
                    if (!ctx->replay_post_recreate_program_only_mode || runtime_trace.m9216_hw_context_generation != 1 ||
                        runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                        runtime_trace.m9216_submissions != 7 || runtime_trace.m9216_wait_completions != 7 ||
                        runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0)
                        throw std::runtime_error("M4096 program-only boundary is not generation 1 with zero submissions/waits");
                    std::fprintf(stderr, "POST_RECREATE_M4096_PROGRAM_ONLY_BOUNDARY_POST context_generation=1 kernel_generation=1 bo_generation=1 generation1_submissions=0 generation1_waits=0 next_generation1_kernel_submission=ID8\n");
                }
                }
            if (is_post_recreate_m4096_program_only) {
                const uint64_t submissions_before = runtime_trace.generation1_submissions;
                const uint64_t waits_before = runtime_trace.generation1_wait_completions;
                const uint64_t runs_before = runtime_trace.run_objects_created;
                std::fprintf(stderr, "TRANSITION_STAGE=M4096_PROGRAM_ONLY_BEGIN source_id=7 instruction_sha=0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904 context_generation=1 kernel_generation=1 bo_generation=1 generation1_submissions=0 generation1_waits=0\n");
                ggml_backend_xdna_replay_program_m4096_instruction_only(ctx, runtime_trace, ctx->q4_m4096, root, r, &program_only_bundle);
                ++ctx->replay_program_only_stages;
                if (ctx->replay_program_only_stages != 1 || runtime_trace.generation1_submissions != submissions_before ||
                    runtime_trace.generation1_wait_completions != waits_before || runtime_trace.run_objects_created != runs_before ||
                    ctx->replay_transition_m4096_submitted != 0 || ctx->replay_transition_m4096_wait_completed != 0 ||
                    program_only_bundle.instruction_init_sync_count != 1 || program_only_bundle.instruction_stage_sync_count != 1)
                    throw std::runtime_error("program-only stage submitted or waited, or sync accounting failed");
                continue;
            }
            const size_t instruction_bytes = r.at("instruction_bytes").get<size_t>();
            const uint64_t execution_ordinal = ++runtime_trace.execution_ordinal;
            if (is_post_recreate_m4096_execution || is_post_recreate_id7_execution || is_post_transition_id8 || is_post_id8_context_recreate_only || (post_recreate_same_shape_mode && (is_post_recreate_control || is_same_shape_id8_second))) ++runtime_trace.generation1_execution_ordinal;
            if ((ctx->replay_post_recreate_context_recreate_id8_mode && (is_post_recreate_id7_first || is_post_context_recreate_id8)) || is_post_id8_context_recreate_only) {
                ++runtime_trace.experiment_submission_ordinal; ++runtime_trace.context_generation_submission_ordinal;
            }
            if ((post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) && is_post_recreate_m4096_no_readback) {
                ++runtime_trace.experiment_submission_ordinal; ++runtime_trace.context_generation_submission_ordinal;
            }
            const char * role_name = selected.role == replay_role::prefix_record ? "PREFIX_RECORD" :
                selected.role == replay_role::probe_record ? "PROBE_RECORD" :
                selected.role == replay_role::injected_history_record ? "INJECTED_HISTORY_RECORD" :
                is_same_shape_id8_second ? "POST_SAME_SHAPE_SECOND_ID8" :
                is_post_recreate_id7_first ? "POST_RECREATE_ID7_FIRST" :
                is_post_context_recreate_id8 ? "POST_CONTEXT_RECREATE_ID8_M9216" :
                is_same_shape_id7_second ? "POST_ID7_SAME_SHAPE_ID7_SECOND" :
                is_same_shape_id8_first ? "POST_RECREATE_ID8_FIRST" :
                is_post_id8_context_recreate_only ? "POST_RECREATE_ID8_FIRST" :
                is_post_recreate_control ? "POST_RECREATE_CONTROL" :
                is_post_recreate_m4096_control ? "POST_RECREATE_M4096_CONTROL" :
                is_post_recreate_m4096_no_readback ? "M4096_EXECUTE_AND_WAIT_NO_READBACK" :
                is_post_recreate_m4096_program_only ? "POST_RECREATE_M4096_PROGRAM_ONLY" :
                is_post_transition_id8 ? (ctx->replay_post_recreate_no_readback_mode ? "POST_M4096_EXEC_NO_READBACK_ID8_M9216" : (ctx->replay_post_recreate_program_only_mode ? "POST_PROGRAM_ONLY_ID8_M9216" : "POST_TRANSITION_ID8_M9216")) : "FULL_RECORD";
            const size_t q4_bytes = r.at("q4_bytes").get<size_t>();
            const size_t q8_bytes = r.at("q8_bytes").get<size_t>();
            const size_t output_bytes = m * sizeof(float);
            std::vector<unsigned char> instruction, q4, q8;
            std::string error;
            if (!ggml_backend_xdna_replay_read_artifact(root, r, "instruction_file", "instruction_bytes", "instruction_sha256", &instruction, &error) ||
                !ggml_backend_xdna_replay_read_artifact(root, r, "q4_file", "q4_bytes", "q4_sha256", &q4, &error) ||
                !ggml_backend_xdna_replay_read_artifact(root, r, "q8_file", "q8_bytes", "q8_sha256", &q8, &error)) throw std::runtime_error(error);
            xrt::kernel * kernel = nullptr;
            xrt::bo * bo_instr = nullptr, * bo_q4 = nullptr, * bo_q8 = nullptr, * bo_out = nullptr;
            void * buf_instr = nullptr, * buf_q4 = nullptr, * buf_q8 = nullptr, * buf_out = nullptr;
            xrt::hw_context * hw_context = nullptr;
            uint32_t group_instr = 0, group_q4 = 0, group_q8 = 0, group_out = 0;
            const size_t instr_capacity = instruction_bytes, q4_capacity = q4_bytes;
            const size_t q8_capacity = q8_bytes, out_capacity = output_bytes;
            if (is_post_recreate_m4096_execution || is_post_recreate_id7_execution) {
                hw_context = transition_bundle.hw_context; kernel = transition_bundle.kernel;
                bo_instr = transition_bundle.bo_instr.get(); bo_q4 = transition_bundle.bo_q4.get();
                bo_q8 = transition_bundle.bo_q8.get(); bo_out = transition_bundle.bo_out.get();
                buf_instr = transition_bundle.buf_instr; buf_q4 = transition_bundle.buf_q4;
                buf_q8 = transition_bundle.buf_q8; buf_out = transition_bundle.buf_out;
                group_instr = transition_bundle.group_instr; group_q4 = transition_bundle.group_q4;
                group_q8 = transition_bundle.group_q8; group_out = transition_bundle.group_out;
            } else if (m == 9216) {
                hw_context = ctx->q4_m9216_hw_context.get();
                kernel = ctx->q4_m9216_kernel.get(); bo_instr = ctx->q4_m9216_bo_instr.get();
                bo_q4 = ctx->q4_m9216_bo_q4.get(); bo_q8 = ctx->q4_m9216_bo_q8.get(); bo_out = ctx->q4_m9216_bo_out.get();
                buf_instr = ctx->q4_m9216_buf_instr; buf_q4 = ctx->q4_m9216_buf_q4;
                buf_q8 = ctx->q4_m9216_buf_q8; buf_out = ctx->q4_m9216_buf_out;
                group_instr = ctx->q4_m9216_group_instr; group_q4 = ctx->q4_m9216_group_q4;
                group_q8 = ctx->q4_m9216_group_q8; group_out = ctx->q4_m9216_group_out;
            } else {
                auto & state = m == 1024 ? ctx->q4_m1024 : (m == 4096 ? ctx->q4_m4096 : ctx->q4_m8192);
                kernel = state.kernel.get(); bo_instr = state.bo_instr.get(); bo_q4 = state.bo_q4.get();
                hw_context = state.hw_context.get();
                bo_q8 = state.bo_q8.get(); bo_out = state.bo_out.get();
                group_instr = state.group_instr; group_q4 = state.group_q4;
                group_q8 = state.group_q8; group_out = state.group_out;
                buf_instr = state.buf_instr; buf_q4 = state.buf_q4; buf_q8 = state.buf_q8; buf_out = state.buf_out;
            }
            if (!kernel || !bo_instr || !bo_q4 || !bo_q8 || !bo_out || !buf_instr || !buf_q4 || !buf_q8 || !buf_out) throw std::runtime_error("replay resource unavailable");
            const size_t output_bo_capacity = bo_out->size();
            std::memcpy(buf_instr, instruction.data(), instruction_bytes);
            const bool is_m9216 = m == 9216;
            const uint64_t source_id = r.at("submission_id").get<uint64_t>();
            const std::string node = r.at("node").get<std::string>();
            const uint64_t m9216_submission_ordinal = is_m9216 ? runtime_trace.m9216_execution_ordinal + 1 : 0;
            auto log_runtime_snapshot = [&](const char * phase, const void * run_wrapper) {
                ggml_backend_xdna_replay_log_runtime_snapshot(
                    phase, execution_ordinal, m9216_submission_ordinal, source_id, role_name, node, m, ctx, runtime_trace,
                    hw_context, kernel, bo_instr, bo_q4, bo_q8, bo_out, buf_instr, buf_q4, buf_q8, buf_out,
                    instr_capacity, q4_capacity, q8_capacity, out_capacity, group_instr, group_q4, group_q8, group_out,
                    run_wrapper, is_m9216);
            };
            const bool m9216_bundle_reused = is_m9216 && runtime_trace.have_previous_m9216 &&
                runtime_trace.previous_m9216_bo_bundle_generation == runtime_trace.m9216_bo_bundle_generation &&
                runtime_trace.previous_m9216_kernel_generation == runtime_trace.m9216_kernel_generation &&
                runtime_trace.previous_m9216_hw_context_generation == runtime_trace.m9216_hw_context_generation &&
                runtime_trace.previous_hw_context == hw_context && runtime_trace.previous_kernel == kernel &&
                runtime_trace.previous_bo_instr == bo_instr && runtime_trace.previous_bo_q4 == bo_q4 &&
                runtime_trace.previous_bo_q8 == bo_q8 && runtime_trace.previous_bo_out == bo_out &&
                runtime_trace.previous_map_instr == buf_instr && runtime_trace.previous_map_q4 == buf_q4 &&
                runtime_trace.previous_map_q8 == buf_q8 && runtime_trace.previous_map_out == buf_out;
            std::memcpy(buf_q4, q4.data(), q4_bytes); std::memcpy(buf_q8, q8.data(), q8_bytes);
            std::memset(buf_out, 0xCC, output_bytes);
            ggml_backend_xdna_sync_host_written_bo(*bo_instr, "XDNA capture replay instruction BO");
            if (is_m9216) ++runtime_trace.m9216_instruction_syncs;
            if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_instruction_stage_syncs;
            if (is_post_recreate_m4096_execution || is_post_recreate_id7_execution) ++transition_bundle.instruction_stage_sync_count;
            if (is_m9216) ++runtime_trace.m9216_input_syncs;
            ggml_backend_xdna_sync_host_written_bo(*bo_q4, "XDNA capture replay Q4 BO");
            if (is_m9216) ++runtime_trace.m9216_input_syncs;
            if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_q4_syncs;
            ggml_backend_xdna_sync_host_written_bo(*bo_q8, "XDNA capture replay Q8 BO");
            if (is_m9216) ++runtime_trace.m9216_input_syncs;
            if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_q8_syncs;
            const bool skip_output_sentinel_sync = (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) && is_post_recreate_m4096_no_readback;
            if (!skip_output_sentinel_sync) {
                ggml_backend_xdna_sync_host_written_bo(*bo_out, "XDNA capture replay output BO sentinel");
                if (is_m9216) ++runtime_trace.m9216_output_sentinel_syncs;
                if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_output_sentinel_syncs;
            }
            const std::string mapped_instruction_sha = ggml_backend_xdna_capture_hash(buf_instr, instruction_bytes);
            const std::string mapped_q4_sha = ggml_backend_xdna_capture_hash(buf_q4, q4_bytes);
            const std::string mapped_q8_sha = ggml_backend_xdna_capture_hash(buf_q8, q8_bytes);
            if (mapped_instruction_sha != r.at("instruction_sha256").get<std::string>() ||
                mapped_q4_sha != r.at("q4_sha256").get<std::string>() || mapped_q8_sha != r.at("q8_sha256").get<std::string>()) {
                throw std::runtime_error("mapped BO SHA mismatch");
            }
            if (is_post_recreate_id7_first) {
                const bool first_boundary_ok = (ctx->replay_post_recreate_same_shape_id7_mode || ctx->replay_post_recreate_context_recreate_id8_mode) && execution.size() == 12 &&
                    execution[10].role == replay_role::post_recreate_id7_first_record && execution[10].record == selected.record &&
                    source_id == 7 && node == "z-2" && m == 4096 && r.at("K").get<size_t>() == 2560 && r.at("rhs").get<size_t>() == 1 &&
                    r.at("instruction_sha256").get<std::string>() == "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" &&
                    r.at("q4_sha256").get<std::string>() == "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" &&
                    r.at("q8_sha256").get<std::string>() == "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a" &&
                    mapped_instruction_sha == r.at("instruction_sha256").get<std::string>() && mapped_q4_sha == r.at("q4_sha256").get<std::string>() && mapped_q8_sha == r.at("q8_sha256").get<std::string>() &&
                    runtime_trace.m9216_hw_context_generation == 1 && runtime_trace.m9216_kernel_generation == 1 && runtime_trace.m9216_bo_bundle_generation == 1 &&
                    runtime_trace.generation1_execution_ordinal == 1 && runtime_trace.experiment_submission_ordinal == 1 && runtime_trace.context_generation_submission_ordinal == 1 && runtime_trace.generation1_submissions == 0 && runtime_trace.generation1_wait_completions == 0 && runtime_trace.generation1_output_syncs == 0 &&
                    transition_bundle.generation == 1 && transition_bundle.id7_submissions == 0 && transition_bundle.id7_waits == 0 && transition_bundle.id7_output_syncs == 0 && transition_bundle.id7_exact_outputs == 0 &&
                    transition_bundle.hw_context == hw_context && transition_bundle.kernel == kernel && transition_bundle.bo_instr.get() == bo_instr && transition_bundle.bo_q4.get() == bo_q4 && transition_bundle.bo_q8.get() == bo_q8 && transition_bundle.bo_out.get() == bo_out &&
                    transition_bundle.buf_instr == buf_instr && transition_bundle.buf_q4 == buf_q4 && transition_bundle.buf_q8 == buf_q8 && transition_bundle.buf_out == buf_out;
                if (!first_boundary_ok) throw std::runtime_error("first ID7 generation-1 pre-submit invariant failed");
                std::fprintf(stderr, "TRANSITION_STAGE=M4096_FIRST_ID7 source_id=7 M=4096 generation=1 generation_local_ordinal=1 submissions=0 waits=0 output_syncs=0 context=%p kernel=%p instr_bo=%p q4_bo=%p q8_bo=%p output_bo=%p instr_map=%p q4_map=%p q8_map=%p output_map=%p\\n",
                    static_cast<const void *>(hw_context), static_cast<const void *>(kernel), static_cast<const void *>(bo_instr), static_cast<const void *>(bo_q4), static_cast<const void *>(bo_q8), static_cast<const void *>(bo_out), buf_instr, buf_q4, buf_q8, buf_out);
            }
            if (is_same_shape_id7_second) {
                const auto & first_id7 = *execution[10].record;
                const bool same_identity = transition_bundle.id7_first_snapshot_valid &&
                    transition_bundle.id7_first_hw_context == hw_context && transition_bundle.id7_first_kernel == kernel &&
                    transition_bundle.id7_first_bo_instr == bo_instr && transition_bundle.id7_first_bo_q4 == bo_q4 &&
                    transition_bundle.id7_first_bo_q8 == bo_q8 && transition_bundle.id7_first_bo_out == bo_out &&
                    transition_bundle.id7_first_map_instr == buf_instr && transition_bundle.id7_first_map_q4 == buf_q4 &&
                    transition_bundle.id7_first_map_q8 == buf_q8 && transition_bundle.id7_first_map_out == buf_out;
                const bool first_exact = transition_bundle.id7_submissions == 1 && transition_bundle.id7_waits == 1 && transition_bundle.id7_output_syncs == 1 &&
                    transition_bundle.id7_validations == 1 && transition_bundle.id7_exact_outputs == 1 &&
                    runtime_trace.generation1_submissions == 1 && runtime_trace.generation1_wait_completions == 1 && runtime_trace.generation1_output_syncs == 1 && runtime_trace.generation1_execution_ordinal == 2;
                const bool frozen_same = execution.size() == 12 && execution[10].role == replay_role::post_recreate_id7_first_record &&
                    execution[11].role == replay_role::post_same_shape_second_id7_record && execution[10].record == execution[11].record &&
                    first_id7.at("submission_id").get<uint64_t>() == 7 && first_id7.at("node").get<std::string>() == "z-2" &&
                    first_id7.at("M").get<size_t>() == 4096 && first_id7.at("K").get<size_t>() == 2560 && first_id7.at("rhs").get<size_t>() == 1 &&
                    source_id == 7 && r.at("M").get<size_t>() == 4096 && r.at("K").get<size_t>() == 2560 && r.at("rhs").get<size_t>() == 1 &&
                    first_id7.at("instruction_sha256").get<std::string>() == r.at("instruction_sha256").get<std::string>() &&
                    first_id7.at("q4_sha256").get<std::string>() == r.at("q4_sha256").get<std::string>() && first_id7.at("q8_sha256").get<std::string>() == r.at("q8_sha256").get<std::string>() &&
                    r.at("instruction_sha256").get<std::string>() == "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" &&
                    r.at("q4_sha256").get<std::string>() == "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" &&
                    r.at("q8_sha256").get<std::string>() == "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a";
                const bool runtime_generation = ctx->replay_post_recreate_same_shape_id7_mode && transition_bundle.generation == 1 &&
                    runtime_trace.m9216_hw_context_generation == 1 && runtime_trace.m9216_kernel_generation == 1 && runtime_trace.m9216_bo_bundle_generation == 1;
                if (!first_exact || !frozen_same || !runtime_generation || !same_identity) throw std::runtime_error("same-shape ID7 second submission requires exact first ID7, ordinal 2, frozen payload, and identical generation-1 resources");
                std::fprintf(stderr, "FIRST_ID7_COMPLETED=1 FIRST_ID7_VALIDATED=1 FIRST_ID7_EXACT=1 generation1_submissions=1 generation1_waits=1 generation1_output_syncs=1 generation1_exact_outputs=1\n");
                std::fprintf(stderr, "TRANSITION_STAGE=M4096_TO_SAME_M4096 previous_source_id=7 next_source_id=7 previous_M=4096 next_M=4096 previous_instruction_sha=%s next_instruction_sha=%s same_shape_as_previous=YES same_instruction_image_as_previous=YES resource_reuse=hw_context:SAME_AS_PREVIOUS,kernel:SAME_AS_PREVIOUS,instr_bo:SAME_AS_PREVIOUS,q4_bo:SAME_AS_PREVIOUS,q8_bo:SAME_AS_PREVIOUS,output_bo:SAME_AS_PREVIOUS,instr_map:SAME_AS_PREVIOUS,q4_map:SAME_AS_PREVIOUS,q8_map:SAME_AS_PREVIOUS,output_map:SAME_AS_PREVIOUS\n", first_id7.at("instruction_sha256").get<std::string>().c_str(), r.at("instruction_sha256").get<std::string>().c_str());
                std::fprintf(stderr, "TRANSITION_STAGE=M4096_SECOND_ID7 role=POST_ID7_SAME_SHAPE_ID7_SECOND source_id=7 M=4096 generation_local_ordinal=2 same_shape_as_previous=YES same_instruction_image_as_previous=YES\n");
            }
            if (is_same_shape_id8_second) {
                const auto & first_id8 = *execution[10].record;
                const bool frozen_same_image = execution.size() == 12 && execution[10].record == execution[11].record &&
                    execution[10].record == selected.record && first_id8.at("submission_id").get<uint64_t>() == 8 &&
                    first_id8.at("node").get<std::string>() == "ffn_gate-2" && first_id8.at("M").get<size_t>() == 9216 &&
                    first_id8.at("K").get<size_t>() == 2560 && first_id8.at("rhs").get<size_t>() == 1 &&
                    first_id8.at("instruction_bytes").get<size_t>() == instruction_bytes &&
                    first_id8.at("instruction_sha256").get<std::string>() == r.at("instruction_sha256").get<std::string>() &&
                    first_id8.at("q4_sha256").get<std::string>() == r.at("q4_sha256").get<std::string>() &&
                    first_id8.at("q8_sha256").get<std::string>() == r.at("q8_sha256").get<std::string>() &&
                    r.at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3";
                const bool first_id8_exact = ctx->replay_post_recreate_control_submitted == 1 &&
                    ctx->replay_post_recreate_control_wait_completed == 1 && ctx->replay_post_recreate_control_output_validated == 1 &&
                    runtime_trace.generation1_execution_ordinal == 2 && runtime_trace.generation1_submissions == 1 &&
                    runtime_trace.generation1_wait_completions == 1 && runtime_trace.generation1_output_syncs == 1 &&
                    runtime_trace.m9216_submissions == 8 && runtime_trace.m9216_wait_completions == 8 && runtime_trace.m9216_output_syncs == 8 &&
                    runtime_trace.m9216_hw_context_generation == 1 && runtime_trace.m9216_kernel_generation == 1 && runtime_trace.m9216_bo_bundle_generation == 1 &&
                    runtime_trace.m9216_hw_context_recreate_count == 1 && runtime_trace.m9216_kernel_recreate_count == 1 && runtime_trace.m9216_bo_bundle_recreate_count == 1;
                if (!post_recreate_same_shape_mode || !frozen_same_image || !first_id8_exact || !m9216_bundle_reused)
                    throw std::runtime_error("same-shape ID8 second submission requires exact first ID8 and reused generation-1 resources");
                std::fprintf(stderr, "FIRST_ID8_COMPLETED=1 FIRST_ID8_VALIDATED=1 FIRST_ID8_EXACT=1 generation1_submissions=1 generation1_waits=1 generation1_exact_outputs=1\n");
                std::fprintf(stderr, "TRANSITION_STAGE=M9216_TO_SAME_M9216 previous_source_id=%llu next_source_id=%llu previous_M=%zu next_M=%zu previous_instruction_sha=%s next_instruction_sha=%s instruction_bytes=%zu same_shape_as_previous=YES same_instruction_image_as_previous=YES\n",
                    static_cast<unsigned long long>(first_id8.at("submission_id").get<uint64_t>()), static_cast<unsigned long long>(source_id),
                    first_id8.at("M").get<size_t>(), m, first_id8.at("instruction_sha256").get<std::string>().c_str(),
                    r.at("instruction_sha256").get<std::string>().c_str(), instruction_bytes);
            }
            if (is_post_recreate_m4096_execution || is_post_transition_id8 || is_same_shape_id8_first || is_same_shape_id8_second || is_post_context_recreate_id8 ||
                (is_post_recreate_id7_first && ctx->replay_post_recreate_context_recreate_id8_mode)) {
                std::fprintf(stderr,
                    "TRANSITION_STAGE=%s role=%s source_id=%llu M=%zu instruction_sha=%s instruction_bo=%p instruction_bytes=%zu mapped_sha=%s "
                    "instruction_init_sync_count=%llu instruction_stage_sync_count=%llu instruction_total_sync_count=%llu "
                    "generation1_instruction_init_sync_count=%llu generation1_instruction_stage_sync_count=%llu generation1_instruction_total_sync_count=%llu "
                    "generation1_submissions=%llu generation1_waits=%llu q4_bo=%p q8_bo=%p output_bo=%p context_generation=%llu kernel_generation=%llu generation1_ordinal=%llu generation_local_ordinal=%llu "
                    "kernel=%p instr_map=%p q4_map=%p q8_map=%p output_map=%p\n",
                    is_post_recreate_m4096_execution ? (is_post_recreate_m4096_no_readback ? "M4096_EXECUTE_NO_READBACK" : "M4096") :
                    (is_same_shape_id8_first ? "M9216_FIRST_ID8" : (is_same_shape_id8_second ? "M9216_SECOND_ID8" :
                        (ctx->replay_post_recreate_no_readback_mode ? "M9216_AFTER_M4096_EXEC_NO_READBACK" : (ctx->replay_post_recreate_program_only_mode ? "M9216_AFTER_M4096_PROGRAM_ONLY" : "M9216_AFTER_M4096")))),
                    role_name,
                    static_cast<unsigned long long>(source_id), m, r.at("instruction_sha256").get<std::string>().c_str(),
                    static_cast<const void *>(bo_instr), instruction_bytes, mapped_instruction_sha.c_str(),
                    static_cast<unsigned long long>(is_post_recreate_m4096_execution ? transition_bundle.instruction_init_sync_count : 0),
                    static_cast<unsigned long long>(is_post_recreate_m4096_execution ? transition_bundle.instruction_stage_sync_count : 0),
                    static_cast<unsigned long long>(is_post_recreate_m4096_execution ? transition_bundle.instruction_init_sync_count + transition_bundle.instruction_stage_sync_count : 0),
                    static_cast<unsigned long long>(runtime_trace.generation1_instruction_init_syncs),
                    static_cast<unsigned long long>(runtime_trace.generation1_instruction_stage_syncs),
                    static_cast<unsigned long long>(runtime_trace.generation1_instruction_init_syncs + runtime_trace.generation1_instruction_stage_syncs),
                    static_cast<unsigned long long>(runtime_trace.generation1_submissions),
                    static_cast<unsigned long long>(runtime_trace.generation1_wait_completions),
                    static_cast<const void *>(bo_q4), static_cast<const void *>(bo_q8), static_cast<const void *>(bo_out),
                    static_cast<unsigned long long>(runtime_trace.m9216_hw_context_generation),
                    static_cast<unsigned long long>(runtime_trace.m9216_kernel_generation),
                    static_cast<unsigned long long>(runtime_trace.generation1_execution_ordinal),
                    static_cast<unsigned long long>(runtime_trace.generation1_execution_ordinal),
                    static_cast<const void *>(kernel), buf_instr, buf_q4, buf_q8, buf_out);
            }
            if (is_probe || is_post_recreate_control || is_post_recreate_m4096_execution || is_post_transition_id8) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PREFIX/POST-RECREATE CONTROL BOUNDARY role=%s prefix_planned=%llu prefix_completed=%llu prefix_outputs_validated=%llu replay_failed=%d records_executed=%llu records_output_validated=%llu\n",
                    is_post_recreate_control ? "POST_RECREATE_CONTROL" : "PROBE",
                    static_cast<unsigned long long>(ctx->replay_prefix_planned),
                    static_cast<unsigned long long>(ctx->replay_prefix_completed),
                    static_cast<unsigned long long>(ctx->replay_prefix_outputs_validated), ctx->exact_history_replay_failed ? 1 : 0,
                    static_cast<unsigned long long>(ctx->replay_prefix_completed),
                    static_cast<unsigned long long>(ctx->replay_prefix_outputs_validated));
                if (ctx->replay_prefix_completed != ctx->replay_prefix_planned ||
                    ctx->replay_prefix_outputs_validated != ctx->replay_prefix_planned || ctx->exact_history_replay_failed) {
                    throw std::runtime_error("prefix/pre-post-recreation-control invariant failed");
                }
                if (is_post_recreate_m4096_execution && (runtime_trace.generation1_execution_ordinal != 1 ||
                    runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0 ||
                    runtime_trace.generation1_output_syncs != 0 || runtime_trace.m9216_hw_context_generation != 1 ||
                    runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_bo_bundle_generation != 1)) {
                    throw std::runtime_error("post-recreation M4096 control is not generation 1 first submission");
                }
                if (is_post_transition_id8) {
                    const bool id7_identity = execution.size() == 12 && execution[10].record->at("submission_id").get<uint64_t>() == 7 &&
                        execution[10].record->at("node").get<std::string>() == "z-2" && execution[10].record->at("M").get<size_t>() == 4096;
                    const bool id8_identity = source_id == 8 && r.at("submission_id").get<uint64_t>() == 8 && node == "ffn_gate-2" &&
                        m == 9216 && r.at("K").get<size_t>() == 2560 && r.at("rhs").get<size_t>() == 1 && !target;
                    const bool history_boundary = runtime_trace.m9216_submissions == 7 && runtime_trace.m9216_wait_completions == 7 &&
                        runtime_trace.m9216_output_syncs == 7;
                    const bool first_submission = runtime_trace.generation1_submissions == 0 && runtime_trace.generation1_wait_completions == 0 &&
                        runtime_trace.generation1_output_syncs == 0 && ctx->replay_program_only_stages == 1 &&
                        ctx->replay_transition_m4096_submitted == 0 && ctx->replay_transition_m4096_wait_completed == 0 &&
                        ctx->replay_transition_m4096_output_validated == 0 && runtime_trace.generation1_execution_ordinal == 1 &&
                        runtime_trace.m9216_hw_context_generation == 1 && runtime_trace.m9216_kernel_generation == 1 && runtime_trace.m9216_bo_bundle_generation == 1;
                    const bool m4096_completed = runtime_trace.generation1_submissions == 1 && runtime_trace.generation1_wait_completions == 1 &&
                        ctx->replay_transition_m4096_submitted == 1 && ctx->replay_transition_m4096_wait_completed == 1 &&
                        runtime_trace.generation1_execution_ordinal == 2 && runtime_trace.m9216_hw_context_generation == 1 &&
                        runtime_trace.m9216_kernel_generation == 1 && runtime_trace.m9216_bo_bundle_generation == 1 &&
                        transition_bundle.generation == 1 && transition_bundle.hw_context == hw_context && transition_bundle.kernel == kernel &&
                        ctx->replay_transition_id8_submitted == 0;
                    const bool ordinary_readback = runtime_trace.generation1_output_syncs == 1 &&
                        runtime_trace.generation1_output_from_device_syncs == 1 && transition_bundle.m4096_output_from_device_syncs == 1 &&
                        transition_bundle.m4096_output_reads == 1 && transition_bundle.m4096_output_sha_calculations == 1 &&
                        transition_bundle.m4096_output_sha_validations == 1 && ctx->replay_transition_m4096_output_validated == 1;
                    const bool no_readback = runtime_trace.generation1_output_syncs == 0 &&
                        runtime_trace.generation1_output_from_device_syncs == 0 && transition_bundle.m4096_output_from_device_syncs == 0 &&
                        transition_bundle.m4096_output_reads == 0 && transition_bundle.m4096_output_sha_calculations == 0 &&
                        transition_bundle.m4096_output_sha_validations == 0 && ctx->replay_transition_m4096_output_validated == 0;
                    bool transition_state_valid = false;
                    if (ctx->replay_post_recreate_program_only_mode) {
                        transition_state_valid = first_submission;
                    } else if (ctx->replay_post_recreate_no_readback_mode) {
                        transition_state_valid = m4096_completed && no_readback;
                    } else if (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) {
                        transition_state_valid = m4096_completed && no_readback;
                    } else if (ctx->replay_post_recreate_transition_mode) {
                        transition_state_valid = m4096_completed && ordinary_readback;
                    }
                    if (!id7_identity || !id8_identity || !history_boundary || !transition_state_valid)
                        throw std::runtime_error("ID8 generation-1 first-submission/transition boundary invariant failed");
                }
            } else if (target && (ctx->replay_kernels_submitted != ctx->replay_historical_records ||
                                 ctx->replay_waits_completed != ctx->replay_historical_records ||
                                 ctx->replay_outputs_validated != ctx->replay_historical_records)) {
                throw std::runtime_error("pre-target telemetry invariant failed");
            }
            if (is_probe || is_post_recreate_control || is_post_recreate_m4096_execution || is_post_transition_id8 || is_same_shape_id8_second) {
                if (is_probe) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PRE-PROBE RUNTIME STATE\n");
                else if (is_same_shape_id8_first) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY SAME-SHAPE FIRST ID8 RUNTIME STATE\n");
                else if (is_same_shape_id8_second) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY SAME-SHAPE SECOND ID8 RUNTIME STATE\n");
                else if (is_post_recreate_control) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-RECREATE CONTROL RUNTIME STATE\n");
                else if (is_post_recreate_m4096_control) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-RECREATE M4096 RUNTIME STATE\n");
                else if (is_post_recreate_m4096_no_readback) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M4096 NO-READBACK RUNTIME STATE\n");
                else if (ctx->replay_post_recreate_program_only_mode) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-PROGRAM-ONLY ID8 RUNTIME STATE\n");
                else std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-TRANSITION ID8 RUNTIME STATE\n");
                log_runtime_snapshot(is_probe ? "PRE_PROBE" : (is_same_shape_id8_first ? "PRE_SAME_SHAPE_ID8_FIRST" : (is_same_shape_id8_second ? "PRE_SAME_SHAPE_ID8_SECOND" : (is_post_recreate_control ? "PRE_POST_RECREATE_CONTROL" : (is_post_recreate_m4096_control ? "PRE_POST_RECREATE_M4096" : (is_post_recreate_m4096_no_readback ? "PRE_M4096_NO_READBACK" : (ctx->replay_post_recreate_program_only_mode ? "PRE_POST_PROGRAM_ONLY_ID8" : (ctx->replay_post_recreate_no_readback_mode ? "PRE_POST_NO_READBACK_ID8" : "PRE_POST_TRANSITION_ID8"))))))), nullptr);
            }
            if (is_post_context_recreate_id8) {
                const bool frozen_id8 = source_id == 8 && node == "ffn_gate-2" && m == 9216 && r.at("K").get<size_t>() == 2560 && r.at("rhs").get<size_t>() == 1 && !target &&
                    r.at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" &&
                    r.at("q4_sha256").get<std::string>() == "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" &&
                    r.at("q8_sha256").get<std::string>() == "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" &&
                    r.at("output_sha256").get<std::string>() == "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3";
                const bool exact_id7 = transition_bundle.generation == 1 && transition_bundle.id7_submissions == 1 && transition_bundle.id7_waits == 1 &&
                    transition_bundle.id7_output_syncs == 1 && transition_bundle.id7_validations == 1 && transition_bundle.id7_exact_outputs == 1;
                const bool recreation_counts = runtime_trace.m9216_hw_context_generation == 2 && runtime_trace.m9216_kernel_generation == 2 && runtime_trace.m9216_bo_bundle_generation == 1 &&
                    runtime_trace.m9216_hw_context_recreate_count == 2 && runtime_trace.m9216_kernel_recreate_count == 2 && runtime_trace.m9216_bo_bundle_recreate_count == 1;
                const bool resources_retained = runtime_trace.post_id7_device == ctx->device_context->xrt_device.get() && runtime_trace.post_id7_xclbin == ctx->q4_m9216_xclbin.get() &&
                    runtime_trace.post_id7_context != nullptr && runtime_trace.post_id7_context != hw_context && runtime_trace.post_id7_kernel != nullptr && runtime_trace.post_id7_kernel != kernel &&
                    runtime_trace.post_id7_bo_instr == bo_instr && runtime_trace.post_id7_bo_q4 == bo_q4 && runtime_trace.post_id7_bo_q8 == bo_q8 && runtime_trace.post_id7_bo_out == bo_out &&
                    runtime_trace.post_id7_map_instr == buf_instr && runtime_trace.post_id7_map_q4 == buf_q4 && runtime_trace.post_id7_map_q8 == buf_q8 && runtime_trace.post_id7_map_out == buf_out &&
                    runtime_trace.post_id7_instr_capacity == instr_capacity && runtime_trace.post_id7_q4_capacity == q4_capacity && runtime_trace.post_id7_q8_capacity == q8_capacity && runtime_trace.post_id7_out_capacity == out_capacity &&
                    runtime_trace.post_id7_xclbin_uuid == ctx->q4_m9216_xclbin->get_uuid().to_string() && runtime_trace.post_id7_kernel_name == kernel->get_name();
                const bool no_intervening_submission = runtime_trace.experiment_submission_ordinal == 2 && runtime_trace.context_generation_submission_ordinal == 1 &&
                    runtime_trace.execution_ordinal == 11 && runtime_trace.m9216_submissions == 7 && runtime_trace.m9216_wait_completions == 7 && runtime_trace.m9216_output_syncs == 7 &&
                    ctx->replay_kernels_submitted == 11 && ctx->replay_waits_completed == 11 && ctx->replay_outputs_validated == 11;
                if (!ctx->replay_post_recreate_context_recreate_id8_mode || !frozen_id8 || !exact_id7 || !recreation_counts || !resources_retained || !no_intervening_submission ||
                    mapped_instruction_sha != r.at("instruction_sha256").get<std::string>() || mapped_q4_sha != r.at("q4_sha256").get<std::string>() || mapped_q8_sha != r.at("q8_sha256").get<std::string>())
                    throw std::runtime_error("XDNA_RECREATE_GUARD ID8 pre-submit context-generation-2 guard failed");
                std::fprintf(stderr, "TRANSITION_STAGE=M4096_TO_CONTEXT_KERNEL_RECREATE_TO_M9216 source_id=8 experiment_ordinal=2 context_generation=2 context_generation_local_ordinal=1 id7_exact=YES no_intervening_submission=YES hw_context_changed=YES kernel_changed=YES device=SAME xclbin_uuid=SAME kernel_name=SAME m9216_bo_bundle=SAME maps=SAME capacities=SAME context_kernel_recreate_delta=+1/+1 bo_recreate_delta=+0\n");
            }
            log_runtime_snapshot("PRE_SUBMIT", nullptr);
            if (is_probe) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PROBE\nsource_id=%llu\nnode=%s\nM=%zu\nprefix_count=%llu\n",
                    static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()),
                    r.at("node").get<std::string>().c_str(), m,
                    static_cast<unsigned long long>(prefix_count));
                if (target) {
                    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY TARGET submission_id=%llu\nnode=ffn_gate-4 M=9216 K=2560 rhs=1\n",
                        static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()));
                }
                std::fprintf(stderr, "BEFORE kernel\n");
            } else if (is_same_shape_id8_first) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY SAME-SHAPE FIRST ID8 source_id=8 node=ffn_gate-2 M=9216\nBEFORE kernel\n");
            } else if (is_post_recreate_control) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-RECREATE CONTROL source_id=8 node=ffn_gate-2 M=9216\nBEFORE kernel\n");
            } else if (is_same_shape_id8_second) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY SAME-SHAPE SECOND ID8 source_id=8 node=ffn_gate-2 M=9216\nBEFORE kernel\n");
            } else if (is_post_recreate_m4096_no_readback) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M4096 EXECUTE NO-READBACK source_id=7 node=z-2 M=4096\nBEFORE kernel\n");
            } else if (is_post_recreate_m4096_control) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-RECREATE M4096 CONTROL source_id=7 node=z-2 M=4096\nBEFORE kernel\n");
            } else if (is_post_transition_id8) {
                std::fprintf(stderr, ctx->replay_post_recreate_no_readback_mode ? "XDNA_CAPTURE_REPLAY POST-M4096-NO-READBACK ID8 source_id=8 node=ffn_gate-2 M=9216\nBEFORE kernel\n" : (ctx->replay_post_recreate_program_only_mode ? "XDNA_CAPTURE_REPLAY POST-PROGRAM-ONLY ID8 source_id=8 node=ffn_gate-2 M=9216\nBEFORE kernel\n" : "XDNA_CAPTURE_REPLAY POST-TRANSITION ID8 source_id=8 node=ffn_gate-2 M=9216\nBEFORE kernel\n"));
            } else if (target) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY TARGET submission_id=%llu\nBEFORE kernel\n", static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()));
                std::fprintf(stderr, "node=ffn_gate-4 M=9216 K=2560 rhs=1\n");
            } else if (is_prefix) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PREFIX source_id=%llu node=%s M=%zu\nBEFORE kernel\n",
                    static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()),
                    r.at("node").get<std::string>().c_str(), m);
            } else if (is_injected) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY INJECTED HISTORY source_id=%llu node=%s M=%zu mapped_input_sha=PASS\nBEFORE kernel\n",
                    static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()),
                    r.at("node").get<std::string>().c_str(), m);
            } else {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY submission_id=%llu node=%s M=%zu\nBEFORE kernel\n",
                    static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()), r.at("node").get<std::string>().c_str(), m);
            }
            std::optional<xrt::run> run;
            run.emplace((*kernel)(3, *bo_instr, instruction_bytes, *bo_q4, *bo_q8, *bo_out));
            ++runtime_trace.run_objects_created;
            if (is_m9216) {
                ++runtime_trace.m9216_execution_ordinal;
                if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_submissions;
                ++runtime_trace.m9216_submissions;
                ++runtime_trace.m9216_bo_bundle_generation_submissions;
                if (m9216_bundle_reused) ++runtime_trace.m9216_bundle_reuses;
            }
            if (is_post_recreate_m4096_execution || is_post_recreate_id7_execution) ++runtime_trace.generation1_submissions;
            if (is_post_recreate_id7_execution) ++transition_bundle.id7_submissions;
            if (is_post_recreate_m4096_execution) ++ctx->replay_transition_m4096_submitted;
            if (is_post_transition_id8) ++ctx->replay_transition_id8_submitted;
            log_runtime_snapshot("POST_KERNEL", &*run);
            ++ctx->replay_kernels_submitted;
            if (is_probe) ++ctx->replay_probe_submitted;
            if (is_post_recreate_control) ++ctx->replay_post_recreate_control_submitted;
            std::fprintf(stderr, "AFTER kernel\nBEFORE run.wait\n");
            const auto completion = run->wait();
            ++runtime_trace.run_wait_calls_returned;
            if (is_m9216) ++runtime_trace.m9216_wait_calls_returned;
            if (completion != ERT_CMD_STATE_COMPLETED) {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY FAILURE RUNTIME STATE wait_state=%d expected=COMPLETED\n", static_cast<int>(completion));
                log_runtime_snapshot("WAIT_FAILURE", &*run);
                throw std::runtime_error("XRT replay wait did not complete");
            }
            ++runtime_trace.run_wait_completions;
            if (is_m9216) ++runtime_trace.m9216_wait_completions;
            if (is_m9216) ++runtime_trace.m9216_bo_bundle_generation_wait_completions;
            if (is_post_recreate_id7_execution) ++transition_bundle.id7_waits;
            if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_wait_completions;
            if (is_post_recreate_m4096_execution) ++ctx->replay_transition_m4096_wait_completed;
            if (is_post_transition_id8) ++ctx->replay_transition_id8_wait_completed;
            log_runtime_snapshot("POST_WAIT", &*run);
            ++ctx->replay_waits_completed;
            if (is_preprobe_history) ++ctx->replay_prefix_completed;
            if (is_probe) ++ctx->replay_probe_wait_completed;
            if (is_post_recreate_control) ++ctx->replay_post_recreate_control_wait_completed;
            std::fprintf(stderr, "AFTER run.wait\n");
            if (is_post_recreate_m4096_no_readback) {
                if (post_recreate_execute_wait_no_readback_context_recreate_mode && (runtime_trace.run_objects_created != 11 ||
                    runtime_trace.run_wait_calls_returned != 11 || runtime_trace.run_wait_completions != 11 ||
                    ctx->replay_kernels_submitted != 11 || ctx->replay_waits_completed != 11 ||
                    runtime_trace.generation1_output_sentinel_syncs != 0 || runtime_trace.generation1_output_from_device_syncs != 0 || runtime_trace.generation1_exact_outputs != 0))
                    throw std::runtime_error("execute/wait no-readback run/wait/readback counter invariant failed");
                if (runtime_trace.generation1_execution_ordinal != 1 || runtime_trace.generation1_submissions != 1 ||
                    runtime_trace.generation1_wait_completions != 1 || runtime_trace.generation1_output_syncs != 0 ||
                    ctx->replay_transition_m4096_submitted != 1 || ctx->replay_transition_m4096_wait_completed != 1 ||
                    ctx->replay_transition_m4096_output_validated != 0 || transition_bundle.m4096_output_from_device_syncs != 0 ||
                    transition_bundle.m4096_output_reads != 0 || transition_bundle.m4096_output_sha_calculations != 0 ||
                    transition_bundle.m4096_output_sha_validations != 0) {
                    throw std::runtime_error("M4096 no-readback boundary invariant failed");
                }
                log_runtime_snapshot("POST_WAIT_NO_READBACK", &*run);
                if (post_recreate_execute_wait_no_readback_context_recreate_mode) {
                    std::fprintf(stderr, "M4096_NO_READBACK_BOUNDARY m4096_submissions=%llu m4096_wait_completions=%llu m4096_output_from_device_syncs=%llu m4096_output_reads=%llu m4096_output_sha_calculations=%llu m4096_output_sha_validations=%llu generation1_submissions=%llu generation1_waits=%llu next_generation1_action=SECOND_CONTEXT_RECREATE\n",
                        static_cast<unsigned long long>(ctx->replay_transition_m4096_submitted), static_cast<unsigned long long>(ctx->replay_transition_m4096_wait_completed),
                        static_cast<unsigned long long>(transition_bundle.m4096_output_from_device_syncs), static_cast<unsigned long long>(transition_bundle.m4096_output_reads),
                        static_cast<unsigned long long>(transition_bundle.m4096_output_sha_calculations), static_cast<unsigned long long>(transition_bundle.m4096_output_sha_validations),
                        static_cast<unsigned long long>(runtime_trace.generation1_submissions), static_cast<unsigned long long>(runtime_trace.generation1_wait_completions));
                } else {
                    std::fprintf(stderr, "M4096_NO_READBACK_BOUNDARY m4096_submissions=%llu m4096_wait_completions=%llu "
                        "m4096_output_from_device_syncs=%llu m4096_output_reads=%llu m4096_output_sha_calculations=%llu m4096_output_sha_validations=%llu "
                        "generation1_submissions=%llu generation1_waits=%llu next_generation1_submission=ID8\n",
                        static_cast<unsigned long long>(ctx->replay_transition_m4096_submitted),
                        static_cast<unsigned long long>(ctx->replay_transition_m4096_wait_completed),
                        static_cast<unsigned long long>(transition_bundle.m4096_output_from_device_syncs),
                        static_cast<unsigned long long>(transition_bundle.m4096_output_reads),
                        static_cast<unsigned long long>(transition_bundle.m4096_output_sha_calculations),
                        static_cast<unsigned long long>(transition_bundle.m4096_output_sha_validations),
                        static_cast<unsigned long long>(runtime_trace.generation1_submissions),
                        static_cast<unsigned long long>(runtime_trace.generation1_wait_completions));
                }
                if (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) {
                    if (ctx->replay_transition_m4096_submitted != 1 || ctx->replay_transition_m4096_wait_completed != 1 ||
                        runtime_trace.generation1_submissions != 1 || runtime_trace.generation1_wait_completions != 1 ||
                        runtime_trace.generation1_output_sentinel_syncs != 0 || runtime_trace.generation1_output_syncs != 0 || runtime_trace.generation1_output_from_device_syncs != 0 ||
                        transition_bundle.m4096_output_from_device_syncs != 0 || transition_bundle.m4096_output_reads != 0 ||
                        transition_bundle.m4096_output_sha_calculations != 0 || transition_bundle.m4096_output_sha_validations != 0 ||
                        ctx->replay_transition_m4096_output_validated != 0 || runtime_trace.run_objects_created != 11 ||
                        runtime_trace.run_wait_calls_returned != 11 || runtime_trace.run_wait_completions != 11)
                        throw std::runtime_error("EXECUTE_WAIT_NO_READBACK_GUARD failed");
                    std::fprintf(stderr, "EXECUTE_WAIT_NO_READBACK_GUARD=PASS experimental_source_id=7 experimental_M=4096 generation1_kernel_submissions=1 generation1_runs=1 generation1_waits=1 generation1_output_sentinel_syncs=0 generation1_output_syncs=0 generation1_output_reads=0 generation1_output_validations=0 wait_completed=YES\n");
                    std::fprintf(stderr, "M4096_TEMP_EXECUTION_RESOURCE_SNAPSHOT source_id=7 M=4096 device_id=%p xclbin_id=%p xclbin_uuid=%s generation1_context=%p generation1_kernel=%p instr_bo=%p instr_map=%p instr_capacity=%zu q4_bo=%p q4_map=%p q4_capacity=%zu q8_bo=%p q8_map=%p q8_capacity=%zu out_bo=%p out_map=%p out_capacity=%zu separate_from_retained_M9216=YES\n",
                        static_cast<const void *>(ctx->device_context->xrt_device.get()), static_cast<const void *>(ctx->q4_m9216_xclbin.get()),
                        ctx->q4_m9216_xclbin->get_uuid().to_string().c_str(), static_cast<const void *>(transition_bundle.hw_context), static_cast<const void *>(transition_bundle.kernel),
                        static_cast<const void *>(transition_bundle.bo_instr.get()), transition_bundle.buf_instr, transition_bundle.bo_instr->size(),
                        static_cast<const void *>(transition_bundle.bo_q4.get()), transition_bundle.buf_q4, transition_bundle.bo_q4->size(),
                        static_cast<const void *>(transition_bundle.bo_q8.get()), transition_bundle.buf_q8, transition_bundle.bo_q8->size(),
                        static_cast<const void *>(transition_bundle.bo_out.get()), static_cast<const void *>(transition_bundle.buf_out), transition_bundle.bo_out->size());
                    if (post_recreate_execute_wait_release_temp_context_recreate_mode) {
                        const bool temp_before = run && transition_bundle.bo_instr && transition_bundle.bo_q4 && transition_bundle.bo_q8 && transition_bundle.bo_out &&
                            transition_bundle.buf_instr && transition_bundle.buf_q4 && transition_bundle.buf_q8 && transition_bundle.buf_out;
                        const bool retained_before = ep1r_retained_device == ctx->device_context->xrt_device.get() &&
                            ep1r_retained_xclbin == ctx->q4_m9216_xclbin.get() && ep1r_retained_context == ctx->q4_m9216_hw_context.get() &&
                            ep1r_retained_kernel == ctx->q4_m9216_kernel.get() && ep1r_retained_bo_instr == ctx->q4_m9216_bo_instr.get() &&
                            ep1r_retained_bo_q4 == ctx->q4_m9216_bo_q4.get() && ep1r_retained_bo_q8 == ctx->q4_m9216_bo_q8.get() &&
                            ep1r_retained_bo_out == ctx->q4_m9216_bo_out.get() && ep1r_retained_map_instr == ctx->q4_m9216_buf_instr &&
                            ep1r_retained_map_q4 == ctx->q4_m9216_buf_q4 && ep1r_retained_map_q8 == ctx->q4_m9216_buf_q8 &&
                            ep1r_retained_map_out == ctx->q4_m9216_buf_out && ep1r_retained_capacity_instr == ctx->q4_m9216_bo_instr->size() &&
                            ep1r_retained_capacity_q4 == ctx->q4_m9216_bo_q4->size() && ep1r_retained_capacity_q8 == ctx->q4_m9216_bo_q8->size() &&
                            ep1r_retained_capacity_out == ctx->q4_m9216_bo_out->size();
                        if (!temp_before || !retained_before) throw std::runtime_error("EP1R pre-release ownership/retained-bundle guard failed");
                        const auto runs_before_release = runtime_trace.run_objects_created;
                        const auto waits_before_release = runtime_trace.run_wait_calls_returned;
                        const auto submissions_before_release = runtime_trace.generation1_submissions;
                        const auto completed_before_release = runtime_trace.generation1_wait_completions;
                        const auto to_device_before_release = runtime_trace.m9216_input_syncs;
                        const auto from_device_before_release = runtime_trace.output_from_device_syncs;
                        std::fprintf(stderr, "BEFORE_COMPLETED_RUN_RELEASE completed_run_alive=1\n");
                        run = std::nullopt;
                        if (run) throw std::runtime_error("completed run lifetime did not end");
                        std::fprintf(stderr, "AFTER_COMPLETED_RUN_RELEASE completed_run_alive=0\n");
                        std::fprintf(stderr, "BEFORE_TEMP_M4096_RESOURCE_RELEASE temp_instr_bo_alive=1 temp_q4_bo_alive=1 temp_q8_bo_alive=1 temp_out_bo_alive=1 maps_alive=1\n");
                        transition_bundle.bo_instr.reset(); transition_bundle.bo_q4.reset();
                        transition_bundle.bo_q8.reset(); transition_bundle.bo_out.reset();
                        transition_bundle.buf_instr = transition_bundle.buf_q4 = transition_bundle.buf_q8 = transition_bundle.buf_out = nullptr;
                        const bool retained_after = ep1r_retained_device == ctx->device_context->xrt_device.get() &&
                            ep1r_retained_xclbin == ctx->q4_m9216_xclbin.get() && ep1r_retained_context == ctx->q4_m9216_hw_context.get() &&
                            ep1r_retained_kernel == ctx->q4_m9216_kernel.get() && ep1r_retained_bo_instr == ctx->q4_m9216_bo_instr.get() &&
                            ep1r_retained_bo_q4 == ctx->q4_m9216_bo_q4.get() && ep1r_retained_bo_q8 == ctx->q4_m9216_bo_q8.get() &&
                            ep1r_retained_bo_out == ctx->q4_m9216_bo_out.get() && ep1r_retained_map_instr == ctx->q4_m9216_buf_instr &&
                            ep1r_retained_map_q4 == ctx->q4_m9216_buf_q4 && ep1r_retained_map_q8 == ctx->q4_m9216_buf_q8 &&
                            ep1r_retained_map_out == ctx->q4_m9216_buf_out && ep1r_retained_capacity_instr == ctx->q4_m9216_bo_instr->size() &&
                            ep1r_retained_capacity_q4 == ctx->q4_m9216_bo_q4->size() && ep1r_retained_capacity_q8 == ctx->q4_m9216_bo_q8->size() &&
                            ep1r_retained_capacity_out == ctx->q4_m9216_bo_out->size();
                        if (run || transition_bundle.bo_instr || transition_bundle.bo_q4 || transition_bundle.bo_q8 || transition_bundle.bo_out ||
                            transition_bundle.buf_instr || transition_bundle.buf_q4 || transition_bundle.buf_q8 || transition_bundle.buf_out || !retained_after ||
                            runtime_trace.run_objects_created != runs_before_release || runtime_trace.run_wait_calls_returned != waits_before_release ||
                            runtime_trace.generation1_submissions != submissions_before_release || runtime_trace.generation1_wait_completions != completed_before_release ||
                            runtime_trace.m9216_input_syncs != to_device_before_release || runtime_trace.output_from_device_syncs != from_device_before_release)
                            throw std::runtime_error("EP1R released-resource, retained-M9216, or zero-extra-XRT guard failed");
                        std::fprintf(stderr, "AFTER_TEMP_M4096_RESOURCE_RELEASE temp_instr_bo_alive=0 temp_q4_bo_alive=0 temp_q8_bo_alive=0 temp_out_bo_alive=0 maps_alive=0 retained_m9216_instruction_alive=1 retained_m9216_q4_alive=1 retained_m9216_q8_alive=1 retained_m9216_output_alive=1 retained_m9216_maps_alive=1 retained_m9216_capacities_unchanged=1 device_same=1 xclbin_same=1 context_kernel_same=1 POST_WAIT_RESOURCE_RELEASE_GUARD=PASS intervening_XRT_operations=0\n");
                    }
                    else if (post_recreate_execute_wait_release_run_context_recreate_mode) {
                        const bool temp_before = run && transition_bundle.bo_instr && transition_bundle.bo_q4 && transition_bundle.bo_q8 && transition_bundle.bo_out &&
                            transition_bundle.buf_instr && transition_bundle.buf_q4 && transition_bundle.buf_q8 && transition_bundle.buf_out;
                        const bool retained_before = ep1r_retained_device == ctx->device_context->xrt_device.get() && ep1r_retained_xclbin == ctx->q4_m9216_xclbin.get() &&
                            ep1r_retained_context == ctx->q4_m9216_hw_context.get() && ep1r_retained_kernel == ctx->q4_m9216_kernel.get() &&
                            ep1r_retained_bo_instr == ctx->q4_m9216_bo_instr.get() && ep1r_retained_bo_q4 == ctx->q4_m9216_bo_q4.get() &&
                            ep1r_retained_bo_q8 == ctx->q4_m9216_bo_q8.get() && ep1r_retained_bo_out == ctx->q4_m9216_bo_out.get() &&
                            ep1r_retained_map_instr == ctx->q4_m9216_buf_instr && ep1r_retained_map_q4 == ctx->q4_m9216_buf_q4 &&
                            ep1r_retained_map_q8 == ctx->q4_m9216_buf_q8 && ep1r_retained_map_out == ctx->q4_m9216_buf_out &&
                            ep1r_retained_capacity_instr == ctx->q4_m9216_bo_instr->size() && ep1r_retained_capacity_q4 == ctx->q4_m9216_bo_q4->size() &&
                            ep1r_retained_capacity_q8 == ctx->q4_m9216_bo_q8->size() && ep1r_retained_capacity_out == ctx->q4_m9216_bo_out->size();
                        if (!temp_before || !retained_before) throw std::runtime_error("XDNA_RESOURCE_RELEASE pre-release ownership/retained-bundle guard failed");
                        resource_release_temp_bo_instr = transition_bundle.bo_instr.get(); resource_release_temp_bo_q4 = transition_bundle.bo_q4.get();
                        resource_release_temp_bo_q8 = transition_bundle.bo_q8.get(); resource_release_temp_bo_out = transition_bundle.bo_out.get();
                        resource_release_temp_map_instr = transition_bundle.buf_instr; resource_release_temp_map_q4 = transition_bundle.buf_q4;
                        resource_release_temp_map_q8 = transition_bundle.buf_q8; resource_release_temp_map_out = transition_bundle.buf_out;
                        resource_release_temp_capacity_instr = transition_bundle.bo_instr->size(); resource_release_temp_capacity_q4 = transition_bundle.bo_q4->size();
                        resource_release_temp_capacity_q8 = transition_bundle.bo_q8->size(); resource_release_temp_capacity_out = transition_bundle.bo_out->size();
                        const auto runs_before_release = runtime_trace.run_objects_created;
                        const auto waits_before_release = runtime_trace.run_wait_calls_returned;
                        const auto submissions_before_release = runtime_trace.generation1_submissions;
                        const auto completed_before_release = runtime_trace.generation1_wait_completions;
                        const auto to_device_before_release = runtime_trace.m9216_input_syncs;
                        const auto from_device_before_release = runtime_trace.output_from_device_syncs;
                        const auto instr_syncs_before_release = runtime_trace.generation1_instruction_stage_syncs;
                        const auto q4_syncs_before_release = runtime_trace.generation1_q4_syncs;
                        const auto q8_syncs_before_release = runtime_trace.generation1_q8_syncs;
                        const auto sentinel_syncs_before_release = runtime_trace.generation1_output_sentinel_syncs;
                        const auto output_syncs_before_release = runtime_trace.generation1_output_syncs;
                        const auto output_from_device_syncs_before_release = runtime_trace.generation1_output_from_device_syncs;
                        std::fprintf(stderr, "BEFORE_COMPLETED_RUN_RELEASE completed_run_alive=1 temp_instr_bo_alive=1 temp_q4_bo_alive=1 temp_q8_bo_alive=1 temp_out_bo_alive=1 maps_alive=1\n");
                        run = std::nullopt;
                        const bool temp_after = !run && transition_bundle.bo_instr.get() == resource_release_temp_bo_instr && transition_bundle.bo_q4.get() == resource_release_temp_bo_q4 &&
                            transition_bundle.bo_q8.get() == resource_release_temp_bo_q8 && transition_bundle.bo_out.get() == resource_release_temp_bo_out &&
                            transition_bundle.buf_instr == resource_release_temp_map_instr && transition_bundle.buf_q4 == resource_release_temp_map_q4 &&
                            transition_bundle.buf_q8 == resource_release_temp_map_q8 && transition_bundle.buf_out == resource_release_temp_map_out &&
                            transition_bundle.bo_instr->size() == resource_release_temp_capacity_instr && transition_bundle.bo_q4->size() == resource_release_temp_capacity_q4 &&
                            transition_bundle.bo_q8->size() == resource_release_temp_capacity_q8 && transition_bundle.bo_out->size() == resource_release_temp_capacity_out;
                        const bool retained_after = retained_before && ep1r_retained_device == ctx->device_context->xrt_device.get() &&
                            ep1r_retained_xclbin == ctx->q4_m9216_xclbin.get() && ep1r_retained_context == ctx->q4_m9216_hw_context.get() &&
                            ep1r_retained_kernel == ctx->q4_m9216_kernel.get() && ep1r_retained_bo_instr == ctx->q4_m9216_bo_instr.get() &&
                            ep1r_retained_bo_q4 == ctx->q4_m9216_bo_q4.get() && ep1r_retained_bo_q8 == ctx->q4_m9216_bo_q8.get() &&
                            ep1r_retained_bo_out == ctx->q4_m9216_bo_out.get() && ep1r_retained_map_instr == ctx->q4_m9216_buf_instr &&
                            ep1r_retained_map_q4 == ctx->q4_m9216_buf_q4 && ep1r_retained_map_q8 == ctx->q4_m9216_buf_q8 &&
                            ep1r_retained_map_out == ctx->q4_m9216_buf_out && ep1r_retained_capacity_instr == ctx->q4_m9216_bo_instr->size() &&
                            ep1r_retained_capacity_q4 == ctx->q4_m9216_bo_q4->size() && ep1r_retained_capacity_q8 == ctx->q4_m9216_bo_q8->size() &&
                            ep1r_retained_capacity_out == ctx->q4_m9216_bo_out->size();
                        if (!temp_after || !retained_after || runtime_trace.run_objects_created != runs_before_release ||
                            runtime_trace.run_wait_calls_returned != waits_before_release || runtime_trace.generation1_submissions != submissions_before_release ||
                            runtime_trace.generation1_wait_completions != completed_before_release || runtime_trace.m9216_input_syncs != to_device_before_release ||
                            runtime_trace.output_from_device_syncs != from_device_before_release || runtime_trace.generation1_instruction_stage_syncs != instr_syncs_before_release ||
                            runtime_trace.generation1_q4_syncs != q4_syncs_before_release || runtime_trace.generation1_q8_syncs != q8_syncs_before_release ||
                            runtime_trace.generation1_output_sentinel_syncs != sentinel_syncs_before_release || runtime_trace.generation1_output_syncs != output_syncs_before_release ||
                            runtime_trace.generation1_output_from_device_syncs != output_from_device_syncs_before_release)
                            throw std::runtime_error("XDNA_RESOURCE_RELEASE run-only release, retained ownership, or zero-extra-XRT guard failed");
                        resource_release_run_release_guard_passed = true;
                        std::fprintf(stderr, "AFTER_COMPLETED_RUN_RELEASE completed_run_alive=0 temp_instr_bo_alive=1 temp_q4_bo_alive=1 temp_q8_bo_alive=1 temp_out_bo_alive=1 maps_alive=1 temp_bo_map_identities_unchanged=1 capacities_unchanged=1 retained_m9216_same=1 extra_submissions=0 extra_waits=0 extra_to_device_syncs=0 extra_from_device_syncs=0 RUN_ONLY_RELEASE_GUARD=PASS intervening_XRT_operations=0\n");
                    }
                    std::fprintf(stderr, "TRANSITION_STAGE=SECOND_CONTEXT_RECREATE_AFTER_EXECUTE_WAIT_NO_READBACK previous_experimental_source=7 generation1_submissions=1 generation1_waits=1 generation1_output_syncs=0\n");
                    std::fflush(stderr);
                    const bool allow_live_run_control = post_recreate_execute_wait_no_readback_context_recreate_mode;
                    ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, false, true, &run, allow_live_run_control);
                    if (runtime_trace.m9216_hw_context_generation != 2 || runtime_trace.m9216_hw_context_recreate_count != 2 ||
                        runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_kernel_recreate_count != 2 ||
                        runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                        runtime_trace.generation1_submissions != 1 || runtime_trace.generation1_wait_completions != 1 ||
                        runtime_trace.generation1_output_syncs != 0 || ctx->replay_transition_m4096_output_validated != 0 ||
                        ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                        throw std::runtime_error("execute/wait no-readback second context-recreate generation invariant failed");
                    if (post_recreate_execute_wait_release_run_context_recreate_mode) {
                        const bool temp_preserved = !run && transition_bundle.bo_instr && transition_bundle.bo_q4 && transition_bundle.bo_q8 && transition_bundle.bo_out &&
                            transition_bundle.buf_instr && transition_bundle.buf_q4 && transition_bundle.buf_q8 && transition_bundle.buf_out &&
                            transition_bundle.bo_instr.get() == resource_release_temp_bo_instr && transition_bundle.bo_q4.get() == resource_release_temp_bo_q4 &&
                            transition_bundle.bo_q8.get() == resource_release_temp_bo_q8 && transition_bundle.bo_out.get() == resource_release_temp_bo_out &&
                            transition_bundle.buf_instr == resource_release_temp_map_instr && transition_bundle.buf_q4 == resource_release_temp_map_q4 &&
                            transition_bundle.buf_q8 == resource_release_temp_map_q8 && transition_bundle.buf_out == resource_release_temp_map_out &&
                            transition_bundle.bo_instr->size() == resource_release_temp_capacity_instr && transition_bundle.bo_q4->size() == resource_release_temp_capacity_q4 &&
                            transition_bundle.bo_q8->size() == resource_release_temp_capacity_q8 && transition_bundle.bo_out->size() == resource_release_temp_capacity_out;
                        const bool retained_bos_preserved = ep1r_retained_device == ctx->device_context->xrt_device.get() &&
                            ep1r_retained_xclbin == ctx->q4_m9216_xclbin.get() && ep1r_retained_bo_instr == ctx->q4_m9216_bo_instr.get() &&
                            ep1r_retained_bo_q4 == ctx->q4_m9216_bo_q4.get() && ep1r_retained_bo_q8 == ctx->q4_m9216_bo_q8.get() &&
                            ep1r_retained_bo_out == ctx->q4_m9216_bo_out.get() && ep1r_retained_map_instr == ctx->q4_m9216_buf_instr &&
                            ep1r_retained_map_q4 == ctx->q4_m9216_buf_q4 && ep1r_retained_map_q8 == ctx->q4_m9216_buf_q8 &&
                            ep1r_retained_map_out == ctx->q4_m9216_buf_out && ep1r_retained_capacity_instr == ctx->q4_m9216_bo_instr->size() &&
                            ep1r_retained_capacity_q4 == ctx->q4_m9216_bo_q4->size() && ep1r_retained_capacity_q8 == ctx->q4_m9216_bo_q8->size() &&
                            ep1r_retained_capacity_out == ctx->q4_m9216_bo_out->size();
                        if (!resource_release_run_release_guard_passed || !temp_preserved || !retained_bos_preserved)
                            throw std::runtime_error("XDNA_RESOURCE_RELEASE temporary M4096 or retained M9216 resources changed across recreation");
                        resource_release_temp_resources_preserved_through_recreate = true;
                        std::fprintf(stderr, "RUN_ONLY_RELEASE_RECREATION_PRESERVATION=PASS temp_bo_owners=UNCHANGED map_aliases=UNCHANGED capacities=UNCHANGED retained_M9216_BOs_maps=SAME run_alive=0\n");
                    }
                    std::fprintf(stderr, "EXECUTE_WAIT_NO_READBACK_SECOND_RECREATE=PASS context_generation=2 kernel_generation=2 BO_generation=1 recreate_counts=2/2/1 retained_M9216_resources=SAME\n");
                }
            } else {
            bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            ++runtime_trace.output_from_device_syncs;
            if (is_post_recreate_m4096_control) ++transition_bundle.m4096_output_from_device_syncs;
            if (is_m9216) ++runtime_trace.m9216_output_syncs;
            if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_output_syncs;
            if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_output_from_device_syncs;
            if (is_post_recreate_id7_execution) ++transition_bundle.id7_output_syncs;
            if (is_post_recreate_m4096_control) {
                ++transition_bundle.m4096_output_reads;
                ++transition_bundle.m4096_output_sha_calculations;
            }
            const bool output_memory_safe = buf_out != nullptr && output_bytes <= output_bo_capacity;
            const std::string output_sha = output_memory_safe ? ggml_backend_xdna_capture_hash(buf_out, output_bytes) : std::string();
            if (output_memory_safe) ++runtime_trace.output_mapped_read_count;
            if (!target) {
                if (output_bytes != r.at("output_bytes").get<size_t>() || output_sha != r.at("output_sha256").get<std::string>()) {
                    if (is_historical_record) {
                        const size_t expected_size = r.at("output_bytes").get<size_t>();
                        const std::string expected_sha = r.at("output_sha256").get<std::string>();
                        std::string first_bytes, last_bytes;
                        if (output_memory_safe) {
                            const auto * output_data = static_cast<const unsigned char *>(buf_out);
                            const size_t window = std::min<size_t>(32, output_bytes);
                            auto hex_window = [](const unsigned char * data, size_t offset, size_t count) {
                                static constexpr char hex[] = "0123456789abcdef";
                                std::string result;
                                result.reserve(count * 2);
                                for (size_t i = 0; i < count; ++i) {
                                    result.push_back(hex[data[offset + i] >> 4]);
                                    result.push_back(hex[data[offset + i] & 0x0f]);
                                }
                                return result;
                            };
                            first_bytes = hex_window(output_data, 0, window);
                            last_bytes = hex_window(output_data, output_bytes - window, window);
                            ++runtime_trace.output_mapped_read_count;
                        }
                        std::fprintf(stderr,
                            "XDNA_CAPTURE_REPLAY HISTORICAL OUTPUT MISMATCH\nsource_id=%llu\nhistory_ordinal=%llu\nnode=%s\nM=%zu\n"
                            "expected_size=%zu\nactual_size=%zu\nexpected_sha=%s\nactual_sha=%s\n"
                            "output_bo_identity=%p\noutput_map_identity=%p\noutput_capacity=%zu\nexpected_output_bytes=%zu\nactual_validation_bytes=%zu\n"
                            "FROM_DEVICE_sync_count=%llu\nmapped_read_count=%llu\ncontext_generation=%llu\nkernel_generation=%llu\nBO_generation=%llu\n",
                            static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()),
                            static_cast<unsigned long long>(is_preprobe_history ? ctx->replay_prefix_completed : ctx->replay_kernels_submitted),
                            node.c_str(), m, expected_size, output_bytes, expected_sha.c_str(),
                            output_memory_safe ? output_sha.c_str() : "NOT_COMPUTED",
                            static_cast<const void *>(bo_out), static_cast<const void *>(buf_out), output_bo_capacity, expected_size, output_bytes,
                            static_cast<unsigned long long>(runtime_trace.output_from_device_syncs),
                            static_cast<unsigned long long>(runtime_trace.output_mapped_read_count),
                            static_cast<unsigned long long>(runtime_trace.m9216_hw_context_generation),
                            static_cast<unsigned long long>(runtime_trace.m9216_kernel_generation),
                            static_cast<unsigned long long>(runtime_trace.m9216_bo_bundle_generation));
                        if (!output_memory_safe) std::fprintf(stderr, "actual_sha=NOT_COMPUTED\nreason=%s\nBYTE_FINGERPRINT=NOT_AVAILABLE\n",
                            buf_out == nullptr ? "output_map_missing" : "actual_validation_bytes_exceed_output_capacity");
                        else std::fprintf(stderr, "BYTE_FINGERPRINT=AVAILABLE first_32_bytes_hex=%s last_32_bytes_hex=%s\n", first_bytes.c_str(), last_bytes.c_str());
                        std::fflush(stderr);
                    }
                    if (is_probe || is_post_recreate_control || is_post_recreate_m4096_control || is_post_transition_id8 || is_same_shape_id8_second || is_post_context_recreate_id8) std::fprintf(stderr, "%s OUTPUT SHA MISMATCH source_id=%llu expected=%s actual=%s\n",
                        is_post_context_recreate_id8 ? "POST-CONTEXT-RECREATE ID8" : (is_same_shape_id8_second ? "POST_SAME_SHAPE_SECOND_ID8" : (is_post_recreate_control ? "POST-RECREATE CONTROL" : (is_post_recreate_m4096_control ? "POST-RECREATE M4096 CONTROL" : (is_post_transition_id8 ? (ctx->replay_post_recreate_program_only_mode ? "POST-PROGRAM-ONLY ID8" : "POST-TRANSITION ID8") : "PROBE")))),
                        static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()),
                        r.at("output_sha256").get<std::string>().c_str(), output_sha.c_str());
                    throw std::runtime_error("historical output size/SHA mismatch at submission " + std::to_string(r.at("submission_id").get<uint64_t>()));
                }
                if (is_post_recreate_m4096_control) ++transition_bundle.m4096_output_sha_validations;
                ++ctx->replay_outputs_validated;
                if (runtime_trace.m9216_hw_context_generation == 1) ++runtime_trace.generation1_exact_outputs;
                if (is_post_recreate_id7_execution) {
                    ++transition_bundle.id7_validations;
                    ++transition_bundle.id7_exact_outputs;
                    if (is_post_recreate_id7_first) {
                        transition_bundle.id7_first_hw_context = hw_context; transition_bundle.id7_first_kernel = kernel;
                        transition_bundle.id7_first_bo_instr = bo_instr; transition_bundle.id7_first_bo_q4 = bo_q4;
                        transition_bundle.id7_first_bo_q8 = bo_q8; transition_bundle.id7_first_bo_out = bo_out;
                        transition_bundle.id7_first_map_instr = buf_instr; transition_bundle.id7_first_map_q4 = buf_q4;
                        transition_bundle.id7_first_map_q8 = buf_q8; transition_bundle.id7_first_map_out = buf_out;
                        transition_bundle.id7_first_snapshot_valid = true;
                    }
                }
                    if (is_post_recreate_id7_first) std::fprintf(stderr, "FIRST_ID7_COMPLETED=1 FIRST_ID7_VALIDATED=1 FIRST_ID7_EXACT=1 generation1_exact_outputs=1\n");
                    if (is_same_shape_id7_second) std::fprintf(stderr, "SECOND_ID7_OUTPUT_SHA_MATCH source_id=7 generation_local_ordinal=2 same_shape_as_previous=YES same_instruction_image_as_previous=YES sha256=%s\n", output_sha.c_str());
                if (is_preprobe_history) ++ctx->replay_prefix_outputs_validated;
                if (is_injected) std::fprintf(stderr, "INJECTED HISTORY OUTPUT SHA MATCH source_id=%llu sha256=%s\n",
                    static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()), output_sha.c_str());
                if (is_probe) {
                    ++ctx->replay_probe_output_validated;
                    std::fprintf(stderr, "PROBE OUTPUT SHA MATCH source_id=%llu sha256=%s\n",
                        static_cast<unsigned long long>(r.at("submission_id").get<uint64_t>()), output_sha.c_str());
                }
                if (is_post_recreate_control) {
                    ++ctx->replay_post_recreate_control_output_validated;
                    std::fprintf(stderr, "POST-RECREATE CONTROL OUTPUT SHA MATCH source_id=8 sha256=%s\n", output_sha.c_str());
                    if (is_same_shape_id8_first) std::fprintf(stderr, "FIRST_ID8_COMPLETED=1 FIRST_ID8_VALIDATED=1 FIRST_ID8_EXACT=1 generation1_exact_outputs=1\n");
                }
                if (is_post_recreate_m4096_control) {
                    ++ctx->replay_transition_m4096_output_validated;
                    std::fprintf(stderr, "POST_RECREATE M4096 OUTPUT SHA MATCH source_id=7 sha256=%s\n", output_sha.c_str());
                }
                if (is_post_transition_id8) {
                    ++ctx->replay_transition_id8_output_validated;
                    std::fprintf(stderr, ctx->replay_post_recreate_no_readback_mode ? "POST-M4096-NO-READBACK ID8 OUTPUT SHA MATCH source_id=8 sha256=%s\n" : (ctx->replay_post_recreate_program_only_mode ? "POST-PROGRAM-ONLY ID8 OUTPUT SHA MATCH source_id=8 sha256=%s\n" : "POST-TRANSITION ID8 OUTPUT SHA MATCH source_id=8 sha256=%s\n"), output_sha.c_str());
                }
                if (is_post_context_recreate_id8) std::fprintf(stderr, "POST_CONTEXT_RECREATE ID8 OUTPUT SHA MATCH source_id=8 experiment_ordinal=2 context_generation=2 context_generation_local_ordinal=1 sha256=%s\n", output_sha.c_str());
                if (is_same_shape_id8_second) std::fprintf(stderr, "SECOND_ID8_OUTPUT_SHA MATCH source_id=8 generation_local_ordinal=2 same_shape_as_previous=YES same_instruction_image_as_previous=YES sha256=%s\n", output_sha.c_str());
                if (is_post_recreate_m4096_control) {
                    if (runtime_trace.generation1_execution_ordinal != 1 || runtime_trace.generation1_submissions != 1 ||
                        runtime_trace.generation1_wait_completions != 1 || runtime_trace.generation1_output_syncs != 1 ||
                        ctx->replay_transition_m4096_submitted != 1 || ctx->replay_transition_m4096_wait_completed != 1 ||
                        ctx->replay_transition_m4096_output_validated != 1) {
                        throw std::runtime_error("M4096 exactness gate failed; refusing transition ID8");
                    }
                    std::fprintf(stderr,
                        "TRANSITION_STAGE=M4096_TO_M9216 previous_source_id=7 previous_M=4096 previous_instruction_sha=%s "
                        "next_source_id=8 next_M=9216 next_instruction_sha=3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3 "
                        "generation1_submissions=1 generation1_waits=1 generation1_exact_outputs=1 no_intervening_submission=YES\n",
                        r.at("instruction_sha256").get<std::string>().c_str());
                }
            } else {
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY TARGET output_bytes=%zu output_sha256=%s\n", output_bytes, output_sha.c_str());
                if (is_probe) std::fprintf(stderr, "PROBE OUTPUT SHA=N/A (SEMANTIC TARGET)\n");
            }
            log_runtime_snapshot("POST_OUTPUT", run ? &*run : nullptr);
            if (is_post_recreate_id7_first && ctx->replay_post_recreate_context_recreate_id8_mode) {
                if (transition_bundle.id7_submissions != 1 || transition_bundle.id7_waits != 1 || transition_bundle.id7_output_syncs != 1 ||
                    transition_bundle.id7_validations != 1 || transition_bundle.id7_exact_outputs != 1 ||
                    runtime_trace.experiment_submission_ordinal != 1 || runtime_trace.context_generation_submission_ordinal != 1 ||
                    ctx->replay_kernels_submitted != 11 || ctx->replay_waits_completed != 11 || ctx->replay_outputs_validated != 11)
                    throw std::runtime_error("XDNA_RECREATE_GUARD ID7 exactness gate failed before context/kernel recreation");
                ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, true);
                std::fprintf(stderr, "TRANSITION_STAGE=POST_ID7_CONTEXT_RECREATE_DONE context_generation=2 kernel_generation=2 bo_generation=1 context_recreate_count=2 kernel_recreate_count=2 bo_recreate_count=1 experiment_ordinal=1 context_generation_1_ordinal=1 next_context_generation_2_ordinal=1 no_intervening_submission=YES id7_exact=YES\n");
            }
            if (is_post_id8_context_recreate_only) {
                if (ctx->replay_post_recreate_control_submitted != 1 || ctx->replay_post_recreate_control_wait_completed != 1 ||
                    ctx->replay_post_recreate_control_output_validated != 1 || runtime_trace.m9216_execution_ordinal != 8 ||
                    runtime_trace.m9216_submissions != 8 || runtime_trace.m9216_wait_completions != 8 || runtime_trace.m9216_output_syncs != 8 ||
                    runtime_trace.experiment_submission_ordinal != 1 || runtime_trace.context_generation_submission_ordinal != 1 ||
                    runtime_trace.generation1_execution_ordinal != 1 || runtime_trace.generation1_submissions != 1 ||
                    runtime_trace.generation1_wait_completions != 1 || runtime_trace.generation1_output_syncs != 1 ||
                    runtime_trace.generation1_exact_outputs != 1 || ctx->replay_kernels_submitted != 11 ||
                    ctx->replay_waits_completed != 11 || ctx->replay_outputs_validated != 11 || ctx->exact_history_replay_failed)
                    throw std::runtime_error("ID8 exactness gate failed before lifecycle-only context recreation");
                std::fprintf(stderr, "FIRST_ID8_COMPLETED=1 FIRST_ID8_WAIT_COMPLETED=1 FIRST_ID8_OUTPUT_SYNC=1 FIRST_ID8_VALIDATED=1 FIRST_ID8_EXACT=1 generation1_exact_outputs=1\n");
                std::fprintf(stderr, "TRANSITION_STAGE=M9216_TO_CONTEXT_RECREATE_CONTROL previous_source_id=8 previous_M=9216 context_recreate_only=YES no_post_recreate_submission=YES\n");
                std::fflush(stderr);
                ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, true);
                std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-ID8 CONTEXT-RECREATE TERMINAL context_generation=2 kernel_generation=2 BO_generation=1 no_post_recreate_submission=YES result=PASS\n");
                std::fflush(stderr);
            }
            }
            if (is_m9216) {
                runtime_trace.have_previous_m9216 = true;
                runtime_trace.previous_m9216_bo_bundle_generation = runtime_trace.m9216_bo_bundle_generation;
                runtime_trace.previous_m9216_kernel_generation = runtime_trace.m9216_kernel_generation;
                runtime_trace.previous_m9216_hw_context_generation = runtime_trace.m9216_hw_context_generation;
                runtime_trace.previous_hw_context = hw_context;
                runtime_trace.previous_kernel = kernel;
                runtime_trace.previous_bo_instr = bo_instr; runtime_trace.previous_bo_q4 = bo_q4;
                runtime_trace.previous_bo_q8 = bo_q8; runtime_trace.previous_bo_out = bo_out;
                runtime_trace.previous_map_instr = buf_instr; runtime_trace.previous_map_q4 = buf_q4;
                runtime_trace.previous_map_q8 = buf_q8; runtime_trace.previous_map_out = buf_out;
            }
        }
        if (post_recreate_program_only_context_recreate_mode) {
            if (ctx->replay_prefix_planned != 10 || ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->exact_history_replay_failed || ctx->replay_kernels_submitted != 10 || ctx->replay_waits_completed != 10 ||
                ctx->replay_outputs_validated != 10 || runtime_trace.execution_ordinal != 10 || runtime_trace.m9216_execution_ordinal != 7 ||
                runtime_trace.m9216_submissions != 7 || runtime_trace.m9216_wait_completions != 7 || runtime_trace.m9216_output_syncs != 7 ||
                runtime_trace.m9216_hw_context_generation != 0 || runtime_trace.m9216_kernel_generation != 0 || runtime_trace.m9216_bo_bundle_generation != 0 ||
                runtime_trace.experiment_submission_ordinal != 0 || runtime_trace.generation1_submissions != 0 ||
                runtime_trace.generation1_wait_completions != 0 || runtime_trace.generation1_output_syncs != 0 ||
                ctx->replay_post_recreate_control_submitted != 0 || ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("program-only second-recreate initial full-bundle guard failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 RECREATION BOUNDARY kind=FULL_RUNTIME_BUNDLE history_completed=10 history_outputs_validated=10 m9216_history_submissions=7 m9216_history_wait_completions=7 m9216_history_output_syncs=7\n");
            ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace);
            if (runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_hw_context_recreate_count != 1 ||
                runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_kernel_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 0 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 0 ||
                runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0 ||
                runtime_trace.generation1_output_syncs != 0 || runtime_trace.generation1_exact_outputs != 0 ||
                runtime_trace.experiment_submission_ordinal != 0 || ctx->replay_kernels_submitted != 10 ||
                ctx->replay_waits_completed != 10 || ctx->replay_outputs_validated != 10 || ctx->replay_probe_submitted != 0 ||
                ctx->replay_target_records != 0) throw std::runtime_error("program-only second-recreate full-bundle guard failed");
            program_only_init_syncs_before = runtime_trace.generation1_instruction_init_syncs;
            program_only_stage_syncs_before = runtime_trace.generation1_instruction_stage_syncs;
            std::fprintf(stderr, "TRANSITION_STAGE=PROGRAM_ONLY_M4096_TO_SECOND_CONTEXT_RECREATE source_id=7 M=4096 instruction_sha=0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904 context_generation=1 kernel_generation=1 bo_generation=1 generation1_kernel_submissions=0 generation1_runs=0 generation1_waits=0 generation1_output_syncs=0\n");
            ggml_backend_xdna_replay_program_m4096_instruction_only(ctx, runtime_trace, ctx->q4_m4096, root, records[6], &program_only_bundle);
            ++ctx->replay_program_only_stages;
            if (ctx->replay_program_only_stages != 1 || program_only_bundle.instruction_init_sync_count != 1 ||
                program_only_bundle.instruction_stage_sync_count != 1 || runtime_trace.generation1_instruction_init_syncs != program_only_init_syncs_before + 1 ||
                runtime_trace.generation1_instruction_stage_syncs != program_only_stage_syncs_before + 1 ||
                runtime_trace.generation1_q4_syncs != 0 || runtime_trace.generation1_q8_syncs != 0 ||
                runtime_trace.generation1_output_sentinel_syncs != 0 || runtime_trace.generation1_output_syncs != 0 ||
                runtime_trace.generation1_output_from_device_syncs != 0 || runtime_trace.generation1_submissions != 0 ||
                runtime_trace.generation1_wait_completions != 0 || runtime_trace.run_objects_created != 10 ||
                runtime_trace.run_wait_completions != 10 || ctx->replay_kernels_submitted != 10 || ctx->replay_waits_completed != 10 ||
                ctx->replay_outputs_validated != 10 || runtime_trace.experiment_submission_ordinal != 0 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("program-only M4096 stage execution/resource guard failed");
            std::fprintf(stderr, "PROGRAM_ONLY_RESOURCE_DELTAS source_id=7 M=4096 instruction_bo_creates=1 instruction_bo_maps=1 instruction_writes=2 instruction_to_device_syncs=2 q4_syncs=0 q8_syncs=0 output_from_device_syncs=0\n");
            std::fprintf(stderr, "PROGRAM_ONLY_COMPLETED=1 KERNEL_SUBMITTED=0 RUN_CREATED=0 WAIT_CALLED=0 generation1_kernel_submissions=0 generation1_runs=0 generation1_waits=0 generation1_output_syncs=0\n");
            std::fprintf(stderr, "PROGRAM_ONLY_CONTEXT_RECREATE_GUARD=PASS source_id=7 generation1_kernel_submissions=0 generation1_runs=0 generation1_waits=0 generation1_output_syncs=0 probe11=0 target13=0\n");
            std::fprintf(stderr, "TRANSITION_STAGE=SECOND_CONTEXT_RECREATE_AFTER_PROGRAM_ONLY previous_experimental_source=NONE second_recreation_requested=YES\n");
            std::fflush(stderr);
            ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, true);
            if (runtime_trace.m9216_hw_context_generation != 2 || runtime_trace.m9216_hw_context_recreate_count != 2 ||
                runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_kernel_recreate_count != 2 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0 ||
                runtime_trace.generation1_output_syncs != 0 || runtime_trace.experiment_submission_ordinal != 0 ||
                ctx->replay_kernels_submitted != 10 || ctx->replay_waits_completed != 10 || ctx->replay_outputs_validated != 10 ||
                runtime_trace.run_objects_created != 10 || runtime_trace.run_wait_completions != 10 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("program-only second context-recreate terminal invariant failed");
            std::fprintf(stderr, "<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nPROGRAM_ONLY_SECOND_CONTEXT_RECREATE_TERMINAL history_completed=10 program_only_completed=1 generation1_kernel_submissions=0 generation1_runs=0 generation1_waits=0 generation1_output_syncs=0 context_generation=2 kernel_generation=2 BO_generation=1 recreate_counts=2/2/1 experiment_submissions=0 post_second_recreation_submissions=0 post_second_recreation_runs=0 post_second_recreation_waits=0 probe11=0 target13=0 result=PASS\n");
            std::fflush(stderr);
        } else if (post_recreate_double_recreate_no_execution_mode) {
            if (ctx->replay_prefix_planned != 10 || ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->exact_history_replay_failed || ctx->replay_kernels_submitted != 10 || ctx->replay_waits_completed != 10 ||
                ctx->replay_outputs_validated != 10 || runtime_trace.execution_ordinal != 10 || runtime_trace.m9216_execution_ordinal != 7 ||
                runtime_trace.m9216_submissions != 7 || runtime_trace.m9216_wait_completions != 7 || runtime_trace.m9216_output_syncs != 7 ||
                runtime_trace.m9216_hw_context_generation != 0 || runtime_trace.m9216_kernel_generation != 0 || runtime_trace.m9216_bo_bundle_generation != 0 ||
                runtime_trace.experiment_submission_ordinal != 0 || runtime_trace.generation1_submissions != 0 ||
                runtime_trace.generation1_wait_completions != 0 || runtime_trace.generation1_output_syncs != 0 ||
                ctx->replay_post_recreate_control_submitted != 0 || ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("double-recreate no-execution initial full-bundle guard failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M9216 RECREATION BOUNDARY kind=FULL_RUNTIME_BUNDLE history_completed=10 history_outputs_validated=10 m9216_history_submissions=7 m9216_history_wait_completions=7 m9216_history_output_syncs=7\n");
            ggml_backend_xdna_replay_recreate_m9216_context_and_bo_bundle(ctx, &runtime_trace);
            if (runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_hw_context_recreate_count != 1 ||
                runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_kernel_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 0 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 0 ||
                runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0 ||
                runtime_trace.generation1_output_syncs != 0 || runtime_trace.generation1_exact_outputs != 0 ||
                runtime_trace.experiment_submission_ordinal != 0 || ctx->replay_kernels_submitted != 10 ||
                ctx->replay_waits_completed != 10 || ctx->replay_outputs_validated != 10 || ctx->replay_post_recreate_control_submitted != 0 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("double-recreate no-execution generation-1 zero-counter guard failed");
            std::fprintf(stderr, "TRANSITION_STAGE=SECOND_CONTEXT_RECREATE_NO_EXECUTION generation1_execution_count=0 previous_experimental_source=NONE second_recreation_requested=YES\n");
            std::fprintf(stderr, "NO_EXECUTION_RECREATE_GUARD=PASS context_generation=1 kernel_generation=1 BO_generation=1 generation1_submissions=0 generation1_waits=0 generation1_output_syncs=0 no_experiment_source=YES ID7_experiment=NO ID8_experiment=NO probe11=NO target13=NO\n");
            std::fflush(stderr);
            ggml_backend_xdna_replay_recreate_m9216_hw_context(ctx, &runtime_trace, false, false, true);
            if (runtime_trace.m9216_hw_context_generation != 2 || runtime_trace.m9216_hw_context_recreate_count != 2 ||
                runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_kernel_recreate_count != 2 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0 ||
                runtime_trace.generation1_output_syncs != 0 || runtime_trace.experiment_submission_ordinal != 0 ||
                ctx->replay_kernels_submitted != 10 || ctx->replay_waits_completed != 10 || ctx->replay_outputs_validated != 10 ||
                runtime_trace.run_objects_created != 10 || runtime_trace.run_wait_completions != 10 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("double-recreate no-execution terminal invariant failed");
            std::fprintf(stderr, "<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nSECOND_CONTEXT_RECREATE_NO_EXECUTION_TERMINAL context_generation=2 kernel_generation=2 BO_generation=1 context_recreate_count=2 kernel_recreate_count=2 BO_recreate_count=1 generation1_submissions=0 generation1_waits=0 generation1_output_syncs=0 experiment_submissions=0 post_second_recreation_submissions=0 post_second_recreation_runs=0 post_second_recreation_waits=0 result=PASS\n");
            std::fflush(stderr);
        }
        const uint64_t expected_replay_outputs_validated = (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) ? 10u : ctx->replay_historical_records;
        if (ctx->replay_kernels_submitted != ctx->replay_records_total || ctx->replay_waits_completed != ctx->replay_records_total ||
            ctx->replay_outputs_validated != expected_replay_outputs_validated) throw std::runtime_error("replay telemetry invariant failed");
        if (post_recreate_double_recreate_no_execution_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_kernels_submitted != 10 || ctx->replay_waits_completed != 10 || ctx->replay_outputs_validated != 10 ||
                runtime_trace.execution_ordinal != 10 || runtime_trace.m9216_execution_ordinal != 7 ||
                runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0 || runtime_trace.generation1_output_syncs != 0 ||
                runtime_trace.m9216_hw_context_generation != 2 || runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                runtime_trace.m9216_hw_context_recreate_count != 2 || runtime_trace.m9216_kernel_recreate_count != 2 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 0 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 0 ||
                runtime_trace.experiment_submission_ordinal != 0 || runtime_trace.run_objects_created != 10 ||
                runtime_trace.run_wait_completions != 10 || ctx->replay_post_recreate_control_submitted != 0 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("double-recreate no-execution final telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY SECOND-RECREATE NO-EXECUTION PASS history_completed=10 history_outputs_validated=10 full_bundle=SUCCESS generation1_submissions=0 generation1_waits=0 generation1_output_syncs=0 context_generation=2 kernel_generation=2 BO_generation=1 recreate_counts=2/2/1 experiment_submissions=0 post_second_recreation_submissions=0 post_second_recreation_runs=0 post_second_recreation_waits=0 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
            *terminal_mode = "double-recreate-no-execution";
            return XDNA_REPLAY_DIAGNOSTIC_TERMINAL;
        } else if (post_recreate_program_only_context_recreate_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_kernels_submitted != 10 || ctx->replay_waits_completed != 10 || ctx->replay_outputs_validated != 10 ||
                runtime_trace.execution_ordinal != 10 || runtime_trace.m9216_execution_ordinal != 7 || runtime_trace.m9216_submissions != 7 ||
                runtime_trace.m9216_wait_completions != 7 || runtime_trace.m9216_output_syncs != 7 ||
                runtime_trace.generation1_submissions != 0 || runtime_trace.generation1_wait_completions != 0 || runtime_trace.generation1_output_syncs != 0 ||
                runtime_trace.generation1_instruction_init_syncs != program_only_init_syncs_before + 1 ||
                runtime_trace.generation1_instruction_stage_syncs != program_only_stage_syncs_before + 1 ||
                runtime_trace.generation1_q4_syncs != 0 || runtime_trace.generation1_q8_syncs != 0 ||
                runtime_trace.generation1_output_sentinel_syncs != 0 || runtime_trace.generation1_output_from_device_syncs != 0 ||
                runtime_trace.m9216_hw_context_generation != 2 || runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                runtime_trace.m9216_hw_context_recreate_count != 2 || runtime_trace.m9216_kernel_recreate_count != 2 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 0 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 0 ||
                runtime_trace.experiment_submission_ordinal != 0 || runtime_trace.run_objects_created != 10 || runtime_trace.run_wait_completions != 10 ||
                ctx->replay_program_only_stages != 1 || program_only_bundle.instruction_init_sync_count != 1 ||
                program_only_bundle.instruction_stage_sync_count != 1 || ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("program-only second-recreate final telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PROGRAM-ONLY SECOND-CONTEXT-RECREATE PASS history_completed=10 history_outputs_validated=10 program_only_source_id=7 program_only_instruction_syncs=2 generation1_kernel_submissions=0 generation1_runs=0 generation1_waits=0 generation1_output_syncs=0 context_generation=2 kernel_generation=2 BO_generation=1 recreate_counts=2/2/1 experiment_submissions=0 post_second_recreation_submissions=0 post_second_recreation_runs=0 post_second_recreation_waits=0 probe11_submitted=0 target13_submitted=0 replay_failed=0 result=PASS\n");
            *terminal_mode = "m4096-program-only-context-recreate";
            return XDNA_REPLAY_DIAGNOSTIC_TERMINAL;
        } else if (post_recreate_execute_wait_no_readback_context_recreate_mode || post_recreate_execute_wait_release_temp_context_recreate_mode || post_recreate_execute_wait_release_run_context_recreate_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 || ctx->replay_historical_records != 11 ||
                ctx->replay_records_total != 11 || ctx->replay_kernels_submitted != 11 || ctx->replay_waits_completed != 11 ||
                ctx->replay_outputs_validated != 10 || ctx->replay_transition_m4096_submitted != 1 ||
                ctx->replay_transition_m4096_wait_completed != 1 || ctx->replay_transition_m4096_output_validated != 0 ||
                runtime_trace.execution_ordinal != 11 || runtime_trace.run_objects_created != 11 ||
                runtime_trace.run_wait_calls_returned != 11 || runtime_trace.run_wait_completions != 11 ||
                runtime_trace.experiment_submission_ordinal != 1 || runtime_trace.context_generation_submission_ordinal != 0 ||
                runtime_trace.generation1_execution_ordinal != 1 || runtime_trace.generation1_submissions != 1 ||
                runtime_trace.generation1_wait_completions != 1 || runtime_trace.generation1_output_sentinel_syncs != 0 || runtime_trace.generation1_output_syncs != 0 ||
                runtime_trace.generation1_output_from_device_syncs != 0 || runtime_trace.generation1_exact_outputs != 0 ||
                runtime_trace.m9216_submissions != 7 || runtime_trace.m9216_wait_completions != 7 || runtime_trace.m9216_output_syncs != 7 ||
                runtime_trace.m9216_hw_context_generation != 2 || runtime_trace.m9216_hw_context_recreate_count != 2 ||
                runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_kernel_recreate_count != 2 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                transition_bundle.generation != 1 || transition_bundle.m4096_output_from_device_syncs != 0 ||
                transition_bundle.m4096_output_reads != 0 || transition_bundle.m4096_output_sha_calculations != 0 ||
                transition_bundle.m4096_output_sha_validations != 0 || ctx->replay_probe_submitted != 0 ||
                ctx->replay_target_records != 0 || ctx->exact_history_replay_failed ||
                (post_recreate_execute_wait_release_run_context_recreate_mode && (!resource_release_run_release_guard_passed || !resource_release_temp_resources_preserved_through_recreate)))
                throw std::runtime_error("execute/wait no-readback second-context-recreate final telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M4096 EXECUTE/WAIT NO-READBACK SECOND-CONTEXT-RECREATE PASS history_completed=10 history_outputs_validated=10 experimental_source_id=7 M=4096 m4096_submitted=1 m4096_wait_completed=1 m4096_output_sentinel_syncs=0 m4096_output_syncs=0 m4096_output_reads=0 m4096_output_validations=0 generation1_submissions=1 generation1_waits=1 context_generation=2 kernel_generation=2 BO_generation=1 recreate_counts=2/2/1 experiment_submissions=1 post_second_recreation_submissions=0 post_second_recreation_runs=0 post_second_recreation_waits=0 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
            *terminal_mode = post_recreate_execute_wait_release_run_context_recreate_mode ? "m4096-execute-wait-release-run-context-recreate" : (post_recreate_execute_wait_release_temp_context_recreate_mode ? "m4096-execute-wait-release-temp-context-recreate" : "m4096-execute-wait-no-readback-context-recreate");
            return XDNA_REPLAY_DIAGNOSTIC_TERMINAL;
        } else if (ctx->replay_post_id8_context_recreate_only_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_post_recreate_control_submitted != 1 || ctx->replay_post_recreate_control_wait_completed != 1 ||
                ctx->replay_post_recreate_control_output_validated != 1 || ctx->replay_kernels_submitted != 11 ||
                ctx->replay_waits_completed != 11 || ctx->replay_outputs_validated != 11 ||
                runtime_trace.execution_ordinal != 11 || runtime_trace.m9216_execution_ordinal != 8 ||
                runtime_trace.m9216_submissions != 8 || runtime_trace.m9216_wait_completions != 8 || runtime_trace.m9216_output_syncs != 8 ||
                runtime_trace.generation1_execution_ordinal != 1 || runtime_trace.generation1_submissions != 1 ||
                runtime_trace.generation1_wait_completions != 1 || runtime_trace.generation1_output_syncs != 1 ||
                runtime_trace.generation1_exact_outputs != 1 || runtime_trace.m9216_hw_context_generation != 2 ||
                runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                runtime_trace.m9216_hw_context_recreate_count != 2 || runtime_trace.m9216_kernel_recreate_count != 2 ||
                runtime_trace.m9216_bo_bundle_recreate_count != 1 || runtime_trace.m9216_bo_bundle_generation_submissions != 1 ||
                runtime_trace.m9216_bo_bundle_generation_wait_completions != 1 || runtime_trace.experiment_submission_ordinal != 1 ||
                runtime_trace.context_generation_submission_ordinal != 1 || runtime_trace.run_objects_created != 11 ||
                runtime_trace.run_wait_completions != 11 || ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("ID8 context-recreate-only terminal invariants failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-ID8 CONTEXT-RECREATE-ONLY PASS history_completed=10 history_outputs_validated=10 ID8_COMPLETED=1 ID8_WAIT_COMPLETED=1 ID8_OUTPUT_SYNC=1 ID8_VALIDATED=1 ID8_EXACT=1 context_generation=2 kernel_generation=2 BO_generation=1 recreate_delta=+1/+1/+0 context_kernel_bo_recreate_counts=2/2/1 lifecycle_only=YES no_post_recreate_submission=YES replay_submissions=11 replay_waits=11 replay_outputs=11 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
            *terminal_mode = "id8-context-recreate-only";
            return XDNA_REPLAY_DIAGNOSTIC_TERMINAL;
        }
        else if (post_recreate_control_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_post_recreate_control_submitted != 1 || ctx->replay_post_recreate_control_wait_completed != 1 ||
                ctx->replay_post_recreate_control_output_validated != 1 || ctx->replay_probe_submitted != 0 ||
                runtime_trace.m9216_execution_ordinal != 8 || runtime_trace.m9216_submissions != 8 ||
                runtime_trace.m9216_wait_completions != 8 || runtime_trace.m9216_output_syncs != 8 ||
                runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_kernel_generation != 1 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_hw_context_recreate_count != 1 ||
                runtime_trace.m9216_kernel_recreate_count != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 1 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 1) {
                throw std::runtime_error("post-recreation control final telemetry invariant failed");
            }
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY POST-RECREATE CONTROL PASS source_id=8 prefix_completed=10 prefix_outputs_validated=10 control_submitted=1 control_wait_completed=1 control_output_validated=1 m9216_submissions=8 m9216_wait_completions=8 m9216_output_syncs=8 fresh_generation_submissions=1 fresh_generation_waits=1 probe11_submitted=0 replay_failed=0\n");
        }
        else if (post_recreate_same_shape_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_post_recreate_control_submitted != 1 || ctx->replay_post_recreate_control_wait_completed != 1 ||
                ctx->replay_post_recreate_control_output_validated != 1 || ctx->replay_transition_m4096_submitted != 0 ||
                ctx->replay_transition_m4096_wait_completed != 0 || ctx->replay_kernels_submitted != 12 ||
                ctx->replay_waits_completed != 12 || ctx->replay_outputs_validated != 12 ||
                runtime_trace.generation1_execution_ordinal != 2 || runtime_trace.generation1_submissions != 2 ||
                runtime_trace.generation1_wait_completions != 2 || runtime_trace.generation1_output_syncs != 2 ||
                runtime_trace.generation1_output_from_device_syncs != 2 || runtime_trace.generation1_instruction_stage_syncs != 2 ||
                runtime_trace.generation1_q4_syncs != 2 || runtime_trace.generation1_q8_syncs != 2 ||
                runtime_trace.generation1_output_sentinel_syncs != 2 ||
                runtime_trace.m9216_execution_ordinal != 9 || runtime_trace.m9216_submissions != 9 ||
                runtime_trace.m9216_wait_completions != 9 || runtime_trace.m9216_output_syncs != 9 ||
                runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                runtime_trace.m9216_hw_context_recreate_count != 1 || runtime_trace.m9216_kernel_recreate_count != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 2 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 2 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0)
                throw std::runtime_error("same-shape ID8-to-ID8 final telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY SAME-SHAPE ID8-TO-ID8 PASS history_completed=10 history_outputs_validated=10 FIRST_ID8_COMPLETED=1 FIRST_ID8_VALIDATED=1 FIRST_ID8_EXACT=1 second_id8_submitted=1 second_id8_wait_completed=1 second_id8_output_exact=1 generation1_submissions=2 generation1_waits=2 generation1_exact_outputs=2 m9216_cumulative=9/9/9 generation1_context_kernel_bo=1/1/1 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
        } else if (ctx->replay_post_recreate_context_recreate_id8_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 || ctx->replay_kernels_submitted != 12 ||
                ctx->replay_waits_completed != 12 || ctx->replay_outputs_validated != 12 || transition_bundle.id7_submissions != 1 || transition_bundle.id7_waits != 1 ||
                transition_bundle.id7_output_syncs != 1 || transition_bundle.id7_validations != 1 || transition_bundle.id7_exact_outputs != 1 ||
                runtime_trace.experiment_submission_ordinal != 2 || runtime_trace.context_generation_submission_ordinal != 1 ||
                runtime_trace.generation1_submissions != 1 || runtime_trace.generation1_wait_completions != 1 || runtime_trace.generation1_output_syncs != 1 || runtime_trace.generation1_exact_outputs != 1 ||
                runtime_trace.m9216_execution_ordinal != 8 || runtime_trace.m9216_submissions != 8 || runtime_trace.m9216_wait_completions != 8 || runtime_trace.m9216_output_syncs != 8 ||
                runtime_trace.m9216_hw_context_generation != 2 || runtime_trace.m9216_kernel_generation != 2 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                runtime_trace.m9216_hw_context_recreate_count != 2 || runtime_trace.m9216_kernel_recreate_count != 2 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 1 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 1 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0 || ctx->replay_post_recreate_control_submitted != 0)
                throw std::runtime_error("XDNA_RECREATE_GUARD M4096/context-kernel-recreate/ID8 final telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M4096 CONTEXT/KERNEL RECREATE TO ID8 PASS history_completed=10 history_outputs_validated=10 id7_submissions=1 id7_waits=1 id7_exact=1 experiment_ordinal=2 context_generation_2_ordinal=1 id8_submitted=1 id8_wait_completed=1 id8_output_exact=1 m9216_cumulative=8/8/8 context_kernel_bo_generation=2/2/1 recreate_counts=2/2/1 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
            *terminal_mode = "m4096-context-recreate-to-id8";
            return XDNA_REPLAY_DIAGNOSTIC_TERMINAL;
        } else if (ctx->replay_post_recreate_same_shape_id7_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_kernels_submitted != 12 || ctx->replay_waits_completed != 12 || ctx->replay_outputs_validated != 12 ||
                transition_bundle.generation != 1 || !transition_bundle.id7_first_snapshot_valid || transition_bundle.id7_submissions != 2 ||
                transition_bundle.id7_waits != 2 || transition_bundle.id7_output_syncs != 2 || transition_bundle.id7_validations != 2 || transition_bundle.id7_exact_outputs != 2 ||
                runtime_trace.generation1_execution_ordinal != 2 || runtime_trace.generation1_submissions != 2 || runtime_trace.generation1_wait_completions != 2 ||
                runtime_trace.generation1_output_syncs != 2 || runtime_trace.generation1_output_from_device_syncs != 2 || runtime_trace.generation1_exact_outputs != 2 ||
                runtime_trace.generation1_instruction_stage_syncs != 2 || runtime_trace.generation1_q4_syncs != 2 || runtime_trace.generation1_q8_syncs != 2 ||
                runtime_trace.generation1_output_sentinel_syncs != 2 || runtime_trace.m9216_submissions != 7 || runtime_trace.m9216_wait_completions != 7 ||
                runtime_trace.m9216_output_syncs != 7 || runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_kernel_generation != 1 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_hw_context_recreate_count != 1 ||
                runtime_trace.m9216_kernel_recreate_count != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                ctx->replay_probe_submitted != 0 || ctx->replay_target_records != 0 || ctx->replay_post_recreate_control_submitted != 0)
                throw std::runtime_error("same-shape ID7-to-ID7 final exactness/resource telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY SAME-SHAPE ID7-TO-ID7 PASS history_completed=10 history_outputs_validated=10 id7_submissions=2 id7_waits=2 id7_output_syncs=2 id7_exact_outputs=2 generation1_submissions=2 generation1_waits=2 generation1_exact_outputs=2 m9216_cumulative=7/7/7 generation1_context_kernel_bo=1/1/1 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
        } else if (ctx->replay_post_recreate_program_only_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_program_only_stages != 1 || ctx->replay_transition_m4096_submitted != 0 ||
                ctx->replay_transition_m4096_wait_completed != 0 || ctx->replay_transition_m4096_output_validated != 0 ||
                ctx->replay_transition_id8_submitted != 1 || ctx->replay_transition_id8_wait_completed != 1 ||
                ctx->replay_transition_id8_output_validated != 1 || runtime_trace.generation1_execution_ordinal != 1 ||
                runtime_trace.generation1_submissions != 1 || runtime_trace.generation1_wait_completions != 1 ||
                runtime_trace.generation1_output_syncs != 1 || runtime_trace.generation1_instruction_init_syncs != 2 ||
                runtime_trace.generation1_instruction_stage_syncs != 2 ||
                runtime_trace.generation1_instruction_init_syncs + runtime_trace.generation1_instruction_stage_syncs != 4 ||
                runtime_trace.generation1_q4_syncs != 1 || runtime_trace.generation1_q8_syncs != 1 ||
                runtime_trace.generation1_output_sentinel_syncs != 1 || runtime_trace.generation1_output_from_device_syncs != 1 ||
                program_only_bundle.instruction_init_sync_count != 1 || program_only_bundle.instruction_stage_sync_count != 1 ||
                runtime_trace.m9216_submissions != 8 || runtime_trace.m9216_wait_completions != 8 || runtime_trace.m9216_output_syncs != 8 ||
                runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 1 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 1 ||
                ctx->replay_probe_submitted != 0) throw std::runtime_error("M4096 program-only to ID8 final telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M4096 PROGRAM-ONLY TO ID8 PASS history_completed=10 history_outputs_validated=10 m4096_submissions=0 m4096_waits=0 id8_first_generation1_submission=YES generation1_submissions=1 generation1_waits=1 generation1_instruction_init_sync_count=2 generation1_instruction_stage_sync_count=2 generation1_instruction_total_sync_count=4 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
        }
        else if (ctx->replay_post_recreate_no_readback_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_transition_m4096_submitted != 1 || ctx->replay_transition_m4096_wait_completed != 1 ||
                ctx->replay_transition_m4096_output_validated != 0 || ctx->replay_transition_id8_submitted != 1 ||
                ctx->replay_transition_id8_wait_completed != 1 || ctx->replay_transition_id8_output_validated != 1 ||
                runtime_trace.generation1_execution_ordinal != 2 || runtime_trace.generation1_submissions != 2 ||
                runtime_trace.generation1_wait_completions != 2 || runtime_trace.generation1_output_syncs != 1 ||
                runtime_trace.generation1_output_from_device_syncs != 1 || runtime_trace.generation1_instruction_init_syncs != 2 ||
                runtime_trace.generation1_instruction_stage_syncs != 2 ||
                runtime_trace.generation1_instruction_init_syncs + runtime_trace.generation1_instruction_stage_syncs != 4 ||
                runtime_trace.generation1_q4_syncs != 2 || runtime_trace.generation1_q8_syncs != 2 ||
                runtime_trace.generation1_output_sentinel_syncs != 2 ||
                transition_bundle.generation != 1 || transition_bundle.instruction_init_sync_count != 1 ||
                transition_bundle.instruction_stage_sync_count != 1 || transition_bundle.m4096_output_from_device_syncs != 0 ||
                transition_bundle.m4096_output_reads != 0 || transition_bundle.m4096_output_sha_calculations != 0 ||
                transition_bundle.m4096_output_sha_validations != 0 ||
                runtime_trace.m9216_submissions != 8 || runtime_trace.m9216_wait_completions != 8 || runtime_trace.m9216_output_syncs != 8 ||
                runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_kernel_generation != 1 || runtime_trace.m9216_bo_bundle_generation != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 1 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 1 ||
                ctx->replay_probe_submitted != 0) throw std::runtime_error("M4096 execute/wait no-readback to ID8 final telemetry invariant failed");
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY M4096 EXECUTE/WAIT NO-READBACK TO ID8 PASS history_completed=10 history_outputs_validated=10 "
                "m4096_submitted=1 m4096_wait_completed=1 m4096_output_from_device_syncs=0 m4096_output_reads=0 "
                "m4096_output_sha_calculations=0 m4096_output_sha_validations=0 id8_submitted=1 id8_wait_completed=1 id8_output_exact=YES "
                "generation1_submissions=2 generation1_waits=2 generation1_output_syncs=1 generation1_instruction_init_syncs=2 generation1_instruction_stage_syncs=2 "
                "m9216_cumulative=8/8/8 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
        }
        else if (ctx->replay_post_recreate_transition_mode) {
            if (ctx->replay_prefix_completed != 10 || ctx->replay_prefix_outputs_validated != 10 ||
                ctx->replay_transition_m4096_submitted != 1 || ctx->replay_transition_m4096_wait_completed != 1 ||
                ctx->replay_transition_m4096_output_validated != 1 || ctx->replay_transition_id8_submitted != 1 ||
                ctx->replay_transition_id8_wait_completed != 1 || ctx->replay_transition_id8_output_validated != 1 ||
                runtime_trace.generation1_execution_ordinal != 2 || runtime_trace.generation1_submissions != 2 ||
                runtime_trace.generation1_wait_completions != 2 || runtime_trace.generation1_output_syncs != 2 ||
                runtime_trace.m9216_submissions != 8 || runtime_trace.m9216_wait_completions != 8 || runtime_trace.m9216_output_syncs != 8 ||
                runtime_trace.m9216_hw_context_generation != 1 || runtime_trace.m9216_kernel_generation != 1 ||
                runtime_trace.m9216_bo_bundle_generation != 1 || runtime_trace.m9216_hw_context_recreate_count != 1 ||
                runtime_trace.m9216_kernel_recreate_count != 1 || runtime_trace.m9216_bo_bundle_recreate_count != 1 ||
                runtime_trace.m9216_bo_bundle_generation_submissions != 1 || runtime_trace.m9216_bo_bundle_generation_wait_completions != 1 ||
                transition_bundle.generation != 1 || transition_bundle.instruction_init_sync_count != 1 ||
                transition_bundle.instruction_stage_sync_count != 1 || ctx->replay_probe_submitted != 0) {
                throw std::runtime_error("post-recreation M4096-to-ID8 transition final telemetry invariant failed");
            }
            std::fprintf(stderr,
                "XDNA_CAPTURE_REPLAY POST-RECREATE M4096-TO-ID8 TRANSITION PASS history_completed=10 history_outputs_validated=10 "
                "generation1_submissions=2 generation1_waits=2 generation1_output_syncs=2 m4096_control=EXACT id8=EXACT "
                "m9216_cumulative=8/8/8 generation1_context_kernel_bo=1/1/1 probe11_submitted=0 target13_submitted=0 replay_failed=0\n");
        }
        else if (ctx->replay_prefix_probe_mode) {
            if (ctx->replay_prefix_completed != ctx->replay_prefix_planned ||
                ctx->replay_prefix_outputs_validated != ctx->replay_prefix_planned ||
                ctx->replay_probe_submitted != 1 || ctx->replay_probe_wait_completed != 1 ||
                ctx->replay_probe_output_validated != (ctx->replay_target_records ? 0u : 1u)) {
                throw std::runtime_error("prefix/probe final telemetry invariant failed");
            }
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PREFIX/PROBE PASS prefix_planned=%llu prefix_completed=%llu prefix_outputs_validated=%llu probe_submitted=%llu probe_wait_completed=%llu probe_output_validated=%llu replay_failed=0\n",
                static_cast<unsigned long long>(ctx->replay_prefix_planned),
                static_cast<unsigned long long>(ctx->replay_prefix_completed),
                static_cast<unsigned long long>(ctx->replay_prefix_outputs_validated),
                static_cast<unsigned long long>(ctx->replay_probe_submitted),
                static_cast<unsigned long long>(ctx->replay_probe_wait_completed),
                static_cast<unsigned long long>(ctx->replay_probe_output_validated));
        } else {
            std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY PASS records_total=%llu historical_records=%llu target_records=%llu kernels_submitted=%llu waits_completed=%llu outputs_validated=%llu\n",
                static_cast<unsigned long long>(ctx->replay_records_total), static_cast<unsigned long long>(ctx->replay_historical_records),
                static_cast<unsigned long long>(ctx->replay_target_records), static_cast<unsigned long long>(ctx->replay_kernels_submitted),
                static_cast<unsigned long long>(ctx->replay_waits_completed), static_cast<unsigned long long>(ctx->replay_outputs_validated));
        }
        return 0;
    } catch (const std::exception & e) {
        ctx->exact_history_replay_failed = true;
        if (ctx->replay_prefix_probe_mode) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PREFIX/PROBE FAILED prefix_planned=%llu prefix_completed=%llu prefix_outputs_validated=%llu probe_submitted=%llu probe_wait_completed=%llu probe_output_validated=%llu replay_failed=1\n",
            static_cast<unsigned long long>(ctx->replay_prefix_planned),
            static_cast<unsigned long long>(ctx->replay_prefix_completed),
            static_cast<unsigned long long>(ctx->replay_prefix_outputs_validated),
            static_cast<unsigned long long>(ctx->replay_probe_submitted),
            static_cast<unsigned long long>(ctx->replay_probe_wait_completed),
            static_cast<unsigned long long>(ctx->replay_probe_output_validated));
        std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY FAILED CLOSED: %s\n", e.what());
        return 1;
    } catch (...) {
        ctx->exact_history_replay_failed = true;
        if (ctx->replay_prefix_probe_mode) std::fprintf(stderr, "XDNA_CAPTURE_REPLAY PREFIX/PROBE FAILED prefix_planned=%llu prefix_completed=%llu prefix_outputs_validated=%llu probe_submitted=%llu probe_wait_completed=%llu probe_output_validated=%llu replay_failed=1\n",
            static_cast<unsigned long long>(ctx->replay_prefix_planned),
            static_cast<unsigned long long>(ctx->replay_prefix_completed),
            static_cast<unsigned long long>(ctx->replay_prefix_outputs_validated),
            static_cast<unsigned long long>(ctx->replay_probe_submitted),
            static_cast<unsigned long long>(ctx->replay_probe_wait_completed),
            static_cast<unsigned long long>(ctx->replay_probe_output_validated));
        std::fprintf(stderr, "XDNA_CAPTURE_REPLAY REPLAY FAILED CLOSED: unknown error\n");
        return 1;
    }
}

int ggml_backend_xdna_diagnostic_replay(ggml_backend_t backend, const char * plan_path) {
    const char * terminal_mode = nullptr;
    const int result = ggml_backend_xdna_replay_exact_history_impl(backend, plan_path, &terminal_mode);
    if (result != XDNA_REPLAY_DIAGNOSTIC_TERMINAL) return result;
    auto * ctx = backend ? static_cast<ggml_backend_xdna_backend_context *>(backend->context) : nullptr;
    if (!ctx || !terminal_mode || ctx->exact_history_replay_failed) return 1;
    std::fprintf(stderr, "XDNA_CAPTURE_REPLAY DIAGNOSTIC TERMINAL SUCCESS mode=%s backend_result=PASS replay_failed=0\n", terminal_mode);
    return 0;
}

static void ggml_backend_xdna_causal_event(const char * event, const char * cell, const char * selected_prefix_ids, const char * mode, const char * pair,
        size_t execution_ordinal, size_t capture_id, size_t rows, const char * node, const char * q4_sha, const char * q8_sha, bool target) {
    std::fprintf(stderr, "XDNA_CAUSAL_EVENT event=%s cell=%s selected_prefix_ids=%s mode=%s pair=%s execution_ordinal=%zu sequence_index=%zu capture_id=%s M=%zu K=2560 node=%s Q4_SHA256=%s Q8_SHA256=%s TARGET=%s\n",
        event, cell, selected_prefix_ids, mode, pair, execution_ordinal, execution_ordinal, capture_id ? std::to_string(capture_id).c_str() : "NONE", rows, node, q4_sha, q8_sha, target ? "YES" : "NO");
    std::fflush(stderr);
}

int ggml_backend_xdna_diagnostic_execute_causal_record(const ggml_xdna_causal::validated_execution_authorization & token,
        const ggml_xdna_causal::plan_identity & identity, ggml_backend_t backend, size_t rows, const void * q4_native, size_t q4_native_bytes,
        const void * q8_bytes, size_t q8_bytes_size, void * output, size_t output_bytes, const char * cell, const char * selected_prefix_ids, const char * mode,
        const char * pair, size_t execution_ordinal, size_t capture_id, const char * node_name, bool target, const char * q4_sha256, const char * q8_sha256) {
    if (!ggml_xdna_causal::authorization_matches(token, identity) || !cell || !mode || !pair || !selected_prefix_ids ||
        token.cell() != cell || token.mode() != mode || token.pair() != pair) return 1;
    std::string expected_ids;
    for (size_t id : token.selected_ids()) { if (!expected_ids.empty()) expected_ids += ','; expected_ids += std::to_string(id); }
    if ((expected_ids.empty() ? std::string("NONE") : expected_ids) != selected_prefix_ids) return 1;
    if (!backend || !backend->context || !q4_native || !q8_bytes || !output || !cell || !selected_prefix_ids || !mode || !pair || !node_name || !q4_sha256 || !q8_sha256 ||
        (rows != 4096 && rows != 9216) || q4_native_bytes != rows * 1440 || q8_bytes_size != 2920 || output_bytes != rows * sizeof(float) ||
        (target && (rows != 9216 || std::strcmp(node_name, "ffn_gate-4") != 0))) return 1;
    auto * ctx = static_cast<ggml_backend_xdna_backend_context *>(backend->context);
    if (!ctx->device_context || !ctx->device_context->xrt_device) return 1;
    const size_t transport_bytes = rows * 1520;
    const uint32_t instruction_count = static_cast<uint32_t>(rows == 4096 ? ctx->q4_m4096.instructions.size() : ctx->q4_m9216_instructions.size());
    const size_t instruction_bytes = static_cast<size_t>(instruction_count) * sizeof(uint32_t);
    if (!instruction_count || (rows == 9216 && !ctx->q4_m9216_available)) return 1;
    auto log = [&](const char * event) { ggml_backend_xdna_causal_event(event, cell, selected_prefix_ids, mode, pair, execution_ordinal, capture_id, rows, node_name, q4_sha256, q8_sha256, target); };
    try {
        xrt::kernel * kernel = nullptr;
        xrt::bo * bo_instr = nullptr, * bo_q4 = nullptr, * bo_q8 = nullptr, * bo_out = nullptr;
        void * buf_instr = nullptr, * buf_q4 = nullptr, * buf_q8 = nullptr, * buf_out = nullptr;
        if (rows == 4096) {
            auto & s = ctx->q4_m4096;
            if (!s.available || !s.kernel) return 1;
            if (!s.bos_initialized) {
                auto i = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, instruction_bytes, XCL_BO_FLAGS_CACHEABLE, s.kernel->group_id(1));
                auto q = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, transport_bytes, XRT_BO_FLAGS_HOST_ONLY, s.kernel->group_id(3));
                auto a = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, 2920, XRT_BO_FLAGS_HOST_ONLY, s.kernel->group_id(4));
                auto o = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, output_bytes, XRT_BO_FLAGS_HOST_ONLY, s.kernel->group_id(5));
                s.buf_instr = i->map<void *>(); s.buf_q4 = q->map<void *>(); s.buf_q8 = a->map<void *>(); s.buf_out = o->map<float *>();
                if (!s.buf_instr || !s.buf_q4 || !s.buf_q8 || !s.buf_out) return 1;
                std::memcpy(s.buf_instr, s.instructions.data(), instruction_bytes); ggml_backend_xdna_sync_host_written_bo(*i, "XDNA_CAUSAL M4096 instruction BO");
                s.bo_instr = std::move(i); s.bo_q4 = std::move(q); s.bo_q8 = std::move(a); s.bo_out = std::move(o);
                s.instruction_count = instruction_count; s.bos_initialized = true;
            } else if (s.instruction_count != instruction_count || std::memcmp(s.buf_instr, s.instructions.data(), instruction_bytes) != 0) return 1;
            kernel = s.kernel.get(); bo_instr = s.bo_instr.get(); bo_q4 = s.bo_q4.get(); bo_q8 = s.bo_q8.get(); bo_out = s.bo_out.get();
            buf_instr = s.buf_instr; buf_q4 = s.buf_q4; buf_q8 = s.buf_q8; buf_out = s.buf_out;
        } else {
            if (!ctx->q4_m9216_kernel) return 1;
            if (!ctx->q4_m9216_bos_initialized) {
                auto i = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, instruction_bytes, XCL_BO_FLAGS_CACHEABLE, ctx->q4_m9216_kernel->group_id(1));
                auto q = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, transport_bytes, XRT_BO_FLAGS_HOST_ONLY, ctx->q4_m9216_kernel->group_id(3));
                auto a = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, 2920, XRT_BO_FLAGS_HOST_ONLY, ctx->q4_m9216_kernel->group_id(4));
                auto o = std::make_unique<xrt::bo>(*ctx->device_context->xrt_device, output_bytes, XRT_BO_FLAGS_HOST_ONLY, ctx->q4_m9216_kernel->group_id(5));
                ctx->q4_m9216_buf_instr = i->map<void *>(); ctx->q4_m9216_buf_q4 = q->map<void *>(); ctx->q4_m9216_buf_q8 = a->map<void *>(); ctx->q4_m9216_buf_out = o->map<float *>();
                if (!ctx->q4_m9216_buf_instr || !ctx->q4_m9216_buf_q4 || !ctx->q4_m9216_buf_q8 || !ctx->q4_m9216_buf_out) return 1;
                std::memcpy(ctx->q4_m9216_buf_instr, ctx->q4_m9216_instructions.data(), instruction_bytes); ggml_backend_xdna_sync_host_written_bo(*i, "XDNA_CAUSAL M9216 instruction BO");
                ctx->q4_m9216_bo_instr = std::move(i); ctx->q4_m9216_bo_q4 = std::move(q); ctx->q4_m9216_bo_q8 = std::move(a); ctx->q4_m9216_bo_out = std::move(o);
                ctx->q4_m9216_instruction_count = instruction_count; ctx->q4_m9216_bos_initialized = true;
            } else if (ctx->q4_m9216_instruction_count != instruction_count || std::memcmp(ctx->q4_m9216_buf_instr, ctx->q4_m9216_instructions.data(), instruction_bytes) != 0) return 1;
            kernel = ctx->q4_m9216_kernel.get(); bo_instr = ctx->q4_m9216_bo_instr.get(); bo_q4 = ctx->q4_m9216_bo_q4.get(); bo_q8 = ctx->q4_m9216_bo_q8.get(); bo_out = ctx->q4_m9216_bo_out.get();
            buf_instr = ctx->q4_m9216_buf_instr; buf_q4 = ctx->q4_m9216_buf_q4; buf_q8 = ctx->q4_m9216_buf_q8; buf_out = ctx->q4_m9216_buf_out;
        }
        if (!ggml_backend_xdna_diagnostic_convert_q4_native(ctx, rows, q4_native, buf_q4)) return 1;
        std::memcpy(buf_q8, q8_bytes, q8_bytes_size); log("SUBMISSION_PREPARED");
        ggml_backend_xdna_sync_host_written_bo(*bo_q4, "XDNA_CAUSAL Q4 transport input");
        ggml_backend_xdna_sync_host_written_bo(*bo_q8, "XDNA_CAUSAL captured Q8 input");
        {
            auto run = (*kernel)(3u, *bo_instr, instruction_bytes, *bo_q4, *bo_q8, *bo_out); log("KERNEL_CALL_RETURNED");
            log("WAIT_ENTER");
            const ert_cmd_state completion_state = run.wait();
            log("WAIT_RETURN");
            if (completion_state != ERT_CMD_STATE_COMPLETED) {
                std::fprintf(stderr, "XDNA_CAUSAL_EXECUTOR_FAIL_CLOSED execution_ordinal=%zu capture_id=%zu M=%zu K=2560 wait_state=%d expected=%d\n",
                    execution_ordinal, capture_id, rows, static_cast<int>(completion_state), static_cast<int>(ERT_CMD_STATE_COMPLETED));
                return 1;
            }
        }
        bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE); log("OUTPUT_SYNC_RETURN");
        std::memcpy(output, buf_out, output_bytes); log("RESULT_COPY_RETURN");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "XDNA_CAUSAL_EXECUTOR_FAIL_CLOSED execution_ordinal=%zu capture_id=%zu error=%s\n", execution_ordinal, capture_id, e.what()); return 1;
    } catch (...) {
        std::fprintf(stderr, "XDNA_CAUSAL_EXECUTOR_FAIL_CLOSED execution_ordinal=%zu capture_id=%zu error=unknown\n", execution_ordinal, capture_id); return 1;
    }
}
