#include "ggml-xdna-causal-payload-capture.h"
#include "ggml-xdna-sha256.h"

#include "../../../vendor/nlohmann/json.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

namespace ggml_xdna_causal_capture {
namespace {

std::string pointer_text(const void * value) {
    if (value == nullptr) return "0x0";
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(static_cast<int>(sizeof(uintptr_t) * 2))
        << std::setfill('0') << reinterpret_cast<uintptr_t>(value);
    return out.str();
}

std::string sha256_hex(const unsigned char digest[32]) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 0x0f];
    }
    return result;
}

std::string bytes_sha256(const void * data, size_t size) {
    unsigned char digest[32];
    ggml_xdna_sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    return sha256_hex(digest);
}

json tensor_json(const tensor_identity & tensor) {
    return json{
        {"name", tensor.name == nullptr ? "" : tensor.name},
        {"tensor_id", pointer_text(tensor.tensor)},
        {"data_id", pointer_text(tensor.data)},
        {"ne", tensor.ne},
        {"nb", tensor.nb},
    };
}

bool safe_relative_path(const fs::path & path) {
    if (path.empty() || path.is_absolute()) return false;
    for (const auto & component : path) {
        if (component == ".." || component == ".") return false;
    }
    return true;
}

bool hash_file(const fs::path & path, uint64_t expected_size, std::string & output) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    ggml_xdna_sha256_t context;
    ggml_xdna_sha256_init(&context);
    char buffer[64 * 1024];
    uint64_t total = 0;
    while (input) {
        input.read(buffer, sizeof(buffer));
        const auto got = input.gcount();
        if (got > 0) {
            total += static_cast<uint64_t>(got);
            ggml_xdna_sha256_update(&context, reinterpret_cast<const unsigned char *>(buffer), static_cast<size_t>(got));
        }
    }
    if (!input.eof() || total != expected_size) return false;
    unsigned char digest[32];
    ggml_xdna_sha256_final(&context, digest);
    output = sha256_hex(digest);
    return true;
}

json optional_u64(const std::optional<uint64_t> & value) {
    return value ? json(*value) : json(nullptr);
}

} // namespace

bool writer::open_session(const fs::path & directory, uint64_t begin, uint64_t end) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (opened_ || directory.empty() || begin == 0 || end < begin ||
        end - begin >= max_window_size) {
        failed_ = true;
        return false;
    }
    std::error_code ec;
    fs::path absolute = fs::absolute(directory, ec).lexically_normal();
    if (ec || absolute.empty()) {
        failed_ = true;
        return false;
    }
    const fs::path parent = absolute.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) {
            failed_ = true;
            return false;
        }
    }
    if (fs::exists(absolute, ec) || ec || !fs::create_directory(absolute, ec) || ec) {
        failed_ = true;
        return false;
    }
    directory_ = absolute;
    begin_ = begin;
    end_ = end;
    if (!fs::create_directory(directory_ / "payloads", ec) || ec) {
        failed_ = true;
        return false;
    }
    const fs::path manifest = directory_ / "manifest.jsonl";
    if (fs::exists(manifest, ec) || ec) {
        failed_ = true;
        return false;
    }
    std::ofstream output(manifest, std::ios::binary | std::ios::out);
    output.flush();
    const bool opened = output.good();
    output.close();
    if (!opened || output.fail()) {
        failed_ = true;
        return false;
    }
    opened_ = true;
    return true;
}

uint64_t writer::reserve_sequence_id() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_ || failed_ || next_sequence_id_ == 0 ||
        next_sequence_id_ == std::numeric_limits<uint64_t>::max()) return 0;
    return next_sequence_id_++;
}

bool writer::selected(uint64_t sequence_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return opened_ && sequence_id >= begin_ && sequence_id <= end_;
}

bool writer::append_json_line(const std::string & line) {
    std::ofstream output(directory_ / "manifest.jsonl", std::ios::binary | std::ios::out | std::ios::app);
    if (!output) return false;
    output.write(line.data(), static_cast<std::streamsize>(line.size()));
    output.put('\n');
    output.flush();
    const bool written = output.good();
    output.close();
    return written && !output.fail();
}

bool writer::write_payload(const std::string & relative_path, const void * data, size_t size,
                           std::string & sha256) {
    const fs::path relative(relative_path);
    if (!safe_relative_path(relative) || data == nullptr || size == 0 ||
        size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;
    const fs::path target = directory_ / relative;
    const fs::path temporary = target.string() + ".tmp";
    std::error_code ec;
    if (fs::exists(target, ec) || ec || fs::exists(temporary, ec) || ec) return false;
    const std::string expected = bytes_sha256(data, size);
    std::ofstream output(temporary, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output) return false;
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    output.flush();
    const bool written = output.good();
    output.close();
    if (!written || output.fail()) return false;
    const auto actual_size = fs::file_size(temporary, ec);
    if (ec || actual_size != size) return false;
    std::string persisted;
    if (!hash_file(temporary, size, persisted) || persisted != expected) return false;
    fs::rename(temporary, target, ec);
    if (ec) return false;
    sha256 = persisted;
    return true;
}

bool writer::prepare(uint64_t sequence_id, const submission_metadata & metadata,
                     const void * q4_native, size_t q4_bytes,
                     const void * q8, size_t q8_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_ || failed_) return false;
    if (sequence_id < begin_ || sequence_id > end_) return true;
    if (prepared_.count(sequence_id) != 0 || metadata.shape_m == 0 || metadata.shape_k <= 0) {
        failed_ = true;
        return false;
    }
    std::ostringstream stem;
    stem << std::setw(6) << std::setfill('0') << sequence_id;
    const std::string q4_path = "payloads/" + stem.str() + ".q4-native.bin";
    const std::string q8_path = "payloads/" + stem.str() + ".q8.bin";
    std::string q4_sha, q8_sha;
    if (!write_payload(q4_path, q4_native, q4_bytes, q4_sha) ||
        !write_payload(q8_path, q8, q8_bytes, q8_sha)) {
        failed_ = true;
        return false;
    }
    json record = {
        {"schema_version", 1},
        {"event", "PREPARED"},
        {"capture_sequence_id", sequence_id},
        {"capture_window", {{"begin", begin_}, {"end", end_}}},
        {"shape", {{"m", metadata.shape_m}, {"k", metadata.shape_k}}},
        {"op", metadata.op == nullptr ? "" : metadata.op},
        {"graph_node_index", metadata.graph_node_index},
        {"node_name", metadata.node_name == nullptr ? "" : metadata.node_name},
        {"node_id", pointer_text(metadata.node)},
        {"src0", tensor_json(metadata.src0)},
        {"src1", tensor_json(metadata.src1)},
        {"dst", tensor_json(metadata.dst)},
        {"correlation_id", optional_u64(metadata.correlation_id)},
        {"m9216_ordinal", optional_u64(metadata.m9216_ordinal)},
        {"q4_native", {{"file", q4_path}, {"bytes", q4_bytes}, {"sha256", q4_sha}}},
        {"q8", {{"file", q8_path}, {"bytes", q8_bytes}, {"sha256", q8_sha}}},
    };
    if (!append_json_line(record.dump())) {
        failed_ = true;
        return false;
    }
    prepared_.insert(sequence_id);
    correlation_[sequence_id] = {metadata.correlation_id, metadata.m9216_ordinal};
    return true;
}

bool writer::append_event(uint64_t sequence_id, const char * event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_ || failed_) return false;
    if (sequence_id < begin_ || sequence_id > end_) return true;
    if (prepared_.count(sequence_id) == 0 || event == nullptr) {
        failed_ = true;
        return false;
    }
    static const std::unordered_set<std::string> allowed = {
        "KERNEL_CALL_RETURNED", "WAIT_ENTER", "WAIT_RETURN",
        "OUTPUT_SYNC_RETURN", "RESULT_COPY_RETURN",
    };
    if (allowed.count(event) == 0) {
        failed_ = true;
        return false;
    }
    const auto ids = correlation_.at(sequence_id);
    json record = {
        {"schema_version", 1},
        {"event", event},
        {"capture_sequence_id", sequence_id},
        {"correlation_id", optional_u64(ids.first)},
        {"m9216_ordinal", optional_u64(ids.second)},
    };
    if (!append_json_line(record.dump())) {
        failed_ = true;
        return false;
    }
    return true;
}

bool writer::failed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_;
}

} // namespace ggml_xdna_causal_capture
