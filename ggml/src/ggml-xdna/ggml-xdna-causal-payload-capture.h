#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ggml_xdna_causal_capture {

constexpr uint64_t max_window_size = 256;

struct tensor_identity {
    const char * name = nullptr;
    const void * tensor = nullptr;
    const void * data = nullptr;
    std::array<int64_t, 4> ne{};
    std::array<size_t, 4> nb{};
};

struct submission_metadata {
    size_t shape_m = 0;
    int64_t shape_k = 0;
    const char * op = nullptr;
    int graph_node_index = -1;
    const char * node_name = nullptr;
    const void * node = nullptr;
    tensor_identity src0;
    tensor_identity src1;
    tensor_identity dst;
    std::optional<uint64_t> correlation_id;
    std::optional<uint64_t> m9216_ordinal;
};

class writer {
public:
    bool open_session(const std::filesystem::path & directory, uint64_t begin, uint64_t end);
    uint64_t reserve_sequence_id();
    bool selected(uint64_t sequence_id) const;
    bool prepare(uint64_t sequence_id, const submission_metadata & metadata,
                 const void * q4_native, size_t q4_bytes,
                 const void * q8, size_t q8_bytes);
    bool append_event(uint64_t sequence_id, const char * event);
    bool failed() const;

private:
    bool append_json_line(const std::string & line);
    bool write_payload(const std::string & relative_path, const void * data, size_t size,
                       std::string & sha256);

    mutable std::mutex mutex_;
    std::filesystem::path directory_;
    uint64_t begin_ = 0;
    uint64_t end_ = 0;
    uint64_t next_sequence_id_ = 1;
    bool opened_ = false;
    bool failed_ = false;
    std::unordered_set<uint64_t> prepared_;
    std::unordered_map<uint64_t, std::pair<std::optional<uint64_t>, std::optional<uint64_t>>> correlation_;
};

} // namespace ggml_xdna_causal_capture
