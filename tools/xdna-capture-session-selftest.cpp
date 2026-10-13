#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../ggml/src/ggml-xdna/ggml-xdna-capture-session.h"
#include "../ggml/src/ggml-xdna/ggml-xdna-sha256.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;
using ggml_xdna_capture::capture_session;
using json = nlohmann::json;

static std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    assert(in.good());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static std::string sha256(const std::string & bytes) {
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    ggml_xdna_sha256_hash(digest, reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < sizeof(digest); ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 15];
    }
    return result;
}

static std::string metadata(const fs::path & xclbin, int schema = 1, const std::string & profile = "q4k_q8k_k2560",
                            const std::string & digest = "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83") {
    return json{{"schema_version", schema}, {"HEAD", "host-self-test"}, {"branch", nullptr},
                {"capture_mode", "GGML_XDNA_CAPTURE_EXACT_HISTORY=1"},
                {"xclbin_path", fs::absolute(xclbin).lexically_normal().string()},
                {"xclbin_sha256", digest}, {"profile", profile}}.dump(2) + "\n";
}

static bool init(capture_session & session, const fs::path & dir, const fs::path & xclbin,
                 const std::string & profile = "q4k_q8k_k2560",
                 const std::string & digest = "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83",
                 int schema = 1, bool * created = nullptr, uint64_t * ordinal = nullptr) {
    return session.attach(dir, xclbin, profile, digest, metadata(xclbin, schema, profile, digest), created, ordinal);
}

static void expect_mismatch(const fs::path & root, const fs::path & xclbin, const char * name,
                            const std::string & profile, const std::string & digest, int schema,
                            bool other_dir, bool other_xclbin = false) {
    capture_session session;
    const fs::path first = root / (std::string(name) + "-first");
    bool created = false;
    assert(init(session, first, xclbin, "q4k_q8k_k2560", "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, &created));
    assert(created);
    const fs::path second = other_dir ? root / (std::string(name) + "-other") : first;
    const fs::path second_xclbin = other_xclbin ? root / (std::string(name) + "-other.xclbin") : xclbin;
    assert(!init(session, second, second_xclbin, profile, digest, schema));
    assert(session.failed());
    assert(session.reserve_submission_id() == 0);
}

int main() {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() / ("xdna-capture-session-selftest-" + std::to_string(tick));
    fs::create_directories(root);
    const fs::path xclbin = root / "frozen.xclbin";
    const fs::path run = root / "run";
    const std::string meta = metadata(xclbin);

    // Same-process A/B/C init, A free, then B continues with one global ID stream.
    capture_session process1;
    bool created = false;
    uint64_t ordinal = 0;
    assert(init(process1, run, xclbin, "q4k_q8k_k2560", "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, &created, &ordinal));
    assert(created && ordinal == 1 && process1.attached_count() == 1);
    assert(read_file(run / "run_metadata.json") == meta);
    assert(read_file(run / "manifest.jsonl").empty());
    assert(!fs::exists(run / ".br1-capture-session-owner"));
    const std::string metadata_sha = sha256(read_file(run / "run_metadata.json"));
    const auto metadata_write_time = fs::last_write_time(run / "run_metadata.json");
    const auto empty_manifest_write_time = fs::last_write_time(run / "manifest.jsonl");
    assert(read_file(run / "manifest.jsonl").empty());
    process1.detach(); // backend A frees; process-local session remains
    assert(process1.attached_count() == 0);
    assert(init(process1, run, xclbin, "q4k_q8k_k2560", "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, &created, &ordinal));
    assert(!created && ordinal == 2 && process1.attached_count() == 1);
    assert(sha256(read_file(run / "run_metadata.json")) == metadata_sha);
    assert(read_file(run / "manifest.jsonl").empty());
    assert(fs::last_write_time(run / "run_metadata.json") == metadata_write_time);
    assert(fs::last_write_time(run / "manifest.jsonl") == empty_manifest_write_time);
    assert(std::distance(fs::directory_iterator(run), fs::directory_iterator{}) == 2);
    assert(process1.reserve_submission_id() == 1);
    assert(process1.append("{\"event\":\"PREPARED\",\"submission_id\":1}\n"));
    const std::string manifest_prefix = read_file(run / "manifest.jsonl");
    const auto metadata_time_before_c = fs::last_write_time(run / "run_metadata.json");
    const auto manifest_time_before_c = fs::last_write_time(run / "manifest.jsonl");
    assert(init(process1, run, xclbin, "q4k_q8k_k2560", "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, &created, &ordinal));
    assert(!created && ordinal == 3 && process1.attached_count() == 2);
    assert(sha256(read_file(run / "run_metadata.json")) == metadata_sha);
    assert(read_file(run / "manifest.jsonl").substr(0, manifest_prefix.size()) == manifest_prefix);
    assert(fs::last_write_time(run / "run_metadata.json") == metadata_time_before_c);
    assert(fs::last_write_time(run / "manifest.jsonl") == manifest_time_before_c);
    assert(process1.reserve_submission_id() == 2);
    assert(process1.append("{\"event\":\"PREPARED\",\"submission_id\":2}\n"));
    assert(process1.reserve_submission_id() == 3);
    assert(process1.append("{\"event\":\"PREPARED\",\"semantic_target\":true}\n", true));
    assert(process1.terminal() && process1.next_submission_id() == 3);
    const std::string terminal_manifest = read_file(run / "manifest.jsonl");
    const auto metadata_time_before_d = fs::last_write_time(run / "run_metadata.json");
    const auto manifest_time_before_d = fs::last_write_time(run / "manifest.jsonl");
    assert(init(process1, run, xclbin, "q4k_q8k_k2560", "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, &created, &ordinal));
    assert(!created && ordinal == 4 && process1.terminal() && process1.reserve_submission_id() == 0);
    assert(read_file(run / "manifest.jsonl") == terminal_manifest);
    assert(fs::last_write_time(run / "run_metadata.json") == metadata_time_before_d);
    assert(fs::last_write_time(run / "manifest.jsonl") == manifest_time_before_d);
    assert(std::distance(fs::directory_iterator(run), fs::directory_iterator{}) == 2);
    assert(sha256(read_file(run / "run_metadata.json")) == metadata_sha);

    // A distinct process-local session has no authorization to reuse the old directory.
    capture_session process2;
    assert(!init(process2, run, xclbin));
    assert(process2.failed() && process2.reserve_submission_id() == 0);

    // capture_failed is process-session-owned and cannot be cleared by a compatible attach.
    capture_session failed_session;
    const fs::path failed_run = root / "failed-run";
    assert(init(failed_session, failed_run, xclbin));
    failed_session.fail();
    assert(!init(failed_session, failed_run, xclbin));
    assert(failed_session.failed() && failed_session.reserve_submission_id() == 0);

    expect_mismatch(root, xclbin, "dir-mismatch", "q4k_q8k_k2560",
                    "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, true);
    expect_mismatch(root, xclbin, "profile-mismatch", "other-profile",
                    "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, false);
    expect_mismatch(root, xclbin, "sha-mismatch", "q4k_q8k_k2560", std::string(64, '0'), 1, false);
    expect_mismatch(root, xclbin, "path-mismatch", "q4k_q8k_k2560",
                    "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 1, false, true);
    expect_mismatch(root, xclbin, "schema-mismatch", "q4k_q8k_k2560",
                    "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83", 2, false);

    fs::remove_all(root);
    std::cout << "XDNA capture session self-test: PASS (multi-init, global ID/latches/terminal, stale-process rejection, compatibility, preservation)\n";
    return 0;
}
