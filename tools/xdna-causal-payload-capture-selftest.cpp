#include "ggml-xdna-causal-payload-capture.h"
#include "ggml-xdna-sha256.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <nlohmann/json.hpp>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string hash_hex(const void * data, size_t size) {
    unsigned char digest[32];
    ggml_xdna_sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        result[i * 2] = hex[digest[i] >> 4];
        result[i * 2 + 1] = hex[digest[i] & 15];
    }
    return result;
}

int main() {
    const fs::path root = fs::temp_directory_path() / "xdna-causal-capture-selftest";
    std::error_code ec;
    fs::remove_all(root, ec);

    ggml_xdna_causal_capture::writer capture;
    assert(capture.open_session(root, 1, 2));
    const uint64_t id1 = capture.reserve_sequence_id();
    const uint64_t id2 = capture.reserve_sequence_id();
    const uint64_t id3 = capture.reserve_sequence_id();
    assert(id1 == 1 && id2 == 2 && id3 == 3);
    assert(capture.selected(id1) && capture.selected(id2) && !capture.selected(id3));

    const unsigned char q4[] = {0, 1, 2, 0xff};
    const unsigned char q8[] = {9, 8, 7, 6, 5};
    ggml_xdna_causal_capture::submission_metadata metadata;
    metadata.shape_m = 9216;
    metadata.shape_k = 2560;
    metadata.op = "MUL_MAT";
    metadata.graph_node_index = 17;
    metadata.node_name = "target-node";
    metadata.node = &metadata;
    metadata.src0.name = "weights";
    metadata.src0.tensor = &q4;
    metadata.src0.data = q4;
    metadata.src0.ne = {2560, 9216, 1, 1};
    metadata.src0.nb = {144, 368640, 3397386240ULL, 3397386240ULL};
    metadata.src1.name = "activation";
    metadata.src1.tensor = &q8;
    metadata.src1.data = q8;
    metadata.src1.ne = {2560, 1, 1, 1};
    metadata.src1.nb = {2920, 2920, 2920, 2920};
    metadata.correlation_id = 9;
    metadata.m9216_ordinal = 3;

    assert(capture.prepare(id1, metadata, q4, sizeof(q4), q8, sizeof(q8)));
    assert(capture.append_event(id1, "KERNEL_CALL_RETURNED"));
    assert(capture.append_event(id1, "WAIT_ENTER"));
    assert(capture.append_event(id1, "WAIT_RETURN"));
    assert(capture.append_event(id1, "OUTPUT_SYNC_RETURN"));
    assert(capture.append_event(id1, "RESULT_COPY_RETURN"));
    assert(capture.prepare(id2, metadata, q4, sizeof(q4), q8, sizeof(q8)));
    assert(capture.prepare(id3, metadata, q4, sizeof(q4), q8, sizeof(q8)));
    assert(!capture.failed());

    std::ifstream manifest(root / "manifest.jsonl", std::ios::binary);
    assert(manifest);
    std::vector<json> records;
    std::string line;
    while (std::getline(manifest, line)) records.push_back(json::parse(line));
    assert(records.size() == 7);
    assert(records[0]["event"] == "PREPARED");
    assert(records[0]["capture_sequence_id"] == 1);
    assert(records[0]["correlation_id"] == 9);
    assert(records[0]["m9216_ordinal"] == 3);
    assert(records[0]["src0"]["ne"][1] == 9216);
    assert(records[0]["q4_native"]["bytes"] == sizeof(q4));
    assert(records[0]["q4_native"]["sha256"] == hash_hex(q4, sizeof(q4)));
    assert(records[0]["q8"]["bytes"] == sizeof(q8));
    assert(records[0]["q8"]["sha256"] == hash_hex(q8, sizeof(q8)));
    assert(records[1]["event"] == "KERNEL_CALL_RETURNED");
    assert(records[2]["event"] == "WAIT_ENTER");
    assert(records[3]["event"] == "WAIT_RETURN");
    assert(records[4]["event"] == "OUTPUT_SYNC_RETURN");
    assert(records[5]["event"] == "RESULT_COPY_RETURN");
    assert(records[6]["capture_sequence_id"] == 2);
    assert(!fs::exists(root / "payloads/000003.q4-native.bin"));
    assert(!fs::exists(root / "payloads/000003.q8.bin"));

    manifest.close();
    ggml_xdna_causal_capture::writer duplicate;
    assert(!duplicate.open_session(root, 1, 2));
    fs::remove_all(root, ec);
    assert(!ec);
    std::cout << "XDNA_CAUSAL_PAYLOAD_CAPTURE_SELFTEST=PASS\n";
    return 0;
}
