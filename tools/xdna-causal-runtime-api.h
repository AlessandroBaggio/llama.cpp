#pragma once

#include <cstddef>
#include "../ggml/src/ggml-xdna/ggml-xdna-execution-authorization.h"
#include <filesystem>
#include <map>
#include <string>
#include <vector>

struct causal_runtime_record {
    size_t sequence = 0, capture_id = 0, m = 0, k = 0;
    bool target = false;
    std::string node, q4_sha, q8_sha, instruction_sha;
    std::vector<unsigned char> q4, q8;
    std::vector<float> output;
};

struct causal_runtime_plan {
    ggml_xdna_causal::plan_identity identity;
    std::vector<causal_runtime_record> records;
    std::filesystem::path xclbin;
    std::map<size_t, std::filesystem::path> instruction_paths;
    std::map<size_t, std::filesystem::path> factory_support_paths;
    size_t prefix_count = 0;
    std::string selected_csv;
};
typedef int (__cdecl * causal_authorized_runtime_fn)(
    const ggml_xdna_causal::validated_execution_authorization *, causal_runtime_plan *, const wchar_t *);
