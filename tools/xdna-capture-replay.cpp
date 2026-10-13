#include "ggml-backend.h"
#include "../ggml/src/ggml-xdna/ggml-xdna-diagnostic-entry.h"
#include "ggml-xdna-sha256.h"
#include "../vendor/nlohmann/json.hpp"

#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <process.h>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
static constexpr const char * kFrozenXclbinSha = "b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83";

enum class execution_role { prefix_record, injected_history_record, probe_record, post_recreate_control_record, post_recreate_m4096_control_record, post_recreate_m4096_program_only_record, post_recreate_m4096_no_readback_record, post_transition_id8_record, post_no_readback_id8_record, post_same_shape_id8_first_record, post_same_shape_id8_second_record, post_same_shape_id7_first_record, post_same_shape_id7_second_record, post_context_recreate_id8_record };

struct execution_item {
    size_t record_index;
    execution_role role;
};

struct cli_options {
    fs::path capture;
    bool dry_run = false;
    bool prefix_probe = false;
    bool recreate_m9216_bo_bundle_before_probe = false;
    bool recreate_m9216_kernel_before_probe = false;
    bool recreate_m9216_hw_context_before_probe = false;
    bool recreate_m9216_context_and_bo_bundle_before_probe = false;
    bool post_recreate_transition_control = false;
    bool post_recreate_program_only_control = false;
    bool post_recreate_execute_no_readback_control = false;
    bool post_recreate_same_shape_control = false;
    bool post_recreate_same_shape_id7_control = false;
    bool post_recreate_context_recreate_id8_control = false;
    bool post_recreate_id8_context_recreate_only_control = false;
    bool post_recreate_second_recreate_no_execution_control = false;
    bool post_recreate_program_only_context_recreate_control = false;
    bool post_recreate_execute_wait_no_readback_context_recreate_control = false;
    bool post_recreate_execute_wait_no_readback_release_temp_context_recreate_control = false;
    bool post_recreate_execute_wait_release_run_context_recreate_control = false;
    std::vector<size_t> inject_ids;
    size_t prefix_count = 0;
    size_t probe_id = 0;
    size_t post_recreate_control_id = 0;
};

static std::string sha256_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open XCLBIN: " + path.string());
    ggml_xdna_sha256_t hash;
    ggml_xdna_sha256_init(&hash);
    std::array<unsigned char, 64 * 1024> buffer{};
    while (in) {
        in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto count = in.gcount();
        if (count > 0) ggml_xdna_sha256_update(&hash, buffer.data(), static_cast<size_t>(count));
    }
    if (in.bad()) throw std::runtime_error("failed while reading XCLBIN: " + path.string());
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    ggml_xdna_sha256_final(&hash, digest);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < sizeof(digest); ++i) {
        result[i * 2] = hex[digest[i] >> 4];
        result[i * 2 + 1] = hex[digest[i] & 15];
    }
    return result;
}

static std::string sha256_bytes(const std::vector<unsigned char> & bytes) {
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    ggml_xdna_sha256_hash(digest, bytes.data(), bytes.size());
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < sizeof(digest); ++i) {
        result[i * 2] = hex[digest[i] >> 4];
        result[i * 2 + 1] = hex[digest[i] & 15];
    }
    return result;
}

static std::vector<unsigned char> read_exact_file(const fs::path & root, const json & record,
                                                   const char * name_key, const char * size_key,
                                                   const char * sha_key) {
    const std::string name = record.at(name_key).get<std::string>();
    const fs::path relative(name);
    if (name.empty() || relative.is_absolute() || relative.filename() != relative ||
        name == "." || name == ".." || name.find('/') != std::string::npos || name.find('\\') != std::string::npos) {
        throw std::runtime_error(std::string("unsafe artifact path: ") + name_key);
    }
    const fs::path path = root / relative;
    if (fs::is_symlink(fs::symlink_status(path)) || !fs::is_regular_file(path)) {
        throw std::runtime_error(std::string("artifact missing/non-regular: ") + name);
    }
    const auto size = record.at(size_key).get<uint64_t>();
    if (fs::file_size(path) != size || size > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error(std::string("artifact size mismatch: ") + name);
    }
    std::vector<unsigned char> data(static_cast<size_t>(size));
    std::ifstream in(path, std::ios::binary);
    if (!in || (size && !in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size)))) {
        throw std::runtime_error(std::string("artifact read failed: ") + name);
    }
    if (sha256_bytes(data) != record.at(sha_key).get<std::string>()) {
        throw std::runtime_error(std::string("artifact SHA mismatch: ") + name);
    }
    return data;
}

static size_t q4_bytes_for_m(size_t m) {
    return m * 1520;
}

static void validate_plan_and_artifacts(const json & plan, const fs::path & root) {
    if (plan.at("schema_version").get<int>() != 1 ||
        plan.at("profile").get<std::string>() != "q4k_q8k_k2560" ||
        plan.at("xclbin_sha256").get<std::string>() != kFrozenXclbinSha) {
        throw std::runtime_error("ReplayPlan identity mismatch");
    }
    const auto & records = plan.at("records");
    if (!records.is_array() || records.empty()) throw std::runtime_error("ReplayPlan has no records");
    size_t targets = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        const auto & r = records[i];
        const uint64_t id = r.at("submission_id").get<uint64_t>();
        const size_t m = r.at("M").get<size_t>();
        if (id != i + 1 || r.at("K").get<size_t>() != 2560 || r.at("rhs").get<size_t>() != 1 ||
            r.at("opcode").get<size_t>() != 3 ||
            (m != 1024 && m != 4096 && m != 8192 && m != 9216) ||
            r.at("instruction_bytes").get<size_t>() == 0 || r.at("instruction_bytes").get<size_t>() % 4 != 0 ||
            r.at("q4_bytes").get<size_t>() != q4_bytes_for_m(m) || r.at("q8_bytes").get<size_t>() != 2920) {
            throw std::runtime_error("ReplayPlan record shape/size/order mismatch");
        }
        if (m == 9216 && r.at("instruction_bytes").get<size_t>() != 189108) {
            throw std::runtime_error("M9216 instruction size mismatch");
        }
        const bool target = r.at("semantic_target").get<bool>();
        if (target) {
            ++targets;
            if (i + 1 != records.size() || r.at("node").get<std::string>() != "ffn_gate-4" || m != 9216 ||
                r.at("K").get<size_t>() != 2560 || r.at("rhs").get<size_t>() != 1 ||
                !r.at("output_file").is_null() || !r.at("output_bytes").is_null() || !r.at("output_sha256").is_null()) {
                throw std::runtime_error("ReplayPlan terminal target mismatch");
            }
        } else {
            if (i + 1 == records.size() || r.at("output_bytes").get<size_t>() != m * 4 ||
                r.at("output_file").get<std::string>().empty() || r.at("output_sha256").get<std::string>().size() != 64) {
                throw std::runtime_error("ReplayPlan historical output mismatch");
            }
        }
        (void) read_exact_file(root, r, "instruction_file", "instruction_bytes", "instruction_sha256");
        (void) read_exact_file(root, r, "q4_file", "q4_bytes", "q4_sha256");
        (void) read_exact_file(root, r, "q8_file", "q8_bytes", "q8_sha256");
        if (!target) (void) read_exact_file(root, r, "output_file", "output_bytes", "output_sha256");
    }
    if (targets != 1 || !records.back().at("semantic_target").get<bool>()) {
        throw std::runtime_error("ReplayPlan must have exactly one final semantic target");
    }
}

static void print_dry_run(const json & plan) {
    const auto & records = plan.at("records");
    const auto & target = records.back();
    std::cout << "XDNA_CAPTURE_REPLAY_DRY_RUN=PASS\n"
              << "records=" << records.size() << "\n"
              << "historical=" << (records.size() - 1) << "\n"
              << "target_id=" << target.at("submission_id") << "\n"
              << "target=" << target.at("node").get<std::string>() << "\n"
              << "| ID | node | M | instr | Q4 | Q8 | expected output | target |\n"
              << "| --: | ---- | -: | ----: | -: | -: | --------------: | :----: |\n";
    for (const auto & r : records) {
        const bool target_record = r.at("semantic_target").get<bool>();
        std::cout << "| " << r.at("submission_id") << " | " << r.at("node").get<std::string>() << " | " << r.at("M")
                  << " | " << r.at("instruction_bytes") << " | " << r.at("q4_bytes") << " | " << r.at("q8_bytes")
                  << " | " << (target_record ? "-" : r.at("output_bytes").dump())
                  << " | " << (target_record ? "YES" : "NO") << " |\n";
    }
}

static size_t parse_count(const char * text, const char * option) {
    const std::string value = text ? text : "";
    size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    return result;
}

static cli_options parse_cli(int argc, char ** argv) {
    if (argc < 2) throw std::runtime_error("missing capture directory");
    cli_options options;
    options.capture = argv[1];
    bool has_prefix = false, has_probe = false, has_post_recreate_control = false, has_post_recreate_transition = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dry-run") {
            if (options.dry_run) throw std::runtime_error("duplicate --dry-run");
            options.dry_run = true;
        } else if (arg == "--recreate-m9216-bo-bundle-before-probe") {
            if (options.recreate_m9216_bo_bundle_before_probe) throw std::runtime_error("duplicate --recreate-m9216-bo-bundle-before-probe");
            options.recreate_m9216_bo_bundle_before_probe = true;
        } else if (arg == "--recreate-m9216-kernel-before-probe") {
            if (options.recreate_m9216_kernel_before_probe) throw std::runtime_error("duplicate --recreate-m9216-kernel-before-probe");
            options.recreate_m9216_kernel_before_probe = true;
        } else if (arg == "--recreate-m9216-hw-context-before-probe") {
            if (options.recreate_m9216_hw_context_before_probe) throw std::runtime_error("duplicate --recreate-m9216-hw-context-before-probe");
            options.recreate_m9216_hw_context_before_probe = true;
        } else if (arg == "--recreate-m9216-context-and-bo-bundle-before-probe") {
            if (options.recreate_m9216_context_and_bo_bundle_before_probe) throw std::runtime_error("duplicate --recreate-m9216-context-and-bo-bundle-before-probe");
            options.recreate_m9216_context_and_bo_bundle_before_probe = true;
        } else if (arg == "--prefix-count") {
            if (has_prefix || i + 1 == argc) throw std::runtime_error("duplicate or missing --prefix-count value");
            options.prefix_count = parse_count(argv[++i], "--prefix-count");
            has_prefix = true;
        } else if (arg == "--inject-id") {
            if (i + 1 == argc) throw std::runtime_error("missing --inject-id value");
            options.inject_ids.push_back(parse_count(argv[++i], "--inject-id"));
        } else if (arg == "--post-recreate-control-id") {
            if (has_post_recreate_control || i + 1 == argc) throw std::runtime_error("duplicate or missing --post-recreate-control-id value");
            options.post_recreate_control_id = parse_count(argv[++i], "--post-recreate-control-id");
            has_post_recreate_control = true;
        } else if (arg == "--post-recreate-transition-control") {
            if (has_post_recreate_transition || i + 1 == argc) throw std::runtime_error("duplicate or missing --post-recreate-transition-control value");
            const std::string control = argv[++i];
            if (control == "m4096-to-id8") options.post_recreate_transition_control = true;
            else if (control == "m4096-program-only-to-id8") options.post_recreate_program_only_control = true;
            else if (control == "m4096-execute-no-readback-to-id8") options.post_recreate_execute_no_readback_control = true;
            else if (control == "id8-to-id8") options.post_recreate_same_shape_control = true;
            else if (control == "id7-to-id7") options.post_recreate_same_shape_id7_control = true;
            else if (control == "id8-context-recreate-only") { options.post_recreate_transition_control = true; options.post_recreate_id8_context_recreate_only_control = true; options.recreate_m9216_context_and_bo_bundle_before_probe = true; }
            else if (control == "double-recreate-no-execution") { options.post_recreate_transition_control = true; options.post_recreate_second_recreate_no_execution_control = true; options.recreate_m9216_context_and_bo_bundle_before_probe = true; }
            else if (control == "m4096-program-only-context-recreate") { options.post_recreate_transition_control = true; options.post_recreate_program_only_context_recreate_control = true; options.recreate_m9216_context_and_bo_bundle_before_probe = true; }
            else if (control == "m4096-context-recreate-to-id8") { options.post_recreate_transition_control = true; options.post_recreate_context_recreate_id8_control = true; }
            else if (control == "m4096-execute-wait-release-temp-context-recreate") { options.post_recreate_transition_control = true; options.post_recreate_execute_wait_no_readback_release_temp_context_recreate_control = true; options.recreate_m9216_context_and_bo_bundle_before_probe = true; }
            else if (control == "m4096-execute-wait-release-run-context-recreate") { options.post_recreate_transition_control = true; options.post_recreate_execute_wait_release_run_context_recreate_control = true; options.recreate_m9216_context_and_bo_bundle_before_probe = true; }
            else if (control == "m4096-execute-wait-no-readback-context-recreate") { options.post_recreate_transition_control = true; options.post_recreate_execute_wait_no_readback_context_recreate_control = true; options.recreate_m9216_context_and_bo_bundle_before_probe = true; }
            else throw std::runtime_error("--post-recreate-transition-control requires a supported frozen transition control");
            has_post_recreate_transition = true;
        } else if (arg == "--probe-id") {
            if (has_probe || i + 1 == argc) throw std::runtime_error("duplicate or missing --probe-id value");
            options.probe_id = parse_count(argv[++i], "--probe-id");
            has_probe = true;
        } else {
            throw std::runtime_error("unknown or malformed option: " + arg);
        }
    }
    const unsigned recreation_modes = static_cast<unsigned>(options.recreate_m9216_bo_bundle_before_probe) +
        static_cast<unsigned>(options.recreate_m9216_kernel_before_probe) +
        static_cast<unsigned>(options.recreate_m9216_hw_context_before_probe) +
        static_cast<unsigned>(options.recreate_m9216_context_and_bo_bundle_before_probe);
    if (recreation_modes > 1) throw std::runtime_error("M9216 recreation diagnostics are mutually exclusive");
    if (has_post_recreate_control) {
        if (has_post_recreate_transition || options.post_recreate_control_id != 8 || has_probe || !has_prefix || options.prefix_count != 8 ||
            options.inject_ids != std::vector<size_t>{8, 8} || !options.recreate_m9216_context_and_bo_bundle_before_probe || recreation_modes != 1) {
            throw std::runtime_error("post-recreation control requires full-bundle recreation, prefix 8, injections 8,8, and control ID 8 only");
        }
    } else if (has_post_recreate_transition) {
        if (has_post_recreate_control || has_probe || !has_prefix || options.prefix_count != 8 ||
            options.inject_ids != std::vector<size_t>{8, 8} || !options.recreate_m9216_context_and_bo_bundle_before_probe || recreation_modes != 1) {
            throw std::runtime_error("post-recreation transition control requires full-bundle recreation, prefix 8, injections 8,8, and one supported transition mode");
        }
    } else {
        if (has_prefix != has_probe || (!options.inject_ids.empty() && !has_prefix)) {
            throw std::runtime_error("--prefix-count and --probe-id are required together; --inject-id requires both");
        }
        if (recreation_modes && (!has_prefix || options.prefix_count != 8 ||
            options.inject_ids != std::vector<size_t>{8, 8} || options.probe_id != 11)) {
            throw std::runtime_error("M9216 recreation diagnostics require prefix 8 + injected IDs 8,8 + probe 11");
        }
    }
    options.prefix_probe = has_prefix;
    return options;
}

static std::vector<execution_item> select_execution(const json & records, size_t prefix_count,
                                                     const std::vector<size_t> & inject_ids, size_t probe_id,
                                                     size_t post_recreate_control_id) {
    if (post_recreate_control_id != 0) {
        const auto & control = records.size() > 7 ? records[7] : json::object();
        if (post_recreate_control_id != 8 || probe_id != 0 || prefix_count != 8 ||
            inject_ids != std::vector<size_t>{8, 8} || records.size() != 13 ||
            control.at("submission_id").get<size_t>() != 8 || control.at("node").get<std::string>() != "ffn_gate-2" ||
            control.at("M").get<size_t>() != 9216 || control.at("K").get<size_t>() != 2560 ||
            control.at("rhs").get<size_t>() != 1 || control.at("semantic_target").get<bool>() ||
            control.at("instruction_sha256").get<std::string>() != "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" ||
            control.at("q4_sha256").get<std::string>() != "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" ||
            control.at("q8_sha256").get<std::string>() != "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" ||
            control.at("output_sha256").get<std::string>() != "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3") {
            throw std::runtime_error("post-recreation control must select the frozen CF1 ID8 payload");
        }
        std::vector<execution_item> execution;
        execution.reserve(prefix_count + inject_ids.size() + 1);
        for (size_t i = 0; i < prefix_count; ++i) execution.push_back({i, execution_role::prefix_record});
        for (const size_t id : inject_ids) execution.push_back({id - 1, execution_role::injected_history_record});
        execution.push_back({7, execution_role::post_recreate_control_record});
        return execution;
    }
    if (probe_id == 0 || probe_id > records.size()) throw std::runtime_error("probe ID is missing/outside source records");
    if (prefix_count >= probe_id) throw std::runtime_error("prefix count must be less than probe ID");
    const bool one_injected = inject_ids.size() == 1 && inject_ids[0] > prefix_count &&
        inject_ids[0] - prefix_count == 2 && probe_id == inject_ids[0] + 1;
    const bool reversed_pair = inject_ids.size() == 2 && inject_ids[0] > prefix_count &&
        inject_ids[0] - prefix_count == 2 && inject_ids[1] == prefix_count + 1 &&
        probe_id == prefix_count + 3;
    const bool duplicate_pair = inject_ids.size() == 2 && inject_ids[0] >= prefix_count &&
        inject_ids[0] < probe_id && inject_ids[1] == inject_ids[0];
    if (!inject_ids.empty() && !one_injected && !reversed_pair && !duplicate_pair) {
        throw std::runtime_error("injections must be one skipped ID, the explicit reversed pair, or its duplicate");
    }
    for (const size_t id : inject_ids) {
        if (id == 0 || id >= records.size() || records[id - 1].at("semantic_target").get<bool>()) {
            throw std::runtime_error("injected IDs must reference historical records");
        }
    }
    std::vector<execution_item> execution;
    execution.reserve(prefix_count + inject_ids.size() + 1);
    for (size_t i = 0; i < prefix_count; ++i) execution.push_back({i, execution_role::prefix_record});
    for (const size_t id : inject_ids) execution.push_back({id - 1, execution_role::injected_history_record});
    execution.push_back({probe_id - 1, execution_role::probe_record});
    return execution;
}

static std::vector<execution_item> select_double_recreate_no_execution(const json & records, size_t prefix_count, const std::vector<size_t> & inject_ids) {
    if (records.size() != 13 || prefix_count != 8 || inject_ids != std::vector<size_t>{8, 8} ||
        records[7].at("submission_id").get<size_t>() != 8 || records[7].at("node").get<std::string>() != "ffn_gate-2" ||
        records[7].at("M").get<size_t>() != 9216 || records[7].at("semantic_target").get<bool>())
        throw std::runtime_error("double-recreate no-execution control requires frozen CF1 history 1..8,8,8");
    std::vector<execution_item> execution;
    execution.reserve(10);
    for (size_t i = 0; i < prefix_count; ++i) execution.push_back({i, execution_role::prefix_record});
    for (const size_t id : inject_ids) execution.push_back({id - 1, execution_role::injected_history_record});
    return execution;
}

static std::vector<execution_item> select_transition_execution(const json & records, size_t prefix_count,
                                                                const std::vector<size_t> & inject_ids, bool program_only, bool no_readback, bool check_m4096_output_hash = true) {
    if (records.size() != 13 || prefix_count != 8 || inject_ids != std::vector<size_t>{8, 8}) {
        throw std::runtime_error("transition control requires frozen CF1 prefix 8 and ID8 injections 8,8");
    }
    const auto & m4096 = records[6];
    const auto & id8 = records[7];
    if (m4096.at("submission_id").get<size_t>() != 7 || m4096.at("node").get<std::string>() != "z-2" ||
        m4096.at("M").get<size_t>() != 4096 || m4096.at("K").get<size_t>() != 2560 || m4096.at("rhs").get<size_t>() != 1 || m4096.at("semantic_target").get<bool>() ||
        m4096.at("instruction_sha256").get<std::string>() != "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" ||
        m4096.at("q4_sha256").get<std::string>() != "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" ||
        m4096.at("q8_sha256").get<std::string>() != "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a" ||
        (check_m4096_output_hash && m4096.at("output_sha256").get<std::string>() != "6da6b7d7ebf672bc203bb00620882d159cc00371900b84ae4f730aef9cb79c66") ||
        id8.at("submission_id").get<size_t>() != 8 || id8.at("node").get<std::string>() != "ffn_gate-2" ||
        id8.at("M").get<size_t>() != 9216 || id8.at("K").get<size_t>() != 2560 || id8.at("rhs").get<size_t>() != 1 ||
        id8.at("instruction_sha256").get<std::string>() != "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" ||
        id8.at("q4_sha256").get<std::string>() != "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" ||
        id8.at("q8_sha256").get<std::string>() != "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" ||
        id8.at("output_sha256").get<std::string>() != "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3") {
        throw std::runtime_error("transition controls do not match frozen CF1 ID7/ID8 payloads");
    }
    std::vector<execution_item> execution;
    execution.reserve(12);
    for (size_t i = 0; i < prefix_count; ++i) execution.push_back({i, execution_role::prefix_record});
    for (const size_t id : inject_ids) execution.push_back({id - 1, execution_role::injected_history_record});
    execution.push_back({6, program_only ? execution_role::post_recreate_m4096_program_only_record : (no_readback ? execution_role::post_recreate_m4096_no_readback_record : execution_role::post_recreate_m4096_control_record)});
    execution.push_back({7, no_readback ? execution_role::post_no_readback_id8_record : execution_role::post_transition_id8_record});
    return execution;
}
static std::vector<execution_item> select_same_shape_id8_execution(const json & records, size_t prefix_count, const std::vector<size_t> & inject_ids) {
    if (records.size() != 13 || prefix_count != 8 || inject_ids != std::vector<size_t>{8, 8})
        throw std::runtime_error("same-shape control requires frozen CF1 prefix 8 and ID8 injections 8,8");
    const auto & id8 = records[7];
    if (id8.at("submission_id").get<size_t>() != 8 || id8.at("node").get<std::string>() != "ffn_gate-2" ||
        id8.at("M").get<size_t>() != 9216 || id8.at("K").get<size_t>() != 2560 || id8.at("rhs").get<size_t>() != 1 || id8.at("semantic_target").get<bool>() ||
        id8.at("instruction_sha256").get<std::string>() != "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" ||
        id8.at("q4_sha256").get<std::string>() != "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" ||
        id8.at("q8_sha256").get<std::string>() != "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" ||
        id8.at("output_sha256").get<std::string>() != "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3")
        throw std::runtime_error("same-shape control does not match frozen CF1 ID8 payload");
    std::vector<execution_item> execution;
    execution.reserve(12);
    for (size_t i = 0; i < prefix_count; ++i) execution.push_back({i, execution_role::prefix_record});
    for (const size_t id : inject_ids) execution.push_back({id - 1, execution_role::injected_history_record});
    execution.push_back({7, execution_role::post_same_shape_id8_first_record});
    execution.push_back({7, execution_role::post_same_shape_id8_second_record});
    return execution;
}

static std::vector<execution_item> select_same_shape_id7_execution(const json & records, size_t prefix_count, const std::vector<size_t> & inject_ids) {
    if (records.size() != 13 || prefix_count != 8 || inject_ids != std::vector<size_t>{8, 8})
        throw std::runtime_error("ID7 same-shape control requires CF1 prefix 8, ID8 injections 8,8");
    const auto & id7 = records[6];
    if (id7.at("submission_id").get<size_t>() != 7 || id7.at("node").get<std::string>() != "z-2" ||
        id7.at("M").get<size_t>() != 4096 || id7.at("K").get<size_t>() != 2560 || id7.at("rhs").get<size_t>() != 1 || id7.at("semantic_target").get<bool>() ||
        id7.at("instruction_sha256").get<std::string>() != "0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904" ||
        id7.at("q4_sha256").get<std::string>() != "66b717d460b71858de0f190e2cc90a8a97377f142d243fbda53546ee30fc5e03" ||
        id7.at("q8_sha256").get<std::string>() != "4a993d15e65e127d63b7b4816fccbb290f257eb7ce8204596b6413e3ce50106a" ||
        id7.at("output_sha256").get<std::string>() != "6da6b7d7ebf672bc203bb00620882d159cc00371900b84ae4f730aef9cb79c66")
        throw std::runtime_error("ID7 same-shape control does not match frozen CF1 ID7");
    std::vector<execution_item> execution;
    execution.reserve(12);
    for (size_t i = 0; i < prefix_count; ++i) execution.push_back({i, execution_role::prefix_record});
    for (const size_t id : inject_ids) execution.push_back({id - 1, execution_role::injected_history_record});
    execution.push_back({6, execution_role::post_same_shape_id7_first_record});
    execution.push_back({6, execution_role::post_same_shape_id7_second_record});
    return execution;
}
static std::vector<execution_item> select_context_recreate_execution(const json & records, size_t prefix_count, const std::vector<size_t> & inject_ids) {
    (void)select_same_shape_id7_execution(records, prefix_count, inject_ids);
    (void)select_same_shape_id8_execution(records, prefix_count, inject_ids);
    std::vector<execution_item> execution;
    execution.reserve(12);
    for (size_t i = 0; i < prefix_count; ++i) execution.push_back({i, execution_role::prefix_record});
    for (const size_t id : inject_ids) execution.push_back({id - 1, execution_role::injected_history_record});
    execution.push_back({6, execution_role::post_same_shape_id7_first_record});
    execution.push_back({7, execution_role::post_context_recreate_id8_record});
    return execution;
}
static void add_execution_plan(json & plan, const std::vector<execution_item> & execution,
                               size_t prefix_count, const std::vector<size_t> & inject_ids, size_t probe_id,
                               bool recreate_m9216_bo_bundle, bool recreate_m9216_kernel, bool recreate_m9216_hw_context,
                               bool recreate_m9216_context_and_bo_bundle, size_t post_recreate_control_id, bool transition_control, bool program_only_control, bool no_readback_control, bool same_shape_control, bool same_shape_id7_control, bool context_recreate_id8_control, bool id8_context_recreate_only_control, bool double_recreate_no_execution_control, bool program_only_context_recreate_control, bool execute_wait_no_readback_context_recreate_control, bool execute_wait_no_readback_release_temp_context_recreate_control, bool execute_wait_release_run_context_recreate_control) {
    const bool post_control = post_recreate_control_id != 0;
    plan["execution_mode"] = "PREFIX_PROBE";
    if (post_control) plan["execution_mode"] = "PREFIX_POST_RECREATE_CONTROL";
    else if (execute_wait_release_run_context_recreate_control) plan["execution_mode"] = "PREFIX_EXECUTE_WAIT_RELEASE_RUN_CONTEXT_RECREATE_CONTROL";
    else if (execute_wait_no_readback_release_temp_context_recreate_control) plan["execution_mode"] = "PREFIX_EXECUTE_WAIT_RELEASE_TEMP_CONTEXT_RECREATE_CONTROL";
    else if (execute_wait_no_readback_context_recreate_control) plan["execution_mode"] = "PREFIX_EXECUTE_WAIT_NO_READBACK_CONTEXT_RECREATE_CONTROL";
    else if (program_only_context_recreate_control) plan["execution_mode"] = "PREFIX_PROGRAM_ONLY_SECOND_RECREATE_CONTROL";
    else if (double_recreate_no_execution_control) plan["execution_mode"] = "PREFIX_SECOND_RECREATE_NO_EXECUTION_CONTROL";
    else if (id8_context_recreate_only_control) plan["execution_mode"] = "PREFIX_POST_ID8_CONTEXT_RECREATE_ONLY";
    else if (context_recreate_id8_control) plan["execution_mode"] = "PREFIX_POST_RECREATE_CONTEXT_RECREATE_CONTROL";
    else if (same_shape_control || same_shape_id7_control) plan["execution_mode"] = "PREFIX_POST_RECREATE_SAME_SHAPE_CONTROL";
    else if (no_readback_control) plan["execution_mode"] = "PREFIX_POST_RECREATE_NO_READBACK_CONTROL";
    else if (program_only_control) plan["execution_mode"] = "PREFIX_POST_RECREATE_PROGRAM_ONLY_CONTROL";
    else if (transition_control) plan["execution_mode"] = "PREFIX_POST_RECREATE_TRANSITION_CONTROL";
    plan["prefix_count"] = prefix_count;
    if (!inject_ids.empty()) plan["inject_ids"] = inject_ids;
    if (post_control) plan["post_recreate_control_id"] = post_recreate_control_id;
    else if (transition_control || program_only_control || no_readback_control || same_shape_control || same_shape_id7_control || context_recreate_id8_control || id8_context_recreate_only_control || double_recreate_no_execution_control || program_only_context_recreate_control || execute_wait_no_readback_context_recreate_control || execute_wait_no_readback_release_temp_context_recreate_control || execute_wait_release_run_context_recreate_control) {
        const char * name = "m4096-to-id8";
        if (execute_wait_release_run_context_recreate_control) name = "m4096-execute-wait-release-run-context-recreate";
        if (execute_wait_no_readback_release_temp_context_recreate_control) name = "m4096-execute-wait-release-temp-context-recreate";
        else if (execute_wait_no_readback_context_recreate_control) name = "m4096-execute-wait-no-readback-context-recreate";
        if (program_only_context_recreate_control) name = "m4096-program-only-context-recreate";
        else if (double_recreate_no_execution_control) name = "double-recreate-no-execution";
        else if (id8_context_recreate_only_control) name = "id8-context-recreate-only";
        else if (context_recreate_id8_control) name = "m4096-context-recreate-to-id8";
        else if (same_shape_id7_control) name = "id7-to-id7";
        else if (same_shape_control) name = "id8-to-id8";
        else if (no_readback_control) name = "m4096-execute-no-readback-to-id8";
        else if (program_only_control) name = "m4096-program-only-to-id8";
        plan["post_recreate_transition_control"] = name;
    }
    else plan["probe_id"] = probe_id;
    if (recreate_m9216_bo_bundle) plan["recreate_m9216_bo_bundle_before_probe"] = true;
    if (recreate_m9216_kernel) plan["recreate_m9216_kernel_before_probe"] = true;
    if (recreate_m9216_hw_context) plan["recreate_m9216_hw_context_before_probe"] = true;
    if (recreate_m9216_context_and_bo_bundle) plan["recreate_m9216_context_and_bo_bundle_before_probe"] = true;
    plan["execution"] = json::array();
    for (const auto & item : execution) {
        const char * role = item.role == execution_role::prefix_record ? "PREFIX_RECORD" :
            item.role == execution_role::injected_history_record ? "INJECTED_HISTORY_RECORD" :
            item.role == execution_role::probe_record ? "PROBE_RECORD" :
            item.role == execution_role::post_recreate_control_record ? (id8_context_recreate_only_control ? "POST_RECREATE_ID8_FIRST" : "POST_RECREATE_CONTROL") :
            item.role == execution_role::post_recreate_m4096_control_record ? "POST_RECREATE_M4096_CONTROL" :
            item.role == execution_role::post_recreate_m4096_program_only_record ? "PROGRAM_M4096_INSTRUCTION_STATE_ONLY" :
            item.role == execution_role::post_recreate_m4096_no_readback_record ? "M4096_EXECUTE_AND_WAIT_NO_READBACK" :
            item.role == execution_role::post_no_readback_id8_record ? "POST_M4096_EXEC_NO_READBACK_ID8_M9216" :
            item.role == execution_role::post_same_shape_id8_first_record ? "POST_RECREATE_ID8_FIRST" :
            item.role == execution_role::post_same_shape_id8_second_record ? "POST_ID8_SAME_SHAPE_ID8_SECOND" :
            item.role == execution_role::post_same_shape_id7_first_record ? "POST_RECREATE_ID7_FIRST" :
            item.role == execution_role::post_same_shape_id7_second_record ? "POST_ID7_SAME_SHAPE_ID7_SECOND" :
            item.role == execution_role::post_context_recreate_id8_record ? "POST_CONTEXT_RECREATE_ID8_M9216" : "POST_TRANSITION_ID8_M9216";
        plan["execution"].push_back({
            {"source_id", plan.at("records")[item.record_index].at("submission_id")}, {"role", role}});
    }
}

static void print_prefix_probe_dry_run(const json & plan) {
    const auto & records = plan.at("records");
    const auto & execution = plan.at("execution");
    const bool post_control = plan.contains("post_recreate_control_id");
    const bool transition_control = plan.contains("post_recreate_transition_control");
    const bool program_only_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-program-only-to-id8";
    const bool no_readback_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-execute-no-readback-to-id8";
    const bool same_shape_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "id8-to-id8";
    const bool same_shape_id7_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "id7-to-id7";
    const bool context_recreate_id8_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-context-recreate-to-id8";
    const bool id8_context_recreate_only_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "id8-context-recreate-only";
    const bool double_recreate_no_execution_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "double-recreate-no-execution";
    const bool program_only_context_recreate_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-program-only-context-recreate";
    const bool execute_wait_no_readback_context_recreate_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-execute-wait-no-readback-context-recreate";
    const bool execute_wait_no_readback_release_temp_context_recreate_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-execute-wait-release-temp-context-recreate";
    const bool execute_wait_release_run_context_recreate_control = transition_control && plan.at("post_recreate_transition_control").get<std::string>() == "m4096-execute-wait-release-run-context-recreate";
    const size_t prefix_count = plan.at("prefix_count").get<size_t>();
    const size_t probe_id = post_control || transition_control ? 0 : plan.at("probe_id").get<size_t>();
    const size_t control_id = post_control ? plan.at("post_recreate_control_id").get<size_t>() : 0;
    const auto inject_ids = plan.contains("inject_ids") ? plan.at("inject_ids").get<std::vector<size_t>>() : std::vector<size_t>{};
    if (execute_wait_no_readback_release_temp_context_recreate_control) std::cout << "XDNA_CAPTURE_REPLAY_EXECUTE_WAIT_RELEASE_TEMP_SECOND_RECREATE_DRY_RUN=PASS\n";
    else if (execute_wait_no_readback_context_recreate_control) std::cout << "XDNA_CAPTURE_REPLAY_EXECUTE_WAIT_NO_READBACK_SECOND_RECREATE_DRY_RUN=PASS\n";
    else if (program_only_context_recreate_control) std::cout << "XDNA_CAPTURE_REPLAY_PROGRAM_ONLY_SECOND_RECREATE_DRY_RUN=PASS\n";
    else if (double_recreate_no_execution_control) std::cout << "XDNA_CAPTURE_REPLAY_SECOND_RECREATE_NO_EXECUTION_DRY_RUN=PASS\n";
    else if (id8_context_recreate_only_control) std::cout << "XDNA_CAPTURE_REPLAY_POST_ID8_CONTEXT_RECREATE_ONLY_DRY_RUN=PASS\n";
    else if (context_recreate_id8_control) std::cout << "XDNA_CAPTURE_REPLAY_M4096_CONTEXT_RECREATE_TO_ID8_DRY_RUN=PASS\n";
    else if (same_shape_id7_control) std::cout << "XDNA_CAPTURE_REPLAY_SAME_SHAPE_ID7_DRY_RUN=PASS\n";
    else if (same_shape_control) std::cout << "XDNA_CAPTURE_REPLAY_SAME_SHAPE_ID8_DRY_RUN=PASS\n";
    else if (no_readback_control) std::cout << "XDNA_CAPTURE_REPLAY_M4096_NO_READBACK_DRY_RUN=PASS\n";
    else if (program_only_control) std::cout << "XDNA_CAPTURE_REPLAY_POST_RECREATE_PROGRAM_ONLY_DRY_RUN=PASS\n";
    else if (transition_control) std::cout << "XDNA_CAPTURE_REPLAY_POST_RECREATE_TRANSITION_DRY_RUN=PASS\n";
    else if (post_control) std::cout << "XDNA_CAPTURE_REPLAY_POST_RECREATE_CONTROL_DRY_RUN=PASS\n";
    else std::cout << "XDNA_CAPTURE_REPLAY_PREFIX_PROBE_DRY_RUN=PASS\n";
    std::cout << "source_records=" << records.size() << "\n" << "prefix_count=" << prefix_count << "\n";
    if (post_control) std::cout << "post_recreate_control_id=" << control_id << "\n";
    else if (transition_control) {
        const char * name = "m4096-to-id8";
        if (execute_wait_release_run_context_recreate_control) name = "m4096-execute-wait-release-run-context-recreate";
        if (execute_wait_no_readback_release_temp_context_recreate_control) name = "m4096-execute-wait-release-temp-context-recreate";
        else if (program_only_context_recreate_control) name = "m4096-program-only-context-recreate";
        else if (execute_wait_no_readback_context_recreate_control) name = "m4096-execute-wait-no-readback-context-recreate";
        else if (double_recreate_no_execution_control) name = "double-recreate-no-execution";
        else if (id8_context_recreate_only_control) name = "id8-context-recreate-only";
        else if (context_recreate_id8_control) name = "m4096-context-recreate-to-id8";
        else if (same_shape_id7_control) name = "id7-to-id7";
        else if (same_shape_control) name = "id8-to-id8";
        else if (no_readback_control) name = "m4096-execute-no-readback-to-id8";
        else if (program_only_control) name = "m4096-program-only-to-id8";
        std::cout << "post_recreate_transition_control=" << name << "\n";
    }
    else std::cout << "probe_id=" << probe_id << "\n";
    std::cout << "executions=" << execution.size() << "\n";
    if (double_recreate_no_execution_control) std::cout << "TRANSITION_STAGE=SECOND_CONTEXT_RECREATE_NO_EXECUTION generation1_execution_count=0 previous_experimental_source=NONE second_recreation_requested=YES\nNO_EXECUTION_RECREATE_GUARD=PASS\n";
    if (id8_context_recreate_only_control) std::cout << "POST_RECREATE_ID8_FIRST source_id=8 M=9216 experiment_ordinal=1 context_generation=1 context_ordinal=1\nTRANSITION_STAGE=M9216_TO_CONTEXT_RECREATE_CONTROL previous_source_id=8 previous_M=9216 context_recreate_only=YES no_post_recreate_submission=YES\n";
    if (context_recreate_id8_control) std::cout << "POST_RECREATE_ID7_FIRST source_id=7 M=4096 experiment_ordinal=1 context_generation=1 context_ordinal=1\nPOST_CONTEXT_RECREATE_ID8 source_id=8 M=9216 experiment_ordinal=2 context_generation=2 context_ordinal=1\nTRANSITION_STAGE=POST_ID7_CONTEXT_RECREATE_BEGIN/END\n";
    else if (same_shape_id7_control) std::cout << "POST_RECREATE_ID7_FIRST source_id=7 M=4096 ordinal=1\nPOST_ID7_SAME_SHAPE_ID7_SECOND source_id=7 M=4096 ordinal=2\nTRANSITION_STAGE=M4096_TO_SAME_M4096\n";
    if (!inject_ids.empty()) {
        std::cout << "inject_ids=";
        for (size_t i = 0; i < inject_ids.size(); ++i) std::cout << (i ? "," : "") << inject_ids[i];
        std::cout << "\n";
    }
    std::cout << "xrt_initialized=NO\n"
              << "| order | source ID | node | M | role |\n"
              << "| ----: | --------: | ---- | -: | ---- |\n";
    for (size_t i = 0; i < execution.size(); ++i) {
        const size_t source_id = execution[i].at("source_id").get<size_t>();
        const std::string role = execution[i].at("role").get<std::string>();
        const auto & r = records[source_id - 1];
        std::cout << "| " << i + 1 << " | " << source_id << " | " << r.at("node").get<std::string>()
                  << " | " << r.at("M") << " | " << role << " |\n";
    }
    if (execute_wait_release_run_context_recreate_control) {
        std::cout << "XDNA_CAPTURE_REPLAY_RUN_ONLY_RELEASE_SECOND_RECREATE_DRY_RUN=PASS\nRUN_ONLY_RELEASE_GUARD=PLANNED\n";
        std::cout << "replay_sequence_begin\n";
        for (size_t i = 0; i < 10; ++i) std::cout << execution[i].at("source_id").get<size_t>() << "\n";
        std::cout << execution[10].at("source_id").get<size_t>() << "\n";
        std::cout << "\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<EXECUTE_WAIT_NO_READBACK_ID7_M4096>\n<RELEASE_COMPLETED_EXPERIMENT_RUN_ONLY>\n<RETAIN_TEMP_M4096_EXECUTION_RESOURCES_AND_MAP_ALIASES>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_RUN_RELEASE>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end\n";
        return;
    }
    if (execute_wait_no_readback_release_temp_context_recreate_control) {
        std::cout << "replay_sequence_begin\n";
        for (size_t i = 0; i < 10; ++i) std::cout << execution[i].at("source_id").get<size_t>() << "\n";
        std::cout << execution[10].at("source_id").get<size_t>() << "\n";
        std::cout << "\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<EXECUTE_WAIT_NO_READBACK_ID7_M4096>\n<RELEASE_COMPLETED_EXPERIMENT_RUN_IF_STILL_ALIVE>\n<RELEASE_TEMP_M4096_EXECUTION_RESOURCES_IF_STILL_ALIVE>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_EXECUTION_RELEASE>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end\n";
        return;
    }
    if (execute_wait_no_readback_context_recreate_control) {
        std::cout << "replay_sequence_begin\n";
        for (size_t i = 0; i < 10; ++i) std::cout << execution[i].at("source_id").get<size_t>() << "\n";
        std::cout << execution[10].at("source_id").get<size_t>() << "\n";
        std::cout << "\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<EXECUTE_WAIT_NO_READBACK_ID7_M4096>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_EXECUTION>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end\n";
        return;
    }
    if (program_only_context_recreate_control) {
        std::cout << "replay_sequence_begin\n";
        for (const auto & item : execution) std::cout << item.at("source_id").get<size_t>() << "\n";
        std::cout << "\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<PROGRAM_ONLY_M4096_TO_SECOND_CONTEXT_RECREATE source_id=7 M=4096>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_PROGRAM_ONLY>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end\n";
        return;
    }
    if (double_recreate_no_execution_control) {
        std::cout << "replay_sequence_begin\n";
        for (const auto & item : execution) std::cout << item.at("source_id").get<size_t>() << "\n";
        std::cout << "\n<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_NO_EXECUTION>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end\n";
        return;
    }
    if (id8_context_recreate_only_control) {
        std::cout << "replay_sequence_begin\n";
        for (size_t i = 0; i < prefix_count + inject_ids.size(); ++i) std::cout << execution[i].at("source_id").get<size_t>() << "\n";
        std::cout << "<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>\n8\n<POST_RECREATE_ID8_FIRST>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_ID8>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\nreplay_sequence_end\n";
        return;
    }
    if (plan.contains("recreate_m9216_bo_bundle_before_probe") || plan.contains("recreate_m9216_kernel_before_probe") ||
        plan.contains("recreate_m9216_hw_context_before_probe") || plan.contains("recreate_m9216_context_and_bo_bundle_before_probe")) {
        std::cout << "replay_sequence_begin\n";
        const char * marker = plan.contains("recreate_m9216_context_and_bo_bundle_before_probe")
            ? "<RECREATE_M9216_CONTEXT_KERNEL_AND_BO_BUNDLE>" : (plan.contains("recreate_m9216_hw_context_before_probe")
                ? "<RECREATE_M9216_HW_CONTEXT>" : (plan.contains("recreate_m9216_kernel_before_probe")
                    ? "<RECREATE_M9216_KERNEL>" : "<RECREATE_M9216_BO_BUNDLE>"));
        const size_t history_size = prefix_count + inject_ids.size();
        if (post_control || transition_control) {
            for (size_t i = 0; i < history_size; ++i) std::cout << execution[i].at("source_id").get<size_t>() << "\n";
            std::cout << marker << "\n";
            if (double_recreate_no_execution_control) std::cout << "<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_NO_EXECUTION>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\n";
            else if (post_control) std::cout << "8\n<POST_RECREATE_CONTROL_ID8>\n";
            else if (id8_context_recreate_only_control) std::cout << "8\n<POST_RECREATE_ID8_FIRST>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_ID8>\n<STOP_AFTER_CONTEXT_RECREATE_DIAGNOSTIC>\n";
            else if (context_recreate_id8_control) std::cout << "7\n<POST_RECREATE_ID7_FIRST>\n<RECREATE_M9216_CONTEXT_AND_KERNEL_ONLY_AFTER_ID7>\n8\n<POST_CONTEXT_RECREATE_ID8_M9216>\n";
            else if (same_shape_id7_control) std::cout << "<POST_RECREATE_ID7_FIRST>\n<POST_ID7_SAME_SHAPE_ID7_SECOND>\n";
            else if (same_shape_control) std::cout << "8\n<POST_RECREATE_ID8_FIRST>\n8\n<POST_ID8_SAME_SHAPE_ID8_SECOND>\n";
            else if (no_readback_control) std::cout << "<M4096_EXECUTE_AND_WAIT_NO_READBACK>\n<POST_M4096_EXEC_NO_READBACK_ID8_M9216>\n";
            else if (program_only_control) std::cout << "<PROGRAM_M4096_INSTRUCTION_STATE_ONLY>\n<POST_PROGRAM_ID8_M9216>\n";
            else if (transition_control) std::cout << "<POST_RECREATE_M4096_CONTROL_ID7>\n<POST_TRANSITION_ID8_M9216>\n";
        } else {
            for (size_t i = 0; i < execution.size(); ++i) {
                if (i == history_size) std::cout << marker << "\n";
                std::cout << execution[i].at("source_id").get<size_t>() << "\n";
            }
        }
        std::cout << "replay_sequence_end\n";
    }
}






static int invoke_planner(const fs::path & exe, const fs::path & capture, const fs::path & output, bool quiet) {
    const fs::path planner = fs::weakly_canonical(exe.parent_path() / "../../scripts/xdna-capture-replay-plan.py");
    const std::string planner_s = planner.string(), capture_s = capture.string(), output_s = output.string();
    std::vector<const char *> argv = {"python", planner_s.c_str(), capture_s.c_str(), "--write-plan", output_s.c_str()};
    if (quiet) argv.push_back("--quiet");
    argv.push_back(nullptr);
    return static_cast<int>(_spawnvp(_P_WAIT, "python", argv.data()));
}

static bool capture_mode_enabled() {
    const char * value = std::getenv("GGML_XDNA_CAPTURE_EXACT_HISTORY");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

int main(int argc, char ** argv) {
    cli_options options;
    try {
        options = parse_cli(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Usage: xdna-capture-replay.exe <capture_dir> [--dry-run] [--prefix-count N --inject-id I --probe-id P | --prefix-count 8 --inject-id 8 --inject-id 8 --post-recreate-control-id 8 | --post-recreate-transition-control m4096-to-id8|m4096-program-only-to-id8|m4096-execute-no-readback-to-id8|id8-to-id8|id7-to-id7|m4096-context-recreate-to-id8|id8-context-recreate-only|double-recreate-no-execution|m4096-program-only-context-recreate|m4096-execute-wait-release-temp-context-recreate|m4096-execute-wait-release-run-context-recreate] [recreation option]\\nXDNA capture replay arguments rejected: %s\\n", e.what());
        return 2;
    }
    if (capture_mode_enabled()) {
        std::fprintf(stderr, "XDNA capture replay refuses while capture mode is enabled\n");
        return 3;
    }
    try {
        const fs::path capture = fs::canonical(options.capture);
        if (!fs::is_directory(capture)) throw std::runtime_error("capture_dir is not a directory");
        const auto exe = fs::absolute(argv[0]);
        const auto temp_plan = fs::temp_directory_path() / ("xdna-capture-plan-" + std::to_string(_getpid()) + ".json");
        if (invoke_planner(exe, capture, temp_plan, options.post_recreate_control_id != 0 || options.post_recreate_transition_control || options.post_recreate_program_only_control || options.post_recreate_execute_no_readback_control || options.post_recreate_same_shape_control || options.post_recreate_same_shape_id7_control || options.post_recreate_context_recreate_id8_control || options.post_recreate_id8_context_recreate_only_control || options.post_recreate_second_recreate_no_execution_control || options.post_recreate_program_only_context_recreate_control || options.post_recreate_execute_wait_no_readback_context_recreate_control || options.post_recreate_execute_wait_no_readback_release_temp_context_recreate_control || options.post_recreate_execute_wait_release_run_context_recreate_control) != 0) throw std::runtime_error("XDNA capture replay planner rejected capture");
        std::ifstream plan_file(temp_plan, std::ios::binary);
        if (!plan_file) throw std::runtime_error("planner did not produce replay-plan.json");
        json plan = json::parse(plan_file);
        plan_file.close();
        const fs::path source = fs::canonical(plan.at("source_capture_dir").get<std::string>());
        if (source != capture) throw std::runtime_error("ReplayPlan source directory mismatch");
        validate_plan_and_artifacts(plan, source);
        std::vector<execution_item> execution;
        if (options.prefix_probe) {
            if (options.post_recreate_second_recreate_no_execution_control)
                execution = select_double_recreate_no_execution(plan.at("records"), options.prefix_count, options.inject_ids);
            else if (options.post_recreate_execute_wait_no_readback_context_recreate_control || options.post_recreate_execute_wait_no_readback_release_temp_context_recreate_control || options.post_recreate_execute_wait_release_run_context_recreate_control) {
                execution = select_transition_execution(plan.at("records"), options.prefix_count, options.inject_ids, false, true, false);
                execution.resize(11);
            }
            else if (options.post_recreate_program_only_context_recreate_control) {
                execution = select_transition_execution(plan.at("records"), options.prefix_count, options.inject_ids, true, false);
                execution.resize(10);
            }
            else if (options.post_recreate_id8_context_recreate_only_control)
                execution = select_execution(plan.at("records"), options.prefix_count, options.inject_ids, 0, 8);
            else if (options.post_recreate_context_recreate_id8_control)
                execution = select_context_recreate_execution(plan.at("records"), options.prefix_count, options.inject_ids);
            else if (options.post_recreate_same_shape_id7_control)
                execution = select_same_shape_id7_execution(plan.at("records"), options.prefix_count, options.inject_ids);
            else if (options.post_recreate_same_shape_control)
                execution = select_same_shape_id8_execution(plan.at("records"), options.prefix_count, options.inject_ids);
            else if (options.post_recreate_transition_control || options.post_recreate_program_only_control || options.post_recreate_execute_no_readback_control)
                execution = select_transition_execution(plan.at("records"), options.prefix_count, options.inject_ids, options.post_recreate_program_only_control, options.post_recreate_execute_no_readback_control);
            else execution = select_execution(plan.at("records"), options.prefix_count, options.inject_ids, options.probe_id, options.post_recreate_control_id);
        }

        if (options.recreate_m9216_bo_bundle_before_probe || options.recreate_m9216_kernel_before_probe ||
            options.recreate_m9216_hw_context_before_probe || options.recreate_m9216_context_and_bo_bundle_before_probe) {
            const auto & records = plan.at("records");
            const auto & id8 = records.size() > 7 ? records.at(7) : json::object();
            const bool frozen_id8 = records.size() == 13 && id8.at("submission_id").get<size_t>() == 8 &&
                id8.at("node").get<std::string>() == "ffn_gate-2" && id8.at("M").get<size_t>() == 9216 &&
                id8.at("instruction_sha256").get<std::string>() == "3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3" &&
                id8.at("q4_sha256").get<std::string>() == "748472f6c07290ac4ffcd6ed3250a13166c8f3f284e17c5edaa4b7e6a3343c1c" &&
                id8.at("q8_sha256").get<std::string>() == "150f43cd0191b83eab9031fa095d3cc65053f9789a831879769bc7af04ea1e9e" &&
                id8.at("output_sha256").get<std::string>() == "06d846b2de3bc807d41adb6e85565e8f6703acd3bb0f2ab97c0c9213cba892a3";
            const bool frozen_id11 = records.size() == 13 && records.at(10).at("M").get<size_t>() == 9216 &&
                records.at(10).at("node").get<std::string>() == "ffn_up-3" && !records.at(10).at("semantic_target").get<bool>();
            if (!frozen_id8 || (!options.post_recreate_control_id && !options.post_recreate_transition_control && !options.post_recreate_program_only_control && !options.post_recreate_execute_no_readback_control && !options.post_recreate_same_shape_control && !options.post_recreate_same_shape_id7_control && !options.post_recreate_context_recreate_id8_control && !frozen_id11)) {
            }
        }
        const char * configured = std::getenv("GGML_XDNA_Q4K_RUNTIME_XCLBIN");
        const fs::path xclbin = configured != nullptr && configured[0] != '\0'
            ? fs::canonical(configured)
            : fs::canonical(plan.at("xclbin_path").get<std::string>());
        if (sha256_file(xclbin) != kFrozenXclbinSha) throw std::runtime_error("runtime XCLBIN SHA mismatch; refusing device initialization");
        if (_putenv_s("GGML_XDNA_Q4K_RUNTIME_XCLBIN", xclbin.string().c_str()) != 0) throw std::runtime_error("failed to set runtime XCLBIN path");
        if (options.prefix_probe) add_execution_plan(plan, execution, options.prefix_count,
            options.inject_ids, options.probe_id, options.recreate_m9216_bo_bundle_before_probe,
            options.recreate_m9216_kernel_before_probe, options.recreate_m9216_hw_context_before_probe,
            options.recreate_m9216_context_and_bo_bundle_before_probe || options.post_recreate_id8_context_recreate_only_control || options.post_recreate_second_recreate_no_execution_control || options.post_recreate_program_only_context_recreate_control,
            options.post_recreate_control_id, options.post_recreate_transition_control, options.post_recreate_program_only_control,
            options.post_recreate_execute_no_readback_control, options.post_recreate_same_shape_control, options.post_recreate_same_shape_id7_control,
            options.post_recreate_context_recreate_id8_control, options.post_recreate_id8_context_recreate_only_control,
            options.post_recreate_second_recreate_no_execution_control, options.post_recreate_program_only_context_recreate_control,
            options.post_recreate_execute_wait_no_readback_context_recreate_control,
            options.post_recreate_execute_wait_no_readback_release_temp_context_recreate_control, options.post_recreate_execute_wait_release_run_context_recreate_control);
        if (options.dry_run) {
            if (options.prefix_probe) print_prefix_probe_dry_run(plan);
            else print_dry_run(plan);
            fs::remove(temp_plan);
            return 0;
        }
        if (options.prefix_probe) {
            std::ofstream derived(temp_plan, std::ios::binary | std::ios::trunc);
            if (!derived) throw std::runtime_error("cannot write derived prefix/probe execution plan");
            derived << plan.dump(2) << '\n';
            derived.close();
            if (!derived) throw std::runtime_error("failed writing derived prefix/probe execution plan");
        }

        // No backend DLL is loaded and no XRT device is created before all host gates above pass.
        const std::string backend_dir = exe.parent_path().string();
        ggml_backend_load_all_from_path(backend_dir.c_str());
        ggml_backend_dev_t device = ggml_backend_dev_by_name("XDNA0");
        if (device == nullptr) throw std::runtime_error("XDNA0 backend not found");
        const std::string init_params = "profile=q4k_q8k_k2560;" + xclbin.string();
        ggml_backend_t backend = ggml_backend_dev_init(device, init_params.c_str());
        if (backend == nullptr) throw std::runtime_error("XDNA resource initialization failed");
        const int result = ggml_backend_xdna_diagnostic_replay(backend, temp_plan.string().c_str());
        ggml_backend_free(backend);
        fs::remove(temp_plan);
        return result == 0 ? 0 : 4;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "XDNA capture replay failed closed: %s\n", e.what());
        return 1;
    }
}
