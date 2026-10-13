#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "xdna-causal-runtime-api.h"
#include "../ggml/src/ggml-xdna/ggml-xdna-sha256.h"
#include "../vendor/nlohmann/json.hpp"

#include <array>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <process.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using ggml_xdna_causal::authorization_request;
using ggml_xdna_causal::authorize_execution;
using ggml_xdna_causal::canonical_cell_label;
using ggml_xdna_causal::enter_authorized_runtime;
using ggml_xdna_causal::plan_identity;
using ggml_xdna_causal::parse_prefix_ids;
using ggml_xdna_causal::validated_execution_authorization;

static const std::map<size_t, std::pair<std::string, std::string>> factory_support = {
    {1024, {"experiments/xdna-q4k-runtime-m-sweep-aq1/aq1_m1024.insts.bin", "58b01921a65b0d11374d4ca3db7d55d5355087be89d542d793125878e1ca2d7e"}},
    {8192, {"experiments/xdna-q4k-runtime-m-sweep-aq1/aq1_m8192.insts.bin", "72f88794f6dbb60f695840b8bec10726f1e2a34edfbb9f5e01079fd392950c8f"}},
};

struct options {
    fs::path config;
    std::string mode, pair, authorized_cell;
    std::vector<size_t> prefix_ids;
    bool prefix_ids_provided = false, execute = false, dry_run = false;
};


static std::string join_ids(const std::vector<size_t> & ids, char separator = ',') {
    std::string out;
    for (size_t id : ids) { if (!out.empty()) out += separator; out += std::to_string(id); }
    return out;
}

static options parse_cli(int argc, char ** argv) {
    options out;
    std::string raw_prefix_ids;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc && out.config.empty()) out.config = argv[++i];
        else if (arg == "--mode" && i + 1 < argc && out.mode.empty()) out.mode = argv[++i];
        else if (arg == "--pair" && i + 1 < argc && out.pair.empty()) out.pair = argv[++i];
        else if (arg == "--prefix-ids" && i + 1 < argc && !out.prefix_ids_provided) { out.prefix_ids_provided = true; raw_prefix_ids = argv[++i]; }
        else if (arg == "--authorize-cell" && i + 1 < argc && out.authorized_cell.empty()) out.authorized_cell = argv[++i];
        else if (arg == "--execute" && !out.execute) out.execute = true;
        else if (arg == "--dry-run" && !out.dry_run) out.dry_run = true;
        else throw std::runtime_error("unknown, duplicate, or incomplete option: " + arg);
    }
    if (out.execute && out.dry_run) throw std::runtime_error("--execute and --dry-run are mutually exclusive");
    if (out.config.empty() || (out.mode != "CLEAN" && out.mode != "CAPTURE_PREFIX") || (out.pair != "A" && out.pair != "B"))
        throw std::runtime_error("required: --config GP1.json --mode CLEAN|CAPTURE_PREFIX --pair A|B [--prefix-ids IDS] [--dry-run|--execute]");
    if (out.prefix_ids_provided) {
        if (out.mode != "CAPTURE_PREFIX") throw std::runtime_error("--prefix-ids is valid only with CAPTURE_PREFIX");
        if (out.pair != "B") throw std::runtime_error("explicit prefix subsets are supported only for Pair B");
        out.prefix_ids = parse_prefix_ids(raw_prefix_ids);
    } else if (out.mode == "CAPTURE_PREFIX") {
        for (size_t id = 1; id <= 12; ++id) out.prefix_ids.push_back(id);
    }
    const std::string selected_cell = canonical_cell_label(out.pair, out.mode, out.prefix_ids);
    if (out.execute && out.authorized_cell != selected_cell) throw std::runtime_error("--execute requires --authorize-cell matching exactly " + selected_cell);
    if (!out.execute && !out.authorized_cell.empty()) {
        if (!out.dry_run) throw std::runtime_error("--authorize-cell without execution is valid only as a dry-run identity check");
        if (out.authorized_cell != selected_cell) throw std::runtime_error("--authorize-cell does not match canonical cell identity " + selected_cell);
    }
    return out;
}

static std::string sha_bytes(const void * data, size_t size) {
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
    ggml_xdna_sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    static constexpr char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < sizeof(digest); ++i) { out[2*i] = hex[digest[i] >> 4]; out[2*i+1] = hex[digest[i] & 15]; }
    return out;
}

static std::vector<unsigned char> read_checked(const fs::path & path, size_t bytes, const std::string & digest) {
    if (!path.is_absolute() || fs::is_symlink(fs::symlink_status(path)) || !fs::is_regular_file(path) || fs::file_size(path) != bytes)
        throw std::runtime_error("missing, unsafe, or wrong-sized artifact: " + path.string());
    std::vector<unsigned char> data(bytes);
    std::ifstream in(path, std::ios::binary);
    if (!in || (bytes && !in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes))))
        throw std::runtime_error("artifact read failed: " + path.string());
    if (sha_bytes(data.data(), data.size()) != digest) throw std::runtime_error("artifact SHA256 mismatch: " + path.string());
    return data;
}

static std::string sha_file(const fs::path & path) {
    if (!path.is_absolute() || fs::is_symlink(fs::symlink_status(path)) || !fs::is_regular_file(path))
        throw std::runtime_error("missing or unsafe artifact: " + path.string());
    ggml_xdna_sha256_t hash; ggml_xdna_sha256_init(&hash);
    std::array<unsigned char, 65536> buffer{};
    std::ifstream in(path, std::ios::binary);
    while (in) { in.read(reinterpret_cast<char *>(buffer.data()), buffer.size()); auto n = in.gcount(); if (n > 0) ggml_xdna_sha256_update(&hash, buffer.data(), static_cast<size_t>(n)); }
    if (in.bad()) throw std::runtime_error("artifact read failed: " + path.string());
    unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE]; ggml_xdna_sha256_final(&hash, digest);
    static constexpr char hex[] = "0123456789abcdef"; std::string out(64, '0');
    for (size_t i = 0; i < sizeof(digest); ++i) { out[2*i] = hex[digest[i] >> 4]; out[2*i+1] = hex[digest[i] & 15]; }
    return out;
}

static fs::path checked_config(const fs::path & input) {
    const fs::path p = fs::canonical(input);
    const fs::path expected = fs::canonical(fs::path(__FILE__).parent_path().parent_path() / "experiments/xdna-gn1-causal-matrix-gp1.json");
    if (p != expected) throw std::runtime_error("only the frozen GP1 config is accepted");
    return p;
}

static int run_planner(const fs::path & repo, const fs::path & config, const options & opt, const fs::path & plan) {
    const fs::path script = repo / "scripts/xdna-causal-matrix-plan.py";
    std::vector<std::string> args = {"python", script.string(), "--config", config.string(), "--mode", opt.mode, "--pair", opt.pair};
    if (opt.prefix_ids_provided) { args.push_back("--prefix-ids"); args.push_back(join_ids(opt.prefix_ids)); }
    args.push_back("--plan-out"); args.push_back(plan.string());
    std::vector<const char *> argv; argv.reserve(args.size() + 1);
    for (const auto & arg : args) argv.push_back(arg.c_str());
    argv.push_back(nullptr);
    return static_cast<int>(_spawnvp(_P_WAIT, "python", argv.data()));
}

static causal_runtime_plan prepare_host_plan(const options & opt) {
    const fs::path config_path = checked_config(opt.config);
    const fs::path repo = config_path.parent_path().parent_path();
    const fs::path project_root = fs::canonical(repo / "../..");
    std::ifstream config_stream(config_path);
    const json config = json::parse(config_stream);
    const fs::path xclbin = fs::canonical(project_root / config.at("xclbin").at("path_from_root").get<std::string>());
    if (sha_file(xclbin) != config.at("xclbin").at("sha256").get<std::string>()) throw std::runtime_error("XCLBIN hash mismatch");

    causal_runtime_plan result;
    result.xclbin = xclbin;
    for (const auto & item : config.at("instructions").items()) {
        const size_t m = std::stoul(item.key()); const auto & artifact = item.value();
        const fs::path path = fs::canonical(project_root / artifact.at("path_from_root").get<std::string>());
        if (sha_file(path) != artifact.at("sha256").get<std::string>()) throw std::runtime_error("instruction hash mismatch");
        result.instruction_paths[m] = path;
    }
    for (const auto & [m, item] : factory_support) {
        const fs::path path = fs::canonical(project_root / item.first);
        if (sha_file(path) != item.second) throw std::runtime_error("backend factory support instruction hash mismatch");
        result.factory_support_paths[m] = path;
    }

    const fs::path plan_path = fs::temp_directory_path() / ("xdna-causal-plan-" + std::to_string(_getpid()) + ".json");
    if (run_planner(repo, config_path, opt, plan_path) != 0) throw std::runtime_error("GP1 host planner rejected config/capture");
    std::ifstream plan_stream(plan_path);
    const json plan = json::parse(plan_stream);
    plan_stream.close();
    fs::remove(plan_path);

    const size_t prefix_count = opt.mode == "CAPTURE_PREFIX" ? opt.prefix_ids.size() : 0;
    const size_t expected_count = prefix_count + 1;
    const json selected_ids = opt.mode == "CAPTURE_PREFIX" ? json(opt.prefix_ids) : json::array();
    const std::string cell = canonical_cell_label(opt.pair, opt.mode, opt.prefix_ids);
    if (plan.at("executor_route") != "XDNA_CAUSAL_DIAGNOSTIC" || plan.at("mode") != opt.mode || plan.at("pair") != opt.pair ||
        plan.at("fixed_target_node") != "ffn_gate-4" || plan.at("execution_authorized").get<bool>() || plan.at("xrt_or_device_touched").get<bool>() ||
        plan.at("selected_prefix_ids") != selected_ids || plan.at("cell_label") != cell)
        throw std::runtime_error("planner identity/stop-policy/subset mismatch");
    const auto & recs = plan.at("execution_records");
    if (!recs.is_array() || recs.size() != expected_count || plan.at("prefix_submission_count") != prefix_count ||
        plan.at("target_submission_count") != 1 || plan.at("total_submission_count") != expected_count)
        throw std::runtime_error("wrong submission count or subset expansion");

    const fs::path capture = fs::canonical(repo / config.at("gn1_capture_dir_from_repo_root").get<std::string>());
    const fs::path payload_root = capture / "payloads";
    result.identity.mode = opt.mode;
    result.identity.pair = opt.pair;
    result.identity.cell = cell;
    result.identity.selected_ids = opt.mode == "CAPTURE_PREFIX" ? opt.prefix_ids : std::vector<size_t>{};
    result.prefix_count = prefix_count;
    result.selected_csv = prefix_count ? join_ids(opt.prefix_ids) : "NONE";
    result.records.reserve(recs.size());
    for (size_t i = 0; i < recs.size(); ++i) {
        const auto & r = recs[i];
        causal_runtime_record x;
        x.sequence = r.at("execution_ordinal").get<size_t>(); x.target = r.at("target").get<bool>(); x.m = r.at("m").get<size_t>(); x.k = r.at("k").get<size_t>();
        x.node = r.at("node_name").get<std::string>(); x.q4_sha = r.at("q4_sha256").get<std::string>(); x.q8_sha = r.at("q8_sha256").get<std::string>();
        x.instruction_sha = r.at("instruction_sha256").get<std::string>(); x.capture_id = r.at("capture_id").get<size_t>();
        if (x.sequence != i + 1 || r.at("sequence_index").get<size_t>() != x.sequence || x.k != 2560 || (x.m != 4096 && x.m != 9216) || x.node.empty())
            throw std::runtime_error("invalid planned shape/order/node");
        if (i < prefix_count) {
            if (x.capture_id != opt.prefix_ids[i] || x.target) throw std::runtime_error("prefix capture identity/order mismatch");
        } else if (!x.target || x.m != 9216 || x.node != "ffn_gate-4" || x.capture_id != plan.at("pair_source_capture").get<size_t>()) {
            throw std::runtime_error("fixed target/payload provenance mismatch");
        }
        const fs::path q4path = fs::canonical(r.at("q4_file").get<std::string>()), q8path = fs::canonical(r.at("q8_file").get<std::string>());
        if (q4path.parent_path() != payload_root || q8path.parent_path() != payload_root) throw std::runtime_error("payload path escapes the causal payload directory");
        x.q4 = read_checked(q4path, r.at("q4_bytes").get<size_t>(), x.q4_sha);
        x.q8 = read_checked(q8path, r.at("q8_bytes").get<size_t>(), x.q8_sha);
        if (x.q4.size() != x.m * 1440 || x.q8.size() != 2920) throw std::runtime_error("native Q4/Q8 size mismatch");
        const fs::path ipath = fs::canonical(r.at("instruction_file").get<std::string>());
        if (!result.instruction_paths.count(x.m) || ipath != result.instruction_paths.at(x.m) ||
            x.instruction_sha != config.at("instructions").at(std::to_string(x.m)).at("sha256")) throw std::runtime_error("instruction mapping mismatch");
        x.output.resize(x.m);
        result.records.push_back(std::move(x));
    }
    if (result.records.back().target == false || result.records.back().capture_id != plan.at("pair_source_capture").get<size_t>())
        throw std::runtime_error("target missing or moved");
    return result;
}

static void print_plan(const causal_runtime_plan & plan) {
    std::printf("CELL_PLAN route=XDNA_CAUSAL_DIAGNOSTIC cell=%s mode=%s pair=%s selected_prefix_ids=%s prefix_count=%zu target_count=1 total=%zu output_reference_validation=NOT_AVAILABLE xrt_initialized=NO\n",
        plan.identity.cell.c_str(), plan.identity.mode.c_str(), plan.identity.pair.c_str(), plan.selected_csv.c_str(), plan.prefix_count, plan.records.size());
    if (plan.identity.mode == "CAPTURE_PREFIX") std::printf("SUBSET_PLAN_EXACT=YES\n");
    for (const auto & x : plan.records)
        std::printf("PLAN execution_ordinal=%zu sequence_index=%zu capture_id=%zu M=%zu K=%zu node=%s Q4_SHA256=%s Q8_SHA256=%s INSTRUCTION_SHA256=%s TARGET=%s\n",
            x.sequence, x.sequence, x.capture_id, x.m, x.k, x.node.c_str(), x.q4_sha.c_str(), x.q8_sha.c_str(), x.instruction_sha.c_str(), x.target ? "YES" : "NO");
}

static int run_authorized_cell(const validated_execution_authorization & token, causal_runtime_plan & plan, const fs::path & executable_dir) {
    return enter_authorized_runtime(token, plan.identity, [&]() -> int {
        const fs::path runtime_path = executable_dir / "xdna-causal-runtime.dll";
        // First runtime-side-effecting call: loading this isolated module, after both authorization gates.
        HMODULE module = LoadLibraryW(runtime_path.c_str());
        if (!module) throw std::runtime_error("authorized runtime module load failed");
        const auto entry = reinterpret_cast<causal_authorized_runtime_fn>(GetProcAddress(module, "xdna_gn1_run_authorized"));
        if (!entry) throw std::runtime_error("authorized runtime entrypoint missing");
        return entry(&token, &plan, executable_dir.c_str());
    });
}

static int run_host_or_authorized(const options & opt, const fs::path & executable_dir) {
    causal_runtime_plan plan = prepare_host_plan(opt);
    print_plan(plan);
    // CLI barrier. If this is accidentally removed, authorize_execution still cannot mint a token
    // unless --execute, exact --authorize-cell, and this exact validated plan are all present.
    if (!opt.execute) {
        std::printf("DRY_RUN=PASS XRT_INITIALIZED=NO DEVICE_ENUMERATION=NONE\n");
        return 0;
    }
    const authorization_request request{opt.execute, opt.prefix_ids_provided, opt.authorized_cell, plan.identity};
    const validated_execution_authorization token = authorize_execution(request);
    return run_authorized_cell(token, plan, executable_dir);
}

int main(int argc, char ** argv) {
    try { return run_host_or_authorized(parse_cli(argc, argv), fs::absolute(argv[0]).parent_path()); }
    catch (const std::exception & e) { std::fprintf(stderr, "XDNA_CAUSAL_CAUSAL_EXECUTOR_FAIL_CLOSED: %s\n", e.what()); return 1; }
}
