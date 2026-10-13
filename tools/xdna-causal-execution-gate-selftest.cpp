#include "../ggml/src/ggml-xdna/ggml-xdna-execution-authorization.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ggml_xdna_causal;
static_assert(!std::is_default_constructible<validated_execution_authorization>::value, "authorization token must not be default constructible");
static_assert(!std::is_constructible<validated_execution_authorization, std::string, std::string, std::string, std::vector<size_t>>::value, "authorization token constructor must remain private");

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static plan_identity make_plan(std::string mode, std::string pair, std::vector<size_t> ids) {
    const std::string cell = canonical_cell_label(pair, mode, ids);
    return {std::move(mode), std::move(pair), cell, std::move(ids)};
}

static authorization_request make_request(bool execute, std::string label, plan_identity identity, bool explicit_subset = true) {
    return {execute, explicit_subset, std::move(label), std::move(identity)};
}

static bool rejected_without_backend(const authorization_request & req, const plan_identity & runtime_plan, int & backend_calls) {
    const int calls_before = backend_calls;
    try {
        const auto token = authorize_execution(req);
        enter_authorized_runtime(token, runtime_plan, [&] { ++backend_calls; return 0; });
    } catch (...) {
        return backend_calls == calls_before;
    }
    return backend_calls == calls_before;
}

int main() {
    try {
        int backend_calls = 0;
        require(parse_prefix_ids("7,8,12") == std::vector<size_t>({7, 8, 12}), "valid subset parse failed");
        for (const char * invalid : {"", "0", "13", "7,7,8", "8,7", "7,,8", "x", "7-8", "+7", "07", " 7"}) {
            bool rejected = false;
            try { (void) parse_prefix_ids(invalid); } catch (...) { rejected = true; }
            require(rejected, "invalid subset parsed");
        }

        const std::vector<size_t> suffix{7, 8, 9, 10, 11, 12};
        auto valid_plan = make_plan("CAPTURE_PREFIX", "B", suffix);
        require(valid_plan.cell == "B_PREFIX_7_12", "suffix label mismatch");
        auto valid_request = make_request(true, "B_PREFIX_7_12", valid_plan);
        const auto token = authorize_execution(valid_request);
        require(token.selected_ids() == suffix && token.cell() == "B_PREFIX_7_12", "token did not bind exact subset");
        require(enter_authorized_runtime(token, valid_plan, [&] { ++backend_calls; return 42; }) == 42, "authorized mock call failed");
        require(backend_calls == 1, "authorized mock backend call count mismatch");

        auto dry_run = make_request(false, "B_PREFIX_7_12", valid_plan);
        require(rejected_without_backend(dry_run, valid_plan, backend_calls), "dry-run reached mock backend");
        auto missing_execute = make_request(false, "", valid_plan);
        require(rejected_without_backend(missing_execute, valid_plan, backend_calls), "missing execute reached mock backend");
        auto missing_authorization = make_request(true, "", valid_plan);
        require(rejected_without_backend(missing_authorization, valid_plan, backend_calls), "missing authorization reached mock backend");
        auto wrong_authorization = make_request(true, "B_PREFIX_12_ONLY", valid_plan);
        require(rejected_without_backend(wrong_authorization, valid_plan, backend_calls), "wrong label reached mock backend");
        const std::vector<size_t> full{1,2,3,4,5,6,7,8,9,10,11,12};
        auto legacy_full = make_plan("CAPTURE_PREFIX", "B", full);
        require(authorize_execution(make_request(true, "B_PREFIX_1_12", legacy_full, false)).selected_ids() == full, "legacy full-prefix capability failed");
        require(rejected_without_backend(make_request(true, "B_PREFIX_7_12", valid_plan, false), valid_plan, backend_calls), "implicit subset reached mock backend");
        auto pair_a = make_plan("CAPTURE_PREFIX", "A", full);
        require(rejected_without_backend(make_request(true, "A_PREFIX_1_12", pair_a, true), pair_a, backend_calls), "explicit Pair A subset reached mock backend");

        auto wrong_subset = make_plan("CAPTURE_PREFIX", "B", {7, 8, 9, 10, 11});
        require(rejected_without_backend(make_request(true, "B_PREFIX_7_12", wrong_subset), wrong_subset, backend_calls), "wrong subset reached mock backend");
        plan_identity invalid_mode{"INVALID", "B", "B_PREFIX_7_12", suffix};
        require(rejected_without_backend(make_request(true, "B_PREFIX_7_12", invalid_mode), invalid_mode, backend_calls), "invalid mode reached mock backend");
        auto invalid_pair = make_plan("CAPTURE_PREFIX", "C", suffix);
        require(rejected_without_backend(make_request(true, "B_PREFIX_7_12", invalid_pair), invalid_pair, backend_calls), "invalid pair reached mock backend");

        auto changed_runtime_plan = make_plan("CAPTURE_PREFIX", "B", {12});
        bool runtime_rejected = false;
        try { enter_authorized_runtime(token, changed_runtime_plan, [&] { ++backend_calls; return 0; }); }
        catch (...) { runtime_rejected = true; }
        require(runtime_rejected && backend_calls == 1, "runtime accepted a token for another subset");

        // Mutation-style proof: bypassing the CLI return still cannot mint a token for dry-run.
        bool mutation_rejected = false;
        try {
            const auto impossible = authorize_execution(dry_run);
            enter_authorized_runtime(impossible, valid_plan, [&] { ++backend_calls; return 0; });
        } catch (...) { mutation_rejected = true; }
        require(mutation_rejected && backend_calls == 1, "removing the CLI guard exposed the mock backend");

        require(canonical_cell_label("B", "CAPTURE_PREFIX", {12}) == "B_PREFIX_12_ONLY", "singleton label mismatch");
        require(canonical_cell_label("B", "CAPTURE_PREFIX", {1,4,7,12}) == "B_PREFIX_IDS_1_4_7_12", "noncontiguous label mismatch");
        require(canonical_cell_label("B", "CAPTURE_PREFIX", full) == "B_PREFIX_1_12", "legacy full-prefix label mismatch");

        std::cout << "BACKEND_ENTRY_MOCK_CALL_COUNT_INVALID=0\n"
                  << "SINGLE_GUARD_REMOVAL_STILL_SAFE=PASS\n"
                  << "TYPE_SAFE_SUBSET_AUTHORIZATION=PASS\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "XDNA_CAUSAL_EXECUTION_GATE_SELFTEST_FAIL: " << e.what() << '\n';
        return 1;
    }
}
