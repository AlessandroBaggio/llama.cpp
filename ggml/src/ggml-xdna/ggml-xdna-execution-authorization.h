#pragma once

#include <cstddef>
#include <charconv>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ggml_xdna_causal {
inline std::vector<size_t> parse_prefix_ids(const std::string & raw) {
    if (raw.empty()) throw std::runtime_error("--prefix-ids must not be empty");
    std::vector<size_t> ids;
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t end = raw.find(',', start);
        const size_t stop = end == std::string::npos ? raw.size() : end;
        if (stop == start) throw std::runtime_error("--prefix-ids contains an empty token");
        const std::string token = raw.substr(start, stop - start);
        if (token.size() > 1 && token.front() == '0') throw std::runtime_error("--prefix-ids requires canonical decimal integers");
        for (unsigned char ch : token) if (ch < '0' || ch > '9') throw std::runtime_error("--prefix-ids accepts ASCII decimal integers only");
        size_t value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || value < 1 || value > 12)
            throw std::runtime_error("--prefix-ids values must be in 1..12");
        if (!ids.empty() && value <= ids.back()) throw std::runtime_error("--prefix-ids must be unique and in original capture order");
        ids.push_back(value);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return ids;
}

struct plan_identity {
    std::string mode;
    std::string pair;
    std::string cell;
    std::vector<size_t> selected_ids;
};

struct authorization_request {
    bool execute_requested = false;
    bool explicit_subset = false;
    std::string authorize_cell;
    plan_identity plan;
};

inline std::string canonical_cell_label(const std::string & pair, const std::string & mode, const std::vector<size_t> & ids) {
    if (mode == "CLEAN") return pair + "_CLEAN";
    if (mode != "CAPTURE_PREFIX" || ids.empty()) throw std::runtime_error("invalid capture prefix identity");
    if (ids.size() == 1) return pair + "_PREFIX_" + std::to_string(ids.front()) + "_ONLY";
    bool contiguous = true;
    for (size_t i = 1; i < ids.size(); ++i) contiguous = contiguous && ids[i] == ids[i - 1] + 1;
    if (contiguous) return pair + "_PREFIX_" + std::to_string(ids.front()) + "_" + std::to_string(ids.back());
    std::string label = pair + "_PREFIX_IDS_";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) label += '_';
        label += std::to_string(ids[i]);
    }
    return label;
}

class validated_execution_authorization {
public:
    validated_execution_authorization(const validated_execution_authorization &) = default;
    validated_execution_authorization & operator=(const validated_execution_authorization &) = default;

    const std::string & cell() const { return cell_; }
    const std::string & mode() const { return mode_; }
    const std::string & pair() const { return pair_; }
    const std::vector<size_t> & selected_ids() const { return selected_ids_; }

private:
    validated_execution_authorization(std::string cell, std::string mode, std::string pair, std::vector<size_t> ids)
        : cell_(std::move(cell)), mode_(std::move(mode)), pair_(std::move(pair)), selected_ids_(std::move(ids)) {}

    std::string cell_;
    std::string mode_;
    std::string pair_;
    std::vector<size_t> selected_ids_;

    friend validated_execution_authorization authorize_execution(const authorization_request & request);
};

inline validated_execution_authorization authorize_execution(const authorization_request & request) {
    const auto & plan = request.plan;
    if (!request.execute_requested) throw std::runtime_error("execution capability requires --execute");
    if (plan.pair != "A" && plan.pair != "B") throw std::runtime_error("invalid authorization pair");
    if (plan.mode != "CLEAN" && plan.mode != "CAPTURE_PREFIX") throw std::runtime_error("invalid authorization mode");
    if (request.explicit_subset && (plan.mode != "CAPTURE_PREFIX" || plan.pair != "B")) throw std::runtime_error("explicit subsets are Pair B CAPTURE_PREFIX only");
    if (plan.mode == "CLEAN" && !plan.selected_ids.empty()) throw std::runtime_error("CLEAN authorization cannot contain prefix IDs");
    if (plan.mode == "CAPTURE_PREFIX") {
        if (plan.selected_ids.empty()) throw std::runtime_error("CAPTURE_PREFIX authorization requires prefix IDs");
        for (size_t i = 0; i < plan.selected_ids.size(); ++i) {
            const size_t id = plan.selected_ids[i];
            if (id < 1 || id > 12 || (i && id <= plan.selected_ids[i - 1])) throw std::runtime_error("invalid or unordered authorization subset");
        }
        if (!request.explicit_subset && plan.selected_ids != std::vector<size_t>{1,2,3,4,5,6,7,8,9,10,11,12})
            throw std::runtime_error("legacy CAPTURE_PREFIX requires the frozen full prefix");
        if (plan.pair == "A" && plan.selected_ids != std::vector<size_t>{1,2,3,4,5,6,7,8,9,10,11,12})
            throw std::runtime_error("Pair A requires the frozen full capture prefix");
    }
    const std::string expected = canonical_cell_label(plan.pair, plan.mode, plan.selected_ids);
    if (plan.cell != expected || request.authorize_cell != expected) throw std::runtime_error("authorization does not bind the exact planned cell/subset");
    return validated_execution_authorization(expected, plan.mode, plan.pair, plan.selected_ids);
}

inline bool authorization_matches(const validated_execution_authorization & token, const plan_identity & plan) {
    return token.cell() == plan.cell && token.mode() == plan.mode && token.pair() == plan.pair && token.selected_ids() == plan.selected_ids;
}

template<class Callback>
auto enter_authorized_runtime(const validated_execution_authorization & token, const plan_identity & plan, Callback && callback)
    -> decltype(std::forward<Callback>(callback)()) {
    if (!authorization_matches(token, plan)) throw std::runtime_error("runtime entry rejected a mismatched execution capability");
    return std::forward<Callback>(callback)();
}

} // namespace ggml_xdna_causal
