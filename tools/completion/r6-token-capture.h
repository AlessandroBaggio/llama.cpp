#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace r6_token_capture {

inline std::string json_string(const std::string & value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 0x0f]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
    return out;
}

class writer {
public:
    writer(const std::string & final_path, const std::string & session_id,
           const std::string & model, const std::string & profile, uint64_t prompt_tokens)
        : final_path_(final_path), partial_path_(final_path + ".partial"), session_id_(session_id) {
        std::error_code ec;
        if (final_path.empty() || session_id.empty() || model.empty() || profile.empty() ||
                std::filesystem::exists(final_path_, ec) || ec ||
                std::filesystem::exists(partial_path_, ec) || ec) {
            error_ = "capture path or metadata invalid, or output already exists";
            return;
        }
        out_.open(partial_path_, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out_) {
            error_ = "cannot open partial token capture";
            return;
        }
        out_ << "{\"schema\":\"ggml-xdna-r6-token-capture-v1\",\"event\":\"session_start\",\"session_id\":"
             << json_string(session_id) << ",\"model\":" << json_string(model)
             << ",\"profile\":" << json_string(profile)
             << ",\"prompt_token_count\":" << prompt_tokens << ",\"state\":\"PARTIAL\"}\n";
        flush();
    }

    writer(const writer &) = delete;
    writer & operator=(const writer &) = delete;
    ~writer() { if (out_.is_open()) out_.close(); }

    bool ok() const { return error_.empty(); }
    const std::string & error() const { return error_; }
    uint64_t accepted_count() const { return accepted_count_; }
    uint64_t shown_count() const { return shown_count_; }

    bool accept(int32_t token_id, bool is_eog) {
        if (!ok()) return false;
        out_ << "{\"schema\":\"ggml-xdna-r6-token-capture-v1\",\"event\":\"generated_token\",\"session_id\":"
             << json_string(session_id_) << ",\"sequence_index\":" << accepted_count_
             << ",\"token_id\":" << token_id << ",\"acceptance\":\"ACCEPTED\",\"is_eog\":"
             << (is_eog ? "true" : "false") << "}\n";
        if (!flush()) return false;
        ++accepted_count_;
        return true;
    }

    bool shown() {
        if (!ok()) return false;
        ++shown_count_;
        return true;
    }

    bool finish(const std::string & reason, bool complete, bool failed = false) {
        if (!ok()) return false;
        out_ << "{\"schema\":\"ggml-xdna-r6-token-capture-v1\",\"event\":\"generation_end\",\"session_id\":"
             << json_string(session_id_) << ",\"accepted_token_count\":" << accepted_count_
             << ",\"shown_token_count\":" << shown_count_
             << ",\"reason\":" << json_string(reason)
             << ",\"state\":\"" << (complete ? "FINALIZING" : failed ? "FAILED" : "PARTIAL") << "\"}\n";
        if (!flush()) return false;
        out_.close();
        if (out_.fail()) { error_ = "failed closing partial token capture"; return false; }
        if (!complete) return true;

        std::error_code ec;
        std::filesystem::rename(partial_path_, final_path_, ec);
        if (ec) { error_ = "failed promoting completed token capture: " + ec.message(); return false; }
        std::ofstream final_out(final_path_, std::ios::out | std::ios::binary | std::ios::app);
        if (!final_out) { std::filesystem::rename(final_path_, partial_path_, ec); error_ = "failed opening final token capture"; return false; }
        final_out << "{\"schema\":\"ggml-xdna-r6-token-capture-v1\",\"event\":\"capture_status\",\"session_id\":"
                  << json_string(session_id_) << ",\"state\":\"COMPLETE\"}\n";
        final_out.flush();
        if (!final_out) {
            final_out.close();
            std::filesystem::rename(final_path_, partial_path_, ec);
            error_ = "failed writing COMPLETE marker";
            return false;
        }
        final_out.close();
        return true;
    }

private:
    bool flush() {
        out_.flush();
        if (!out_) { error_ = "token capture write/flush failed"; return false; }
        return true;
    }

    std::string final_path_;
    std::string partial_path_;
    std::string session_id_;
    std::ofstream out_;
    std::string error_;
    uint64_t accepted_count_ = 0;
    uint64_t shown_count_ = 0;
};

} // namespace r6_token_capture
