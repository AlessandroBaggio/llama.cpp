#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace r6_response_capture {

class writer {
public:
    writer(const std::string & path, const std::string & session_id, const std::string & profile)
        : final_path_(path), partial_path_(path + ".partial"), session_id_(session_id), profile_(profile) {
        const std::filesystem::path final_path(path);
        if (path.empty() || !final_path.is_absolute() || session_id.empty() || profile.empty()) {
            error_ = "capture path or metadata invalid";
            return;
        }
        std::error_code path_ec;
        if (std::filesystem::exists(final_path_, path_ec) || path_ec) {
            error_ = "response capture final path already exists or cannot be checked";
            return;
        }
#ifdef _WIN32
        const int fd = _open(partial_path_.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY, _S_IREAD | _S_IWRITE);
        if (fd >= 0) out_ = _fdopen(fd, "wb");
        else error_ = "cannot exclusively create partial response capture";
#else
        const int fd = ::open(partial_path_.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
        if (fd >= 0) out_ = fdopen(fd, "wb");
        else error_ = "cannot exclusively create partial response capture";
#endif
        if (!out_) {
            if (error_.empty()) error_ = "cannot open partial response capture stream";
            return;
        }
        if (!write_record("{\"schema\":\"ggml-xdna-r6-response-capture-v1\",\"event\":\"session_start\",\"session_id\":" +
                          quote(session_id_) + ",\"profile\":" + quote(profile_) + ",\"state\":\"PARTIAL\"}\n")) return;
    }

    writer(const writer &) = delete;
    writer & operator=(const writer &) = delete;
    ~writer() { if (out_) std::fclose(out_); }

    bool ok() const { return error_.empty(); }
    const std::string & error() const { return error_; }
    uint64_t piece_count() const { return piece_count_; }

    bool piece(uint64_t accepted_index, const std::string & bytes) {
        if (!ok()) return false;
        if (bytes.find('\0') != std::string::npos) return fail_with_status("NUL byte cannot be correlated with LOG C-string output");
        const std::string encoded = base64(bytes);
        const bool written = write_record("{\"schema\":\"ggml-xdna-r6-response-capture-v1\",\"event\":\"text_piece\",\"session_id\":" +
            quote(session_id_) + ",\"profile\":" + quote(profile_) + ",\"piece_index\":" + std::to_string(piece_count_) +
            ",\"accepted_token_index\":" + std::to_string(accepted_index) + ",\"byte_length\":" + std::to_string(bytes.size()) +
            ",\"payload_base64\":" + quote(encoded) + "}\n");
        if (written) ++piece_count_;
        return written;
    }

    bool finish(bool complete) {
        if (!ok()) return false;
        const std::string terminal = "{\"schema\":\"ggml-xdna-r6-response-capture-v1\",\"event\":\"capture_status\",\"session_id\":" +
            quote(session_id_) + ",\"profile\":" + quote(profile_) + ",\"state\":\"" + (complete ? "COMPLETE" : "FAILED") + "\"}\n";
        if (!complete) {
            if (!write_record(terminal)) return false;
            if (std::fclose(out_) != 0) { out_ = nullptr; return fail("failed closing failed response capture"); }
            out_ = nullptr;
            return true;
        }
        if (std::fclose(out_) != 0) { out_ = nullptr; return fail("failed closing partial response capture"); }
        out_ = nullptr;
        std::error_code ec;
        if (std::filesystem::exists(final_path_, ec) || ec) return fail("response capture final path already exists or cannot be checked");
        std::filesystem::rename(partial_path_, final_path_, ec);
        if (ec) return fail("failed promoting response capture: " + ec.message());
        out_ = std::fopen(final_path_.c_str(), "ab");
        if (!out_) { std::filesystem::rename(final_path_, partial_path_, ec); return fail("failed opening promoted response capture"); }
        if (!write_record(terminal)) {
            std::fclose(out_);
            out_ = nullptr;
            std::filesystem::rename(final_path_, partial_path_, ec);
            return false;
        }
        if (std::fclose(out_) != 0) {
            out_ = nullptr;
            std::filesystem::rename(final_path_, partial_path_, ec);
            return fail("failed closing completed response capture");
        }
        out_ = nullptr;
        return true;
    }

private:
    static std::string quote(const std::string & s) {
        std::string out = "\"";
        for (unsigned char c : s) {
            if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(static_cast<char>(c)); }
            else if (c < 0x20) return "\"\"";
            else out.push_back(static_cast<char>(c));
        }
        out.push_back('"');
        return out;
    }

    static std::string base64(const std::string & bytes) {
        static constexpr char chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        for (size_t i = 0; i < bytes.size(); i += 3) {
            const uint32_t a = static_cast<unsigned char>(bytes[i]);
            const uint32_t b = i + 1 < bytes.size() ? static_cast<unsigned char>(bytes[i + 1]) : 0;
            const uint32_t c = i + 2 < bytes.size() ? static_cast<unsigned char>(bytes[i + 2]) : 0;
            out.push_back(chars[a >> 2]);
            out.push_back(chars[((a & 3) << 4) | (b >> 4)]);
            out.push_back(i + 1 < bytes.size() ? chars[((b & 15) << 2) | (c >> 6)] : '=');
            out.push_back(i + 2 < bytes.size() ? chars[c & 63] : '=');
        }
        return out;
    }

    bool write_record(const std::string & line) {
        if (!out_ || std::fwrite(line.data(), 1, line.size(), out_) != line.size() || std::fflush(out_) != 0) return fail("response capture write/flush failed");
#ifdef _WIN32
        if (_commit(_fileno(out_)) != 0) return fail("response capture durable flush failed");
#else
        if (fsync(fileno(out_)) != 0) return fail("response capture durable flush failed");
#endif
        return true;
    }

    bool fail_with_status(const std::string & message) {
        write_record("{\"schema\":\"ggml-xdna-r6-response-capture-v1\",\"event\":\"capture_status\",\"session_id\":" +
                     quote(session_id_) + ",\"profile\":" + quote(profile_) + ",\"state\":\"FAILED\"}\n");
        error_ = message;
        return false;
    }
    bool fail(const std::string & message) { error_ = message; return false; }
    std::string final_path_, partial_path_, session_id_, profile_, error_;
    std::FILE * out_ = nullptr;
    uint64_t piece_count_ = 0;
};

} // namespace r6_response_capture
