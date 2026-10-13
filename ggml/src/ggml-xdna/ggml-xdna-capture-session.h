#pragma once

#include "../../../vendor/nlohmann/json.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ggml_xdna_capture {

class capture_session {
public:
    bool attach(const std::filesystem::path & dir, const std::filesystem::path & xclbin_path,
                const std::string & profile, const std::string & xclbin_sha256,
                const std::string & metadata, bool * created = nullptr, uint64_t * ordinal = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (created) *created = false;
        if (ordinal) *ordinal = 0;
        if (failed_) return false;
        try {
            const auto normalized_dir = std::filesystem::absolute(dir).lexically_normal();
            const auto normalized_xclbin = std::filesystem::absolute(xclbin_path).lexically_normal();
            const auto parsed = nlohmann::json::parse(metadata);
            if (parsed.at("schema_version").get<int>() != 1 ||
                parsed.at("profile").get<std::string>() != profile ||
                parsed.at("capture_mode").get<std::string>() != "GGML_XDNA_CAPTURE_EXACT_HISTORY=1" ||
                parsed.at("xclbin_sha256").get<std::string>() != xclbin_sha256 ||
                std::filesystem::absolute(parsed.at("xclbin_path").get<std::string>()).lexically_normal() != normalized_xclbin) {
                failed_ = true;
                return false;
            }
            const auto metadata_path = normalized_dir / "run_metadata.json";
            const auto manifest_path = normalized_dir / "manifest.jsonl";
            if (!established_) {
                if (std::filesystem::exists(normalized_dir)) {
                    if (!std::filesystem::is_directory(normalized_dir) || !std::filesystem::is_empty(normalized_dir)) {
                        failed_ = true;
                        return false;
                    }
                } else {
                    std::filesystem::create_directories(normalized_dir);
                }
                if (!std::filesystem::is_directory(normalized_dir) || !std::filesystem::is_empty(normalized_dir)) {
                    failed_ = true;
                    return false;
                }
                const auto owner = normalized_dir / ".br1-capture-session-owner";
                std::error_code owner_error;
                if (!std::filesystem::create_directory(owner, owner_error) || owner_error) {
                    failed_ = true;
                    return false;
                }
                identity_dir_ = normalized_dir;
                identity_xclbin_ = normalized_xclbin;
                identity_profile_ = profile;
                identity_sha_ = xclbin_sha256;
                metadata_ = metadata;
                if (!write_new_file(metadata_path, metadata_) || !write_new_file(manifest_path, "")) {
                    failed_ = true;
                    return false;
                }
                owner_error.clear();
                if (!std::filesystem::remove(owner, owner_error) || owner_error) {
                    failed_ = true;
                    return false;
                }
                manifest_ = "";
                established_ = true;
                attached_ = 1;
                if (created) *created = true;
                if (ordinal) *ordinal = ++attach_ordinal_;
                return true;
            }
            if (normalized_dir != identity_dir_ || normalized_xclbin != identity_xclbin_ ||
                profile != identity_profile_ || xclbin_sha256 != identity_sha_ || metadata != metadata_ ||
                !regular_non_symlink(metadata_path) || !regular_non_symlink(manifest_path) ||
                read_file(metadata_path) != metadata_ || read_file(manifest_path) != manifest_) {
                failed_ = true;
                return false;
            }
            ++attached_;
            if (ordinal) *ordinal = ++attach_ordinal_;
            return true;
        } catch (...) {
            failed_ = true;
            return false;
        }
    }

    uint64_t reserve_submission_id() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!established_ || attached_ == 0 || failed_ || terminal_) return 0;
        return ++next_submission_id_;
    }

    bool can_submit() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return established_ && attached_ != 0 && !failed_ && !terminal_;
    }

    bool append(const std::string & line, bool terminal_record = false, bool allow_failed = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!established_ || attached_ == 0 || terminal_ || (failed_ && !allow_failed)) return false;
        try {
            std::ofstream out(identity_dir_ / "manifest.jsonl", std::ios::binary | std::ios::out | std::ios::app);
            if (!out) { failed_ = true; return false; }
            out.write(line.data(), static_cast<std::streamsize>(line.size()));
            out.flush();
            out.close();
            if (out.fail()) { failed_ = true; return false; }
            manifest_ += line;
            if (terminal_record) terminal_ = true;
            return true;
        } catch (...) {
            failed_ = true;
            return false;
        }
    }

    void fail() {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
    }

    size_t detach() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (attached_ != 0) --attached_;
        return attached_;
    }

    bool failed() const { std::lock_guard<std::mutex> lock(mutex_); return failed_; }
    bool terminal() const { std::lock_guard<std::mutex> lock(mutex_); return terminal_; }
    uint64_t next_submission_id() const { std::lock_guard<std::mutex> lock(mutex_); return next_submission_id_; }
    size_t attached_count() const { std::lock_guard<std::mutex> lock(mutex_); return attached_; }

private:
    static bool regular_non_symlink(const std::filesystem::path & path) {
        return !std::filesystem::is_symlink(std::filesystem::symlink_status(path)) &&
               std::filesystem::is_regular_file(path);
    }

    static std::string read_file(const std::filesystem::path & path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("capture session file open failed");
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    static bool write_new_file(const std::filesystem::path & path, const std::string & bytes) {
        const auto temp = std::filesystem::path(path.string() + ".tmp");
        if (std::filesystem::exists(path) || std::filesystem::exists(temp)) return false;
        std::ofstream out(temp, std::ios::binary | std::ios::out);
        if (!out) return false;
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        out.close();
        if (out.fail()) { std::filesystem::remove(temp); return false; }
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) { std::filesystem::remove(temp); return false; }
        return true;
    }

    mutable std::mutex mutex_;
    bool established_ = false;
    bool failed_ = false;
    bool terminal_ = false;
    std::filesystem::path identity_dir_, identity_xclbin_;
    // ponytail: retain the short manifest for exact attach checks; switch to an incremental digest if captures grow materially.
    std::string identity_profile_, identity_sha_, metadata_, manifest_;
    uint64_t next_submission_id_ = 0;
    uint64_t attach_ordinal_ = 0;
    size_t attached_ = 0;
};

} // namespace ggml_xdna_capture
