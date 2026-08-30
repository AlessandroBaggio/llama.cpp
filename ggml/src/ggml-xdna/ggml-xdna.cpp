#include "ggml-xdna.h"

#include "ggml-backend-impl.h"
#include "ggml-quants.h"
#include "ggml-xdna-sha256.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>

enum class ggml_backend_xdna_kernel_profile {
    none,
    i16_256,
    q4k_q8k_k2560,
};

static const char * ggml_backend_xdna_kernel_profile_name(
        ggml_backend_xdna_kernel_profile profile) {
    switch (profile) {
        case ggml_backend_xdna_kernel_profile::none:
            return "none";
        case ggml_backend_xdna_kernel_profile::i16_256:
            return "i16_256";
        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560:
            return "q4k_q8k_k2560";
    }

    return "unknown";
}

struct ggml_backend_xdna_context {
    std::unique_ptr<xrt::device> xrt_device;
    std::string description = "AMD XDNA NPU (XRT device 0)";
    std::string device_id;

    // supports_op() is a device-level callback, therefore every live
    // backend instance sharing this device must expose the same profile.
    ggml_backend_xdna_kernel_profile active_profile =
        ggml_backend_xdna_kernel_profile::none;

    size_t active_backend_count = 0;
};


// Persistent host-side Q4_K -> 152-byte transport cache.
//
// This cache exists to test ONE variable: eliminating repeated CPU-side
// native Q4_K -> transport conversion for immutable model weight tensors
// across tokens. It intentionally does NOT change Q4 BO sync/XRT/output
// behavior: a cache hit still ends up producing the exact same bytes in
// the existing mapped Q4 execution BO, followed by the same sync/run/
// wait/output path as a cache miss.
//
// Keyed by the q4 source tensor data pointer (stable for the lifetime of
// an immutable weight tensor within one model/context), validated against
// M, native byte size and transport byte size to reject stale/incorrect
// reuse if a pointer were ever recycled for a different tensor.
struct ggml_backend_xdna_q4_cache_entry {
    size_t m = 0;
    size_t native_bytes = 0;
    size_t transport_bytes = 0;
    std::vector<uint8_t> transport;
};

enum class ggml_backend_xdna_q4_shape {
    m9216,
    m4096,
    m8192,
    m1024,
};

struct ggml_backend_xdna_q4_transport_cache {
    bool enabled = false;
    bool env_checked = false;

    std::unordered_map<const void *, ggml_backend_xdna_q4_cache_entry> entries;

    uint64_t hits = 0;
    uint64_t misses = 0;

    uint64_t hits_m9216 = 0;
    uint64_t hits_m4096 = 0;
    uint64_t hits_m8192 = 0;
    uint64_t hits_m1024 = 0;

    uint64_t blocks_converted = 0;
    uint64_t native_bytes_converted = 0;
    uint64_t transport_bytes_generated = 0;
    uint64_t transport_bytes_copied = 0;
};

// Minimal accumulated per-token-run timers (milliseconds), summed across
// all accepted Q4 nodes for the process lifetime. Diagnostic only.
struct ggml_backend_xdna_timers {
    double q4_conversion_ms = 0.0;
    double q4_cached_memcpy_ms = 0.0;
    double q4_sync_ms = 0.0;
    double q8_quant_ms = 0.0;
    double q8_sync_ms = 0.0;
    double xrt_submit_wait_ms = 0.0;
    double output_ms = 0.0;
};

struct ggml_backend_xdna_shape_timing {
    uint64_t ops = 0;
    double submit_ms = 0.0;
    double wait_ms = 0.0;
};

static bool ggml_backend_xdna_env_flag_enabled(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0;
}

struct ggml_backend_xdna_batched_state {
    size_t rows = 0;
    std::string xclbin_path;
    std::vector<uint32_t> instructions;
    std::unique_ptr<xrt::xclbin> xclbin;
    std::unique_ptr<xrt::hw_context> hw_context;
    std::unique_ptr<xrt::kernel> kernel;
    std::unique_ptr<xrt::bo> bo_instr, bo_q4, bo_q8, bo_out;
    void * buf_instr = nullptr, * buf_q4 = nullptr, * buf_q8 = nullptr;
    float * buf_out = nullptr;
    uint32_t instruction_count = 0;
    bool bos_initialized = false;
    bool available = false;
};
struct ggml_backend_xdna_backend_context {
    ggml_backend_xdna_context * device_context = nullptr;

    ggml_backend_xdna_kernel_profile kernel_profile =
        ggml_backend_xdna_kernel_profile::none;

    bool device_profile_registered = false;

    std::string xclbin_path;
    std::string instructions_path;
    std::vector<uint32_t> instructions;

    std::string kernel_name = "MLIR_AIE";

    std::unique_ptr<xrt::xclbin> xclbin;
    std::unique_ptr<xrt::hw_context> hw_context;
    std::unique_ptr<xrt::kernel> kernel;

    // Persistent Q4_K K=2560 BOs. They belong to this backend/XRT
    // context and are released automatically when the backend is freed.
    std::unique_ptr<xrt::bo> q4_bo_instr;
    std::unique_ptr<xrt::bo> q4_bo_q4;
    std::unique_ptr<xrt::bo> q4_bo_q8;
    std::unique_ptr<xrt::bo> q4_bo_out;

    void * q4_buf_instr = nullptr;
    void * q4_buf_q4 = nullptr;
    void * q4_buf_q8 = nullptr;
    float * q4_buf_out = nullptr;

    uint32_t q4_instruction_count = 0;
    bool q4_bos_initialized = false;

    // Optional full-shape Q4_K x Q8_K K=2560 M=9216 fast path.
    // This is a secondary artifact belonging to the same
    // q4k_q8k_k2560 backend profile. The existing per-row kernel
    // remains the fallback for every other M.
    std::string q4_m9216_xclbin_path;
    std::string q4_m9216_instructions_path;
    std::vector<uint32_t> q4_m9216_instructions;

    std::unique_ptr<xrt::xclbin> q4_m9216_xclbin;
    std::unique_ptr<xrt::hw_context> q4_m9216_hw_context;
    std::unique_ptr<xrt::kernel> q4_m9216_kernel;

    std::unique_ptr<xrt::bo> q4_m9216_bo_instr;
    std::unique_ptr<xrt::bo> q4_m9216_bo_q4;
    std::unique_ptr<xrt::bo> q4_m9216_bo_q8;
    std::unique_ptr<xrt::bo> q4_m9216_bo_out;

    void * q4_m9216_buf_instr = nullptr;
    void * q4_m9216_buf_q4 = nullptr;
    void * q4_m9216_buf_q8 = nullptr;
    float * q4_m9216_buf_out = nullptr;

    uint32_t q4_m9216_instruction_count = 0;
    bool q4_m9216_bos_initialized = false;
    bool q4_m9216_available = false;

    ggml_backend_xdna_batched_state q4_m1024;
    ggml_backend_xdna_batched_state q4_m4096;
    ggml_backend_xdna_batched_state q4_m8192;
    uint64_t accepted_q4_nodes = 0;
    uint64_t batched_m9216 = 0, batched_m4096 = 0, batched_m8192 = 0, batched_m1024 = 0;
    uint64_t fallback_rows = 0, xrt_runs = 0, run_waits = 0;
    uint64_t q4_to_device = 0, q8_to_device = 0, output_from_device = 0;

    // Persistent Q4_K transport cache (experimental, off by default).
    // See GGML_XDNA_Q4K_TRANSPORT_CACHE. Owned by the backend/model-
    // context lifetime, matching q4_m9216_* and q4_m{1024,4096,8192} BOs.
    ggml_backend_xdna_q4_transport_cache q4_cache;
    ggml_backend_xdna_timers timers;
    ggml_backend_xdna_shape_timing shape_timing[4];
};


static ggml_backend_xdna_context ggml_backend_xdna_create_context() {
    ggml_backend_xdna_context ctx;

    try {
        ctx.xrt_device = std::make_unique<xrt::device>(0);
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to open XRT device 0: %s\n",
            e.what());
        return ctx;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to open XRT device 0: unknown error\n");
        return ctx;
    }

    try {
        const std::string xrt_name =
            ctx.xrt_device->get_info<xrt::info::device::name>();

        if (!xrt_name.empty()) {
            ctx.description = "AMD XDNA NPU (" + xrt_name + ")";
        }
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: XRT device 0 opened, but name query failed: %s\n",
            e.what());
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: XRT device 0 opened, but name query failed\n");
    }

    try {
        ctx.device_id =
            ctx.xrt_device->get_info<xrt::info::device::bdf>();
    } catch (...) {
        // BDF is optional for this milestone.
    }

    return ctx;
}

static ggml_backend_xdna_context * ggml_backend_xdna_get_context(
        ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return nullptr;
    }

    return static_cast<ggml_backend_xdna_context *>(dev->context);
}

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    (void) dev;
    return "XDNA0";
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    const auto * ctx = ggml_backend_xdna_get_context(dev);

    if (ctx == nullptr) {
        return "AMD XDNA NPU";
    }

    return ctx->description.c_str();
}

static void ggml_backend_xdna_device_get_memory(
        ggml_backend_dev_t dev,
        size_t * free,
        size_t * total) {
    (void) dev;

    // XDNA does not expose conventional dedicated VRAM through this
    // registration-only interface.
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    (void) dev;
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xdna_device_get_props(
        ggml_backend_dev_t dev,
        struct ggml_backend_dev_props * props) {
    const auto * ctx = ggml_backend_xdna_get_context(dev);

    props->name         = ggml_backend_xdna_device_get_name(dev);
    props->description  = ggml_backend_xdna_device_get_description(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->type         = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id    =
        ctx != nullptr && !ctx->device_id.empty()
            ? ctx->device_id.c_str()
            : nullptr;
    props->caps         = {};
}

static bool ggml_backend_xdna_parse_init_params(
        const char * params,
        ggml_backend_xdna_kernel_profile * profile,
        std::string * xclbin_path) {
    if (profile == nullptr || xclbin_path == nullptr) {
        return false;
    }

    *profile =
        ggml_backend_xdna_kernel_profile::none;

    xclbin_path->clear();

    if (params == nullptr || params[0] == '\0') {
        return true;
    }

    const std::string value(params);

    static constexpr char i16_prefix[] =
        "profile=i16_256;";

    static constexpr char q4_prefix[] =
        "profile=q4k_q8k_k2560;";

    if (value.rfind(i16_prefix, 0) == 0) {
        *profile =
            ggml_backend_xdna_kernel_profile::i16_256;

        *xclbin_path =
            value.substr(sizeof(i16_prefix) - 1);

        if (xclbin_path->empty()) {
            std::fprintf(
                stderr,
                "ggml_xdna: i16_256 profile requires an XCLBIN path\n");
            return false;
        }

        return true;
    }

    if (value.rfind(q4_prefix, 0) == 0) {
        *profile =
            ggml_backend_xdna_kernel_profile::q4k_q8k_k2560;

        *xclbin_path =
            value.substr(sizeof(q4_prefix) - 1);

        if (xclbin_path->empty()) {
            std::fprintf(
                stderr,
                "ggml_xdna: q4k_q8k_k2560 profile requires an XCLBIN path\n");
            return false;
        }

        return true;
    }

    if (value.rfind("profile=", 0) == 0) {
        std::fprintf(
            stderr,
            "ggml_xdna: unknown kernel profile in backend params: %s\n",
            params);
        return false;
    }

    // Backward-compatible direct/probe mode.
    //
    // A plain path loads the artifact but deliberately carries no
    // advertised kernel profile.
    *xclbin_path = value;
    return true;
}

static bool ggml_backend_xdna_init_kernel(
        ggml_backend_xdna_backend_context * ctx,
        const char * params) {
    if (params == nullptr || params[0] == '\0') {
        // Kernel initialization is optional for now. This preserves the
        // registration/backend-instance milestones when no XCLBIN is supplied.
        return true;
    }

    ctx->xclbin_path = params;

    try {
        ctx->xclbin =
            std::make_unique<xrt::xclbin>(ctx->xclbin_path);

        auto xkernels = ctx->xclbin->get_kernels();

        auto it = std::find_if(
            xkernels.begin(),
            xkernels.end(),
            [&](xrt::xclbin::kernel k) {
                return k.get_name().rfind(ctx->kernel_name, 0) == 0;
            });

        if (it == xkernels.end()) {
            std::fprintf(
                stderr,
                "ggml_xdna: kernel '%s' not found in XCLBIN: %s\n",
                ctx->kernel_name.c_str(),
                ctx->xclbin_path.c_str());
            return false;
        }

        const std::string resolved_kernel_name = it->get_name();

        ctx->device_context->xrt_device->register_xclbin(
            *ctx->xclbin);

        ctx->hw_context =
            std::make_unique<xrt::hw_context>(
                *ctx->device_context->xrt_device,
                ctx->xclbin->get_uuid());

        ctx->kernel =
            std::make_unique<xrt::kernel>(
                *ctx->hw_context,
                resolved_kernel_name);

        ctx->kernel_name = resolved_kernel_name;

        std::fprintf(
            stderr,
            "ggml_xdna: XRT kernel ready: %s\n",
            ctx->kernel_name.c_str());

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to initialize XCLBIN '%s': %s\n",
            ctx->xclbin_path.c_str(),
            e.what());
        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to initialize XCLBIN '%s': unknown error\n",
            ctx->xclbin_path.c_str());
        return false;
    }
}

static bool ggml_backend_xdna_load_instructions(
        ggml_backend_xdna_backend_context * ctx) {
    if (ctx == nullptr) {
        return false;
    }

    if (!ctx->instructions.empty()) {
        return true;
    }

    if (ctx->xclbin_path.empty()) {
        std::fprintf(
            stderr,
            "ggml_xdna: cannot load instructions without an XCLBIN path\n");
        return false;
    }

    try {
        const std::filesystem::path xclbin_path(ctx->xclbin_path);
        const std::filesystem::path instructions_path =
            xclbin_path.parent_path() / "insts.bin";

        ctx->instructions_path = instructions_path.string();

        std::ifstream file(
            instructions_path,
            std::ios::binary | std::ios::ate);

        if (!file) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to open instruction file: %s\n",
                ctx->instructions_path.c_str());
            return false;
        }

        const std::streamsize size = file.tellg();

        if (size <= 0 ||
            size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: invalid instruction file size: %s\n",
                ctx->instructions_path.c_str());
            return false;
        }

        ctx->instructions.resize(
            static_cast<size_t>(size) / sizeof(uint32_t));

        file.seekg(0, std::ios::beg);

        if (!file.read(
                reinterpret_cast<char *>(ctx->instructions.data()),
                size)) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to read instruction file: %s\n",
                ctx->instructions_path.c_str());

            ctx->instructions.clear();
            return false;
        }

        std::fprintf(
            stderr,
            "ggml_xdna: loaded %zu instruction words from %s\n",
            ctx->instructions.size(),
            ctx->instructions_path.c_str());

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to load instructions: %s\n",
            e.what());
        ctx->instructions.clear();
        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to load instructions: unknown error\n");
        ctx->instructions.clear();
        return false;
    }
}

static const unsigned char *
ggml_backend_xdna_kernel_profile_instruction_sha256(
        ggml_backend_xdna_kernel_profile profile) {
    static constexpr unsigned char i16_256_sha256[
        GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xac, 0xa7, 0xc1, 0xab, 0x8a, 0xe9, 0x0f, 0x83,
        0xc8, 0x24, 0xeb, 0xb5, 0x6e, 0x0e, 0x8d, 0x55,
        0x9b, 0x1b, 0x2f, 0x21, 0x7e, 0xd6, 0x6e, 0x08,
        0x51, 0xdf, 0x37, 0xd3, 0x15, 0xce, 0x1d, 0x14,
    };

    static constexpr unsigned char q4k_q8k_k2560_sha256[
        GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xc1, 0xc8, 0x45, 0xd8, 0x3f, 0xcd, 0x97, 0xa1,
        0xaf, 0xff, 0x80, 0xab, 0x15, 0x0f, 0xc4, 0xe9,
        0x10, 0x89, 0x30, 0x6a, 0x9e, 0xfa, 0xe1, 0x67,
        0x82, 0xb5, 0xea, 0xdc, 0x38, 0x7c, 0x29, 0x14,
    };

    switch (profile) {
        case ggml_backend_xdna_kernel_profile::none:
            return nullptr;

        case ggml_backend_xdna_kernel_profile::i16_256:
            return i16_256_sha256;

        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560:
            return q4k_q8k_k2560_sha256;
    }

    return nullptr;
}

static void ggml_backend_xdna_format_sha256(
        const unsigned char digest[
            GGML_XDNA_SHA256_DIGEST_SIZE],
        char output[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1]) {
    static constexpr char hex[] =
        "0123456789abcdef";

    for (size_t i = 0;
         i < GGML_XDNA_SHA256_DIGEST_SIZE;
         ++i) {
        output[2 * i] =
            hex[(digest[i] >> 4) & 0x0f];

        output[2 * i + 1] =
            hex[digest[i] & 0x0f];
    }

    output[
        GGML_XDNA_SHA256_DIGEST_SIZE * 2] = '\0';
}

static bool ggml_backend_xdna_hash_file_sha256(
        const std::string & path,
        unsigned char digest[
            GGML_XDNA_SHA256_DIGEST_SIZE]) {
    try {
        std::ifstream file(
            path,
            std::ios::binary);

        if (!file) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to open file for SHA256: %s\n",
                path.c_str());

            return false;
        }

        ggml_xdna_sha256_t hash;
        ggml_xdna_sha256_init(&hash);

        char buffer[64 * 1024];

        while (file) {
            file.read(
                buffer,
                sizeof(buffer));

            const std::streamsize count =
                file.gcount();

            if (count > 0) {
                ggml_xdna_sha256_update(
                    &hash,
                    reinterpret_cast<
                        const unsigned char *>(buffer),
                    static_cast<size_t>(count));
            }
        }

        if (file.bad()) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed while hashing file: %s\n",
                path.c_str());

            return false;
        }

        ggml_xdna_sha256_final(
            &hash,
            digest);

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to hash file '%s': %s\n",
            path.c_str(),
            e.what());

        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to hash file '%s': "
            "unknown error\n",
            path.c_str());

        return false;
    }
}

static bool
ggml_backend_xdna_kernel_profile_xclbin_sha256_matches(
        ggml_backend_xdna_kernel_profile profile,
        const unsigned char digest[
            GGML_XDNA_SHA256_DIGEST_SIZE]) {
    static constexpr unsigned char
        i16_256_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xd4, 0x24, 0x66, 0xe1, 0x54, 0xed, 0xe8, 0xc2,
        0x02, 0x05, 0x67, 0x8d, 0x9f, 0x29, 0x2a, 0xfc,
        0x8a, 0x05, 0xf2, 0x24, 0x09, 0xd5, 0x8d, 0x29,
        0x89, 0x5a, 0x9f, 0xe0, 0x64, 0x9e, 0x79, 0x33,
    };

    static constexpr unsigned char
        q4k_q8k_k2560_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xe6, 0xba, 0x92, 0xbf, 0x3e, 0xa6, 0xed, 0x35,
        0x70, 0x77, 0xb7, 0xc8, 0x36, 0xd5, 0x3e, 0x78,
        0x4b, 0xc3, 0xa4, 0xf1, 0x5f, 0x79, 0x31, 0x20,
        0xb8, 0xf3, 0xb9, 0x6f, 0x56, 0xec, 0xc6, 0x8f,
    };

    switch (profile) {
        case ggml_backend_xdna_kernel_profile::none:
            return false;

        case ggml_backend_xdna_kernel_profile::i16_256:
            return std::memcmp(
                digest,
                i16_256_sha256,
                GGML_XDNA_SHA256_DIGEST_SIZE) == 0;

        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560:
            return std::memcmp(
                digest,
                q4k_q8k_k2560_sha256,
                GGML_XDNA_SHA256_DIGEST_SIZE) == 0;
    }

    return false;
}

static bool ggml_backend_xdna_validate_kernel_profile(
        ggml_backend_xdna_backend_context * ctx) {
    if (ctx == nullptr) {
        return false;
    }

    if (ctx->kernel_profile ==
        ggml_backend_xdna_kernel_profile::none) {
        return true;
    }

    const unsigned char * expected_instructions =
        ggml_backend_xdna_kernel_profile_instruction_sha256(
            ctx->kernel_profile);

    if (expected_instructions == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: no instruction fingerprint "
            "registered for profile '%s'\n",
            ggml_backend_xdna_kernel_profile_name(
                ctx->kernel_profile));

        return false;
    }

    if (!ggml_backend_xdna_load_instructions(ctx)) {
        return false;
    }

    unsigned char actual_instructions[
        GGML_XDNA_SHA256_DIGEST_SIZE];

    ggml_xdna_sha256_hash(
        actual_instructions,
        reinterpret_cast<const unsigned char *>(
            ctx->instructions.data()),
        ctx->instructions.size() *
            sizeof(uint32_t));

    if (std::memcmp(
            actual_instructions,
            expected_instructions,
            GGML_XDNA_SHA256_DIGEST_SIZE) != 0) {
        char actual_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        char expected_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        ggml_backend_xdna_format_sha256(
            actual_instructions,
            actual_hex);

        ggml_backend_xdna_format_sha256(
            expected_instructions,
            expected_hex);

        std::fprintf(
            stderr,
            "ggml_xdna: instruction fingerprint mismatch "
            "for profile '%s'\n"
            "ggml_xdna: expected SHA256: %s\n"
            "ggml_xdna: actual   SHA256: %s\n"
            "ggml_xdna: instruction file: %s\n",
            ggml_backend_xdna_kernel_profile_name(
                ctx->kernel_profile),
            expected_hex,
            actual_hex,
            ctx->instructions_path.c_str());

        return false;
    }

    std::fprintf(
        stderr,
        "ggml_xdna: instruction fingerprint "
        "validated for profile '%s'\n",
        ggml_backend_xdna_kernel_profile_name(
            ctx->kernel_profile));

    unsigned char actual_xclbin[
        GGML_XDNA_SHA256_DIGEST_SIZE];

    if (!ggml_backend_xdna_hash_file_sha256(
            ctx->xclbin_path,
            actual_xclbin)) {
        return false;
    }

    if (!ggml_backend_xdna_kernel_profile_xclbin_sha256_matches(
            ctx->kernel_profile,
            actual_xclbin)) {
        char actual_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        ggml_backend_xdna_format_sha256(
            actual_xclbin,
            actual_hex);

        std::fprintf(
            stderr,
            "ggml_xdna: XCLBIN fingerprint mismatch "
            "for profile '%s'\n"
            "ggml_xdna: actual SHA256: %s\n"
            "ggml_xdna: XCLBIN file: %s\n",
            ggml_backend_xdna_kernel_profile_name(
                ctx->kernel_profile),
            actual_hex,
            ctx->xclbin_path.c_str());

        return false;
    }

    std::fprintf(
        stderr,
        "ggml_xdna: XCLBIN fingerprint "
        "validated for profile '%s'\n",
        ggml_backend_xdna_kernel_profile_name(
            ctx->kernel_profile));

    return true;
}

// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    (void) backend;
    return GGML_XDNA_NAME;
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(backend->context);

    if (ctx != nullptr &&
        ctx->device_context != nullptr &&
        ctx->device_profile_registered) {
        auto * device_ctx =
            ctx->device_context;

        if (device_ctx->active_profile !=
            ctx->kernel_profile) {
            std::fprintf(
                stderr,
                "ggml_xdna: backend/device profile mismatch during free\n");
        }

        if (device_ctx->active_backend_count == 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: active backend count already zero during free\n");
        }
        else {
            --device_ctx->active_backend_count;
        }

        if (device_ctx->active_backend_count == 0) {
            device_ctx->active_profile =
                ggml_backend_xdna_kernel_profile::none;
        }

        ctx->device_profile_registered = false;
    }

    if (ctx != nullptr && ctx->kernel_profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
        std::fprintf(stderr, "ggml_xdna: ledger accepted=%llu batched_m9216=%llu batched_m4096=%llu batched_m8192=%llu batched_m1024=%llu fallback=%llu xrt_runs=%llu run_waits=%llu q4_to_device=%llu q8_to_device=%llu output_from_device=%llu\n",
            static_cast<unsigned long long>(ctx->accepted_q4_nodes), static_cast<unsigned long long>(ctx->batched_m9216), static_cast<unsigned long long>(ctx->batched_m4096), static_cast<unsigned long long>(ctx->batched_m8192), static_cast<unsigned long long>(ctx->batched_m1024), static_cast<unsigned long long>(ctx->fallback_rows), static_cast<unsigned long long>(ctx->xrt_runs), static_cast<unsigned long long>(ctx->run_waits), static_cast<unsigned long long>(ctx->q4_to_device), static_cast<unsigned long long>(ctx->q8_to_device), static_cast<unsigned long long>(ctx->output_from_device));
    }

    if (ctx != nullptr && ctx->kernel_profile == ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
        size_t cache_bytes_total = 0;
        for (const auto & kv : ctx->q4_cache.entries) {
            cache_bytes_total += kv.second.transport.size();
        }

        std::fprintf(stderr,
            "ggml_xdna: q4_cache enabled=%d entries=%zu bytes=%zu hits=%llu misses=%llu hits_m9216=%llu hits_m4096=%llu hits_m8192=%llu hits_m1024=%llu blocks_converted=%llu native_bytes_converted=%llu transport_bytes_generated=%llu transport_bytes_copied=%llu\n",
            ctx->q4_cache.enabled ? 1 : 0,
            ctx->q4_cache.entries.size(),
            cache_bytes_total,
            static_cast<unsigned long long>(ctx->q4_cache.hits),
            static_cast<unsigned long long>(ctx->q4_cache.misses),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m9216),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m4096),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m8192),
            static_cast<unsigned long long>(ctx->q4_cache.hits_m1024),
            static_cast<unsigned long long>(ctx->q4_cache.blocks_converted),
            static_cast<unsigned long long>(ctx->q4_cache.native_bytes_converted),
            static_cast<unsigned long long>(ctx->q4_cache.transport_bytes_generated),
            static_cast<unsigned long long>(ctx->q4_cache.transport_bytes_copied));

        std::fprintf(stderr,
            "ggml_xdna: timers_ms q4_conversion=%.3f q4_cached_memcpy=%.3f q4_sync=%.3f q8_quant=%.3f q8_sync=%.3f xrt_submit_wait=%.3f output=%.3f\n",
            ctx->timers.q4_conversion_ms,
            ctx->timers.q4_cached_memcpy_ms,
            ctx->timers.q4_sync_ms,
            ctx->timers.q8_quant_ms,
            ctx->timers.q8_sync_ms,
            ctx->timers.xrt_submit_wait_ms,
            ctx->timers.output_ms);
    }

        static const char * shape_names[] = {"M9216", "M4096", "M8192", "M1024"};
        for (size_t i = 0; i < 4; ++i) {
            const auto & t = ctx->shape_timing[i];
            std::fprintf(stderr, "ggml_xdna: xrt_shape=%s ops=%llu submit_ms=%.3f submit_avg_ms=%.6f wait_ms=%.3f wait_avg_ms=%.6f combined_ms=%.3f\n", shape_names[i], static_cast<unsigned long long>(t.ops), t.submit_ms, t.ops ? t.submit_ms / t.ops : 0.0, t.wait_ms, t.ops ? t.wait_ms / t.ops : 0.0, t.submit_ms + t.wait_ms);
        }

    // kernel, hw_context and xclbin belong to the backend instance.
    // The physical xrt::device remains owned by the static device context.
    delete ctx;
    delete backend;
}

static bool ggml_backend_xdna_run_i16_f32_matmul(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const int16_t * a,
        size_t a_elements,
        const int16_t * b,
        size_t b_elements,
        float * c,
        size_t c_elements);

static bool ggml_backend_xdna_run_q4k_q8k_k2560(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const void * q4_row,
        size_t q4_row_bytes,
        const float * activations,
        size_t activation_elements,
        bool refresh_activation,
        float * result);

static bool ggml_backend_xdna_run_batched_shape(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_batched_state & state,
        const void * q4_rows, size_t q4_rows_bytes, const float * activations,
        size_t activation_elements, float * result, size_t result_bytes);

static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216(
        ggml_backend_t backend,
        const void * q4_rows,
        size_t q4_rows_bytes,
        const float * activations,
        size_t activation_elements,
        float * result,
        size_t result_bytes);

static bool ggml_backend_xdna_is_view_op(enum ggml_op op) {
    return
        op == GGML_OP_VIEW ||
        op == GGML_OP_RESHAPE ||
        op == GGML_OP_PERMUTE ||
        op == GGML_OP_TRANSPOSE;
}

static enum ggml_status ggml_backend_xdna_graph_compute_single_node(
        ggml_backend_t backend,
        struct ggml_tensor * node) {
    if (backend == nullptr) {
        return GGML_STATUS_FAILED;
    }

    if (node == nullptr ||
        node->op != GGML_OP_MUL_MAT ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled graph path requires one MUL_MAT node\n");
        return GGML_STATUS_FAILED;
    }

    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    // Controlled Q4_K graph path.
    //
    // This intentionally does not change supports_op().
    // It can only be reached when graph_compute() is invoked
    // directly on an XDNA backend initialized with the
    // K=2560 Q4_K x Q8_K artifact.
    if (src0->type == GGML_TYPE_Q4_K) {
        constexpr int64_t k = 2560;

        constexpr size_t q4_native_row_bytes =
            10 * sizeof(block_q4_K);

        constexpr size_t f32_activation_bytes =
            static_cast<size_t>(k) * sizeof(float);

        if (src1->type != GGML_TYPE_F32 ||
            node->type != GGML_TYPE_F32) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires Q4_K x F32 -> F32\n");
            return GGML_STATUS_FAILED;
        }

        const int64_t m = src0->ne[1];

        if (src0->ne[0] != k ||
            m <= 0 ||
            src0->ne[2] != 1 ||
            src0->ne[3] != 1 ||
            src1->ne[0] != k ||
            src1->ne[1] != 1 ||
            src1->ne[2] != 1 ||
            src1->ne[3] != 1 ||
            node->ne[0] != m ||
            node->ne[1] != 1 ||
            node->ne[2] != 1 ||
            node->ne[3] != 1) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires [2560,M] x [2560,1] -> [M,1]\n");
            return GGML_STATUS_FAILED;
        }

        if (!ggml_is_contiguous(src0) ||
            !ggml_is_contiguous(src1) ||
            !ggml_is_contiguous(node)) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires contiguous tensors\n");
            return GGML_STATUS_FAILED;
        }

        if (src0->nb[1] != q4_native_row_bytes ||
            src1->nb[1] != f32_activation_bytes ||
            node->nb[0] != sizeof(float)) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT has unexpected tensor strides\n");
            return GGML_STATUS_FAILED;
        }

        if (src0->data == nullptr ||
            src1->data == nullptr ||
            node->data == nullptr) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT requires host-accessible tensor data\n");
            return GGML_STATUS_FAILED;
        }

        const size_t rows =
            static_cast<size_t>(m);

        const size_t expected_q4_bytes =
            rows * q4_native_row_bytes;

        const size_t expected_dst_bytes =
            rows * sizeof(float);

        const size_t actual_q4_bytes =
            ggml_nbytes(src0);

        const size_t actual_src1_bytes =
            ggml_nbytes(src1);

        const size_t actual_dst_bytes =
            ggml_nbytes(node);

        if (actual_q4_bytes != expected_q4_bytes ||
            actual_src1_bytes != f32_activation_bytes ||
            actual_dst_bytes != expected_dst_bytes) {
            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K MUL_MAT has unexpected tensor byte sizes\n");
            return GGML_STATUS_FAILED;
        }

        auto * ctx =
            static_cast<ggml_backend_xdna_backend_context *>(
                backend->context);

        if (ctx == nullptr ||
            !ggml_backend_xdna_load_instructions(ctx)) {
            return GGML_STATUS_FAILED;
        }

        const auto * q4_rows =
            static_cast<const uint8_t *>(
                src0->data);

        const auto * activations =
            static_cast<const float *>(
                src1->data);

        auto * result =
            static_cast<float *>(
                node->data);

        std::fprintf(
            stderr,
            "ggml_xdna: controlled Q4_K K=2560 MUL_MAT graph accepted for M=%lld\n",
            static_cast<long long>(m));

        ++ctx->accepted_q4_nodes;
        if (m == 9216 &&
            ctx->q4_m9216_available) {
            if (!ggml_backend_xdna_run_q4k_q8k_k2560_m9216(
                    backend,
                    q4_rows,
                    expected_q4_bytes,
                    activations,
                    static_cast<size_t>(k),
                    result,
                    expected_dst_bytes)) {
                std::fprintf(
                    stderr,
                    "ggml_xdna: M=9216 multi-row Q4_K K=2560 execution failed\n");
                return GGML_STATUS_FAILED;
            }

            std::fprintf(
                stderr,
                "ggml_xdna: controlled Q4_K K=2560 MUL_MAT executed via M=9216 multi-row fast path\n");

            ++ctx->batched_m9216;
            return GGML_STATUS_SUCCESS;
        }
        if ((m == 1024 && ctx->q4_m1024.available) || (m == 4096 && ctx->q4_m4096.available) || (m == 8192 && ctx->q4_m8192.available)) {
            auto & state = m == 1024 ? ctx->q4_m1024 : (m == 4096 ? ctx->q4_m4096 : ctx->q4_m8192);
            if (!ggml_backend_xdna_run_batched_shape(ctx, state, q4_rows, expected_q4_bytes, activations, static_cast<size_t>(k), result, expected_dst_bytes)) return GGML_STATUS_FAILED;
            if (m == 1024) ++ctx->batched_m1024; else if (m == 4096) ++ctx->batched_m4096; else ++ctx->batched_m8192;
            std::fprintf(stderr, "ggml_xdna: controlled Q4_K K=2560 MUL_MAT executed via M=%lld batched fast path\n", static_cast<long long>(m));
            return GGML_STATUS_SUCCESS;
        }


        // Fallback primitive: one K=2560 dot product per launch.
        ++ctx->fallback_rows;
        // Quantize/upload the shared activation on the first row, then
        // reuse the persistent Q8_K BO for all remaining rows.
        for (size_t row = 0; row < rows; ++row) {
            const void * q4_row =
                q4_rows +
                row * q4_native_row_bytes;

            if (!ggml_backend_xdna_run_q4k_q8k_k2560(
                    backend,
                    ctx->instructions.data(),
                    static_cast<uint32_t>(
                        ctx->instructions.size()),
                    q4_row,
                    q4_native_row_bytes,
                    activations,
                    static_cast<size_t>(k),
                    row == 0,
                    &result[row])) {
                std::fprintf(
                    stderr,
                    "ggml_xdna: controlled Q4_K K=2560 MUL_MAT execution failed at row %zu\n",
                    row);
                return GGML_STATUS_FAILED;
            }
        }

        std::fprintf(
            stderr,
            "ggml_xdna: controlled Q4_K K=2560 MUL_MAT executed on XDNA for M=%lld\n",
            static_cast<long long>(m));

        return GGML_STATUS_SUCCESS;
    }

    if (src0->type != GGML_TYPE_I16 ||
        src1->type != GGML_TYPE_I16 ||
        node->type != GGML_TYPE_F32) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires I16 x I16 -> F32\n");
        return GGML_STATUS_FAILED;
    }

    constexpr int64_t size = 256;

    if (src0->ne[0] != size ||
        src0->ne[1] != size ||
        src0->ne[2] != 1 ||
        src0->ne[3] != 1 ||
        src1->ne[0] != size ||
        src1->ne[1] != size ||
        src1->ne[2] != 1 ||
        src1->ne[3] != 1 ||
        node->ne[0] != size ||
        node->ne[1] != size ||
        node->ne[2] != 1 ||
        node->ne[3] != 1) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires 256x256 tensors\n");
        return GGML_STATUS_FAILED;
    }

    if (!ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(node)) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires contiguous tensors\n");
        return GGML_STATUS_FAILED;
    }

    if (src0->data == nullptr ||
        src1->data == nullptr ||
        node->data == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT requires host-accessible tensor data\n");
        return GGML_STATUS_FAILED;
    }

    std::fprintf(
        stderr,
        "ggml_xdna: controlled 256x256 I16 MUL_MAT graph accepted\n");

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(backend->context);

    if (ctx == nullptr || !ggml_backend_xdna_load_instructions(ctx)) {
        return GGML_STATUS_FAILED;
    }

    constexpr size_t elements = 256 * 256;

    const auto * src0_data =
        static_cast<const int16_t *>(src0->data);
    const auto * src1_data =
        static_cast<const int16_t *>(src1->data);

    auto * dst_data =
        static_cast<float *>(node->data);


    // GGML MUL_MAT semantics:
    //
    // dst[m, n] = dot(src0[n, :], src1[m, :])
    //
    // The NPU kernel computes C = A * B, therefore:
    // A = src1
    // B = transpose(src0)
    std::vector<int16_t> src0_transposed(elements);


    for (int64_t row = 0; row < size; ++row) {
        for (int64_t col = 0; col < size; ++col) {
            src0_transposed[
                static_cast<size_t>(col) * size + row] =
                src0_data[
                    static_cast<size_t>(row) * size + col];
        }
    }

    if (!ggml_backend_xdna_run_i16_f32_matmul(
            backend,
            ctx->instructions.data(),
            static_cast<uint32_t>(ctx->instructions.size()),
            src1_data,
            elements,
            src0_transposed.data(),
            elements,
            dst_data,
            elements)) {
        std::fprintf(
            stderr,
            "ggml_xdna: controlled MUL_MAT execution failed\n");
        return GGML_STATUS_FAILED;
    }


    std::fprintf(
        stderr,
        "ggml_xdna: controlled 256x256 I16 MUL_MAT executed on XDNA\n");

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_xdna_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    if (backend == nullptr || cgraph == nullptr) {
        return GGML_STATUS_FAILED;
    }

    const int n_nodes =
        ggml_graph_n_nodes(cgraph);

    if (n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    int compute_nodes = 0;
    int view_nodes = 0;

    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node =
            ggml_graph_node(
                cgraph,
                i);

        if (node == nullptr) {
            std::fprintf(
                stderr,
                "ggml_xdna: null graph node at index %d\n",
                i);
            return GGML_STATUS_FAILED;
        }

        // Keep this classification exactly aligned with the GGML
        // scheduler's private ggml_is_view_op() helper.
        if (ggml_backend_xdna_is_view_op(node->op)) {
            ++view_nodes;
            continue;
        }

        const enum ggml_status status =
            ggml_backend_xdna_graph_compute_single_node(
                backend,
                node);

        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(
                stderr,
                "ggml_xdna: graph node %d failed "
                "(op=%d)\n",
                i,
                static_cast<int>(node->op));
            return status;
        }

        ++compute_nodes;
    }

    std::fprintf(
        stderr,
        "ggml_xdna: graph dispatch complete: "
        "%d compute node(s), %d view node(s) skipped\n",
        compute_nodes,
        view_nodes);

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_xdna_run_i16_f32_matmul(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const int16_t * a,
        size_t a_elements,
        const int16_t * b,
        size_t b_elements,
        float * c,
        size_t c_elements) {
    if (backend == nullptr ||
        instructions == nullptr ||
        instruction_count == 0 ||
        a == nullptr ||
        a_elements == 0 ||
        b == nullptr ||
        b_elements == 0 ||
        c == nullptr ||
        c_elements == 0) {
        std::fprintf(stderr, "ggml_xdna: invalid i16-f32 matmul arguments\n");
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(backend->context);

    if (ctx == nullptr ||
        ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr ||
        ctx->kernel == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16-f32 matmul requires an initialized XRT kernel\n");
        return false;
    }

    try {
        xrt::device & device =
            *ctx->device_context->xrt_device;

        xrt::kernel & kernel =
            *ctx->kernel;

        xrt::bo bo_instr(
            device,
            static_cast<size_t>(instruction_count) * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE,
            kernel.group_id(1));

        xrt::bo bo_a(
            device,
            a_elements * sizeof(int16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(3));

        xrt::bo bo_b(
            device,
            b_elements * sizeof(int16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(4));

        xrt::bo bo_c(
            device,
            c_elements * sizeof(float),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(5));

        void * buf_instr = bo_instr.map<void *>();
        std::memcpy(
            buf_instr,
            instructions,
            static_cast<size_t>(instruction_count) * sizeof(uint32_t));
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        int16_t * buf_a = bo_a.map<int16_t *>();
        std::memcpy(
            buf_a,
            a,
            a_elements * sizeof(int16_t));
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        int16_t * buf_b = bo_b.map<int16_t *>();
        std::memcpy(
            buf_b,
            b,
            b_elements * sizeof(int16_t));
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        float * buf_c = bo_c.map<float *>();
        std::memset(
            buf_c,
            0,
            c_elements * sizeof(float));
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        constexpr unsigned int opcode = 3;

        auto run = kernel(
            opcode,
            bo_instr,
            instruction_count,
            bo_a,
            bo_b,
            bo_c);

        run.wait();

        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        std::memcpy(
            c,
            buf_c,
            c_elements * sizeof(float));

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16-f32 matmul execution failed: %s\n",
            e.what());
        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16-f32 matmul execution failed: unknown error\n");
        return false;
    }
}

static bool ggml_backend_xdna_init_batched_shape(
        ggml_backend_xdna_backend_context * ctx,
        ggml_backend_xdna_batched_state & state, size_t rows,
        const char * path, const char * expected_xclbin,
        const char * expected_instructions) {
    if (ctx == nullptr || ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr || path == nullptr || path[0] == '\0') return false;
    try {
        unsigned char digest[GGML_XDNA_SHA256_DIGEST_SIZE];
        if (!ggml_backend_xdna_hash_file_sha256(path, digest)) return false;
        char actual[GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];
        ggml_backend_xdna_format_sha256(digest, actual);
        if (std::string(actual) != expected_xclbin) {
            std::fprintf(stderr, "ggml_xdna: M=%zu batched XCLBIN fingerprint mismatch: %s\n", rows, actual);
            return false;
        }
        const std::filesystem::path artifact_path(path);
        const std::string inst_path = (artifact_path.parent_path() / "insts.bin").string();
        std::ifstream file(inst_path, std::ios::binary | std::ios::ate);
        const std::streamsize size = file ? file.tellg() : 0;
        if (size <= 0 || size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) return false;
        state.instructions.resize(static_cast<size_t>(size) / sizeof(uint32_t));
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(state.instructions.data()), size)) return false;
        ggml_xdna_sha256_hash(digest, reinterpret_cast<const unsigned char *>(state.instructions.data()), static_cast<size_t>(size));
        ggml_backend_xdna_format_sha256(digest, actual);
        if (std::string(actual) != expected_instructions) {
            std::fprintf(stderr, "ggml_xdna: M=%zu batched instruction fingerprint mismatch: %s\n", rows, actual);
            state.instructions.clear(); return false;
        }
        state.rows = rows; state.xclbin_path = path;
        state.xclbin = std::make_unique<xrt::xclbin>(state.xclbin_path);
        auto kernels = state.xclbin->get_kernels();
        auto it = std::find_if(kernels.begin(), kernels.end(), [](xrt::xclbin::kernel k) { return k.get_name().rfind("MLIR_AIE", 0) == 0; });
        if (it == kernels.end()) return false;
        ctx->device_context->xrt_device->register_xclbin(*state.xclbin);
        state.hw_context = std::make_unique<xrt::hw_context>(*ctx->device_context->xrt_device, state.xclbin->get_uuid());
        state.kernel = std::make_unique<xrt::kernel>(*state.hw_context, it->get_name());
        state.available = true;
        std::fprintf(stderr, "ggml_xdna: M=%zu batched fast path ready\n", rows);
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ggml_xdna: M=%zu batched init failed: %s\n", rows, e.what());
        state.available = false; return false;
    } catch (...) { state.available = false; return false; }
}

static ggml_backend_xdna_q4_shape ggml_backend_xdna_q4_shape_from_rows(size_t rows) {
    switch (rows) {
        case 9216: return ggml_backend_xdna_q4_shape::m9216;
        case 4096: return ggml_backend_xdna_q4_shape::m4096;
        case 8192: return ggml_backend_xdna_q4_shape::m8192;
        default:   return ggml_backend_xdna_q4_shape::m1024;
    }
}

// Applies the persistent Q4_K -> 152-byte transport cache (experimental,
// GGML_XDNA_Q4K_TRANSPORT_CACHE) or, if disabled/uninitialized, performs the
// exact existing native -> transport conversion directly. Either way,
// `dst_transport` (the mapped Q4 execution BO buffer) ends up holding valid
// transport bytes for `rows` rows; nothing about Q4 BO sync/XRT/output
// changes. `q4_native_rows` is used as the cache key: it is the ggml
// tensor's data pointer, stable for the lifetime of an immutable weight
// tensor within one model/context.
static bool ggml_backend_xdna_q4_cache_apply(
        ggml_backend_xdna_backend_context * ctx,
        ggml_backend_xdna_q4_shape shape,
        const void * q4_native_rows,
        size_t rows,
        void * dst_transport) {
    constexpr size_t block_count = 10;
    constexpr size_t q4_transport_block_bytes = 152;
    constexpr size_t q4_native_row_bytes = block_count * sizeof(block_q4_K);
    constexpr size_t q4_transport_row_bytes = block_count * q4_transport_block_bytes;

    if (ctx == nullptr || q4_native_rows == nullptr || dst_transport == nullptr || rows == 0) {
        return false;
    }

    const size_t native_bytes    = rows * q4_native_row_bytes;
    const size_t transport_bytes = rows * q4_transport_row_bytes;

    auto & cache = ctx->q4_cache;
    if (!cache.env_checked) {
        cache.enabled = ggml_backend_xdna_env_flag_enabled("GGML_XDNA_Q4K_TRANSPORT_CACHE");
        cache.env_checked = true;
        std::fprintf(stderr, "ggml_xdna: Q4 transport cache %s\n", cache.enabled ? "ENABLED" : "disabled");
    }

    auto convert_into = [&](uint8_t * dst) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto * native = static_cast<const uint8_t *>(q4_native_rows);
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t * native_row    = native + row * q4_native_row_bytes;
            uint8_t       * transport_row = dst    + row * q4_transport_row_bytes;
            for (size_t block = 0; block < block_count; ++block) {
                block_q4_K b;
                std::memcpy(&b, native_row + block * sizeof(b), sizeof(b));
                uint8_t * out = transport_row + block * q4_transport_block_bytes;
                std::memcpy(out, &b, sizeof(b));
                const float d    = ggml_fp16_to_fp32(b.d);
                const float dmin = ggml_fp16_to_fp32(b.dmin);
                std::memcpy(out + sizeof(b), &d, sizeof(d));
                std::memcpy(out + sizeof(b) + sizeof(d), &dmin, sizeof(dmin));
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        ctx->timers.q4_conversion_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        cache.blocks_converted          += rows * block_count;
        cache.native_bytes_converted    += native_bytes;
        cache.transport_bytes_generated += transport_bytes;
    };

    if (!cache.enabled) {
        convert_into(static_cast<uint8_t *>(dst_transport));
        return true;
    }

    auto it = cache.entries.find(q4_native_rows);
    const bool valid_hit =
        it != cache.entries.end() &&
        it->second.m               == rows &&
        it->second.native_bytes    == native_bytes &&
        it->second.transport_bytes == transport_bytes &&
        it->second.transport.size() == transport_bytes;

    ggml_backend_xdna_q4_cache_entry * entry = nullptr;

    if (valid_hit) {
        ++cache.hits;
        switch (shape) {
            case ggml_backend_xdna_q4_shape::m9216: ++cache.hits_m9216; break;
            case ggml_backend_xdna_q4_shape::m4096: ++cache.hits_m4096; break;
            case ggml_backend_xdna_q4_shape::m8192: ++cache.hits_m8192; break;
            case ggml_backend_xdna_q4_shape::m1024: ++cache.hits_m1024; break;
        }
        entry = &it->second;
    } else {
        // Miss, or a stale entry whose metadata no longer matches (e.g. a
        // recycled pointer): rebuild safely rather than trust stale bytes.
        ++cache.misses;
        ggml_backend_xdna_q4_cache_entry fresh;
        fresh.m               = rows;
        fresh.native_bytes    = native_bytes;
        fresh.transport_bytes = transport_bytes;
        fresh.transport.assign(transport_bytes, 0);
        convert_into(fresh.transport.data());
        entry = &(cache.entries[q4_native_rows] = std::move(fresh));
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::memcpy(dst_transport, entry->transport.data(), transport_bytes);
    const auto t1 = std::chrono::steady_clock::now();
    ctx->timers.q4_cached_memcpy_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    cache.transport_bytes_copied    += transport_bytes;

    return true;
}

static bool ggml_backend_xdna_run_batched_shape(
        ggml_backend_xdna_backend_context * ctx, ggml_backend_xdna_batched_state & state,
        const void * q4_rows, size_t q4_rows_bytes, const float * activations,
        size_t activation_elements, float * result, size_t result_bytes) {
    constexpr size_t block_count = 10, q4_transport_block_bytes = 152;
    constexpr size_t q4_native_row_bytes = block_count * sizeof(block_q4_K);
    constexpr size_t q4_transport_row_bytes = block_count * q4_transport_block_bytes;
    constexpr size_t q8_row_bytes = block_count * sizeof(block_q8_K);
    if (!state.available || state.kernel == nullptr || q4_rows == nullptr || activations == nullptr || result == nullptr ||
        q4_rows_bytes != state.rows * q4_native_row_bytes || activation_elements != 2560 || result_bytes != state.rows * sizeof(float)) return false;
    try {
        auto & device = *ctx->device_context->xrt_device;
        auto & kernel = *state.kernel;
        const uint32_t instruction_count = static_cast<uint32_t>(state.instructions.size());
        if (!state.bos_initialized) {
            state.bo_instr = std::make_unique<xrt::bo>(device, instruction_count * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
            state.bo_q4 = std::make_unique<xrt::bo>(device, state.rows * q4_transport_row_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
            state.bo_q8 = std::make_unique<xrt::bo>(device, q8_row_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
            state.bo_out = std::make_unique<xrt::bo>(device, state.rows * sizeof(float), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
            state.buf_instr = state.bo_instr->map<void *>(); state.buf_q4 = state.bo_q4->map<void *>();
            state.buf_q8 = state.bo_q8->map<void *>(); state.buf_out = state.bo_out->map<float *>();
            std::memcpy(state.buf_instr, state.instructions.data(), instruction_count * sizeof(uint32_t));
            state.bo_instr->sync(XCL_BO_SYNC_BO_TO_DEVICE); state.instruction_count = instruction_count; state.bos_initialized = true;
        }
        else if (state.instruction_count != instruction_count || std::memcmp(state.buf_instr, state.instructions.data(), instruction_count * sizeof(uint32_t)) != 0) return false;
        if (!ggml_backend_xdna_q4_cache_apply(ctx, ggml_backend_xdna_q4_shape_from_rows(state.rows), q4_rows, state.rows, state.buf_q4)) {
            return false;
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<block_q8_K> q8(block_count); std::memset(q8.data(), 0, q8_row_bytes);
            quantize_row_q8_K_ref(activations, q8.data(), 2560); std::memcpy(state.buf_q8, q8.data(), q8_row_bytes);
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.q8_quant_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            state.bo_q4->sync(XCL_BO_SYNC_BO_TO_DEVICE); ++ctx->q4_to_device;
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.q4_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            state.bo_q8->sync(XCL_BO_SYNC_BO_TO_DEVICE); ++ctx->q8_to_device;
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.q8_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        {
            const size_t shape_index = static_cast<size_t>(ggml_backend_xdna_q4_shape_from_rows(state.rows));
            auto & timing = ctx->shape_timing[shape_index];
            const auto submit_begin = std::chrono::steady_clock::now();
            auto run = (*state.kernel)(3u, *state.bo_instr, instruction_count, *state.bo_q4, *state.bo_q8, *state.bo_out);
            const auto submit_end = std::chrono::steady_clock::now();
            timing.submit_ms += std::chrono::duration<double, std::milli>(submit_end - submit_begin).count();
            const auto wait_begin = std::chrono::steady_clock::now();
            ++ctx->xrt_runs; run.wait(); ++ctx->run_waits;
            const auto wait_end = std::chrono::steady_clock::now();
            timing.wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();
            ++timing.ops;
            ctx->timers.xrt_submit_wait_ms += std::chrono::duration<double, std::milli>(wait_end - submit_begin).count();
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            state.bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE); ++ctx->output_from_device;
            std::memcpy(result, state.buf_out, result_bytes);
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.output_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        return true;
    } catch (...) { return false; }
}
static bool ggml_backend_xdna_run_q4k_q8k_k2560_m9216(
        ggml_backend_t backend,
        const void * q4_rows,
        size_t q4_rows_bytes,
        const float * activations,
        size_t activation_elements,
        float * result,
        size_t result_bytes) {
    constexpr size_t rows = 9216;
    constexpr int64_t elements = 2560;
    constexpr size_t block_count = 10;
    constexpr size_t q4_transport_block_bytes = 152;

    static_assert(QK_K == 256, "unexpected QK_K");
    static_assert(sizeof(block_q4_K) == 144, "unexpected Q4_K block size");
    static_assert(sizeof(block_q8_K) == 292, "unexpected Q8_K block size");

    constexpr size_t q4_native_row_bytes =
        block_count * sizeof(block_q4_K);

    constexpr size_t q4_transport_row_bytes =
        block_count * q4_transport_block_bytes;

    constexpr size_t q4_transport_bytes =
        rows * q4_transport_row_bytes;

    constexpr size_t q8_row_bytes =
        block_count * sizeof(block_q8_K);

    constexpr size_t output_bytes =
        rows * sizeof(float);

    if (backend == nullptr ||
        q4_rows == nullptr ||
        activations == nullptr ||
        result == nullptr ||
        q4_rows_bytes != rows * q4_native_row_bytes ||
        activation_elements != static_cast<size_t>(elements) ||
        result_bytes != output_bytes) {
        std::fprintf(
            stderr,
            "ggml_xdna: invalid M=9216 multi-row Q4_K x Q8_K arguments\n");
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(
            backend->context);

    if (ctx == nullptr ||
        ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr ||
        !ctx->q4_m9216_available ||
        ctx->q4_m9216_kernel == nullptr ||
        ctx->q4_m9216_instructions.empty()) {
        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row fast path is not initialized\n");
        return false;
    }

    try {
        xrt::device & device =
            *ctx->device_context->xrt_device;

        xrt::kernel & kernel =
            *ctx->q4_m9216_kernel;

        const uint32_t instruction_count =
            static_cast<uint32_t>(
                ctx->q4_m9216_instructions.size());

        const size_t instruction_bytes =
            static_cast<size_t>(instruction_count) *
            sizeof(uint32_t);

        if (!ctx->q4_m9216_bos_initialized) {
            auto bo_instr = std::make_unique<xrt::bo>(
                device,
                instruction_bytes,
                XCL_BO_FLAGS_CACHEABLE,
                kernel.group_id(1));

            auto bo_q4 = std::make_unique<xrt::bo>(
                device,
                q4_transport_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(3));

            auto bo_q8 = std::make_unique<xrt::bo>(
                device,
                q8_row_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(4));

            auto bo_out = std::make_unique<xrt::bo>(
                device,
                output_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(5));

            void * buf_instr =
                bo_instr->map<void *>();

            void * buf_q4 =
                bo_q4->map<void *>();

            void * buf_q8 =
                bo_q8->map<void *>();

            float * buf_out =
                bo_out->map<float *>();

            std::memcpy(
                buf_instr,
                ctx->q4_m9216_instructions.data(),
                instruction_bytes);

            bo_instr->sync(
                XCL_BO_SYNC_BO_TO_DEVICE);

            ctx->q4_m9216_bo_instr =
                std::move(bo_instr);

            ctx->q4_m9216_bo_q4 =
                std::move(bo_q4);

            ctx->q4_m9216_bo_q8 =
                std::move(bo_q8);

            ctx->q4_m9216_bo_out =
                std::move(bo_out);

            ctx->q4_m9216_buf_instr =
                buf_instr;

            ctx->q4_m9216_buf_q4 =
                buf_q4;

            ctx->q4_m9216_buf_q8 =
                buf_q8;

            ctx->q4_m9216_buf_out =
                buf_out;

            ctx->q4_m9216_instruction_count =
                instruction_count;

            ctx->q4_m9216_bos_initialized =
                true;
        }
        else if (
            ctx->q4_m9216_instruction_count != instruction_count ||
            std::memcmp(
                ctx->q4_m9216_buf_instr,
                ctx->q4_m9216_instructions.data(),
                instruction_bytes) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: persistent M=9216 instruction BO mismatch\n");
            return false;
        }

        xrt::bo & bo_q4 =
            *ctx->q4_m9216_bo_q4;

        xrt::bo & bo_q8 =
            *ctx->q4_m9216_bo_q8;

        xrt::bo & bo_out =
            *ctx->q4_m9216_bo_out;

        if (!ggml_backend_xdna_q4_cache_apply(
                ctx, ggml_backend_xdna_q4_shape::m9216,
                q4_rows, rows, ctx->q4_m9216_buf_q4)) {
            return false;
        }

        {
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<block_q8_K> q8_blocks(block_count);
            std::memset(q8_blocks.data(), 0, q8_row_bytes);
            quantize_row_q8_K_ref(activations, q8_blocks.data(), elements);
            std::memcpy(ctx->q4_m9216_buf_q8, q8_blocks.data(), q8_row_bytes);
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.q8_quant_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        {
            const auto t0 = std::chrono::steady_clock::now();
            bo_q4.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            ++ctx->q4_to_device;
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.q4_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        {
            const auto t0 = std::chrono::steady_clock::now();
            bo_q8.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            ++ctx->q8_to_device;
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.q8_sync_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        constexpr unsigned int opcode = 3;
        {
            auto & timing = ctx->shape_timing[static_cast<size_t>(ggml_backend_xdna_q4_shape::m9216)];
            const auto submit_begin = std::chrono::steady_clock::now();
            auto run = kernel(
                opcode,
                *ctx->q4_m9216_bo_instr,
                instruction_count,
                bo_q4,
                bo_q8,
                bo_out);
            const auto submit_end = std::chrono::steady_clock::now();
            timing.submit_ms += std::chrono::duration<double, std::milli>(submit_end - submit_begin).count();
            const auto wait_begin = std::chrono::steady_clock::now();
            run.wait();
            ++ctx->xrt_runs; ++ctx->run_waits;
            const auto wait_end = std::chrono::steady_clock::now();
            timing.wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();
            ++timing.ops;
            ctx->timers.xrt_submit_wait_ms += std::chrono::duration<double, std::milli>(wait_end - submit_begin).count();
        }

        {
            const auto t0 = std::chrono::steady_clock::now();
            bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            ++ctx->output_from_device;
            std::memcpy(result, ctx->q4_m9216_buf_out, output_bytes);
            const auto t1 = std::chrono::steady_clock::now();
            ctx->timers.output_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        return true;
    }
    catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row execution failed: %s\n",
            e.what());
        return false;
    }
    catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row execution failed: unknown error\n");
        return false;
    }
}
static bool ggml_backend_xdna_run_q4k_q8k_k2560(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const void * q4_row,
        size_t q4_row_bytes,
        const float * activations,
        size_t activation_elements,
        bool refresh_activation,
        float * result) {
    constexpr int64_t elements = 2560;
    constexpr size_t block_count = 10;
    constexpr size_t q4_transport_block_bytes = 152;

    static_assert(QK_K == 256, "unexpected QK_K");
    static_assert(sizeof(block_q4_K) == 144, "unexpected Q4_K block size");
    static_assert(sizeof(block_q8_K) == 292, "unexpected Q8_K block size");

    constexpr size_t q4_native_row_bytes =
        block_count * sizeof(block_q4_K);

    constexpr size_t q4_transport_row_bytes =
        block_count * q4_transport_block_bytes;

    constexpr size_t q8_row_bytes =
        block_count * sizeof(block_q8_K);

    if (backend == nullptr ||
        instructions == nullptr ||
        instruction_count == 0 ||
        q4_row == nullptr ||
        activations == nullptr ||
        result == nullptr ||
        q4_row_bytes != q4_native_row_bytes ||
        activation_elements != static_cast<size_t>(elements)) {
        std::fprintf(
            stderr,
            "ggml_xdna: invalid Q4_K x Q8_K K=2560 arguments\n");
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(
            backend->context);

    if (ctx == nullptr ||
        ctx->device_context == nullptr ||
        ctx->device_context->xrt_device == nullptr ||
        ctx->kernel == nullptr) {
        std::fprintf(
            stderr,
            "ggml_xdna: Q4_K x Q8_K K=2560 requires an initialized XRT kernel\n");
        return false;
    }

    const auto * q4_native =
        static_cast<const uint8_t *>(q4_row);

    std::vector<uint8_t> q4_transport(
        q4_transport_row_bytes);

    for (size_t block = 0; block < block_count; ++block) {
        block_q4_K native_block;

        std::memcpy(
            &native_block,
            q4_native + block * sizeof(block_q4_K),
            sizeof(native_block));

        uint8_t * dst =
            q4_transport.data() +
            block * q4_transport_block_bytes;

        std::memcpy(
            dst,
            &native_block,
            sizeof(native_block));

        const float d =
            ggml_fp16_to_fp32(native_block.d);

        const float dmin =
            ggml_fp16_to_fp32(native_block.dmin);

        std::memcpy(
            dst + sizeof(block_q4_K),
            &d,
            sizeof(d));

        std::memcpy(
            dst + sizeof(block_q4_K) + sizeof(d),
            &dmin,
            sizeof(dmin));
    }

    std::vector<block_q8_K> q8_blocks;

    if (refresh_activation) {
        q8_blocks.resize(block_count);

        // quantize_row_q8_K_ref leaves bsums untouched for an
        // all-zero block, so initialize the complete native row.
        std::memset(
            q8_blocks.data(),
            0,
            q8_row_bytes);

        quantize_row_q8_K_ref(
            activations,
            q8_blocks.data(),
            elements);
    }

    try {
        xrt::device & device =
            *ctx->device_context->xrt_device;

        xrt::kernel & kernel =
            *ctx->kernel;

        const size_t instruction_bytes =
            static_cast<size_t>(instruction_count) *
                sizeof(uint32_t);

        if (!ctx->q4_bos_initialized) {
            auto bo_instr = std::make_unique<xrt::bo>(
                device,
                instruction_bytes,
                XCL_BO_FLAGS_CACHEABLE,
                kernel.group_id(1));

            auto bo_q4 = std::make_unique<xrt::bo>(
                device,
                q4_transport_row_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(3));

            auto bo_q8 = std::make_unique<xrt::bo>(
                device,
                q8_row_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(4));

            auto bo_out = std::make_unique<xrt::bo>(
                device,
                sizeof(float),
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(5));

            void * buf_instr =
                bo_instr->map<void *>();

            void * buf_q4 =
                bo_q4->map<void *>();

            void * buf_q8 =
                bo_q8->map<void *>();

            float * buf_out =
                bo_out->map<float *>();

            std::memcpy(
                buf_instr,
                instructions,
                instruction_bytes);

            bo_instr->sync(
                XCL_BO_SYNC_BO_TO_DEVICE);

            ctx->q4_bo_instr = std::move(bo_instr);
            ctx->q4_bo_q4 = std::move(bo_q4);
            ctx->q4_bo_q8 = std::move(bo_q8);
            ctx->q4_bo_out = std::move(bo_out);

            ctx->q4_buf_instr = buf_instr;
            ctx->q4_buf_q4 = buf_q4;
            ctx->q4_buf_q8 = buf_q8;
            ctx->q4_buf_out = buf_out;

            ctx->q4_instruction_count =
                instruction_count;

            ctx->q4_bos_initialized = true;
        }
        else if (
            ctx->q4_instruction_count != instruction_count ||
            std::memcmp(
                ctx->q4_buf_instr,
                instructions,
                instruction_bytes) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: persistent Q4_K BO instruction mismatch\n");
            return false;
        }

        xrt::bo & bo_instr =
            *ctx->q4_bo_instr;

        xrt::bo & bo_q4 =
            *ctx->q4_bo_q4;

        xrt::bo & bo_q8 =
            *ctx->q4_bo_q8;

        xrt::bo & bo_out =
            *ctx->q4_bo_out;

        void * buf_q4 =
            ctx->q4_buf_q4;

        void * buf_q8 =
            ctx->q4_buf_q8;

        float * buf_out =
            ctx->q4_buf_out;

        std::memcpy(
            buf_q4,
            q4_transport.data(),
            q4_transport_row_bytes);

        bo_q4.sync(
            XCL_BO_SYNC_BO_TO_DEVICE);

        if (refresh_activation) {
            std::memcpy(
                buf_q8,
                q8_blocks.data(),
                q8_row_bytes);

            bo_q8.sync(
                XCL_BO_SYNC_BO_TO_DEVICE);
        }


        constexpr unsigned int opcode = 3;

        auto run = kernel(
            opcode,
            bo_instr,
            instruction_count,
            bo_q4,
            bo_q8,
            bo_out);

        run.wait();

        bo_out.sync(
            XCL_BO_SYNC_BO_FROM_DEVICE);

        *result = *buf_out;

        return true;
    }
    catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: Q4_K x Q8_K K=2560 execution failed: %s\n",
            e.what());
        return false;
    }
    catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: Q4_K x Q8_K K=2560 execution failed: unknown error\n");
        return false;
    }
}

static struct ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = {
        0x58, 0x44, 0x4e, 0x41,
        0x2d, 0x47, 0x47, 0x4d,
        0x4c, 0x2d, 0x58, 0x52,
        0x54, 0x2d, 0x30, 0x31
    };

    return &guid;
}

static bool
ggml_backend_xdna_init_q4k_q8k_k2560_m9216(
        ggml_backend_xdna_backend_context * ctx,
        const char * xclbin_path) {
    if (ctx == nullptr ||
        xclbin_path == nullptr ||
        xclbin_path[0] == '\0') {
        return false;
    }

    static constexpr unsigned char
        expected_xclbin_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0x33, 0x9c, 0x39, 0xe5, 0x43, 0xd0, 0x02, 0x45,
        0x7a, 0x43, 0x9f, 0xee, 0x8e, 0x80, 0x9a, 0xbf,
        0xdb, 0x8f, 0xa0, 0x05, 0xe2, 0x02, 0x25, 0x3d,
        0xfc, 0xca, 0x22, 0x89, 0x0e, 0xe9, 0x98, 0x28,
    };

    static constexpr unsigned char
        expected_instructions_sha256[
            GGML_XDNA_SHA256_DIGEST_SIZE] = {
        0xbc, 0x8c, 0x03, 0x1c, 0x85, 0x42, 0x17, 0xa6,
        0xaf, 0x5d, 0xce, 0xce, 0x5c, 0x5b, 0xfc, 0x5b,
        0xcf, 0xda, 0xfe, 0xb0, 0xfe, 0xba, 0x44, 0xa5,
        0x00, 0x79, 0xb4, 0xba, 0x81, 0x9b, 0x81, 0x4b,
    };

    ctx->q4_m9216_xclbin_path =
        xclbin_path;

    const std::filesystem::path artifact_path(
        ctx->q4_m9216_xclbin_path);

    ctx->q4_m9216_instructions_path =
        (
            artifact_path.parent_path() /
            "insts.bin"
        ).string();

    unsigned char actual_xclbin[
        GGML_XDNA_SHA256_DIGEST_SIZE];

    if (!ggml_backend_xdna_hash_file_sha256(
            ctx->q4_m9216_xclbin_path,
            actual_xclbin)) {
        return false;
    }

    if (std::memcmp(
            actual_xclbin,
            expected_xclbin_sha256,
            GGML_XDNA_SHA256_DIGEST_SIZE) != 0) {
        char actual_hex[
            GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

        ggml_backend_xdna_format_sha256(
            actual_xclbin,
            actual_hex);

        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 XCLBIN fingerprint mismatch\n"
            "ggml_xdna: actual SHA256: %s\n"
            "ggml_xdna: XCLBIN file: %s\n",
            actual_hex,
            ctx->q4_m9216_xclbin_path.c_str());

        return false;
    }

    try {
        std::ifstream file(
            ctx->q4_m9216_instructions_path,
            std::ios::binary |
                std::ios::ate);

        if (!file) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to open M=9216 instructions: %s\n",
                ctx->q4_m9216_instructions_path.c_str());
            return false;
        }

        const std::streamsize size =
            file.tellg();

        if (size <= 0 ||
            size % static_cast<std::streamsize>(
                sizeof(uint32_t)) != 0) {
            std::fprintf(
                stderr,
                "ggml_xdna: invalid M=9216 instruction file size\n");
            return false;
        }

        ctx->q4_m9216_instructions.resize(
            static_cast<size_t>(size) /
            sizeof(uint32_t));

        file.seekg(
            0,
            std::ios::beg);

        if (!file.read(
                reinterpret_cast<char *>(
                    ctx->q4_m9216_instructions.data()),
                size)) {
            std::fprintf(
                stderr,
                "ggml_xdna: failed to read M=9216 instructions\n");

            ctx->q4_m9216_instructions.clear();
            return false;
        }

        unsigned char actual_instructions[
            GGML_XDNA_SHA256_DIGEST_SIZE];

        ggml_xdna_sha256_hash(
            actual_instructions,
            reinterpret_cast<const unsigned char *>(
                ctx->q4_m9216_instructions.data()),
            ctx->q4_m9216_instructions.size() *
                sizeof(uint32_t));

        if (std::memcmp(
                actual_instructions,
                expected_instructions_sha256,
                GGML_XDNA_SHA256_DIGEST_SIZE) != 0) {
            char actual_hex[
                GGML_XDNA_SHA256_DIGEST_SIZE * 2 + 1];

            ggml_backend_xdna_format_sha256(
                actual_instructions,
                actual_hex);

            std::fprintf(
                stderr,
                "ggml_xdna: M=9216 instruction fingerprint mismatch\n"
                "ggml_xdna: actual SHA256: %s\n"
                "ggml_xdna: instruction file: %s\n",
                actual_hex,
                ctx->q4_m9216_instructions_path.c_str());

            ctx->q4_m9216_instructions.clear();
            return false;
        }

        ctx->q4_m9216_xclbin =
            std::make_unique<xrt::xclbin>(
                ctx->q4_m9216_xclbin_path);

        auto kernels =
            ctx->q4_m9216_xclbin->get_kernels();

        auto it =
            std::find_if(
                kernels.begin(),
                kernels.end(),
                [](xrt::xclbin::kernel kernel) {
                    return kernel.get_name().rfind(
                        "MLIR_AIE",
                        0) == 0;
                });

        if (it == kernels.end()) {
            std::fprintf(
                stderr,
                "ggml_xdna: MLIR_AIE kernel not found in M=9216 XCLBIN\n");
            return false;
        }

        const std::string resolved_name =
            it->get_name();

        ctx->device_context->xrt_device->
            register_xclbin(
                *ctx->q4_m9216_xclbin);

        ctx->q4_m9216_hw_context =
            std::make_unique<xrt::hw_context>(
                *ctx->device_context->xrt_device,
                ctx->q4_m9216_xclbin->get_uuid());

        ctx->q4_m9216_kernel =
            std::make_unique<xrt::kernel>(
                *ctx->q4_m9216_hw_context,
                resolved_name);

        ctx->q4_m9216_available =
            true;

        std::fprintf(
            stderr,
            "ggml_xdna: M=9216 multi-row fast path ready: "
            "%zu instruction words, kernel %s\n",
            ctx->q4_m9216_instructions.size(),
            resolved_name.c_str());

        return true;
    }
    catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to initialize M=9216 fast path: %s\n",
            e.what());

        ctx->q4_m9216_available =
            false;

        return false;
    }
    catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: failed to initialize M=9216 fast path: "
            "unknown error\n");

        ctx->q4_m9216_available =
            false;

        return false;
    }
}
static ggml_backend_t ggml_backend_xdna_device_init(
        ggml_backend_dev_t dev,
        const char * params) {
    ggml_backend_xdna_context * device_ctx =
        ggml_backend_xdna_get_context(dev);

    if (device_ctx == nullptr || device_ctx->xrt_device == nullptr) {
        return nullptr;
    }

    const char * resolved_params = params;

    // Normal llama.cpp backend initialization passes nullptr.
    // Allow an XDNA-specific opt-in without changing generic llama
    // initialization or command-line parameter plumbing.
    //
    // Explicit non-empty backend params always take precedence.
    if (resolved_params == nullptr || resolved_params[0] == '\0') {
        const char * env_params =
            std::getenv("GGML_XDNA_BACKEND_PARAMS");

        if (env_params != nullptr && env_params[0] != '\0') {
            resolved_params = env_params;
        }
    }

    ggml_backend_xdna_kernel_profile requested_profile =
        ggml_backend_xdna_kernel_profile::none;

    std::string xclbin_path;

    if (!ggml_backend_xdna_parse_init_params(
            resolved_params,
            &requested_profile,
            &xclbin_path)) {
        return nullptr;
    }

    if (device_ctx->active_backend_count != 0 &&
        device_ctx->active_profile != requested_profile) {
        std::fprintf(
            stderr,
            "ggml_xdna: cannot initialize profile '%s'; "
            "device already has %zu backend instance(s) using profile '%s'\n",
            ggml_backend_xdna_kernel_profile_name(requested_profile),
            device_ctx->active_backend_count,
            ggml_backend_xdna_kernel_profile_name(
                device_ctx->active_profile));
        return nullptr;
    }

    auto * backend_ctx =
        new ggml_backend_xdna_backend_context;

    backend_ctx->device_context = device_ctx;
    backend_ctx->kernel_profile = requested_profile;

    if (requested_profile !=
        ggml_backend_xdna_kernel_profile::none) {
        backend_ctx->xclbin_path =
            xclbin_path;

        if (!ggml_backend_xdna_validate_kernel_profile(
                backend_ctx)) {
            delete backend_ctx;
            return nullptr;
        }
    }

    const char * kernel_params =
        xclbin_path.empty()
            ? nullptr
            : xclbin_path.c_str();

    if (!ggml_backend_xdna_init_kernel(
            backend_ctx,
            kernel_params)) {
        delete backend_ctx;
        return nullptr;
    }

    if (requested_profile ==
        ggml_backend_xdna_kernel_profile::q4k_q8k_k2560) {
        const std::filesystem::path primary_xclbin_path(
            backend_ctx->xclbin_path);
        const std::filesystem::path kernel_root =
            primary_xclbin_path.parent_path().parent_path();
        auto resolve_shape_path = [&](const char * env_name, size_t rows) {
            const char * env = std::getenv(env_name);
            if (env != nullptr && env[0] != '\0') {
                std::fprintf(stderr,
                    "ggml_xdna: using explicit M=%zu XCLBIN override: %s\n",
                    rows, env);
                return std::string(env);
            }
            const auto path = kernel_root /
                (std::string("q4k_q8k_k2560_m") + std::to_string(rows)) /
                "final.xclbin";
            if (std::filesystem::exists(path)) {
                std::fprintf(stderr,
                    "ggml_xdna: auto-discovered M=%zu XCLBIN: %s\n",
                    rows, path.string().c_str());
                return path.string();
            }
            return std::string();
        };

        std::string m9216_xclbin_path =
            resolve_shape_path("GGML_XDNA_Q4K_M9216_XCLBIN", 9216);
        if (!m9216_xclbin_path.empty() &&
            !ggml_backend_xdna_init_q4k_q8k_k2560_m9216(
                backend_ctx, m9216_xclbin_path.c_str())) {
            delete backend_ctx;
            return nullptr;
        }
        if (m9216_xclbin_path.empty()) {
            std::fprintf(stderr,
                "ggml_xdna: M=9216 artifact not found; per-row fallback remains active\n");
        }

        const struct {
            const char * env_name;
            size_t rows;
            ggml_backend_xdna_batched_state ggml_backend_xdna_backend_context::* state;
            const char * xclbin_sha;
            const char * instructions_sha;
        } shapes[] = {
            {"GGML_XDNA_Q4K_M1024_XCLBIN", 1024, &ggml_backend_xdna_backend_context::q4_m1024,
                "b4238e2ec9c302832f7d0b592b7c8c004779403473f6a2bfe4512e663de081c9",
                "896b176c47abc9456b6744df36097982cc373f38306ec4cde39a51e1a3422b50"},
            {"GGML_XDNA_Q4K_M4096_XCLBIN", 4096, &ggml_backend_xdna_backend_context::q4_m4096,
                "07fc35af0d8203fb2d7cd748e88bcd99a4febbcb7eb09a6e2e8e389af38b82ff",
                "965611200fb2bd4bfeb216726a464303ed9b88de4defffb6c4811c5f9b35a0be"},
            {"GGML_XDNA_Q4K_M8192_XCLBIN", 8192, &ggml_backend_xdna_backend_context::q4_m8192,
                "1d066a247221ec16817ac7fa2ad5f096c8bc0a78652d757f17c9353006bb314c",
                "b6c2bedb58c105e616b203be561136b81d2789b90ec14307817dad6504490695"},
        };
        for (const auto & shape : shapes) {
            const std::string path = resolve_shape_path(shape.env_name, shape.rows);
            if (!path.empty() && !ggml_backend_xdna_init_batched_shape(
                    backend_ctx, backend_ctx->*shape.state, shape.rows, path.c_str(),
                    shape.xclbin_sha, shape.instructions_sha)) {
                delete backend_ctx;
                return nullptr;
            }
            if (path.empty()) {
                std::fprintf(stderr,
                    "ggml_xdna: M=%zu artifact not found; per-row fallback remains active\n",
                    shape.rows);
            }
        }
    }

    if (device_ctx->active_backend_count == 0) {
        device_ctx->active_profile =
            requested_profile;
    }

    ++device_ctx->active_backend_count;
    backend_ctx->device_profile_registered = true;

    std::fprintf(
        stderr,
        "ggml_xdna: backend profile '%s' registered "
        "(%zu active instance(s))\n",
        ggml_backend_xdna_kernel_profile_name(
            requested_profile),
        device_ctx->active_backend_count);

    return new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ dev,
        /* .context = */ backend_ctx,
    };
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(
        ggml_backend_dev_t dev) {
    (void) dev;
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    (void) dev;
    (void) max_tensor_size;

    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_xdna_device_supports_op(
        ggml_backend_dev_t dev,
        const struct ggml_tensor * op) {
    const ggml_backend_xdna_context * device_ctx =
        ggml_backend_xdna_get_context(dev);

    if (device_ctx == nullptr ||
        device_ctx->active_backend_count == 0) {
        return false;
    }

    if (op == nullptr ||
        op->op != GGML_OP_MUL_MAT ||
        op->src[0] == nullptr ||
        op->src[1] == nullptr) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (device_ctx->active_profile) {
        case ggml_backend_xdna_kernel_profile::none:
            // Plain-path/direct-probe mode intentionally does not
            // advertise operations to the scheduler.
            return false;

        case ggml_backend_xdna_kernel_profile::i16_256: {
            if (src0->type != GGML_TYPE_I16 ||
                src1->type != GGML_TYPE_I16 ||
                op->type != GGML_TYPE_F32) {
                return false;
            }

            constexpr int64_t size = 256;

            if (src0->ne[0] != size ||
                src0->ne[1] != size ||
                src0->ne[2] != 1 ||
                src0->ne[3] != 1 ||
                src1->ne[0] != size ||
                src1->ne[1] != size ||
                src1->ne[2] != 1 ||
                src1->ne[3] != 1 ||
                op->ne[0] != size ||
                op->ne[1] != size ||
                op->ne[2] != 1 ||
                op->ne[3] != 1) {
                return false;
            }

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   ggml_is_contiguous(op);
        }

        case ggml_backend_xdna_kernel_profile::q4k_q8k_k2560: {
            if (src0->type != GGML_TYPE_Q4_K ||
                src1->type != GGML_TYPE_F32 ||
                op->type != GGML_TYPE_F32) {
                return false;
            }

            constexpr int64_t k = 2560;

            constexpr size_t q4_native_row_bytes =
                10 * sizeof(block_q4_K);

            constexpr size_t f32_activation_bytes =
                static_cast<size_t>(k) * sizeof(float);

            const int64_t m = src0->ne[1];

            if (src0->ne[0] != k ||
                m <= 0 ||
                src0->ne[2] != 1 ||
                src0->ne[3] != 1 ||
                src1->ne[0] != k ||
                src1->ne[1] != 1 ||
                src1->ne[2] != 1 ||
                src1->ne[3] != 1 ||
                op->ne[0] != m ||
                op->ne[1] != 1 ||
                op->ne[2] != 1 ||
                op->ne[3] != 1) {
                return false;
            }

            if (!ggml_is_contiguous(src0) ||
                !ggml_is_contiguous(src1) ||
                !ggml_is_contiguous(op)) {
                return false;
            }

            if (src0->nb[1] != q4_native_row_bytes ||
                src1->nb[1] != f32_activation_bytes ||
                op->nb[0] != sizeof(float)) {
                return false;
            }

            const size_t rows =
                static_cast<size_t>(m);

            return
                ggml_nbytes(src0) ==
                    rows * q4_native_row_bytes &&
                ggml_nbytes(src1) ==
                    f32_activation_bytes &&
                ggml_nbytes(op) ==
                    rows * sizeof(float);
        }
    }

    return false;
}

static bool ggml_backend_xdna_device_supports_buft(
        ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft) {
    (void) dev;

    return buft != nullptr && ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    (void) reg;
    return GGML_XDNA_NAME;
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    const auto * device =
        static_cast<ggml_backend_device *>(reg->context);

    if (device == nullptr || device->context == nullptr) {
        return 0;
    }

    const auto * ctx =
        static_cast<ggml_backend_xdna_context *>(device->context);

    return ctx->xrt_device != nullptr ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(
        ggml_backend_reg_t reg,
        size_t index) {
    if (index != 0 || ggml_backend_xdna_reg_get_device_count(reg) == 0) {
        return nullptr;
    }

    return static_cast<ggml_backend_device *>(reg->context);
}

static void * ggml_backend_xdna_reg_get_proc_address(
        ggml_backend_reg_t reg,
        const char * name) {
    (void) reg;

    if (name != nullptr &&
        std::strcmp(name, "ggml_backend_xdna_run_i16_f32_matmul") == 0) {
        return (void *) ggml_backend_xdna_run_i16_f32_matmul;
    }

    if (name != nullptr &&
        std::strcmp(name, "ggml_backend_xdna_run_q4k_q8k_k2560") == 0) {
        return (void *) ggml_backend_xdna_run_q4k_q8k_k2560;
    }

    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static ggml_backend_xdna_context context =
        ggml_backend_xdna_create_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ nullptr,
    };

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ &reg,
        /* .context = */ &context,
    };

    reg.context = &device;
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)