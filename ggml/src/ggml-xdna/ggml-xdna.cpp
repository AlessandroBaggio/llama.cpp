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
        float * result);

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

        // Correctness-first multi-row path.
        //
        // The current primitive computes one K=2560 dot product.
        // Execute one weight row at a time. This intentionally
        // re-quantizes/re-uploads the shared activation for each row;
        // activation reuse is a later performance optimization.
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

static bool ggml_backend_xdna_run_q4k_q8k_k2560(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const void * q4_row,
        size_t q4_row_bytes,
        const float * activations,
        size_t activation_elements,
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

    std::vector<block_q8_K> q8_blocks(block_count);

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

        std::memcpy(
            buf_q8,
            q8_blocks.data(),
            q8_row_bytes);

        bo_q8.sync(
            XCL_BO_SYNC_BO_TO_DEVICE);

        *buf_out = 0.0f;

        bo_out.sync(
            XCL_BO_SYNC_BO_TO_DEVICE);

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