#include "ggml-xdna.h"

#include "ggml-backend-impl.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

struct ggml_backend_xdna_context {
    std::unique_ptr<xrt::device> xrt_device;
    std::string description = "AMD XDNA NPU (XRT device 0)";
    std::string device_id;
};

struct ggml_backend_xdna_backend_context {
    ggml_backend_xdna_context * device_context = nullptr;

    std::string xclbin_path;
    std::string kernel_name = "MLIR_AIE";

    std::unique_ptr<xrt::xclbin> xclbin;
    std::unique_ptr<xrt::hw_context> hw_context;
    std::unique_ptr<xrt::kernel> kernel;
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

// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    (void) backend;
    return GGML_XDNA_NAME;
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    auto * ctx =
        static_cast<ggml_backend_xdna_backend_context *>(backend->context);

    // kernel, hw_context and xclbin belong to the backend instance.
    // The physical xrt::device remains owned by the static device context.
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xdna_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    (void) backend;
    (void) cgraph;

    // No GGML operations are supported yet.
    return GGML_STATUS_FAILED;
}

static bool ggml_backend_xdna_run_i16_matmul(
        ggml_backend_t backend,
        const uint32_t * instructions,
        uint32_t instruction_count,
        const int16_t * a,
        size_t a_elements,
        const int16_t * b,
        size_t b_elements,
        int16_t * c,
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
        std::fprintf(stderr, "ggml_xdna: invalid i16 matmul arguments\n");
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
            "ggml_xdna: i16 matmul requires an initialized XRT kernel\n");
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
            c_elements * sizeof(int16_t),
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

        int16_t * buf_c = bo_c.map<int16_t *>();
        std::memset(
            buf_c,
            0,
            c_elements * sizeof(int16_t));
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
            c_elements * sizeof(int16_t));

        return true;
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16 matmul execution failed: %s\n",
            e.what());
        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "ggml_xdna: i16 matmul execution failed: unknown error\n");
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

    auto * backend_ctx =
        new ggml_backend_xdna_backend_context;

    backend_ctx->device_context = device_ctx;

    if (!ggml_backend_xdna_init_kernel(backend_ctx, params)) {
        delete backend_ctx;
        return nullptr;
    }

    return new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ dev,
        /* .context = */ backend_ctx,
    };
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void) dev;
    return nullptr;
}

static bool ggml_backend_xdna_device_supports_op(
        ggml_backend_dev_t dev,
        const struct ggml_tensor * op) {
    (void) dev;
    (void) op;
    return false;
}

static bool ggml_backend_xdna_device_supports_buft(
        ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft) {
    (void) dev;
    (void) buft;
    return false;
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
    /* .buffer_from_host_ptr = */ nullptr,
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
        std::strcmp(name, "ggml_backend_xdna_run_i16_matmul") == 0) {
        return (void *) ggml_backend_xdna_run_i16_matmul;
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