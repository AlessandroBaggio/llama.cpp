#include "ggml-xdna.h"

#include "ggml-backend-impl.h"

#include <xrt/xrt_device.h>

#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

struct ggml_backend_xdna_context {
    std::unique_ptr<xrt::device> xrt_device;
    std::string description = "AMD XDNA NPU (XRT device 0)";
    std::string device_id;
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

static ggml_backend_t ggml_backend_xdna_device_init(
        ggml_backend_dev_t dev,
        const char * params) {
    (void) dev;
    (void) params;

    // XRT device detection only. A real GGML backend object is introduced
    // in a later milestone.
    return nullptr;
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
    (void) name;
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