#include "ggml-backend.h"

#include <cstdio>
#include <exception>

int main(int argc, char ** argv) {
    if (argc != 1) {
        std::fprintf(stderr, "XDNA_PREFLIGHT_HARNESS_RESULT=HARNESS_SETUP_ERROR reason=unexpected_arguments\n");
        return 10;
    }

    try {
        ggml_backend_load_all();

        ggml_backend_dev_t device = ggml_backend_dev_by_name("XDNA0");
        if (device == nullptr) {
            const bool any_backend = ggml_backend_reg_count() != 0;
            std::fprintf(stderr, "XDNA_PREFLIGHT_HARNESS_RESULT=%s reason=xdna0_not_registered\n",
                any_backend ? "DEVICE_NOT_FOUND" : "HARNESS_SETUP_ERROR");
            return any_backend ? 11 : 12;
        }

        // nullptr deliberately exercises the backend's environment-parameter fallback.
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (backend != nullptr) {
            std::fprintf(stderr, "XDNA_PREFLIGHT_HARNESS_RESULT=UNEXPECTED_BACKEND_INIT_SUCCESS\n");
            return 13;
        }

        // PASS versus expected rejection is determined by the supervisor from same-PID DLL markers.
        std::fprintf(stderr,
            "XDNA_PREFLIGHT_HARNESS_RESULT=BACKEND_INIT_RETURNED_NULL classification=DEFER_TO_SUPERVISOR\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "XDNA_PREFLIGHT_HARNESS_RESULT=HARNESS_SETUP_ERROR reason=exception detail=%s\n", e.what());
        return 14;
    } catch (...) {
        std::fprintf(stderr, "XDNA_PREFLIGHT_HARNESS_RESULT=HARNESS_SETUP_ERROR reason=unknown_exception\n");
        return 15;
    }
}
