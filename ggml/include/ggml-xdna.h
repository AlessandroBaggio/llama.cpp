#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_XDNA_NAME "XDNA"

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

#ifdef __cplusplus
}
#endif
