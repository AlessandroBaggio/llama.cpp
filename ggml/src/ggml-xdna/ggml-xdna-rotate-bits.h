/*
 * Private rotate helper used by ggml-xdna-sha256.c.
 */

#ifndef GGML_XDNA_ROTATE_BITS_H
#define GGML_XDNA_ROTATE_BITS_H

#include <stdint.h>

#define ROTR32(value, shift) \
    ((uint32_t) ( \
        ((uint32_t) (value) >> (shift)) | \
        ((uint32_t) (value) << (32U - (shift))) \
    ))

#endif