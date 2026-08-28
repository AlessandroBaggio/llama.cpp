/*
 * Private SHA-256 helper for the GGML XDNA backend.
 *
 * The implementation is adapted from llama.cpp/vendor/hash/sha256,
 * originally by Igor Pavlov and released to the public domain.
 */

#ifndef GGML_XDNA_SHA256_H
#define GGML_XDNA_SHA256_H

#include <stddef.h>
#include <stdint.h>

#define GGML_XDNA_SHA256_DIGEST_SIZE 32

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_xdna_sha256_t {
    uint32_t state[8];
    uint64_t count;
    unsigned char buffer[64];
} ggml_xdna_sha256_t;

void ggml_xdna_sha256_init(
    ggml_xdna_sha256_t * p);

void ggml_xdna_sha256_update(
    ggml_xdna_sha256_t * p,
    const unsigned char * data,
    size_t size);

void ggml_xdna_sha256_final(
    ggml_xdna_sha256_t * p,
    unsigned char * digest);

void ggml_xdna_sha256_hash(
    unsigned char * digest,
    const unsigned char * data,
    size_t size);

#ifdef __cplusplus
}
#endif

#endif