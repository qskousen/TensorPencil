// Golden-fixture generator: dequantize one block of each ggml quant format
// with the reference ggml-quants.c and print bytes + f32 bit patterns.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ggml-quants.h"
#include "ggml-impl.h"

// Link stubs for symbols pulled in by quantize_* paths we never call.
#include <stdlib.h>
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void)type; (void)ne; abort(); }
size_t ggml_type_size(enum ggml_type type) { (void)type; abort(); }
void ggml_abort(const char *file, int line, const char *fmt, ...) { (void)file; (void)line; (void)fmt; abort(); }
int64_t ggml_blck_size(enum ggml_type type) { (void)type; abort(); }
const char *ggml_type_name(enum ggml_type type) { (void)type; abort(); }

// Deterministic byte stream (LCG).
static uint32_t lcg_state = 0x12345678;
static uint8_t next_byte(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (uint8_t)(lcg_state >> 16);
}

static void fill(uint8_t *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = next_byte();
}

static void dump(const char *name, const uint8_t *block, int nbytes, const float *y, int n) {
    printf("=== %s block_bytes(%d) ===\n", name, nbytes);
    for (int i = 0; i < nbytes; i++) printf("0x%02x,%s", block[i], (i % 16 == 15) ? "\n" : " ");
    if (nbytes % 16) printf("\n");
    printf("--- expected f32 bits (%d) ---\n", n);
    for (int i = 0; i < n; i++) {
        uint32_t b; memcpy(&b, &y[i], 4);
        printf("0x%08x,%s", b, (i % 8 == 7) ? "\n" : " ");
    }
    if (n % 8) printf("\n");
}

int main(void) {
    float y[256];
    // Sane f16 scales: d = 0.05f, dmin = 0.02f
    ggml_fp16_t d16 = GGML_FP32_TO_FP16(0.05f);
    ggml_fp16_t m16 = GGML_FP32_TO_FP16(0.02f);

    { // Q8_0: [f16 d][32 x i8]
        block_q8_0 b;
        fill((uint8_t *)&b, sizeof(b));
        memcpy(&b.d, &d16, 2);
        dequantize_row_q8_0(&b, y, 32);
        dump("q8_0", (const uint8_t *)&b, sizeof(b), y, 32);
    }
    { // Q4_K: [f16 d][f16 dmin][12 scales][128 qs]
        block_q4_K b;
        fill((uint8_t *)&b, sizeof(b));
        memcpy(&b.d, &d16, 2);
        memcpy(&b.dmin, &m16, 2);
        dequantize_row_q4_K(&b, y, 256);
        dump("q4_k", (const uint8_t *)&b, sizeof(b), y, 256);
    }
    { // Q5_K: [f16 d][f16 dmin][12 scales][32 qh][128 qs]
        block_q5_K b;
        fill((uint8_t *)&b, sizeof(b));
        memcpy(&b.d, &d16, 2);
        memcpy(&b.dmin, &m16, 2);
        dequantize_row_q5_K(&b, y, 256);
        dump("q5_k", (const uint8_t *)&b, sizeof(b), y, 256);
    }
    { // Q6_K: [128 ql][64 qh][16 i8 scales][f16 d]
        block_q6_K b;
        fill((uint8_t *)&b, sizeof(b));
        memcpy(&b.d, &d16, 2);
        dequantize_row_q6_K(&b, y, 256);
        dump("q6_k", (const uint8_t *)&b, sizeof(b), y, 256);
    }
    return 0;
}
