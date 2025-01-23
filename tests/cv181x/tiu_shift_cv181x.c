//test for cvkcv181x_tiu_shift
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "../../src/cv181x/cvkcv181x.h"
#include "../../include/cvikernel/cvikernel.h"
#include "../../include/cvikernel/cv181x/cv181x_tpu_cfg.h"  // Include hardware configuration macro definitionsns

// Helper to create a cvk_tl_t with allocated data
cvk_tl_t* create_tensor(int n, int c, int h, int w,
                        int stride_n, int stride_c, int stride_h, int stride_w,
                        cvk_fmt_t fmt) {
    cvk_tl_t *tensor = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    tensor->shape.n = n;
    tensor->shape.c = c;
    tensor->shape.h = h;
    tensor->shape.w = w;
    tensor->stride.n = stride_n;
    tensor->stride.c = stride_c;
    tensor->stride.h = stride_h;
    tensor->stride.w = stride_w;
    tensor->fmt = fmt;

    // Allocate memory for tensor data in int8_t units
    size_t elem_count = (size_t)n * c * h * w;
    tensor->start_address = (uintptr_t)malloc(elem_count * sizeof(int8_t));
    return tensor;
}


// Create the param struct for arithmetic shift

cvk_tiu_arith_shift_param_t
create_arith_shift_param(cvk_tl_t *a_low, cvk_tl_t *a_high,
                         cvk_tl_t *res_low, cvk_tl_t *res_high,
                         cvk_tl_t *bits) {
    cvk_tiu_arith_shift_param_t param;
    memset(&param, 0, sizeof(param));
    param.a_low = a_low;
    param.a_high = a_high;
    param.res_low = res_low;
    param.res_high = res_high;
    param.bits = bits;
    param.layer_id = 1; // Example layer ID
    return param;
}


// Fill the input low/high parts with predictable patterns
static void fill_input_16bit(cvk_tl_t* low, cvk_tl_t* high, int base_low, int base_high) {
    int8_t* low_ptr  = (int8_t*)low->start_address;
    int8_t* high_ptr = (int8_t*)high->start_address;
    int N = low->shape.n;  // same shape as high->shape
    int C = low->shape.c;
    int H = low->shape.h;
    int W = low->shape.w;
    size_t total = (size_t)N * C * H * W;

    // Example pattern: each element's 16-bit value = (base_high << 8) + (base_low + i)
    // We'll store the low byte in 'low_ptr' and the high byte in 'high_ptr'
    for (size_t i = 0; i < total; i++) {
        // Some incremental pattern so each element differs
        int16_t val = (int16_t)(((base_high << 8) & 0xFF00) | ((base_low + i) & 0x00FF));
        // Low byte
        low_ptr[i]  = (int8_t)(val & 0xFF);
        // High byte
        high_ptr[i] = (int8_t)((val >> 8) & 0xFF);
    }
}


// Fill the shift amounts
static void fill_shift_bits(cvk_tl_t* bits) {
    int8_t* ptr = (int8_t*)bits->start_address;
    int N = bits->shape.n;  // e.g. 1
    int C = bits->shape.c;  // e.g. 3
    int H = bits->shape.h;  // e.g. 1
    int W = bits->shape.w;  // e.g. 1
    // total shift values
    size_t total = (size_t)N*C*H*W;

    // Example: if we have 3 channels, store shift amounts: -2, 3, 1
    // Or cycle through some pattern
    int8_t pattern[] = { -2, 3, 1, -1, 4, 0 }; // pick any you like
    size_t pattern_len = sizeof(pattern)/sizeof(pattern[0]);

    for (size_t i = 0; i < total; i++) {
        ptr[i] = pattern[i % pattern_len];
    }
}


// Reference arithmetic shift function
static void ref_arith_shift(const cvk_tl_t* a_low, const cvk_tl_t* a_high,
                            const cvk_tl_t* bits,
                            int8_t* ref_out_low, int8_t* ref_out_high) {
    const int8_t* al = (const int8_t*)a_low->start_address;
    const int8_t* ah = (const int8_t*)a_high->start_address;
    const int8_t* b  = (const int8_t*)bits->start_address;

    int N = a_low->shape.n;
    int C = a_low->shape.c;
    int H = a_low->shape.h;
    int W = a_low->shape.w;

    // We'll assume bits->shape.n=1, shape.c matches C, shape.h=1, shape.w=1
    // so that each channel c gets a single shift value.
    // If your design is different, you can adapt the indexing.
    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            // The shift amount for this channel
            int8_t shift_val = b[c];
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    size_t idx = (size_t)(((n*C + c)*H + h)*W + w);

                    // Combine low/high into signed 16
                    int16_t val = (int16_t)(((uint8_t)ah[idx] << 8) | (uint8_t)al[idx]);

                    // If shift_val > 0 => left shift, < 0 => right shift
                    if (shift_val > 0) {
                        // left shift (val << shift_val)
                        int amt = shift_val;
                        // clamp shift range if needed, or let it overflow
                        // We'll do a naive shift
                        val = (int16_t)(val << amt);
                    } else if (shift_val < 0) {
                        // arithmetic right shift => replicate sign bit
                        int amt = -shift_val;
                        // sign-extend shift
                        uint16_t tmp = (uint16_t)val;
                        // replicate sign bit
                        uint16_t mask = 0x8000;
                        // If negative
                        if (val < 0) {
                            // set top bits for shift
                            for(int i=0; i<amt; i++){
                                tmp |= mask;
                                mask >>= 1;
                            }
                        }
                        // then do logical shift
                        tmp = tmp >> amt;
                        val = (int16_t)tmp;
                    }
                    // Split back to low/high
                    ref_out_low[idx]  = (int8_t)(val & 0xFF);
                    ref_out_high[idx] = (int8_t)((val >> 8) & 0xFF);
                }
            }
        }
    }
}


// Compare the hardware result with the reference

static void compare_arith_shift(const cvk_tl_t* res_low, const cvk_tl_t* res_high,
                                const int8_t* ref_out_low, const int8_t* ref_out_high) {
    const int8_t* rl = (const int8_t*)res_low->start_address;
    const int8_t* rh = (const int8_t*)res_high->start_address;

    int N = res_low->shape.n;
    int C = res_low->shape.c;
    int H = res_low->shape.h;
    int W = res_low->shape.w;
    size_t total = (size_t)N*C*H*W;

    for (size_t i = 0; i < total; i++) {
        if (rl[i] != ref_out_low[i]) {
            printf("Mismatch in SHIFT Low at index=%zu: expected=%d, got=%d\n",
                   i, ref_out_low[i], rl[i]);
            assert(0 && "Arithmetic SHIFT Low mismatch!");
        }
        if (rh[i] != ref_out_high[i]) {
            printf("Mismatch in SHIFT High at index=%zu: expected=%d, got=%d\n",
                   i, ref_out_high[i], rh[i]);
            assert(0 && "Arithmetic SHIFT High mismatch!");
        }
    }
    printf("[compare_arith_shift] All results match the reference.\n");
}


// Main test function

int main() {
    // Create mock tensors for a_low, a_high, res_low, res_high, and bits
    cvk_tl_t *a_low   = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *a_high  = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *res_low = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *res_high= create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    // Suppose bits->shape = {1,3,1,1}, meaning one shift value per channel
    cvk_tl_t *bits    = create_tensor(1, 3, 1, 1, 1, 1, 0, 0, CVK_FMT_I8);

    cvk_tiu_arith_shift_param_t shift_param =
        create_arith_shift_param(a_low, a_high, res_low, res_high, bits);

    cvk_context_t *ctx = (cvk_context_t *)malloc(sizeof(cvk_context_t));
    memset(ctx, 0, sizeof(*ctx));

    fill_input_16bit(a_low, a_high, /*base_low=*/0x30, /*base_high=*/0x01);

    fill_shift_bits(bits);

    printf("Running arithmetic shift operation...\n");
    cvkcv181x_tiu_arith_shift(ctx, &shift_param);
    printf("Arithmetic Shift operation executed.\n");

    int N = a_low->shape.n, C = a_low->shape.c, H = a_low->shape.h, W = a_low->shape.w;
    size_t total = (size_t)N*C*H*W;

    int8_t* ref_out_low  = (int8_t*)malloc(total * sizeof(int8_t));
    int8_t* ref_out_high = (int8_t*)malloc(total * sizeof(int8_t));
    memset(ref_out_low,  0, total);
    memset(ref_out_high, 0, total);

    ref_arith_shift(a_low, a_high, bits, ref_out_low, ref_out_high);
    compare_arith_shift(res_low, res_high, ref_out_low, ref_out_high);

    // Clean up
    free(ref_out_low);
    free(ref_out_high);

    free((void*)a_low->start_address);
    free((void*)a_high->start_address);
    free((void*)res_low->start_address);
    free((void*)res_high->start_address);
    free((void*)bits->start_address);
    free(a_low);
    free(a_high);
    free(res_low);
    free(res_high);
    free(bits);
    free(ctx);

    printf("Arithmetic shift test verified successfully!\n");
    return 0;
}
