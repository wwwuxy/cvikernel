// test for cvkcv181x_tiu_xor
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
#include "../../include/cvikernel/cv181x/cv181x_tpu_cfg.h"  // Include hardware configuration macro definitions

// Create a helper function for building cvk_tl_t with allocated data
cvk_tl_t* create_tensor(int n, int c, int h, int w,
                        int stride_n, int stride_c, int stride_h, int stride_w,
                        cvk_fmt_t fmt) {
    cvk_tl_t *tensor = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    if (!tensor) {
        fprintf(stderr, "Failed to allocate memory for tensor.\n");
        exit(EXIT_FAILURE);
    }

    tensor->shape.n = n;
    tensor->shape.c = c;
    tensor->shape.h = h;
    tensor->shape.w = w;
    tensor->stride.n = stride_n;
    tensor->stride.c = stride_c;
    tensor->stride.h = stride_h;
    tensor->stride.w = stride_w;
    tensor->fmt = fmt;

    // Allocate memory for the data (int8_t for simplicity)
    size_t count = (size_t)n * c * h * w;
    tensor->start_address = (uint32_t)(uintptr_t)malloc(count * sizeof(int8_t));
    if (!tensor->start_address) {
        fprintf(stderr, "Failed to allocate memory for tensor data.\n");
        free(tensor);
        exit(EXIT_FAILURE);
    }
    return tensor;
}

// Create parameter structs for XOR operations

cvk_tiu_xor_int8_param_t create_xor_int8_param(cvk_tl_t *a, cvk_tl_t *b, cvk_tl_t *res) {
    cvk_tiu_xor_int8_param_t param;
    memset(&param, 0, sizeof(param));
    param.a = a;
    param.b = b;
    param.res = res;
    param.layer_id = 1; // Example layer ID
    return param;
}

cvk_tiu_xor_int16_param_t create_xor_int16_param(cvk_tl_t *a_low, cvk_tl_t *a_high,
                                                 cvk_tl_t *b_low, cvk_tl_t *b_high,
                                                 cvk_tl_t *res_low, cvk_tl_t *res_high) {
    cvk_tiu_xor_int16_param_t param;
    memset(&param, 0, sizeof(param));
    param.a_low = a_low;
    param.a_high = a_high;
    param.b_low = b_low;
    param.b_high = b_high;
    param.res_low = res_low;
    param.res_high = res_high;
    return param;
}

// Fill the int8 data with a simple pattern for debugging and verification
static void fill_data_int8(cvk_tl_t* tensor, int8_t base_value) {
    int8_t* ptr = (int8_t*)(uintptr_t)(tensor->start_address);
    int N = tensor->shape.n;
    int C = tensor->shape.c;
    int H = tensor->shape.h;
    int W = tensor->shape.w;

    // Fill each element with (base_value + linear_index) & 0xFF to have a distinct pattern per tensor
    size_t total = (size_t)N * C * H * W;
    for (size_t i = 0; i < total; i++) {
        ptr[i] = (int8_t)((base_value + i) & 0xFF);
    }
}

// Software reference for XOR int8
static void ref_xor_int8(const cvk_tl_t* a,
                         const cvk_tl_t* b,
                         int8_t* ref_output) {
    const int8_t* a_data = (int8_t*)(uintptr_t)(a->start_address);
    const int8_t* b_data = (int8_t*)(uintptr_t)(b->start_address);

    int N = a->shape.n;
    int C = a->shape.c;
    int H = a->shape.h;
    int W = a->shape.w;

    size_t total = (size_t)N * C * H * W;
    for (size_t i = 0; i < total; i++) {
        // XOR at the byte level
        // Treat them as unsigned for XOR:
        uint8_t aa = (uint8_t)a_data[i];
        uint8_t bb = (uint8_t)b_data[i];
        ref_output[i] = (int8_t)(aa ^ bb);
    }
}

// Compare the int8 XOR hardware output with reference
static void compare_xor_int8(const cvk_tl_t* res,
                             const int8_t* ref_output) {
    const int8_t* hw_data = (int8_t*)(uintptr_t)(res->start_address);
    int N = res->shape.n;
    int C = res->shape.c;
    int H = res->shape.h;
    int W = res->shape.w;
    size_t total = (size_t)N * C * H * W;

    for (size_t i = 0; i < total; i++) {
        if (hw_data[i] != ref_output[i]) {
            printf("Mismatch in int8 XOR at index %zu: expected %d, got %d\n",
                   i, ref_output[i], hw_data[i]);
            assert(0 && "int8 XOR mismatch!");
        }
    }
    printf("[compare_xor_int8] All int8 XOR results match.\n");
}

// Software reference for XOR int16
static void ref_xor_int16(const cvk_tl_t* a_low, const cvk_tl_t* a_high,
                          const cvk_tl_t* b_low, const cvk_tl_t* b_high,
                          int8_t* ref_out_low, int8_t* ref_out_high) {
    const int8_t* al = (int8_t*)(uintptr_t)(a_low->start_address);
    const int8_t* ah = (int8_t*)(uintptr_t)(a_high->start_address);
    const int8_t* bl = (int8_t*)(uintptr_t)(b_low->start_address);
    const int8_t* bh = (int8_t*)(uintptr_t)(b_high->start_address);

    int N = a_low->shape.n;  // same shape as a_high
    int C = a_low->shape.c;
    int H = a_low->shape.h;
    int W = a_low->shape.w;

    size_t total = (size_t)N * C * H * W;

    for (size_t i = 0; i < total; i++) {
        // Reconstruct 16-bit "a" and 16-bit "b" from splitted int8
        // a16 = (a_high << 8) | (a_low & 0xFF)
        uint16_t A = ((uint16_t)((uint8_t)ah[i]) << 8) | (uint8_t)al[i];
        uint16_t B = ((uint16_t)((uint8_t)bh[i]) << 8) | (uint8_t)bl[i];

        // XOR 16-bit
        uint16_t R = (uint16_t)(A ^ B);

        // Split result
        ref_out_low[i]  = (int8_t)(R & 0xFF);
        ref_out_high[i] = (int8_t)((R >> 8) & 0xFF);
    }
}

// Compare the int16 XOR hardware output with reference
static void compare_xor_int16(const cvk_tl_t* res_low, const cvk_tl_t* res_high,
                              const int8_t* ref_out_low, const int8_t* ref_out_high) {
    const int8_t* hl = (int8_t*)(uintptr_t)(res_low->start_address);
    const int8_t* hh = (int8_t*)(uintptr_t)(res_high->start_address);

    int N = res_low->shape.n;
    int C = res_low->shape.c;
    int H = res_low->shape.h;
    int W = res_low->shape.w;
    size_t total = (size_t)N * C * H * W;

    for (size_t i = 0; i < total; i++) {
        if (hl[i] != ref_out_low[i]) {
            printf("Mismatch in int16 XOR Low at index=%zu: expected=%d, got=%d\n",
                   i, ref_out_low[i], hl[i]);
            assert(0 && "int16 XOR Low mismatch!");
        }
        if (hh[i] != ref_out_high[i]) {
            printf("Mismatch in int16 XOR High at index=%zu: expected=%d, got=%d\n",
                   i, ref_out_high[i], hh[i]);
            assert(0 && "int16 XOR High mismatch!");
        }
    }
    printf("[compare_xor_int16] All int16 XOR results match.\n");
}

// Main test function

int main() {
    cvk_tl_t *a_int8   = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *b_int8   = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *res_int8 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);

    cvk_tl_t *a_low_int16    = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *a_high_int16   = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *b_low_int16    = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *b_high_int16   = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *res_low_int16  = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t *res_high_int16 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);

    cvk_tiu_xor_int8_param_t xor_int8_param = create_xor_int8_param(a_int8, b_int8, res_int8);
    cvk_tiu_xor_int16_param_t xor_int16_param = create_xor_int16_param(
        a_low_int16, a_high_int16, b_low_int16, b_high_int16,
        res_low_int16, res_high_int16);

    cvk_context_t *ctx = (cvk_context_t *)malloc(sizeof(cvk_context_t));
    if (!ctx) {
        fprintf(stderr, "Failed to allocate memory for context.\n");
        exit(EXIT_FAILURE);
    }
    memset(ctx, 0, sizeof(cvk_context_t));
    // Fill ctx with any needed info for real hardware usage

    fill_data_int8(a_int8,  0x10);
    fill_data_int8(b_int8,  0x20);
    fill_data_int8(a_low_int16,   0x01);
    // 将 fill_data_int16 替换为 fill_data_int8，因为 fill_data_int16 未定义
    fill_data_int8(a_high_int16,  0x02);
    fill_data_int8(b_low_int16,   0x03);
    fill_data_int8(b_high_int16,  0x04);
    // res_int8, res_low_int16, res_high_int16 将被覆盖，无需填充

    printf("Running int8 XOR operation...\n");
    cvkcv181x_tiu_xor_int8(ctx, &xor_int8_param);
    printf("Int8 XOR operation executed.\n");

    printf("Running int16 XOR operation...\n");
    cvkcv181x_tiu_xor_int16(ctx, &xor_int16_param);
    printf("Int16 XOR operation executed.\n");

    int N = a_int8->shape.n, C = a_int8->shape.c, H = a_int8->shape.h, W = a_int8->shape.w;
    size_t total_int8 = (size_t)N * C * H * W;
    int8_t* ref_int8 = (int8_t*)malloc(total_int8 * sizeof(int8_t));
    if (!ref_int8) {
        fprintf(stderr, "Failed to allocate memory for reference int8.\n");
        exit(EXIT_FAILURE);
    }
    // Compute software XOR
    ref_xor_int8(a_int8, b_int8, ref_int8);
    // Compare hardware vs. reference
    compare_xor_int8(res_int8, ref_int8);
    free(ref_int8);

    // For int16:
    size_t total_int16 = (size_t)a_low_int16->shape.n * a_low_int16->shape.c
                       * a_low_int16->shape.h * a_low_int16->shape.w;
    int8_t* ref_int16_low  = (int8_t*)malloc(total_int16 * sizeof(int8_t));
    int8_t* ref_int16_high = (int8_t*)malloc(total_int16 * sizeof(int8_t));
    if (!ref_int16_low || !ref_int16_high) {
        fprintf(stderr, "Failed to allocate memory for reference int16.\n");
        exit(EXIT_FAILURE);
    }
    // Compute software XOR
    ref_xor_int16(a_low_int16, a_high_int16,
                  b_low_int16, b_high_int16,
                  ref_int16_low, ref_int16_high);
    // Compare hardware vs. reference
    compare_xor_int16(res_low_int16, res_high_int16, ref_int16_low, ref_int16_high);
    free(ref_int16_low);
    free(ref_int16_high);

    // Clean up
    // 修正函数调用，传递 ctx 和 tensor 指针，而不是 &ctx 和 start_address
    cvkcv181x_lmem_free_tensor(ctx, a_int8);
    cvkcv181x_lmem_free_tensor(ctx, b_int8);
    cvkcv181x_lmem_free_tensor(ctx, res_int8);
    cvkcv181x_lmem_free_tensor(ctx, a_low_int16);
    cvkcv181x_lmem_free_tensor(ctx, a_high_int16);
    cvkcv181x_lmem_free_tensor(ctx, b_low_int16);
    cvkcv181x_lmem_free_tensor(ctx, b_high_int16);
    cvkcv181x_lmem_free_tensor(ctx, res_low_int16);
    cvkcv181x_lmem_free_tensor(ctx, res_high_int16);

    // 释放 tensor 结构体和上下文
    free(a_int8);
    free(b_int8);
    free(res_int8);
    free(a_low_int16);
    free(a_high_int16);
    free(b_low_int16);
    free(b_high_int16);
    free(res_low_int16);
    free(res_high_int16);
    free(ctx);

    printf("XOR test (int8 + int16) verified successfully!\n");
    return 0;
}
