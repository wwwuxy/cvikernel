#include "cvkcv181x.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

// Mock function to simulate check_tiu_tensor
int8_t check_tiu_tensor(cvk_tl_t *t) {
    return 0;  // Simulate always passing
}

// Mock function to simulate check_stride_type_0
int8_t check_stride_type_0(cvk_context_t *ctx, cvk_tl_t *t) {
    return 0;  // Simulate always passing
}

// Mock function to simulate emit_tiu_cmdbuf
void* emit_tiu_cmdbuf(cvk_context_t *ctx, tiu_reg_t *reg) {
    printf("Emitting command buffer with layer_info: %d\n", reg->layer_info);
    return NULL;  // Simulate successful emission
}

// Mock function to simulate matrix_is_signed
int matrix_is_signed(cvk_ml_t *m) {
    return 1;  // Simulate signed matrix
}

// Mock function to simulate check_matrix (just returns 0 for simplicity)
int8_t check_matrix(cvk_context_t *ctx, const cvk_ml_t *m) {
    return 0;  // Simulate always passing
}

// Mocked cvk_tensor_t and cvk_ml_t for testing
cvk_ml_t* create_ml(cvk_fmt_t fmt, int n, int c, int h, int w, int start_address) {
    cvk_ml_t *ml = (cvk_ml_t*)malloc(sizeof(cvk_ml_t));
    ml->fmt = fmt;
    ml->shape.n = n;
    ml->shape.c = c;
    ml->shape.h = h;
    ml->shape.w = w;
    ml->start_address = start_address;
    ml->stride.n = 1;
    ml->stride.c = 1;
    ml->stride.h = 1;
    ml->stride.w = 1;
    return ml;
}

// Test function for cvkcv181x_tiu_matrix_multiplication_qm
void test_cvkcv181x_tiu_matrix_multiplication_qm() {
    cvk_context_t ctx;
    ctx.info.eu_num = 4;  // Example number of execution units
    ctx.info.npu_num = 2; // Example number of NPUs

    // Create tensors (matrices) for testing
    cvk_ml_t *left = create_ml(CVK_FMT_I8, 1, 8, 1, 4, 0x1000); // Left matrix (1, 8, 1, 4)
    cvk_ml_t *right = create_ml(CVK_FMT_I8, 1, 8, 1, 4, 0x2000); // Right matrix (1, 8, 1, 4)
    cvk_ml_t *res = create_ml(CVK_FMT_I8, 1, 8, 1, 4, 0x3000);  // Result matrix (1, 8, 1, 4)
    cvk_ml_t *bias = create_ml(CVK_FMT_I8, 1, 8, 1, 4, 0x4000);  // Bias matrix (1, 8, 1, 4)

    // Set up the parameters for matrix multiplication
    cvk_tiu_matrix_multiplication_qm_param_t param;
    param.res = res;
    param.left = left;
    param.right = right;
    param.bias = bias;
    param.lshift_bits = 0;
    param.rshift_bits = 0;
    param.relu_enable = 0;
    param.add_result = 0;
    param.ps32_mode = 0;
    param.layer_id = 1;
    param.quan_m = 0;

    // Call the matrix multiplication function
    cvkcv181x_tiu_matrix_multiplication_qm(&ctx, &param);
}

// Main function to run the test
int main() {
    test_cvkcv181x_tiu_matrix_multiplication_qm();
    printf("Matrix multiplication test passed.\n");
    return 0;
}

