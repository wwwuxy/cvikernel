#include <stdio.h>
#include <stdint.h>
#include "cvkcv181x.h"

// Mock initialization functions (replace with actual initialization if available)
cvk_tensor_t* create_tensor(int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, int stride_w, cvk_fmt_t fmt) {
    cvk_tensor_t *tensor = (cvk_tensor_t *)malloc(sizeof(cvk_tensor_t));
    tensor->shape.n = n;
    tensor->shape.c = c;
    tensor->shape.h = h;
    tensor->shape.w = w;
    tensor->stride.n = stride_n;
    tensor->stride.c = stride_c;
    tensor->stride.h = stride_h;
    tensor->stride.w = stride_w;
    tensor->fmt = fmt;
    tensor->start_address = (uintptr_t)malloc(n * c * h * w * sizeof(int8_t)); // Allocate memory for tensor data
    return tensor;
}

cvk_tiu_xor_int8_param_t create_xor_int8_param(cvk_tensor_t *a, cvk_tensor_t *b, cvk_tensor_t *res) {
    cvk_tiu_xor_int8_param_t param;
    param.a = a;
    param.b = b;
    param.res = res;
    param.layer_id = 1; // Example layer ID
    return param;
}

cvk_tiu_xor_int16_param_t create_xor_int16_param(cvk_tensor_t *a_low, cvk_tensor_t *a_high, 
                                                  cvk_tensor_t *b_low, cvk_tensor_t *b_high, 
                                                  cvk_tensor_t *res_low, cvk_tensor_t *res_high) {
    cvk_tiu_xor_int16_param_t param;
    param.a_low = a_low;
    param.a_high = a_high;
    param.b_low = b_low;
    param.b_high = b_high;
    param.res_low = res_low;
    param.res_high = res_high;
    param.layer_id = 1; // Example layer ID
    return param;
}

int main() {
    // Set up mock tensors for int8 XOR operation
    cvk_tensor_t *a_int8 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);
    cvk_tensor_t *b_int8 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);
    cvk_tensor_t *res_int8 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);

    // Set up mock tensors for int16 XOR operation
    cvk_tensor_t *a_low_int16 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);
    cvk_tensor_t *a_high_int16 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);
    cvk_tensor_t *b_low_int16 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);
    cvk_tensor_t *b_high_int16 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);
    cvk_tensor_t *res_low_int16 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);
    cvk_tensor_t *res_high_int16 = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_INT8);

    // Create XOR int8 and int16 parameters
    cvk_tiu_xor_int8_param_t xor_int8_param = create_xor_int8_param(a_int8, b_int8, res_int8);
    cvk_tiu_xor_int16_param_t xor_int16_param = create_xor_int16_param(a_low_int16, a_high_int16, 
                                                                        b_low_int16, b_high_int16, 
                                                                        res_low_int16, res_high_int16);

    // Create a cvk_context_t for the operations
    cvk_context_t *ctx = (cvk_context_t *)malloc(sizeof(cvk_context_t));

    // Run the int8 XOR operation
    printf("Running int8 XOR operation...\n");
    cvkcv181x_tiu_xor_int8(ctx, &xor_int8_param);
    printf("Int8 XOR operation executed.\n");

    // Run the int16 XOR operation
    printf("Running int16 XOR operation...\n");
    cvkcv181x_tiu_xor_int16(ctx, &xor_int16_param);
    printf("Int16 XOR operation executed.\n");

    // Clean up (free memory allocated for tensors)
    free((void*)a_int8->start_address);
    free((void*)b_int8->start_address);
    free((void*)res_int8->start_address);
    free((void*)a_low_int16->start_address);
    free((void*)a_high_int16->start_address);
    free((void*)b_low_int16->start_address);
    free((void*)b_high_int16->start_address);
    free((void*)res_low_int16->start_address);
    free((void*)res_high_int16->start_address);
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

    return 0;
}

