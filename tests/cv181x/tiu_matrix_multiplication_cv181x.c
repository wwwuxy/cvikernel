#include "cvkcv181x.h"
#include <stdio.h>
#include <assert.h>

#define CHECK_ERROR(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("ERROR: %s\n", msg); \
            assert(0); \
        } \
    } while(0)

void test_cvkcv181x_tiu_matrix_multiplication() {
    // Step 1: Initialize the context (for the purposes of this test, we are assuming it's already set up)
    cvk_context_t ctx;
    // Assume ctx is initialized appropriately elsewhere in your environment
    
    // Step 2: Define the matrix shapes and parameters
    cvk_ml_t left, right, res, bias;
    
    // Initialize left matrix (shape: 2x3)
    left.start_address = 0x1000;  // example address
    left.fmt = CVK_FMT_I8;         // format (example: int8)
    left.shape.n = 2;              // rows
    left.shape.c = 3;              // channels
    left.shape.h = 1;              // height
    left.shape.w = 3;              // width
    left.stride.n = 3;             // stride
    left.stride.c = 1;
    left.stride.h = 1;
    left.stride.w = 1;
    
    // Initialize right matrix (shape: 3x4)
    right.start_address = 0x2000;  // example address
    right.fmt = CVK_FMT_I8;         // format (example: int8)
    right.shape.n = 3;              // rows
    right.shape.c = 4;              // channels
    right.shape.h = 1;              // height
    right.shape.w = 4;              // width
    right.stride.n = 4;             // stride
    right.stride.c = 1;
    right.stride.h = 1;
    right.stride.w = 1;
    
    // Initialize result matrix (shape: 2x4)
    res.start_address = 0x3000;    // example address
    res.fmt = CVK_FMT_I8;          // format (example: int8)
    res.shape.n = 2;               // rows
    res.shape.c = 4;               // channels
    res.shape.h = 1;               // height
    res.shape.w = 4;               // width
    res.stride.n = 4;              // stride
    res.stride.c = 1;
    res.stride.h = 1;
    res.stride.w = 1;
    
    // Initialize bias matrix (shape: 1x4)
    bias.start_address = 0x4000;   // example address
    bias.fmt = CVK_FMT_I8;          // format (example: int8)
    bias.shape.n = 1;               // rows
    bias.shape.c = 4;               // channels
    bias.shape.h = 1;               // height
    bias.shape.w = 4;               // width
    bias.stride.n = 4;              // stride
    bias.stride.c = 1;
    bias.stride.h = 1;
    bias.stride.w = 1;
    
    // Step 3: Define multiplication parameters
    cvk_tiu_matrix_multiplication_param_t param;
    param.res = &res;
    param.left = &left;
    param.right = &right;
    param.bias = &bias;
    param.lshift_bits = 0;  // No left shift
    param.rshift_bits = 0;  // No right shift
    param.relu_enable = 0;  // No ReLU activation
    param.add_result = 0;   // Do not add result
    param.ps32_mode = 0;    // Use regular mode
    param.res_is_int8 = 1;  // Result is in int8 format
    param.layer_id = 1;     // Layer ID for identification
    
    // Step 4: Call the matrix multiplication function
    cvkcv181x_tiu_matrix_multiplication(&ctx, &param);
    
    // Step 5: Verify the result (check if matrix multiplication was done correctly)
    // For this test, we will just verify that no assertion errors occurred.
    // In a real scenario, you might want to compare the content of the 'res' matrix.
    printf("Matrix multiplication completed successfully!\n");
}

int main() {
    // Run the test
    test_cvkcv181x_tiu_matrix_multiplication();
    return 0;
}

