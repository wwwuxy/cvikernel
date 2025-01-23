//test for cvkcv181x_tiu_matrix_multiplication
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

// Helper macro for checking errors
#define CHECK_ERROR(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("ERROR: %s\n", msg); \
            assert(0); \
        } \
    } while(0)

// Placeholder functions to write/read data to/from memory at a given address.
// In a real environment, you should replace these with the proper memory access APIs.
static void write_memory(uint64_t address, const void* data, size_t size) {
    (void)address;  // Avoid unused variable warnings
    (void)data;
    (void)size;
}

static void read_memory(uint64_t address, void* buffer, size_t size) {
    (void)address;
    (void)buffer;
    (void)size;
}

// A simple reference function to compute (Left x Right + Bias) in software
static void compute_reference_result(const int8_t* left_data, int left_rows, int left_cols,
                                     const int8_t* right_data, int right_rows, int right_cols,
                                     const int8_t* bias_data,
                                     int8_t* result_data) {
    // left: shape = left_rows x left_cols
    // right: shape = right_rows x right_cols
    // bias: shape = 1 x right_cols
    // result: shape = left_rows x right_cols

    // For each element result[i, j]:
    //   result[i, j] = sum_{k=0..(left_cols-1)} left[i, k] * right[k, j] + bias[j]
    for (int i = 0; i < left_rows; ++i) {
        for (int j = 0; j < right_cols; ++j) {
            int sum = 0;
            for (int k = 0; k < left_cols; ++k) {
                sum += left_data[i * left_cols + k] * right_data[k * right_cols + j];
            }
            sum += bias_data[j];

            result_data[i * right_cols + j] = (int8_t)sum;
        }
    }
}

void test_cvkcv181x_tiu_matrix_multiplication() {
    cvk_context_t ctx;

    // Left matrix: shape = 2 x 3  (n=2, c=3)
    cvk_tl_t left;
    left.start_address = 0x1000;
    left.fmt = CVK_FMT_I8;
    left.shape.n = 2;  // # of rows (logical)
    left.shape.c = 3;  // # of columns (logical)
    left.shape.h = 1;
    left.shape.w = 3;  // Typically the same as 'c' in a flattened 2D
    left.stride.n = 3; // Because each row has 3 elements
    left.stride.c = 1;
    left.stride.h = 1;
    left.stride.w = 1;

    // Right matrix: shape = 3 x 4 (n=3, c=4)
    cvk_tl_t right;
    right.start_address = 0x2000;
    right.fmt = CVK_FMT_I8;
    right.shape.n = 3;  // # of rows
    right.shape.c = 4;  // # of columns
    right.shape.h = 1;
    right.shape.w = 4;
    right.stride.n = 4;
    right.stride.c = 1;
    right.stride.h = 1;
    right.stride.w = 1;

    // Result matrix: shape = 2 x 4 (n=2, c=4)
    cvk_tl_t res;
    res.start_address = 0x3000;
    res.fmt = CVK_FMT_I8;
    res.shape.n = 2; 
    res.shape.c = 4;
    res.shape.h = 1;
    res.shape.w = 4;
    res.stride.n = 4;
    res.stride.c = 1;
    res.stride.h = 1;
    res.stride.w = 1;

    // Bias matrix: shape = 1 x 4 (n=1, c=4)
    cvk_tl_t bias;
    bias.start_address = 0x4000;
    bias.fmt = CVK_FMT_I8;
    bias.shape.n = 1;
    bias.shape.c = 4;
    bias.shape.h = 1;
    bias.shape.w = 4;
    bias.stride.n = 4;
    bias.stride.c = 1;
    bias.stride.h = 1;
    bias.stride.w = 1;

    // Left: 2x3
    int8_t left_data[2 * 3] = {
        1, 2, 3,
        4, 5, 6
    };

    // Right: 3x4
    int8_t right_data[3 * 4] = {
        7,  8,  9,  10,
        11, 12, 13, 14,
        15, 16, 17, 18
    };

    // Bias: 1x4
    int8_t bias_data[4] = {1, 2, 3, 4};

    // We'll compute the reference result on the CPU side for comparison.
    int8_t expected_res[2 * 4];
    compute_reference_result(left_data, 2, 3,
                             right_data, 3, 4,
                             bias_data,
                             expected_res);

    write_memory(left.start_address, left_data, sizeof(left_data));
    write_memory(right.start_address, right_data, sizeof(right_data));
    write_memory(bias.start_address, bias_data, sizeof(bias_data));

    cvk_tiu_matrix_multiplication_param_t param;
    memset(&param, 0, sizeof(param));
    param.res = &res;
    param.left = &left;
    param.right = &right;
    param.bias = &bias;

    // No shift, no ReLU, no accumulation, INT8 result
    param.lshift_bits = 0;
    param.rshift_bits = 0;
    param.relu_enable = 0;
    param.add_result = 0;
    param.ps32_mode = 0;
    param.res_is_int8 = 1;
    param.layer_id = 1;

    cvkcv181x_tiu_matrix_multiplication(&ctx, &param);

    int8_t actual_res[2 * 4];
    memset(actual_res, 0, sizeof(actual_res));
    read_memory(res.start_address, actual_res, sizeof(actual_res));

    for (int i = 0; i < (2 * 4); ++i) {
        if (actual_res[i] != expected_res[i]) {
            printf("ERROR: Mismatch at index %d. Expected %d but got %d\n",
                   i, expected_res[i], actual_res[i]);
            assert(0);
        }
    }

    printf("Matrix multiplication test passed successfully!\n");
}

int main() {
    test_cvkcv181x_tiu_matrix_multiplication();
    return 0;
}
