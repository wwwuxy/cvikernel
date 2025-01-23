//test for cvkcv181x_tiu_matrix_multiplication_qm
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

static void compute_reference_result_add_relu(
    const int8_t* left_data,   // shape = (left_rows x left_cols)
    int left_rows, int left_cols,
    const int8_t* right_data,  // shape = (right_rows x right_cols)
    int right_rows, int right_cols,
    const int8_t* bias_data,   // shape = (1 x right_cols)
    const int8_t* old_res_data,// shape = (left_rows x right_cols)
    int8_t* result_data,       // shape = (left_rows x right_cols)
    int add_result,
    int relu_enable)
{
    for (int i = 0; i < left_rows; ++i) {
        for (int j = 0; j < right_cols; ++j) {
            int sum = 0;

            // Dot product
            for (int k = 0; k < left_cols; ++k) {
                int left_val  = left_data[i * left_cols + k];
                int right_val = right_data[k * right_cols + j];
                sum += left_val * right_val;
            }

            // Add bias
            sum += bias_data[j];

            // If add_result is enabled, add old_res as well
            if (add_result) {
                sum += old_res_data[i * right_cols + j];
            }

            // Apply ReLU if enabled
            if (relu_enable && sum < 0) {
                sum = 0;
            }

            // Store final result (no shift logic in this example)
            result_data[i * right_cols + j] = (int8_t)sum;
        }
    }
}

void test_cvkcv181x_tiu_matrix_multiplication_add_relu()
{
    cvk_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));

    cvk_tl_t left;
    memset(&left, 0, sizeof(left));
    left.start_address = 0x1000;
    left.fmt = CVK_FMT_I8;
    left.shape.n = 4;  // "rows" = 4
    left.shape.c = 2;  // "cols" = 2
    left.shape.h = 1;
    left.shape.w = 2;
    // Usually stride.n = #cols when using shape.n x shape.c for 2D
    left.stride.n = 2;
    left.stride.c = 1;
    left.stride.h = 1;
    left.stride.w = 1;

    cvk_tl_t right;
    memset(&right, 0, sizeof(right));
    right.start_address = 0x2000;
    right.fmt = CVK_FMT_I8;
    right.shape.n = 2;  // "rows" = 2
    right.shape.c = 3;  // "cols" = 3
    right.shape.h = 1;
    right.shape.w = 3;
    right.stride.n = 3;
    right.stride.c = 1;
    right.stride.h = 1;
    right.stride.w = 1;

    cvk_tl_t bias;
    memset(&bias, 0, sizeof(bias));
    bias.start_address = 0x3000;
    bias.fmt = CVK_FMT_I8;
    bias.shape.n = 1;  // 1 row
    bias.shape.c = 3;  // 3 columns
    bias.shape.h = 1;
    bias.shape.w = 3;
    bias.stride.n = 3;
    bias.stride.c = 1;
    bias.stride.h = 1;
    bias.stride.w = 1;

    cvk_tl_t res;
    memset(&res, 0, sizeof(res));
    res.start_address = 0x4000;
    res.fmt = CVK_FMT_I8;
    res.shape.n = 4;  // 4 rows
    res.shape.c = 3;  // 3 columns
    res.shape.h = 1;
    res.shape.w = 3;
    res.stride.n = 3;
    res.stride.c = 1;
    res.stride.h = 1;
    res.stride.w = 1;

    int8_t left_data[4 * 2] = {
         1, 2,
         3, 4,
         5, 6,
         7, 8
    };

    int8_t right_data[2 * 3] = {
        -1, -2, -3,
         2,  3,  4
    };

    int8_t bias_data[3] = {1, 1, 1};

    int8_t old_res_data[4 * 3] = {
        10,  2,  3,
         4,  5,  6,
         7,  8,  9,
        10, 11, 12
    };

    int8_t expected_res[4 * 3];
    memset(expected_res, 0, sizeof(expected_res));

    compute_reference_result_add_relu(
        left_data,
        4, 2,
        right_data,
        2, 3,
        bias_data,
        old_res_data,
        expected_res,
        /*add_result=*/1,
        /*relu_enable=*/1
    );

    write_memory(left.start_address,  left_data,     sizeof(left_data));
    write_memory(right.start_address, right_data,    sizeof(right_data));
    write_memory(bias.start_address,  bias_data,     sizeof(bias_data));
    write_memory(res.start_address,   old_res_data,  sizeof(old_res_data));

    cvk_tiu_matrix_multiplication_param_t param;
    memset(&param, 0, sizeof(param));
    param.res         = &res;
    param.left        = &left;
    param.right       = &right;
    param.bias        = &bias;

    // SHIFT & RELU & ADD
    param.lshift_bits = 0;       // No left shift in this example
    param.rshift_bits = 0;       // No right shift
    param.relu_enable = 1;       // Enable ReLU
    param.add_result  = 1;       // We want to add old_res into the result
    param.ps32_mode   = 0;       // 0 => normal int8 mode (not 32-bit partial sum)
    param.res_is_int8 = 1;       // Store final result in int8
    param.layer_id    = 2;       // Just an example layer ID

    cvkcv181x_tiu_matrix_multiplication(&ctx, &param);

    int8_t actual_res[4 * 3];
    memset(actual_res, 0, sizeof(actual_res));
    read_memory(res.start_address, actual_res, sizeof(actual_res));

    int total_size = 4 * 3;
    for (int i = 0; i < total_size; ++i) {
        if (actual_res[i] != expected_res[i]) {
            printf("ERROR: Mismatch at index %d. Expected %d but got %d\n",
                   i, expected_res[i], actual_res[i]);
            assert(0);
        }
    }

    printf("Matrix multiplication (with add + ReLU) test passed successfully!\n");
}

int main()
{
    test_cvkcv181x_tiu_matrix_multiplication_add_relu();
    return 0;
}
