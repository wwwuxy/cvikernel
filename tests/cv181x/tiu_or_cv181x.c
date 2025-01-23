//test for cvkcv181x_tiu_or
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


// Mocking a local memory region so we can store input/output tensors.
#define FAKE_LMEM_SIZE  (0x30000)
static int8_t g_lmem[FAKE_LMEM_SIZE];

// Creates a tensor descriptor with the specified format, shape, and base offset
cvk_tl_t* create_tensor(cvk_fmt_t fmt, int n, int c, int h, int w, int start_address) {
    cvk_tl_t* tensor = (cvk_tl_t*)malloc(sizeof(cvk_tl_t));
    tensor->fmt = fmt;
    tensor->shape.n = n;
    tensor->shape.c = c;
    tensor->shape.h = h;
    tensor->shape.w = w;
    tensor->start_address = start_address;
    return tensor;
}

// Mock function to reset a TIU register (not strictly necessary for this demo)
void reset_tiu_reg(tiu_reg_t* reg) {
    memset(reg, 0, sizeof(tiu_reg_t));
}


static void fill_ifmap_data(cvk_tl_t *tensor) {
    int8_t* data = g_lmem + tensor->start_address;
    int n = tensor->shape.n;
    int c = tensor->shape.c;
    int h = tensor->shape.h;
    int w = tensor->shape.w;

    // For demonstration, fill with row*width + col (wrapped to int8)
    // This ensures each 3x3 window can be checked easily
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            for (int hi = 0; hi < h; hi++) {
                for (int wi = 0; wi < w; wi++) {
                    int idx = ni*c*h*w + ci*h*w + hi*w + wi;
                    // Just a small pattern so we don't overflow int8 too easily
                    int val = (hi * w + wi) & 0x7F;  // keep it positive
                    data[idx] = (int8_t)val;
                }
            }
        }
    }
}


static void verify_min_pooling_result(const cvk_tl_t* ifmap,
                                      const cvk_tl_t* ofmap,
                                      const cvk_tiu_min_pooling_param_t *param) {
    const int8_t* ifmap_data = g_lmem + ifmap->start_address;
    const int8_t* ofmap_data = g_lmem + ofmap->start_address;

    int N = ifmap->shape.n, C = ifmap->shape.c;
    int H = ifmap->shape.h, W = ifmap->shape.w;
    int outH = ofmap->shape.h, outW = ofmap->shape.w;

    int kh = param->kh, kw = param->kw;
    int stride_h = param->stride_h, stride_w = param->stride_w;
    int pad_top = param->pad_top, pad_left = param->pad_left;

    // For each output pixel, determine the “true” minimum in its window
    // and compare against ofmap_data.
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    // Index into the ofmap
                    int out_index = n*C*outH*outW + c*outH*outW + oh*outW + ow;
                    int8_t hw_min_val = ofmap_data[out_index];

                    // Compute the input window start (account for pad=0 in this example)
                    int in_h_start = oh * stride_h - pad_top;
                    int in_w_start = ow * stride_w - pad_left;

                    int8_t ref_val = 127; // largest int8 to start
                    for (int r = 0; r < kh; r++) {
                        for (int s = 0; s < kw; s++) {
                            int in_h = in_h_start + r;
                            int in_w = in_w_start + s;
                            // Boundary check
                            if (in_h < 0 || in_h >= H || in_w < 0 || in_w >= W) {
                                continue;  // skip if out of bounds
                            }
                            int in_index = n*C*H*W + c*H*W + in_h*W + in_w;
                            int8_t val = ifmap_data[in_index];
                            if (val < ref_val) {
                                ref_val = val;
                            }
                        }
                    }

                    // Compare reference min vs. hardware min
                    if (hw_min_val != ref_val) {
                        printf("Error: Mismatch at (n=%d, c=%d, oh=%d, ow=%d). "
                               "Expected %d, got %d\n",
                               n, c, oh, ow, ref_val, hw_min_val);
                        assert(0 && "Min pooling mismatch!");
                    }
                }
            }
        }
    }
    printf("[verify_min_pooling_result] All results match expected min values!\n");
}


// The main test function that sets up your context, creates tensors,
// fills the input data, calls min pooling, and verifies the results
void test_cvkcv181x_tiu_min_pooling() {
    // Create a mock context
    cvk_context_t ctx;
    ctx.info.eu_num = 4;  // Mock EU number
    ctx.info.npu_num = 2; // Mock NPU number

    // Create ifmap (1x8x64x64) and ofmap (1x8x32x32)
    cvk_tl_t *ifmap = create_tensor(CVK_FMT_I8, 1, 8, 64, 64, 0x1000);
    cvk_tl_t *ofmap = create_tensor(CVK_FMT_I8, 1, 8, 32, 32, 0x2000);

    // Fill the input data with a known pattern
    fill_ifmap_data(ifmap);

    // Prepare parameters for min pooling
    cvk_tiu_min_pooling_param_t param;
    memset(&param, 0, sizeof(param));
    param.ifmap = ifmap;
    param.ofmap = ofmap;
    param.kh = 3;
    param.kw = 3;
    param.stride_h = 2;
    param.stride_w = 2;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.layer_id = 1;
    param.ins_fp = 0.0f; // not used in this mock

    // Call the min pooling function (mocked or actual hardware)
    cvkcv181x_tiu_min_pooling(&ctx, &param);
    printf("[test_cvkcv181x_tiu_min_pooling] Min pooling operation completed.\n");

    // Verify the results with a software-based reference
    verify_min_pooling_result(ifmap, ofmap, &param);

    // Free resources
    cvkcv181x_lmem_free_tensor(&ctx, ifmap);
    cvkcv181x_lmem_free_tensor(&ctx, ofmap);
}


// Entry point

int main() {
    test_cvkcv181x_tiu_min_pooling();
    printf("Min pooling test passed.\n");
    return 0;
}
