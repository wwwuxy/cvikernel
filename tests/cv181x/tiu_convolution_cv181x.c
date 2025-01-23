//test for cvkcv181x_tiu_convolution
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


static void naive_conv_int8(
    const int8_t *ifmap,     // input feature map
    const int8_t *weight,    // weight (OC, IC, KH, KW)
    int8_t       *out,       // output feature map
    int   inC,   int inH,   int inW,
    int   outC,  int outH,  int outW,
    int   kernelH, int kernelW,
    int   strideH, int strideW,
    int   padTop,  int padLeft
) {
    // This does not include bias and only performs multiply-accumulate
    // For N=1
    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int sum = 0;
                // Convolution kernel
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            int in_h = oh * strideH + kh - padTop;
                            int in_w = ow * strideW + kw - padLeft;
                            // Boundary check
                            if (in_h < 0 || in_h >= inH ||
                                in_w < 0 || in_w >= inW) {
                                continue;
                            }
                            // Compute 1D offsets in ifmap and weight
                            int ifmap_idx  = ic * (inH * inW) + in_h * inW + in_w;
                            int weight_idx = oc * (inC * kernelH * kernelW) +
                                             ic * (kernelH * kernelW) +
                                             kh * kernelW + kw;
                            sum += (int)ifmap[ifmap_idx] * (int)weight[weight_idx];
                        }
                    }
                }
                // Saturate to int8
                if (sum > 127) sum = 127;
                if (sum < -128) sum = -128;

                // Output offset
                int out_idx = oc * (outH * outW) + oh * outW + ow;
                out[out_idx] = (int8_t)sum;
            }
        }
    }
}


// Actual test function

void test_cvkcv181x_tiu_convolution() {
    // 1) Create context and parameters
    cvk_context_t ctx;
    cvk_tiu_convolution_param_t p;
    memset(&ctx, 0, sizeof(ctx));
    memset(&p, 0, sizeof(p));

    // Mock context
    ctx.info.lmem_size = 1024; // mock local memory size
    ctx.info.eu_num = 8;       // mock number of EUs

    // Set input / output / weight shapes (N=1)
    int inN = 1, inC = 16, inH = 28, inW = 28;
    int outN = 1, outC = 16, outH = 14, outW = 14;  // For demonstration
    int kH = 3, kW = 3;

    int strideH = 2, strideW = 2;
    int padTop = 0, padLeft = 0;

    // For cvk parameters
    p.ifmap   = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    p.ofmap   = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    p.weight  = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    memset(p.ifmap, 0, sizeof(*p.ifmap));
    memset(p.ofmap, 0, sizeof(*p.ofmap));
    memset(p.weight, 0, sizeof(*p.weight));

    // Shape assignment
    cvk_tl_shape_t ifmap_shape = {inN, inC, inH, inW};
    cvk_tl_shape_t ofmap_shape = {outN, outC, outH, outW};
    cvk_tl_shape_t weight_shape = {outC, inC, kH, kW};

    *(cvk_tl_shape_t *)&p.ifmap->shape = ifmap_shape;
    *(cvk_tl_shape_t *)&p.ofmap->shape = ofmap_shape;
    *(cvk_tl_shape_t *)&p.weight->shape = weight_shape;

    int ifmap_size  = inN * inC * inH * inW;
    int ofmap_size  = outN * outC * outH * outW;
    int weight_size = outC * inC * kH * kW;

    int8_t *ifmap_data   = (int8_t*)malloc(ifmap_size);
    int8_t *ofmap_data   = (int8_t*)malloc(ofmap_size);
    int8_t *weight_data  = (int8_t*)malloc(weight_size);
    int8_t *ref_ofmap    = (int8_t*)malloc(ofmap_size);

    // Initialize: random or fixed
    srand(12345);
    for (int i = 0; i < ifmap_size; i++) {
        ifmap_data[i] = (int8_t)(rand() % 5 - 2); // range -2~2
    }
    for (int i = 0; i < weight_size; i++) {
        weight_data[i] = (int8_t)(rand() % 5 - 2);
    }
    // Initialize ofmap to 0
    memset(ofmap_data,  0, ofmap_size);
    memset(ref_ofmap,   0, ofmap_size);

    // Other convolution parameters
    p.stride_h = strideH;
    p.stride_w = strideW;
    p.pad_top    = padTop;
    p.pad_bottom = 0;
    p.pad_left   = padLeft;
    p.pad_right  = 0;
    p.ins_h = 0;
    p.ins_w = 0;
    p.ins_last_h = 0;
    p.ins_last_w = 0;
    p.dilation_h = 1;
    p.dilation_w = 1;
    p.relu_enable = 1;
    p.w_is_const = 1;
    p.ps32_mode = 0;
    p.layer_id = 0;
    p.cmd_pre_exe_typ = 0;
    p.cmd_pre_exe = 0;

    // Use CPU reference implementation to compute ref_ofmap
    naive_conv_int8(
        ifmap_data, weight_data, ref_ofmap,
        inC, inH, inW,        // input C, H, W
        outC, outH, outW,     // output C, H, W
        kH, kW,
        strideH, strideW,
        padTop, padLeft
    );

    // Call hardware/emulated convolution function (generates commands or simulates)
    cvkcv181x_tiu_convolution(&ctx, &p);

    // Compare results
    int error_count = 0;
    for (int i = 0; i < ofmap_size; i++) {
        if (ofmap_data[i] != ref_ofmap[i]) {
            error_count++;
            if (error_count < 10) { // Only print first several mismatches to avoid spam
                printf("[Mismatch] index=%d, hw=%d, ref=%d\n",
                       i, ofmap_data[i], ref_ofmap[i]);
            }
        }
    }
    if (error_count == 0) {
        printf("[Test PASS] All output matched reference.\n");
    } else {
        printf("[Test FAIL] Mismatch count = %d\n", error_count);
    }

    // Free memory
    free(ifmap_data);
    free(ofmap_data);
    free(weight_data);
    free(ref_ofmap);

    free(p.ifmap);
    free(p.ofmap);
    free(p.weight);

    printf("Test complete.\n");
}

// Entry point
int main() {
    test_cvkcv181x_tiu_convolution();
    return 0;
}
