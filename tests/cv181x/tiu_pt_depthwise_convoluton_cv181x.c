//test for cvkcv181x_tiu_pt_depthwise_convolution
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


// Helper function for creating a tensor with allocated data
// The data pointer is stored in 'start_address'.

cvk_tl_t* create_tensor(int n, int c, int h, int w,
                        int stride_n, int stride_c, int stride_h, int stride_w,
                        cvk_fmt_t fmt) {
    cvk_tl_t* tensor = (cvk_tl_t*)malloc(sizeof(cvk_tl_t));
    tensor->shape.n = n;
    tensor->shape.c = c;
    tensor->shape.h = h;
    tensor->shape.w = w;
    tensor->stride.n = stride_n;
    tensor->stride.c = stride_c;
    tensor->stride.h = stride_h;
    tensor->stride.w = stride_w;
    tensor->fmt = fmt;
    
    // Allocate memory for tensor data (int8_t for simplicity)
    size_t elem_count = (size_t)n * c * h * w;
    tensor->start_address = (uintptr_t)malloc(elem_count * sizeof(int8_t));
    return tensor;
}


// Helper function to build depthwise convolution parameters

cvk_tiu_depthwise_pt_convolution_param_t
create_conv_param(cvk_tl_t* ifmap, cvk_tl_t* ofmap, cvk_tl_t* weight, cvk_tl_t* bias) {
    cvk_tiu_depthwise_pt_convolution_param_t param;
    memset(&param, 0, sizeof(param));
    param.ifmap = ifmap;
    param.ofmap = ofmap;
    param.weight = weight;
    param.bias   = bias;
    
    param.weight_is_const = 0;
    param.relu_enable     = 1;  // We'll test with ReLU = ON
    param.stride_h        = 1;
    param.stride_w        = 1;
    param.pad_top         = 0;
    param.pad_bottom      = 0;
    param.pad_left        = 0;
    param.pad_right       = 0;
    param.ins_h           = 0;
    param.ins_last_h      = 0;
    param.ins_w           = 0;
    param.ins_last_w      = 0;
    param.dilation_h      = 1;
    param.dilation_w      = 1;
    param.rshift_bits     = 0;  // No shift in this example
    param.layer_id        = 1;
    param.cmd_pre_exe_typ = 0;
    param.cmd_pre_exe     = 0;
    return param;
}


// Fill the tensors with predictable patterns
// so we know what results to expect.

static void fill_ifmap(cvk_tl_t* ifmap) {
    int8_t* data = (int8_t*)ifmap->start_address;
    int N = ifmap->shape.n;
    int C = ifmap->shape.c;
    int H = ifmap->shape.h;
    int W = ifmap->shape.w;

    // Example pattern: val = (c*5 + h*W + w) & 0x7F
    // ensures values stay in a known int8 range
    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    size_t idx = (size_t)((((n*C) + c)*H + h)*W + w);
                    int val = (c * 5) + (h * W) + w;
                    data[idx] = (int8_t)(val & 0x7F);
                }
            }
        }
    }
}

static void fill_weight_depthwise(cvk_tl_t* weight) {

    int8_t* data = (int8_t*)weight->start_address;
    int N = weight->shape.n; // typically 1 in depthwise
    int C = weight->shape.c;
    int H = weight->shape.h; // kernel height
    int W = weight->shape.w; // kernel width

    // Example pattern: val = c + h + w
    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            for(int kh = 0; kh < H; kh++) {
                for(int kw = 0; kw < W; kw++) {
                    size_t idx = (size_t)((((n*C) + c)*H + kh)*W + kw);
                    int val = c + kh + kw;  // small number
                    data[idx] = (int8_t)(val & 0x7F);
                }
            }
        }
    }
}

static void fill_bias_depthwise(cvk_tl_t* bias) {
    // For depthwise, bias->shape might be {1, C, 1, 1}
    int8_t* data = (int8_t*)bias->start_address;
    int N = bias->shape.n;  // typically 1
    int C = bias->shape.c;
    // Since shape.h=1 and shape.w=1, total elements = N*C
    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            size_t idx = (size_t)(n*C + c);
            // Example bias = c * 2
            data[idx] = (int8_t)((c * 2) & 0x7F);
        }
    }
}


// A reference software depthwise convolution function


static void reference_depthwise_conv(const cvk_tiu_depthwise_pt_convolution_param_t* param,
                                     int8_t* ref_output) {
    const cvk_tl_t* ifmap  = param->ifmap;
    const cvk_tl_t* weight = param->weight;
    const cvk_tl_t* bias   = param->bias;
    const cvk_tl_t* ofmap  = param->ofmap;

    const int8_t* ifmap_data  = (int8_t*)ifmap->start_address;
    const int8_t* weight_data = (int8_t*)weight->start_address;
    const int8_t* bias_data   = (bias ? (int8_t*)bias->start_address : NULL);

    int N = ifmap->shape.n;  // e.g., 1
    int C = ifmap->shape.c;  // e.g., 3
    int IH= ifmap->shape.h;
    int IW= ifmap->shape.w;

    int OH= ofmap->shape.h;
    int OW= ofmap->shape.w;

    // Kernel shape is 3x3 for each channel
    int KH = weight->shape.h;
    int KW = weight->shape.w;

    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            // bias for channel c
            int bias_val = bias_data ? bias_data[c] : 0;

            for(int oh = 0; oh < OH; oh++) {
                for(int ow = 0; ow < OW; ow++) {
                    // We'll do a 32-bit accumulation to avoid int8 overflow
                    int sum = 0;
                    for(int kh = 0; kh < KH; kh++) {
                        for(int kw = 0; kw < KW; kw++) {
                            int ih = oh + kh;  // pad=0, stride=1
                            int iw = ow + kw;
                            // index in ifmap
                            size_t if_idx = (size_t)((((n*C) + c)*IH + ih)*IW + iw);
                            // index in weight (for channel c)
                            size_t wt_idx = (size_t)((((0*C) + c)*KH + kh)*KW + kw);
                            // multiply
                            int a = ifmap_data[if_idx];
                            int b = weight_data[wt_idx];
                            sum += (a * b);
                        }
                    }
                    // add bias
                    sum += bias_val;

                    // ReLU
                    if (param->relu_enable && sum < 0) {
                        sum = 0;
                    }

                    // clamp to int8
                    if (sum > 127)  sum = 127;
                    if (sum < -128) sum = -128;

                    // store to output
                    size_t out_idx = (size_t)((((n*C) + c)*OH + oh)*OW + ow);
                    ref_output[out_idx] = (int8_t)sum;
                }
            }
        }
    }
}


// Compare hardware (or mocked) output with the software reference

static void compare_output(const cvk_tl_t* ofmap, const int8_t* ref_output) {
    const int8_t* hw_data = (int8_t*)ofmap->start_address;
    int N = ofmap->shape.n;
    int C = ofmap->shape.c;
    int H = ofmap->shape.h;
    int W = ofmap->shape.w;

    size_t total = (size_t)N * C * H * W;
    for(size_t i = 0; i < total; i++) {
        if (hw_data[i] != ref_output[i]) {
            printf("Mismatch at index=%zu. Expected=%d, Got=%d\n",
                   i, ref_output[i], hw_data[i]);
            assert(0 && "Depthwise conv result mismatch!");
        }
    }
    printf("[compare_output] All outputs match the reference result.\n");
}


// Main test function

int main() {
    cvk_tl_t* ifmap  = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t* ofmap  = create_tensor(1, 3, 32, 32, 32, 32, 1, 1, CVK_FMT_I8);
    cvk_tl_t* weight= create_tensor(1, 3, 3, 3,  3,  3,  1, 1, CVK_FMT_I8);
    cvk_tl_t* bias  = create_tensor(1, 3, 1, 1,  1,  1,  0, 0, CVK_FMT_I8);

    cvk_tiu_depthwise_pt_convolution_param_t conv_param = create_conv_param(ifmap, ofmap, weight, bias);

    fill_ifmap(ifmap);
    fill_weight_depthwise(weight);
    fill_bias_depthwise(bias);

    cvk_context_t *ctx = (cvk_context_t*)malloc(sizeof(cvk_context_t));
    memset(ctx, 0, sizeof(cvk_context_t));

    printf("Running depthwise convolution...\n");
    cvkcv181x_tiu_pt_depthwise_convolution(ctx, &conv_param);
    printf("Depthwise convolution completed.\n");

    size_t out_size = (size_t)ofmap->shape.n * ofmap->shape.c
                    * ofmap->shape.h * ofmap->shape.w;
    int8_t* ref_out = (int8_t*)malloc(out_size * sizeof(int8_t));
    memset(ref_out, 0, out_size);

    reference_depthwise_conv(&conv_param, ref_out);

    compare_output(ofmap, ref_out);

    // Clean up
    cvkcv181x_lmem_free_tensor(&ctx, ref_out);
    cvkcv181x_lmem_free_tensor(&ctx, (void*)ifmap->start_address);
    cvkcv181x_lmem_free_tensor(&ctx, (void*)ofmap->start_address);
    cvkcv181x_lmem_free_tensor(&ctx, (void*)weight->start_address);
    cvkcv181x_lmem_free_tensor(&ctx, (void*)bias->start_address);
    cvkcv181x_lmem_free_tensor(&ctx, ifmap);
    cvkcv181x_lmem_free_tensor(&ctx, ofmap);
    cvkcv181x_lmem_free_tensor(&ctx, weight);
    cvkcv181x_lmem_free_tensor(&ctx, bias);
    cvkcv181x_lmem_free_tensor(&ctx, ctx);

    printf("Depthwise Convolution test verified successfully!\n");
    return 0;
}
