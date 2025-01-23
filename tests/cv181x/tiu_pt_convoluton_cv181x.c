//test for cvkcv181x_tiu_pt_convolution
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


// Mock global buffer simulating local memory (LMEM)
#define FAKE_LMEM_SIZE  0x40000  // Example large enough size
static int8_t g_lmem[FAKE_LMEM_SIZE];


// Helper function to get a pointer into the mock LMEM at a specific offset
static inline int8_t* lmem_ptr(uint32_t start_address) {
    // In a real system, you'd do more error checking
    return &g_lmem[start_address];
}


// Fill the ifmap and weight tensors with a predictable pattern
static void fill_ifmap(const cvk_tl_t* ifmap) {
    int8_t* data = lmem_ptr(ifmap->start_address);
    int N = ifmap->shape.n, C = ifmap->shape.c;
    int H = ifmap->shape.h, W = ifmap->shape.w;
    // Example pattern: value = (c*5 + h*W + w) & 0x7F
    // so it's in a known range for int8
    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    int idx = ((n*C + c)*H + h)*W + w;
                    int val = (c * 5) + (h * W) + w;
                    data[idx] = (int8_t)(val & 0x7F);
                }
            }
        }
    }
}

static void fill_weight(const cvk_tl_t* weight) {
    int8_t* data = lmem_ptr(weight->start_address);
    // For weights: shape {OutChannels=32, InChannels=32, KernelH=3, KernelW=3}
    int OC = weight->shape.n, IC = weight->shape.c;
    int KH = weight->shape.h, KW = weight->shape.w;
    // Example pattern: value = (oc + ic + kh + kw) & 0x7F
    for(int oc = 0; oc < OC; oc++) {
        for(int ic = 0; ic < IC; ic++) {
            for(int kh = 0; kh < KH; kh++) {
                for(int kw = 0; kw < KW; kw++) {
                    int idx = oc * (IC*KH*KW)
                            + ic * (KH*KW)
                            + kh * KW
                            + kw;
                    int val = oc + ic + kh + kw;
                    data[idx] = (int8_t)(val & 0x7F);
                }
            }
        }
    }
}


// Software reference convolution (no bias, no shift, no ReLU) to verify
static void reference_convolution(const cvk_tiu_pt_convolution_param_t* param,
                                  int8_t* ref_output) {
    const cvk_tl_t* ifmap  = param->ifmap;
    const cvk_tl_t* weight= param->weight;
    const cvk_tl_t* ofmap  = param->ofmap;

    const int8_t* ifmap_data  = lmem_ptr(ifmap->start_address);
    const int8_t* weight_data = lmem_ptr(weight->start_address);

    int N   = ifmap->shape.n;   // e.g. 1
    int IC  = ifmap->shape.c;   // e.g. 32
    int IH  = ifmap->shape.h;   // e.g. 32
    int IW  = ifmap->shape.w;   // e.g. 32
    int OC  = weight->shape.n;  // e.g. 32
    int KH  = weight->shape.h;  // e.g. 3
    int KW  = weight->shape.w;  // e.g. 3
    int OH  = ofmap->shape.h;   // e.g. 30
    int OW  = ofmap->shape.w;   // e.g. 30

    // In this example, N=1, so we only do n=0
    for(int n = 0; n < N; n++) {
        for(int oc = 0; oc < OC; oc++) {
            for(int oh = 0; oh < OH; oh++) {
                for(int ow = 0; ow < OW; ow++) {
                    int sum = 0;  // accumulate in 32-bit
                    for(int ic = 0; ic < IC; ic++) {
                        for(int kh = 0; kh < KH; kh++) {
                            for(int kw = 0; kw < KW; kw++) {
                                int ih = oh + kh;  // stride=1, pad=0
                                int iw = ow + kw;
                                int if_idx = ((n*IC + ic)*IH + ih)*IW + iw;
                                int wt_idx = ((oc*IC + ic)*KH + kh)*KW + kw;
                                int a = ifmap_data[if_idx];
                                int b = weight_data[wt_idx];
                                sum += (a * b);
                            }
                        }
                    }
                    // clamp to int8 range
                    if (sum > 127)  sum = 127;
                    if (sum < -128) sum = -128;
                    // store into ref_out
                    int out_idx = ((n*OC + oc)*OH + oh)*OW + ow;
                    ref_output[out_idx] = (int8_t)sum;
                }
            }
        }
    }
}


// Compare the hardware (or mocked) output with the reference
static void compare_output(const cvk_tl_t* ofmap, const int8_t* ref_out) {
    const int8_t* hw_out = lmem_ptr(ofmap->start_address);
    int N = ofmap->shape.n, C = ofmap->shape.c;
    int H = ofmap->shape.h, W = ofmap->shape.w;
    int total = N*C*H*W;
    for(int i = 0; i < total; i++) {
        if (hw_out[i] != ref_out[i]) {
            printf("Mismatch at index %d: expected %d, got %d\n",
                   i, ref_out[i], hw_out[i]);
            assert(0 && "Convolution result mismatch!");
        }
    }
    printf("[compare_output] All outputs match the reference result.\n");
}


// Mockup for tensor data initialization
void init_tensor(cvk_tl_t *tensor, uint32_t start_address, uint8_t fmt,
                 cvk_tl_shape_t shape, cvk_tl_stride_t stride) {
    tensor->start_address = start_address;
    tensor->fmt = fmt;
    tensor->shape = shape;
    tensor->stride = stride;
}


// Mockup for convolution parameter initialization
void init_convolution_param(cvk_tiu_pt_convolution_param_t *param) {
    // Initialize Ifmap
    cvk_tl_t *ifmap = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    cvk_tl_shape_t ifmap_shape = {1, 32, 32, 32};  // N, C, H, W
    cvk_tl_stride_t ifmap_stride = {1024, 256, 8, 1};
    init_tensor(ifmap, 0x1000, CVK_FMT_I8, ifmap_shape, ifmap_stride);
    
    // Initialize Weight
    cvk_tl_t *weight = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    cvk_tl_shape_t weight_shape = {32, 32, 3, 3};  // N (OC), C (IC), H, W
    cvk_tl_stride_t weight_stride = {1024, 256, 8, 1};
    init_tensor(weight, 0x2000, CVK_FMT_I8, weight_shape, weight_stride);
    
    // Initialize Output Feature Map
    cvk_tl_t *ofmap = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    cvk_tl_shape_t ofmap_shape = {1, 32, 30, 30};  // N, C, H, W
    cvk_tl_stride_t ofmap_stride = {1024, 256, 8, 1};
    init_tensor(ofmap, 0x3000, CVK_FMT_I8, ofmap_shape, ofmap_stride);
    
    // Initialize Convolution Parameters
    param->ifmap = ifmap;
    param->weight = weight;
    param->ofmap = ofmap;
    param->bias = NULL;
    param->pad_top = 0;
    param->pad_bottom = 0;
    param->pad_left = 0;
    param->pad_right = 0;
    param->stride_h = 1;
    param->stride_w = 1;
    param->dilation_h = 1;
    param->dilation_w = 1;
    param->ins_h = 0;
    param->ins_last_h = 0;
    param->ins_w = 0;
    param->ins_last_w = 0;
    param->relu_enable = 0;
    param->rshift_bits = 0;
    param->ps32_mode = 0;
    param->cmd_pre_exe_typ = 0;
    param->cmd_pre_exe = 0;
    param->layer_id = 1;
}


// Main test function
int main() {
    // Create a mock context
    cvk_context_t ctx;
    memset(&ctx, 0, sizeof(cvk_context_t));
    ctx.info.lmem_size = FAKE_LMEM_SIZE; // example size
    ctx.info.eu_num = 16;                // example # of execution units
    ctx.info.npu_num = 4;                // example # of NPUs
    
    // Initialize convolution parameters
    cvk_tiu_pt_convolution_param_t conv_param;
    init_convolution_param(&conv_param);
    
    // Fill ifmap & weights with known patterns
    fill_ifmap(conv_param.ifmap);
    fill_weight(conv_param.weight);

    // Call the convolution function
    printf("Running convolution test...\n");
    cvkcv181x_tiu_pt_convolution(&ctx, &conv_param);
    printf("Convolution test completed.\n");

    // Compute a software reference for verification
    int out_n  = conv_param.ofmap->shape.n;
    int out_c  = conv_param.ofmap->shape.c;
    int out_h  = conv_param.ofmap->shape.h;
    int out_w  = conv_param.ofmap->shape.w;
    int out_len= out_n * out_c * out_h * out_w;
    int8_t* ref_result = (int8_t*)malloc(out_len * sizeof(int8_t));
    memset(ref_result, 0, out_len * sizeof(int8_t));

    reference_convolution(&conv_param, ref_result);

    // Compare the hardware output with reference
    compare_output(conv_param.ofmap, ref_result);

    // Cleanup
    cvkcv181x_lmem_free_tensor(&ctx, ref_result);
    cvkcv181x_lmem_free_tensor(&ctx, conv_param.ifmap);
    cvkcv181x_lmem_free_tensor(&ctx, conv_param.weight);
    cvkcv181x_lmem_free_tensor(&ctx, conv_param.ofmap);

    printf("Convolution test verified successfully!\n");
    return 0;
}
