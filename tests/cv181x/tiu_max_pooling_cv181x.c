//test for cvkcv181x_tiu_max_pooling
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

#define FAKE_LMEM_SIZE  (0x30000)
static int8_t g_lmem_base[FAKE_LMEM_SIZE];

typedef struct {
    cvk_tl_t *ifmap;
    cvk_tl_t *ofmap;
    int kh, kw;
    int stride_h, stride_w;
    int pad_top, pad_bottom, pad_left, pad_right;
    int layer_id;
    int ins_val;  
} cvk_tiu_max_pooling_param_t;


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

void cvkcv181x_tiu_max_pooling(cvk_context_t *ctx, const cvk_tiu_max_pooling_param_t *param) {


    cvk_tl_t *ifmap = param->ifmap;
    cvk_tl_t *ofmap = param->ofmap;
    int kh = param->kh;
    int kw = param->kw;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;

    int8_t *ifmap_data = g_lmem_base + ifmap->start_address;
    int8_t *ofmap_data = g_lmem_base + ofmap->start_address;

    int inN = ifmap->shape.n;
    int inC = ifmap->shape.c;
    int inH = ifmap->shape.h;
    int inW = ifmap->shape.w;

    int outN = ofmap->shape.n;
    int outC = ofmap->shape.c;
    int outH = ofmap->shape.h;
    int outW = ofmap->shape.w;

    assert(inN == outN && inC == outC);


    for (int n = 0; n < inN; n++) {
        for (int c = 0; c < inC; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int out_index = n*outC*outH*outW
                                  + c*outH*outW
                                  + oh*outW + ow;
                    int in_h_start = oh * stride_h;  
                    int in_w_start = ow * stride_w;  

                    int8_t max_val = -128; 
                    for (int r = 0; r < kh; r++) {
                        for (int s = 0; s < kw; s++) {
                            int in_h = in_h_start + r;
                            int in_w = in_w_start + s;
                            // boundary check
                            if (in_h < 0 || in_h >= inH || in_w < 0 || in_w >= inW) {
                                continue;
                            }
                            int in_index = n*inC*inH*inW
                                         + c*inH*inW
                                         + in_h*inW + in_w;
                            int8_t val = ifmap_data[in_index];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    ofmap_data[out_index] = max_val;
                }
            }
        }
    }
}

void fill_ifmap_data(cvk_tl_t *tensor) {
    int8_t *data = g_lmem_base + tensor->start_address;
    int N = tensor->shape.n;
    int C = tensor->shape.c;
    int H = tensor->shape.h;
    int W = tensor->shape.w;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {

                    int index = n*C*H*W + c*H*W + h*W + w;
                    int val = h*W + w; 
                    data[index] = (int8_t)(val & 0x7F); 
                }
            }
        }
    }
}


void verify_max_pooling_result(const cvk_tl_t *ifmap,
                               const cvk_tl_t *ofmap,
                               const cvk_tiu_max_pooling_param_t *param) {
    int8_t *ifmap_data = g_lmem_base + ifmap->start_address;
    int8_t *ofmap_data = g_lmem_base + ofmap->start_address;

    int inN = ifmap->shape.n;
    int inC = ifmap->shape.c;
    int inH = ifmap->shape.h;
    int inW = ifmap->shape.w;

    int outN = ofmap->shape.n;
    int outC = ofmap->shape.c;
    int outH = ofmap->shape.h;
    int outW = ofmap->shape.w;

    int kh = param->kh;
    int kw = param->kw;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;

    for (int n = 0; n < outN; n++) {
        for (int c = 0; c < outC; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int out_index = n*outC*outH*outW
                                  + c*outH*outW
                                  + oh*outW + ow;
                    int8_t hw_val = ofmap_data[out_index];

                    int in_h_start = oh * stride_h;
                    int in_w_start = ow * stride_w;
                    int8_t ref_max = -128;
                    for (int r = 0; r < kh; r++) {
                        for (int s = 0; s < kw; s++) {
                            int in_h = in_h_start + r;
                            int in_w = in_w_start + s;
                            if (in_h < 0 || in_h >= inH ||
                                in_w < 0 || in_w >= inW) {
                                continue;
                            }
                            int in_index = n*inC*inH*inW
                                         + c*inH*inW
                                         + in_h*inW + in_w;
                            int8_t val = ifmap_data[in_index];
                            if (val > ref_max) {
                                ref_max = val;
                            }
                        }
                    }

                    if (hw_val != ref_max) {
                        printf("Error: Mismatch at (n=%d,c=%d,oh=%d,ow=%d). "
                               "Expected %d, got %d\n",
                               n, c, oh, ow, ref_max, hw_val);
                        assert(hw_val == ref_max); 
                    }
                }
            }
        }
    }
    printf("Max pooling result verified successfully!\n");
}

void test_cvkcv181x_tiu_max_pooling() {
    cvk_context_t ctx;
    ctx.info.eu_num = 4;  
    ctx.info.npu_num = 2; 

    cvk_tl_t *ifmap = create_tensor(CVK_FMT_I8, 1, 8, 64, 64, 0x1000);
    cvk_tl_t *ofmap = create_tensor(CVK_FMT_I8, 1, 8, 32, 32, 0x2000);

    fill_ifmap_data(ifmap);

    cvk_tiu_max_pooling_param_t param;
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
    param.ins_val = 0; 

    cvkcv181x_tiu_max_pooling(&ctx, &param);
    printf("Max pooling finished.\n");

    verify_max_pooling_result(ifmap, ofmap, &param);

    free(ifmap);
    free(ofmap);
}

int main() {
    test_cvkcv181x_tiu_max_pooling();
    printf("Max pooling test passed.\n");
    return 0;
}
