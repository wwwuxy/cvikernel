#include <stdio.h>
#include <assert.h>
#include "cvkcv181x.h"

// Mocked tensor structures and utility functions
cvk_tensor_t* create_tensor(cvk_fmt_t fmt, int n, int c, int h, int w, int start_address) {
    cvk_tensor_t* tensor = (cvk_tensor_t*)malloc(sizeof(cvk_tensor_t));
    tensor->fmt = fmt;
    tensor->shape.n = n;
    tensor->shape.c = c;
    tensor->shape.h = h;
    tensor->shape.w = w;
    tensor->start_address = start_address;
    return tensor;
}

void reset_tiu_reg(tiu_reg_t* reg) {
    // Initialize or reset the register structure
    memset(reg, 0, sizeof(tiu_reg_t));
}

void test_cvkcv181x_tiu_min_pooling() {
    // Create context
    cvk_context_t ctx;
    ctx.info.eu_num = 4;  // Mock EU number
    ctx.info.npu_num = 2; // Mock NPU number

    // Set up the parameters for the pooling operation
    cvk_tensor_t *ifmap = create_tensor(CVK_FMT_I8, 1, 8, 64, 64, 0x1000);
    cvk_tensor_t *ofmap = create_tensor(CVK_FMT_I8, 1, 8, 32, 32, 0x2000);
    
    cvk_tiu_min_pooling_param_t param;
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
    param.ins_fp = 0.0f;

    // Call the min pooling function
    cvkcv181x_tiu_min_pooling(&ctx, &param);
}

int main() {
    test_cvkcv181x_tiu_min_pooling();
    printf("Min pooling test passed.\n");
    return 0;
}

