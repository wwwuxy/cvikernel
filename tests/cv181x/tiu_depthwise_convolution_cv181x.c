#include <stdio.h>
#include <assert.h>
#include "cvkcv181x.h"

// Mock for emitting the command buffer
void *emit_tiu_cmdbuf(cvk_context_t *ctx, tiu_reg_t *reg) {
    printf("Emitting command buffer for Depthwise Convolution:\n");
    printf("Command enabled: %d\n", reg->cmd_en);
    printf("Task type: %d\n", reg->tsk_typ);
    printf("Source Address (ifmap): 0x%lx\n", reg->opd0_addr);
    printf("Destination Address (ofmap): 0x%lx\n", reg->res0_addr);
    printf("Weight Address: 0x%lx\n", reg->opd1_addr);
    printf("Stride: %d x %d\n", reg->conv_op_x_str, reg->conv_op_y_str);
    printf("Layer Info: %d\n", reg->layer_info);
    return NULL;
}

// Mock for tensor checks (validates tensor address and size)
int check_tiu_tensor_2(cvk_tensor_t *a, cvk_tensor_t *b) {
    return (a->shape.n == b->shape.n && a->shape.c == b->shape.c &&
            a->shape.h == b->shape.h && a->shape.w == b->shape.w) ? 0 : -1;
}

int check_tiu_tensor_3(cvk_tensor_t *a, cvk_tensor_t *b, cvk_tensor_t *c) {
    return check_tiu_tensor_2(a, b) | check_tiu_tensor_2(a, c);
}

int check_stride_type_0(cvk_context_t *ctx, cvk_tensor_t *tensor) {
    return (tensor->stride.n > 0 && tensor->stride.c > 0 &&
            tensor->stride.h > 0 && tensor->stride.w > 0) ? 0 : -1;
}

int check_stride_type_2(cvk_context_t *ctx, cvk_tensor_t *tensor) {
    return (tensor->stride.n > 0 && tensor->stride.c > 0 &&
            tensor->stride.h == 0 && tensor->stride.w == 0) ? 0 : -1;
}

// Mock for tensor signedness check
int tensor_is_signed(cvk_tensor_t *tensor) {
    return tensor->fmt == CVK_FMT_INT8 ? 1 : 0;
}

// Test function for depthwise convolution
void test_cvkcv181x_tiu_depthwise_convolution() {
    cvk_context_t ctx;
    cvk_tiu_depthwise_convolution_param_t p;

    // Setup mock context
    ctx.info.lmem_size = 1024; // Local memory size
    ctx.info.eu_num = 8; // Number of execution units

    // Setup mock tensor shapes and data for ifmap, ofmap, weight
    p.ifmap = (cvk_tensor_t *)malloc(sizeof(cvk_tensor_t));
    p.ofmap = (cvk_tensor_t *)malloc(sizeof(cvk_tensor_t));
    p.weight = (cvk_tensor_t *)malloc(sizeof(cvk_tensor_t));

    // Mock shape values (example)
    p.ifmap->shape.n = 1;
    p.ifmap->shape.c = 16;
    p.ifmap->shape.h = 28;
    p.ifmap->shape.w = 28;

    p.ofmap->shape.n = 1;
    p.ofmap->shape.c = 16;
    p.ofmap->shape.h = 28;
    p.ofmap->shape.w = 28;

    p.weight->shape.n = 1;
    p.weight->shape.c = 16;
    p.weight->shape.h = 3;
    p.weight->shape.w = 3;

    // Mock tensor start addresses
    p.ifmap->start_address = 0x1000;
    p.ofmap->start_address = 0x2000;
    p.weight->start_address = 0x3000;

    // Mock strides (for simplicity)
    p.ifmap->stride.n = 16;
    p.ifmap->stride.c = 16;
    p.ifmap->stride.h = 28;
    p.ifmap->stride.w = 28;

    p.ofmap->stride.n = 16;
    p.ofmap->stride.c = 16;
    p.ofmap->stride.h = 28;
    p.ofmap->stride.w = 28;

    p.weight->stride.n = 16;
    p.weight->stride.c = 16;
    p.weight->stride.h = 3;
    p.weight->stride.w = 3;

    // Other parameters for convolution
    p.relu_enable = 1;
    p.stride_h = 1;
    p.stride_w = 1;
    p.pad_top = 0;
    p.pad_bottom = 0;
    p.pad_left = 0;
    p.pad_right = 0;
    p.ins_h = 1;
    p.ins_last_h = 1;
    p.ins_w = 1;
    p.ins_last_w = 1;
    p.dilation_h = 1;
    p.dilation_w = 1;
    p.chl_quan_param = (cvk_tensor_t *)malloc(sizeof(cvk_tensor_t));
    p.chl_quan_param->start_address = 0x4000;
    p.chl_quan_param->shape.n = 1;
    p.chl_quan_param->shape.c = 16;
    p.chl_quan_param->shape.h = 1;
    p.chl_quan_param->shape.w = 1;

    p.has_bias = 1;
    p.layer_id = 1;
    p.cmd_pre_exe_typ = 0;
    p.cmd_pre_exe = 0;

    // Call depthwise convolution function
    cvkcv181x_tiu_depthwise_convolution(&ctx, &p);

    // Verify that the function emitted the expected command buffer
    // In this test, we mainly print out the emitted command buffer values
    printf("Test complete.\n");

    // Clean up
    free(p.ifmap);
    free(p.ofmap);
    free(p.weight);
    free(p.chl_quan_param);
}

int main() {
    test_cvkcv181x_tiu_depthwise_convolution();
    return 0;
}

