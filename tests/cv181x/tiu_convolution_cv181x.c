#include <stdio.h>
#include <assert.h>
#include "cvkcv181x.h"

// Mock for checking if tensors are signed
int tensor_is_signed(cvk_tensor_t *tensor) {
    // Return a random value for the sake of testing
    return 0; // assuming unsigned for simplicity
}

// Mock for emitting the command buffer (can be extended for actual verification)
void *emit_tiu_cmdbuf(cvk_context_t *ctx, tiu_reg_t *reg) {
    printf("Emitting command buffer:\n");
    printf("Command enabled: %d\n", reg->cmd_en);
    printf("Task type: %d\n", reg->tsk_typ);
    return NULL; // No real action, just a placeholder
}

// Test function for convolution
void test_cvkcv181x_tiu_convolution() {
    cvk_context_t ctx;
    param_t p;

    // Setup mock context (mocking values for the sake of testing)
    ctx.info.lmem_size = 1024; // Mock value for local memory size
    ctx.info.eu_num = 8; // Mock value for the number of execution units

    // Setup mock tensor shapes and data for ifmap, ofmap, and weight
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
    p.ofmap->shape.h = 14;
    p.ofmap->shape.w = 14;

    p.weight->shape.n = 16;
    p.weight->shape.c = 16;
    p.weight->shape.h = 3;
    p.weight->shape.w = 3;

    // Mock tensor start addresses (aligned to some boundary for simplicity)
    p.ifmap->start_address = 0x1000;
    p.ofmap->start_address = 0x2000;
    p.weight->start_address = 0x3000;

    // Other parameters
    p.stride_h = 1;
    p.stride_w = 1;
    p.pad_top = 0;
    p.pad_bottom = 0;
    p.pad_left = 0;
    p.pad_right = 0;
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

    // Call the convolution function
    cvkcv181x_tiu_convolution(&ctx, &p);

    // Verify that the function emitted the expected command buffer
    // You could expand this test by validating the contents of the generated register
    // (by checking the `reg` content before calling `emit_tiu_cmdbuf`).
    printf("Test complete.\n");

    // Clean up
    free(p.ifmap);
    free(p.ofmap);
    free(p.weight);
}

int main() {
    test_cvkcv181x_tiu_convolution();
    return 0;
}

