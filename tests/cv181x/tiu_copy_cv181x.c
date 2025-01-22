#include <stdio.h>
#include <assert.h>
#include "cvkcv181x.h"

// Mock for emitting the command buffer (can be extended for actual verification)
void *emit_tiu_cmdbuf(cvk_context_t *ctx, tiu_reg_t *reg) {
    printf("Emitting command buffer for Tensor Copy:\n");
    printf("Command enabled: %d\n", reg->cmd_en);
    printf("Task type: %d\n", reg->tsk_typ);
    printf("Source Address: 0x%lx\n", reg->opd0_addr);
    printf("Destination Address: 0x%lx\n", reg->res0_addr);
    printf("Layer Info: %d\n", reg->layer_info);
    return NULL; // No real action, just a placeholder
}

// Mock for checking tensor shapes (validates the dimensions are the same)
int check_same_shape(cvk_tensor_t *dst, cvk_tensor_t *src) {
    return (dst->shape.n == src->shape.n &&
            dst->shape.c == src->shape.c &&
            dst->shape.h == src->shape.h &&
            dst->shape.w == src->shape.w) ? 0 : -1;
}

// Mock for checking tensor strides (validates the strides are within the range)
int check_stride_range(cvk_tensor_stride_t stride) {
    return (stride.n > 0 && stride.c > 0 && stride.h > 0 && stride.w > 0) ? 0 : -1;
}

// Test function for tensor copy
void test_cvkcv181x_tiu_copy() {
    cvk_context_t ctx;
    cvk_tiu_copy_param_t p;

    // Setup mock context (mocking values for the sake of testing)
    ctx.info.lmem_size = 1024; // Mock value for local memory size
    ctx.info.eu_num = 8; // Mock value for the number of execution units

    // Setup mock tensor shapes and data for src and dst
    p.src = (cvk_tensor_t *)malloc(sizeof(cvk_tensor_t));
    p.dst = (cvk_tensor_t *)malloc(sizeof(cvk_tensor_t));

    // Mock shape values (example)
    p.src->shape.n = 1;
    p.src->shape.c = 16;
    p.src->shape.h = 28;
    p.src->shape.w = 28;

    p.dst->shape.n = 1;
    p.dst->shape.c = 16;
    p.dst->shape.h = 28;
    p.dst->shape.w = 28;

    // Mock tensor start addresses (aligned to some boundary for simplicity)
    p.src->start_address = 0x1000;
    p.dst->start_address = 0x2000;

    // Mock stride values
    p.src->stride.n = 16;
    p.src->stride.c = 16;
    p.src->stride.h = 28;
    p.src->stride.w = 28;

    p.dst->stride.n = 16;
    p.dst->stride.c = 16;
    p.dst->stride.h = 28;
    p.dst->stride.w = 28;

    // Mock format and layer ID
    p.src->fmt = CVK_FMT_BF16;
    p.dst->fmt = CVK_FMT_BF16;
    p.layer_id = 1;

    // Call the copy function
    cvkcv181x_tiu_copy(&ctx, &p);

    // Verify that the function emitted the expected command buffer
    // You could expand this test by validating the contents of the generated register
    // (by checking the `reg` content before calling `emit_tiu_cmdbuf`).
    printf("Test complete.\n");

    // Clean up
    free(p.src);
    free(p.dst);
}

int main() {
    test_cvkcv181x_tiu_copy();
    return 0;
}

