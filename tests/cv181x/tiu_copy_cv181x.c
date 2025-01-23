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

static uint8_t* g_src_buffer = NULL;
static uint8_t* g_dst_buffer = NULL;

// Mock: Checks whether two tensors have the same shape.
int check_same_shape(const cvk_tl_t *dst, const cvk_tl_t *src) {
    return (dst->shape.n == src->shape.n &&
            dst->shape.c == src->shape.c &&
            dst->shape.h == src->shape.h &&
            dst->shape.w == src->shape.w) ? 0 : -1;
}

// Mock: Checks if tensor stride values are within a valid range.
int check_stride_range(cvk_tl_stride_t stride) {
    return (stride.n > 0 && stride.c > 0 && stride.h > 0 && stride.w > 0) ? 0 : -1;
}

// Calculates the total number of elements in the tensor.
static size_t tensor_element_count(const cvk_tl_t* t) {
    return (size_t)t->shape.n * t->shape.c * t->shape.h * t->shape.w;
}

// Computes the memory size in bytes for a given tensor.
static size_t tensor_byte_size(const cvk_tl_t* t) {
    if (t->fmt == CVK_FMT_BF16) {
        return tensor_element_count(t) * 2;
    }
    // For other formats, add the respective calculation...
    return 0;
}

void test_cvkcv181x_tiu_copy() {
    cvk_context_t ctx;
    cvk_tiu_copy_param_t p;

    // Initialize mock context
    ctx.info.lmem_size = 1024; // Mock local memory size
    ctx.info.eu_num = 8;       // Mock number of EUs

    cvk_tl_t* src_tl = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    cvk_tl_t* dst_tl = (cvk_tl_t *)malloc(sizeof(cvk_tl_t));
    assert(src_tl && dst_tl);

    // Set shapes
    src_tl->shape.n = 1;
    src_tl->shape.c = 16;
    src_tl->shape.h = 28;
    src_tl->shape.w = 28;

    dst_tl->shape.n = 1;
    dst_tl->shape.c = 16;
    dst_tl->shape.h = 28;
    dst_tl->shape.w = 28;

    // Set strides
    src_tl->stride.n = 16;
    src_tl->stride.c = 16;
    src_tl->stride.h = 28;
    src_tl->stride.w = 28;

    dst_tl->stride.n = 16;
    dst_tl->stride.c = 16;
    dst_tl->stride.h = 28;
    dst_tl->stride.w = 28;

    // Set format
    src_tl->fmt = CVK_FMT_BF16;
    dst_tl->fmt = CVK_FMT_BF16;

    p.src = src_tl;
    p.dst = dst_tl;
    p.layer_id = 1;

    // Allocate actual storage buffers (simulating hardware memory)
    size_t src_size = tensor_byte_size(src_tl);
    size_t dst_size = tensor_byte_size(dst_tl);

    // Allocate and zero the buffers
    g_src_buffer = (uint8_t*)malloc(src_size);
    g_dst_buffer = (uint8_t*)malloc(dst_size);
    memset(g_src_buffer, 0, src_size);
    memset(g_dst_buffer, 0, dst_size);

    // Assign start addresses (simulate hardware local memory or address offsets).
    src_tl->start_address = (uint64_t)(uintptr_t)g_src_buffer;
    dst_tl->start_address = (uint64_t)(uintptr_t)g_dst_buffer;

    // Fill the src buffer with known data for verification
    for (size_t i = 0; i < src_size; i++) {
        g_src_buffer[i] = (uint8_t)(i & 0xFF);
    }

    // Call the copy function
    cvkcv181x_tiu_copy(&ctx, &p);

    // Compare the destination buffer and source buffer to check if they match.
    if (memcmp(g_src_buffer, g_dst_buffer, src_size) == 0) {
        printf("Tensor copy test PASS: destination data matches source.\n");
    } else {
        printf("Tensor copy test FAIL: destination data does NOT match source.\n");
    }

    // Cleanup
    printf("Test complete.\n");
    free(src_tl);
    free(dst_tl);
    free(g_src_buffer);
    free(g_dst_buffer);
    g_src_buffer = NULL;
    g_dst_buffer = NULL;
}

int main() {
    test_cvkcv181x_tiu_copy();
    return 0;
}
