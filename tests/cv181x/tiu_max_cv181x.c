//test for cvkcv181x_tiu_max
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

uint8_t *g_lmem_base = NULL;

// Initialize tensor data
void init_tensor_data(cvk_tl_t *tensor, int8_t value) {
    if (tensor == NULL) {
        printf("Error: In init_tensor_data, tensor is NULL.\n");
        exit(1);
    }

    // Ensure start_address Within the scope of LMEM
    if (tensor->start_address >= CV181X_HW_LMEM_SIZE) { // LMEM is 32KB
        printf("Error: In init_tensor_data, start_address exceeds LMEM range.\n");
        exit(1);
    }

    printf("Initializing tensor data: start_address = %u\n", tensor->start_address);
    printf("Tensor shape: n=%u, c=%u, h=%u, w=%u\n", tensor->shape.n, tensor->shape.c, tensor->shape.h, tensor->shape.w);

    // Calculate the actual memory address
    uint32_t offset = tensor->start_address;
    int8_t *data = ((int8_t *)g_lmem_base) + offset;
    size_t size = tensor->shape.n * tensor->shape.c * tensor->shape.h * tensor->shape.w;

    printf("Initialization data size: %zu bytes\n", size);

    for (size_t i = 0; i < size; ++i) {
        data[i] = value;
    }
}

int main() {
    printf("Running test: cvkcv181x_tiu_max...\n");

    // Open /dev/mem to access physical memory
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("Failed to open /dev/mem");
        return -1;
    }

    // Map physical memory to user space
    g_lmem_base = mmap(NULL, CV181X_HW_LMEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, CV181X_HW_LMEM_START_ADDR);
    if (g_lmem_base == MAP_FAILED) {
        perror("mmap failed");
        close(mem_fd);
        return -1;
    }
    close(mem_fd);  // After mapping, the file descriptor can be closed.

    printf("Successfully mapped physical memory to user space: %p\n", (void*)g_lmem_base);

    // Initialize context
    cvk_context_t ctx;
    cvkcv181x_reset(&ctx);

    // Define tensor shape and format
    cvk_tl_shape_t tl_shape = { .n = 1, .c = 4, .h = 4, .w = 4 };
    cvk_fmt_t fmt = CVK_FMT_I8;
    int eu_align = 1;

    printf("a.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);
    printf("b.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);
    printf("max.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);

    //Create and assign tensors a, b and max
    cvk_tl_t *a = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (a == NULL) {
        printf("Failed to allocate tensor a.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    cvk_tl_t *b = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (b == NULL) {
        printf("Failed to allocate tensor b.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE);
        cvkcv181x_lmem_free_tensor(&ctx, a);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    cvk_tl_t *max = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (max == NULL) {
        printf("Failed to allocate tensor max.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE);
        cvkcv181x_lmem_free_tensor(&ctx, a);
        cvkcv181x_lmem_free_tensor(&ctx, b);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    // Print initial start_address
    printf("a.start_address: %u\n", a->start_address);
    printf("b.start_address: %u\n", b->start_address);
    printf("max.start_address: %u\n", max->start_address);

    // Initialize tensor data
    // Set all elements in 1
    init_tensor_data(a, 1);
    // Set all elements in 2
    init_tensor_data(b, 2);

    // configuration max operation parameters
    cvk_tiu_max_param_t max_param;
    memset(&max_param, 0, sizeof(cvk_tiu_max_param_t));

    max_param.max = max;
    max_param.a = a;
    max_param.b = b;
    max_param.b_is_const = 0;       // b Not a constant
    max_param.layer_id = 0;

    printf("Calling cvkcv181x_tiu_max...\n");
    // Call max function
    cvkcv181x_tiu_max(&ctx, &max_param);
    printf("Call successful\n");

    // Check results
    int8_t *max_data = ((int8_t *)g_lmem_base) + max->start_address;
    size_t total_size = max->shape.n * max->shape.c * max->shape.h * max->shape.w;

    printf("Verifying max results...\n");
    for (size_t i = 0; i < total_size; ++i) {
        printf("max_data[%zu] = %d\n", i, max_data[i]);
        assert(max_data[i] == 2);  // max(1, 2) should be equal to 2
    }
    printf("Max tensor test passed!\n");

    printf("Test passed!\n");

    // Release resources
    munmap(g_lmem_base, CV181X_HW_LMEM_SIZE); // Unmap memory
    cvkcv181x_lmem_free_tensor(&ctx, a);
    cvkcv181x_lmem_free_tensor(&ctx, b);
    cvkcv181x_lmem_free_tensor(&ctx, max);
    cvkcv181x_cleanup(&ctx);

    return 0;
}
