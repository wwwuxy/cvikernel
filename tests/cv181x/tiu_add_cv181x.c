//test for cvkcv181x_tiu_add
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "../src/cv181x/cvkcv181x.h"
#include "../include/cvikernel/cvikernel.h"
#include "../include/cvikernel/cv181x/cv181x_tpu_cfg.h"  // Include hardware configuration macro definitions

uint8_t *g_lmem_base = NULL;

// Initialize tensor data
void init_tensor_data(cvk_tl_t *tensor, int8_t value) {
    if (tensor == NULL) {
        printf("Error: In init_tensor_data, tensor is NULL.\n");
        exit(1);
    }

    // Ensure start_address is within LMEM range
    if (tensor->start_address >= CV181X_HW_LMEM_SIZE ) { // LMEM size is 32KB
        printf("Error: In init_tensor_data, start_address exceeds LMEM range.\n");
        exit(1);
    }

    printf("Initializing tensor data: start_address = %u\n", tensor->start_address);
    printf("Tensor shape: n=%u, c=%u, h=%u, w=%u\n", tensor->shape.n, tensor->shape.c, tensor->shape.h, tensor->shape.w);

    // Calculate actual memory address
    uint32_t offset = tensor->start_address;
    int8_t *data = ((int8_t *)g_lmem_base) + offset;
    size_t size = tensor->shape.n * tensor->shape.c * tensor->shape.h * tensor->shape.w;

    printf("Initialization data size: %zu bytes\n", size);

    for (size_t i = 0; i < size; ++i) {
        data[i] = value;
    }
}

int main() {
    printf("Running test: cvkcv181x_tiu_add...\n");

    // Open /dev/mem to access physical memory
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("Failed to open /dev/mem");
        return -1;
    }

    // Map physical memory to user space
    g_lmem_base = mmap(NULL, CV181X_HW_LMEM_SIZE , PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, CV181X_HW_LMEM_START_ADDR);
    if (g_lmem_base == MAP_FAILED) {
        perror("mmap failed");
        close(mem_fd);
        return -1;
    }
    close(mem_fd);  // Can close the file descriptor after mapping

    printf("Successfully mapped physical memory to user space: %p\n", (void*)g_lmem_base);

    // Initialize context
    cvk_context_t ctx;
    cvkcv181x_reset(&ctx);  // Use the custom initialization function

    // Define tensor shape and format
    cvk_tl_shape_t tl_shape = { .n = 1, .c = 4, .h = 4, .w = 4 };
    cvk_fmt_t fmt = CVK_FMT_I8;
    int eu_align = 1;

    printf("a_low.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);

    // Create and allocate tensors a_low, b_low, and res_low
    cvk_tl_t *a_low = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (a_low == NULL) {
        printf("Failed to allocate tensor a_low.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE );
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    cvk_tl_t *b_low = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (b_low == NULL) {
        printf("Failed to allocate tensor b_low.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE );
        cvkcv181x_lmem_free_tensor(&ctx,a_low);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    cvk_tl_t *res_low = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (res_low == NULL) {
        printf("Failed to allocate tensor res_low.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE );
        cvkcv181x_lmem_free_tensor(&ctx,a_low);
        cvkcv181x_lmem_free_tensor(&ctx,b_low);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    // Print initial start_address
    printf("a_low.start_address: %u\n", a_low->start_address);
    printf("b_low.start_address: %u\n", b_low->start_address);
    printf("res_low.start_address: %u\n", res_low->start_address);

    // Initialize tensor data
    init_tensor_data(a_low, 1);  // Initialize all elements of a_low to 1
    init_tensor_data(b_low, 2);  // Initialize all elements of b_low to 2

    // **Define high part tensors**
    cvk_tl_t *a_high = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (a_high == NULL) {
        printf("Failed to allocate tensor a_high.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE );
        cvkcv181x_lmem_free_tensor(&ctx,a_low);
        cvkcv181x_lmem_free_tensor(&ctx,b_low);
        cvkcv181x_lmem_free_tensor(&ctx,res_low);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    cvk_tl_t *b_high = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (b_high == NULL) {
        printf("Failed to allocate tensor b_high.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE );
        cvkcv181x_lmem_free_tensor(&ctx,a_low);
        cvkcv181x_lmem_free_tensor(&ctx,b_low);
        cvkcv181x_lmem_free_tensor(&ctx,res_low);
        cvkcv181x_lmem_free_tensor(&ctx,a_high);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    cvk_tl_t *res_high = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (res_high == NULL) {
        printf("Failed to allocate tensor res_high.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE );
        cvkcv181x_lmem_free_tensor(&ctx,a_low);
        cvkcv181x_lmem_free_tensor(&ctx,b_low);
        cvkcv181x_lmem_free_tensor(&ctx,res_low);
        cvkcv181x_lmem_free_tensor(&ctx,a_high);
        cvkcv181x_lmem_free_tensor(&ctx,b_high);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    // Initialize high part data
    init_tensor_data(a_high, 1);  // Example initialization
    init_tensor_data(b_high, 2);  // Example initialization

    // Configure addition operation parameters
    cvk_tiu_add_param_t add_param;
    memset(&add_param, 0, sizeof(cvk_tiu_add_param_t));

    add_param.a_low = a_low;
    add_param.a_high = a_high;      // Set high part
    add_param.b.low = b_low;
    add_param.b.high = b_high;      // Set high part
    add_param.b_is_const = 0;       // b is not a constant
    add_param.res_low = res_low;
    add_param.res_high = res_high;  // Set high part
    add_param.relu_enable = 0;
    add_param.rshift_bits = 0;
    add_param.layer_id = 0;

    printf("Calling cvkcv181x_tiu_add...\n");
    // Call addition function
    cvkcv181x_tiu_add(&ctx, &add_param);
    printf("Call successful\n");

    // Check results
    int8_t *res_data = ((int8_t *)g_lmem_base) + res_low->start_address;
    size_t total_size = res_low->shape.n * res_low->shape.c * res_low->shape.h * res_low->shape.w;
    for (size_t i = 0; i < total_size; ++i) {
        printf("res_data[%zu] = %d\n", i, res_data[i]);
        assert(res_data[i] == 3);  // 1 + 2 should equal 3
    }

    // Check res_high data
    int8_t *res_high_data = ((int8_t *)g_lmem_base) + res_high->start_address;
    size_t high_total_size = res_high->shape.n * res_high->shape.c * res_high->shape.h * res_high->shape.w;
    for (size_t i = 0; i < high_total_size; ++i) {
        printf("res_high_data[%zu] = %d\n", i, res_high_data[i]);
        assert(res_high_data[i] == 3);  // 1 + 2 should equal 3
    }
    printf("res_high test passed!\n");

    printf("Test passed!\n");

    // Free resources
    munmap(g_lmem_base, CV181X_HW_LMEM_SIZE ); // Unmap memory
    cvkcv181x_lmem_free_tensor(&ctx,a_low);
    cvkcv181x_lmem_free_tensor(&ctx,b_low);
    cvkcv181x_lmem_free_tensor(&ctx,res_low);
    cvkcv181x_lmem_free_tensor(&ctx,a_high);
    cvkcv181x_lmem_free_tensor(&ctx,b_high);
    cvkcv181x_lmem_free_tensor(&ctx,res_high);
    cvkcv181x_cleanup(&ctx);

    return 0;
}
