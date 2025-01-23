//test for cvkcv181x_tiu_mac
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

// Global pointer for accessing mapped local memory
uint8_t *g_lmem_base = NULL;

// Initialize tensor data and set all elements to the specified value
void init_tensor_data(cvk_tl_t *tensor, int8_t value) {
    if (tensor == NULL) {
        printf("Error: In init_tensor_data, tensor is NULL.\n");
        exit(1);
    }

    // Ensure start_address is within the LMEM range
    if (tensor->start_address >= CV181X_HW_LMEM_SIZE) { // 32KB
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
    printf("Running test: cvkcv181x_tiu_mac...\n");

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
    cvkcv181x_reset(&ctx);  // Use custom initialization function

    // Define tensor shape and format
    cvk_tl_shape_t tl_shape = { .n = 1, .c = 4, .h = 4, .w = 4 };
    cvk_fmt_t fmt = CVK_FMT_I8; // Can be changed to CVK_FMT_BF16 as needed
    int eu_align = 1;

    printf("a.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);
    printf("b.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);
    printf("res_low.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);
    printf("res_high.shape: %u %u %u %u\n", tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);

    // Create and assign tensors a, b, res_low and res_high
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

    cvk_tl_t *res_low = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
    if (res_low == NULL) {
        printf("Failed to allocate tensor res_low.\n");
        munmap(g_lmem_base, CV181X_HW_LMEM_SIZE);
        cvkcv181x_lmem_free_tensor(&ctx, a);
        cvkcv181x_lmem_free_tensor(&ctx, b);
        cvkcv181x_cleanup(&ctx);
        return -1;
    }

    cvk_tl_t *res_high = NULL;
    if (fmt != CVK_FMT_BF16) { // If not BF16, res_high is required
        res_high = cvkcv181x_lmem_alloc_tensor(&ctx, tl_shape, fmt, eu_align);
        if (res_high == NULL) {
            printf("Failed to allocate tensor res_high.\n");
            munmap(g_lmem_base, CV181X_HW_LMEM_SIZE);
            cvkcv181x_lmem_free_tensor(&ctx, a);
            cvkcv181x_lmem_free_tensor(&ctx, b);
            cvkcv181x_lmem_free_tensor(&ctx, res_low);
            cvkcv181x_cleanup(&ctx);
            return -1;
        }
    }
    // Print initial start_address
    printf("a.start_address: %u\n", a->start_address);
    printf("b.start_address: %u\n", b->start_address);
    printf("res_low.start_address: %u\n", res_low->start_address);
    if (res_high != NULL) {
        printf("res_high.start_address: %u\n", res_high->start_address);
    }

    // Initialize tensor data
    // Set all elements in  2
    init_tensor_data(a, 2);
    // Set all elements in 3
    init_tensor_data(b, 3);
    // Initialization res_low is 0
    init_tensor_data(res_low, 0);
    if (res_high != NULL) {
        init_tensor_data(res_high, 0);
    }

    // Configure macOS operation parameters
    cvk_tiu_mac_param_t mac_param;
    memset(&mac_param, 0, sizeof(cvk_tiu_mac_param_t));

    mac_param.a = a;
    mac_param.b = b;
    mac_param.res_low = res_low;
    mac_param.res_high = res_high;  // Optional, set to NULL to test the case with only res_low
    mac_param.b_is_const = 0;       // b Not a constant
    mac_param.relu_enable = 0;       // Disabled ReLU
    mac_param.lshift_bits = 0;       // No left shift
    mac_param.rshift_bits = 0;       // Do not shift right
    mac_param.res_is_int8 = (fmt != CVK_FMT_BF16) ? 1 : 0; // Set res_is_int8 according to the format
    mac_param.layer_id = 0;

    printf("Calling cvkcv181x_tiu_mac...\n");
    // Call mac function
    cvkcv181x_tiu_mac(&ctx, &mac_param);
    printf("Call successful\n");

    // Check res_low result
    int8_t *res_low_data = ((int8_t *)g_lmem_base) + res_low->start_address;
    size_t total_size = res_low->shape.n * res_low->shape.c * res_low->shape.h * res_low->shape.w;

    printf("Verifying res_low results...\n");
    for (size_t i = 0; i < total_size; ++i) {
        printf("res_low_data[%zu] = %d\n", i, res_low_data[i]);
        assert(res_low_data[i] == 6);  // 2 * 3 * 1 should be equal to 6
    }
    printf("res_low tensor test passed!\n");

    // Check res_high result
    if (res_high != NULL) {
        int8_t *res_high_data = ((int8_t *)g_lmem_base) + res_high->start_address;
        size_t high_total_size = res_high->shape.n * res_high->shape.c * res_high->shape.h * res_high->shape.w;

        printf("Verifying res_high results...\n");
        for (size_t i = 0; i < high_total_size; ++i) {
            printf("res_high_data[%zu] = %d\n", i, res_high_data[i]);
            assert(res_high_data[i] == 6);  // 2 * 3 * 1 should be equal to 6
        }
        printf("res_high tensor test passed!\n");
    }

    printf("Test passed!\n");

    // Release resources
    munmap(g_lmem_base, CV181X_HW_LMEM_SIZE); // Unmap memory
    cvkcv181x_lmem_free_tensor(&ctx, a);
    cvkcv181x_lmem_free_tensor(&ctx, b);
    cvkcv181x_lmem_free_tensor(&ctx, res_low);
    if (res_high != NULL) {
        cvkcv181x_lmem_free_tensor(&ctx, res_high);
    }
    cvkcv181x_cleanup(&ctx);

    return 0;
}
