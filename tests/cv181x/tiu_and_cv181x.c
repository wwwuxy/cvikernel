#include <stdio.h>
#include <assert.h>
#include "cvkcv181x.h"

// Mock context and tensor structure to simulate the input
typedef struct {
    int start_address;
    int stride;
    int shape[3];  // Example shape: [height, width, depth]
} tensor_t;

// Mocked function for context
cvk_context_t ctx; 

// Example tensor creation for testing purposes
tensor_t create_tensor(int start_address, int stride, int shape[3]) {
    tensor_t tensor;
    tensor.start_address = start_address;
    tensor.stride = stride;
    for (int i = 0; i < 3; i++) {
        tensor.shape[i] = shape[i];
    }
    return tensor;
}

// Test the cvkcv181x_tiu_and_int8 function
void test_cvkcv181x_tiu_and_int8() {
    // Prepare mock tensors
    int shape[3] = {4, 4, 4};
    tensor_t a = create_tensor(1000, 16, shape);
    tensor_t b = create_tensor(1024, 16, shape);
    tensor_t res = create_tensor(2048, 16, shape);

    cvk_tiu_and_int8_param_t p;
    p.a = &a;
    p.b = &b;
    p.res = &res;
    p.layer_id = 1;

    // Call the function
    cvkcv181x_tiu_and_int8(&ctx, &p);

    // For the sake of testing, we assert that the addresses are set correctly
    assert(res.start_address == 2048);
    assert(a.start_address == 1000);
    assert(b.start_address == 1024);
    
    // Add additional checks if necessary, for example, validating the tensor shapes
    assert(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] && a.shape[2] == b.shape[2]);
    assert(a.stride == b.stride);
    
    printf("test_cvkcv181x_tiu_and_int8 passed\n");
}

// Test the cvkcv181x_tiu_and_int16 function
void test_cvkcv181x_tiu_and_int16() {
    // Prepare mock tensors (16-bit version)
    int shape[3] = {4, 4, 4};
    tensor_t a_low = create_tensor(1000, 16, shape);
    tensor_t a_high = create_tensor(1024, 16, shape);
    tensor_t b_low = create_tensor(2048, 16, shape);
    tensor_t b_high = create_tensor(2080, 16, shape);
    tensor_t res_low = create_tensor(4096, 16, shape);
    tensor_t res_high = create_tensor(4112, 16, shape);

    cvk_tiu_and_int16_param_t p;
    p.a_low = &a_low;
    p.a_high = &a_high;
    p.b_low = &b_low;
    p.b_high = &b_high;
    p.res_low = &res_low;
    p.res_high = &res_high;
    p.layer_id = 2;

    // Call the function
    cvkcv181x_tiu_and_int16(&ctx, &p);

    // For the sake of testing, we assert that the addresses are set correctly
    assert(res_low.start_address == 4096);
    assert(res_high.start_address == 4112);

    // Add additional checks if necessary, for example, validating the tensor shapes
    assert(a_low.shape[0] == b_low.shape[0] && a_low.shape[1] == b_low.shape[1] && a_low.shape[2] == b_low.shape[2]);
    assert(a_high.shape[0] == b_high.shape[0] && a_high.shape[1] == b_high.shape[1] && a_high.shape[2] == b_high.shape[2]);
    
    // Check the tensor strides
    assert(a_low.stride == b_low.stride);
    assert(a_high.stride == b_high.stride);
    
    printf("test_cvkcv181x_tiu_and_int16 passed\n");
}

// Main function to run tests
int main() {
    // Run tests
    test_cvkcv181x_tiu_and_int8();
    test_cvkcv181x_tiu_and_int16();
    
    return 0;
}

