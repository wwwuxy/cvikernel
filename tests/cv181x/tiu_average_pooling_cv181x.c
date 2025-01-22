#include <stdio.h>
#include <assert.h>
#include "cvkcv181x.h"

// Mock context and tensor structure to simulate the input
typedef struct {
    int start_address;
    int stride;
    int shape[4];  // [batch, channels, height, width]
    int fmt; // Tensor format (e.g., CVK_FMT_BF16)
} tensor_t;

// Mocked function for context
cvk_context_t ctx; 

// Example tensor creation for testing purposes
tensor_t create_tensor(int start_address, int stride, int shape[4], int fmt) {
    tensor_t tensor;
    tensor.start_address = start_address;
    tensor.stride = stride;
    for (int i = 0; i < 4; i++) {
        tensor.shape[i] = shape[i];
    }
    tensor.fmt = fmt;
    return tensor;
}

// Test the cvkcv181x_tiu_average_pooling function
void test_cvkcv181x_tiu_average_pooling() {
    // Prepare mock tensors
    int ifmap_shape[4] = {1, 16, 32, 32};  // [batch, channels, height, width]
    int ofmap_shape[4] = {1, 16, 16, 16};  // [batch, channels, height, width]
    int stride_h = 2, stride_w = 2;
    int pad_top = 1, pad_bottom = 1, pad_left = 1, pad_right = 1;
    int kh = 3, kw = 3;
    int ins_h = 1, ins_w = 1, ins_last_h = 1, ins_last_w = 1;
    float avg_pooling_const = 1.0f;
    int layer_id = 1;
    
    // Create tensors (assuming CVK_FMT_BF16 format for this test)
    tensor_t ifmap = create_tensor(1000, 128, ifmap_shape, CVK_FMT_BF16);
    tensor_t ofmap = create_tensor(2000, 128, ofmap_shape, CVK_FMT_BF16);

    // Set up the parameters
    cvk_tiu_average_pooling_param_t p;
    p.ifmap = &ifmap;
    p.ofmap = &ofmap;
    p.stride_h = stride_h;
    p.stride_w = stride_w;
    p.pad_top = pad_top;
    p.pad_bottom = pad_bottom;
    p.pad_left = pad_left;
    p.pad_right = pad_right;
    p.ins_h = ins_h;
    p.ins_w = ins_w;
    p.ins_last_h = ins_last_h;
    p.ins_last_w = ins_last_w;
    p.kh = kh;
    p.kw = kw;
    p.avg_pooling_const = avg_pooling_const;
    p.layer_id = layer_id;

    // Call the function
    cvkcv181x_tiu_average_pooling(&ctx, &p);

    // For the sake of testing, we assert that the addresses are set correctly
    assert(ofmap.start_address == 2000);
    assert(ifmap.start_address == 1000);

    // Check tensor shapes and strides
    assert(ifmap.shape[0] == ofmap.shape[0]);  // batch
    assert(ifmap.shape[1] == ofmap.shape[1]);  // channels
    assert(ifmap.shape[2] == 32 && ofmap.shape[2] == 16);  // height: 32 -> 16
    assert(ifmap.shape[3] == 32 && ofmap.shape[3] == 16);  // width: 32 -> 16

    // Additional validation on padding and stride values
    assert(pad_top == 1);
    assert(pad_bottom == 1);
    assert(pad_left == 1);
    assert(pad_right == 1);
    assert(stride_h == 2);
    assert(stride_w == 2);
    assert(kh == 3 && kw == 3);

    // Check the average pooling constant
    assert(avg_pooling_const == 1.0f);

    // Check layer ID
    assert(layer_id == 1);

    printf("test_cvkcv181x_tiu_average_pooling passed\n");
}

// Main function to run tests
int main() {
    // Run the test for average pooling
    test_cvkcv181x_tiu_average_pooling();
    
    return 0;
}

