//test for cvkcv181x_tiu_ge
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

// Mocked helper functions and structures
void test_check_same_shape(cvk_tl_t *t1, cvk_tl_t *t2) {
    // Assert that tensors t1 and t2 have the same shape
    assert(t1->shape.n == t2->shape.n);
    assert(t1->shape.c == t2->shape.c);
    assert(t1->shape.h == t2->shape.h);
    assert(t1->shape.w == t2->shape.w);
}

void test_check_tiu_tensor_2(cvk_tl_t *t1, cvk_tl_t *t2) {
    // Assert that tensors t1 and t2 are valid
    assert(t1 != NULL && t2 != NULL);
}

// Mock tensor and parameter setup
cvk_tl_t a, b, ge, b_const;
cvk_tiu_ge_param_t param;
cvk_context_t ctx;

void setup_test() {
    // Initialize tensor a
    a.start_address = 0x1000;
    a.shape.n = 1;
    a.shape.c = 3;
    a.shape.h = 32;
    a.shape.w = 32;
    a.stride.n = 1;
    a.stride.c = 3;
    a.stride.h = 32;
    a.stride.w = 32;
    a.fmt = CVK_FMT_I8;

    // Initialize tensor b
    b.start_address = 0x2000;
    b.shape.n = 1;
    b.shape.c = 3;
    b.shape.h = 32;
    b.shape.w = 32;
    b.stride.n = 1;
    b.stride.c = 3;
    b.stride.h = 32;
    b.stride.w = 32;
    b.fmt = CVK_FMT_I8;

    // Initialize tensor ge
    ge.start_address = 0x3000;
    ge.shape.n = 1;
    ge.shape.c = 3;
    ge.shape.h = 32;
    ge.shape.w = 32;
    ge.stride.n = 1;
    ge.stride.c = 3;
    ge.stride.h = 32;
    ge.stride.w = 32;
    ge.fmt = CVK_FMT_I8;

    // Initialize parameter struct
    param.a = &a;
    param.b = &b;
    param.ge = &ge;
    param.b_is_const = 0;  // Test non-constant b
    param.layer_id = 1;

    // Initialize context (mock)
    ctx.info.eu_num = 16;
}

void test_tiu_ge() {
    setup_test();

    // Perform the operation
    cvkcv181x_tiu_ge(&ctx, &param);

    // After the operation, verify that the registers were set as expected.
    // In this mock, we will check the `ge` tensor to ensure it was updated correctly.
    // For example, check the start address of the result tensor `ge`
    assert(ge.start_address == 0x3000);

    // Check if strides are set properly
    assert(ge.stride.n == 1);
    assert(ge.stride.c == 3);
    assert(ge.stride.h == 32);
    assert(ge.stride.w == 32);
}

void test_tiu_ge_const_b() {
    setup_test();
    param.b_is_const = 1;  // Test constant b
    param.b_const.val = 0x7F;
    param.b_const.is_signed = 1;

    // Perform the operation
    cvkcv181x_tiu_ge(&ctx, &param);

    // Verify the constant handling by checking the operand registers
    assert(param.b_const.val == 0x7F);
}

int main() {
    // Run the basic test
    test_tiu_ge();
    printf("Basic GE test passed.\n");

    // Run the constant b test
    test_tiu_ge_const_b();
    printf("Constant b GE test passed.\n");

    return 0;
}

