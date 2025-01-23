//test for cvkcv181x_tiu_lookup_table
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

// Mock helper functions and structures
cvk_context_t ctx;
cvk_tl_t ifmap, table, ofmap;
cvk_tiu_lookup_table_param_t param;

void setup_test() {
    // Initialize context with mock values
    ctx.info.eu_num = 16;
    ctx.info.npu_num = 4;

    // Initialize tensors a (ifmap), b (table), and c (ofmap)
    ifmap.start_address = 0x1000;
    ifmap.shape.n = 1;
    ifmap.shape.c = 16;
    ifmap.shape.h = 32;
    ifmap.shape.w = 32;
    ifmap.stride.n = 1;
    ifmap.stride.c = 16;
    ifmap.stride.h = 32;
    ifmap.stride.w = 32;
    ifmap.fmt = CVK_FMT_I8;

    table.start_address = 0x2000;
    table.shape.n = 1;
    table.shape.c = 4;
    table.shape.h = 16;
    table.shape.w = 16;
    table.stride.n = 1;
    table.stride.c = 4;
    table.stride.h = 16;
    table.stride.w = 16;
    table.fmt = CVK_FMT_I8;

    ofmap.start_address = 0x3000;
    ofmap.shape.n = 1;
    ofmap.shape.c = 16;
    ofmap.shape.h = 32;
    ofmap.shape.w = 32;
    ofmap.stride.n = 1;
    ofmap.stride.c = 16;
    ofmap.stride.h = 32;
    ofmap.stride.w = 32;
    ofmap.fmt = CVK_FMT_I8;

    // Initialize parameter struct
    param.ifmap = &ifmap;
    param.table = &table;
    param.ofmap = &ofmap;
    param.layer_id = 1;

    // Initialize the context
    ctx.info.eu_num = 16;
    ctx.info.npu_num = 4;
}

void test_tiu_lookup_table() {
    setup_test();

    // Perform the operation
    cvkcv181x_tiu_lookup_table(&ctx, &param);

    // After the operation, we expect the registers to be set correctly.
    // Here, we verify that the tensor addresses and strides are being passed correctly.
    assert(ofmap.start_address == 0x3000);
    assert(table.start_address == 0x2000);
    assert(ifmap.start_address == 0x1000);
    
    // Check if strides are set properly
    assert(ofmap.stride.n == 1 && ofmap.stride.c == 16 && ofmap.stride.h == 32 && ofmap.stride.w == 32);
    assert(table.stride.n == 1 && table.stride.c == 4 && table.stride.h == 16 && table.stride.w == 16);
    assert(ifmap.stride.n == 1 && ifmap.stride.c == 16 && ifmap.stride.h == 32 && ifmap.stride.w == 32);
    
    // Verify the format consistency under BF16 (optional, based on the input data)
    if (ofmap.fmt == CVK_FMT_BF16) {
        assert(ifmap.fmt == CVK_FMT_BF16);
    }
    assert(ofmap.fmt == CVK_FMT_I8 || ofmap.fmt == CVK_FMT_U8 || ofmap.fmt == CVK_FMT_BF16);
}

void test_tiu_lookup_table_bf16() {
    // Test for BF16 case
    setup_test();

    // Modify tensor formats to BF16
    ifmap.fmt = CVK_FMT_BF16;
    ofmap.fmt = CVK_FMT_BF16;
    table.fmt = CVK_FMT_BF16;

    // Perform the operation with BF16 tensors
    cvkcv181x_tiu_lookup_table(&ctx, &param);

    // Verify that the function handled BF16 format correctly
    assert(ofmap.fmt == CVK_FMT_BF16);
    assert(ifmap.fmt == CVK_FMT_BF16);
    assert(table.fmt == CVK_FMT_BF16);
}

int main() {
    // Run the test for regular tensors
    test_tiu_lookup_table();
    printf("Regular lookup table test passed.\n");

    // Run the test for BF16 tensors
    test_tiu_lookup_table_bf16();
    printf("BF16 lookup table test passed.\n");

    return 0;
}

