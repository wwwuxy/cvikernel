#include "cvkcv181x.h"
#include <stdio.h>
#include <stdlib.h>

// 测试 cvkcv181x_tiu_or_int8 函数
void test_tiu_or_int8() {
    // 创建 cvk_context_t 上下文
    cvk_context_t ctx;
    ctx.info.eu_num = 16; // 假设有16个执行单元

    // 创建 mock tensor 对象（cvk_ml_t）
    cvk_ml_t a, b, res;
    
    // 设置 tensor a 的属性
    a.start_address = 0x1000;
    a.fmt = CVK_FMT_I8;
    a.shape.n = 1;
    a.shape.c = 1;
    a.shape.h = 1;
    a.shape.w = 4; // 示例形状
    a.stride.n = 1;
    a.stride.c = 1;
    a.stride.h = 1;
    a.stride.w = 4;
    
    // 设置 tensor b 的属性
    b.start_address = 0x2000;
    b.fmt = CVK_FMT_I8;
    b.shape.n = 1;
    b.shape.c = 1;
    b.shape.h = 1;
    b.shape.w = 4; // 示例形状
    b.stride.n = 1;
    b.stride.c = 1;
    b.stride.h = 1;
    b.stride.w = 4;
    
    // 设置 result tensor 的属性
    res.start_address = 0x3000;
    res.fmt = CVK_FMT_I8;
    res.shape.n = 1;
    res.shape.c = 1;
    res.shape.h = 1;
    res.shape.w = 4; // 与输入 tensor 相同的形状
    res.stride.n = 1;
    res.stride.c = 1;
    res.stride.h = 1;
    res.stride.w = 4;
    
    // 准备 cvk_tiu_or_int8_param_t 参数结构体
    cvk_tiu_or_int8_param_t p;
    p.res = &res;
    p.a = &a;
    p.b = &b;
    p.layer_id = 1;

    // 调用 OR 操作函数
    cvkcv181x_tiu_or_int8(&ctx, &p);

    // 打印测试结果（此处应替换为实际的检查/验证逻辑）
    printf("Test passed for cvkcv181x_tiu_or_int8\n");
}

// 测试 cvkcv181x_tiu_or_int16 函数
void test_tiu_or_int16() {
    // 创建 cvk_context_t 上下文
    cvk_context_t ctx;
    ctx.info.eu_num = 16; // 假设有16个执行单元

    // 创建 16-bit mock tensor 对象（cvk_ml_t）
    cvk_ml_t a_low, a_high, b_low, b_high, res_low, res_high;
    
    // 设置 tensor a_low 和 a_high 的属性
    a_low.start_address = 0x1000;
    a_high.start_address = 0x2000;
    a_low.fmt = CVK_FMT_I16;
    a_high.fmt = CVK_FMT_I16;
    a_low.shape.n = 1;
    a_high.shape.n = 1;
    a_low.shape.c = 1;
    a_high.shape.c = 1;
    a_low.shape.h = 1;
    a_high.shape.h = 1;
    a_low.shape.w = 4;
    a_high.shape.w = 4;
    a_low.stride.n = 1;
    a_high.stride.n = 1;
    a_low.stride.c = 1;
    a_high.stride.c = 1;
    a_low.stride.h = 1;
    a_high.stride.h = 1;
    a_low.stride.w = 4;
    a_high.stride.w = 4;
    
    // 设置 tensor b_low 和 b_high 的属性
    b_low.start_address = 0x3000;
    b_high.start_address = 0x4000;
    b_low.fmt = CVK_FMT_I16;
    b_high.fmt = CVK_FMT_I16;
    b_low.shape.n = 1;
    b_high.shape.n = 1;
    b_low.shape.c = 1;
    b_high.shape.c = 1;
    b_low.shape.h = 1;
    b_high.shape.h = 1;
    b_low.shape.w = 4;
    b_high.shape.w = 4;
    b_low.stride.n = 1;
    b_high.stride.n = 1;
    b_low.stride.c = 1;
    b_high.stride.c = 1;
    b_low.stride.h = 1;
    b_high.stride.h = 1;
    b_low.stride.w = 4;
    b_high.stride.w = 4;
    
    // 设置 result tensor 的属性
    res_low.start_address = 0x5000;
    res_high.start_address = 0x6000;
    res_low.fmt = CVK_FMT_I16;
    res_high.fmt = CVK_FMT_I16;
    res_low.shape.n = 1;
    res_high.shape.n = 1;
    res_low.shape.c = 1;
    res_high.shape.c = 1;
    res_low.shape.h = 1;
    res_high.shape.h = 1;
    res_low.shape.w = 4;
    res_high.shape.w = 4;
    res_low.stride.n = 1;
    res_high.stride.n = 1;
    res_low.stride.c = 1;
    res_high.stride.c = 1;
    res_low.stride.h = 1;
    res_high.stride.h = 1;
    res_low.stride.w = 4;
    res_high.stride.w = 4;

    // 准备 cvk_tiu_or_int16_param_t 参数结构体
    cvk_tiu_or_int16_param_t p;
    p.res_low = &res_low;
    p.res_high = &res_high;
    p.a_low = &a_low;
    p.a_high = &a_high;
    p.b_low = &b_low;
    p.b_high = &b_high;
    p.layer_id = 2;

    // 调用 OR 操作函数
    cvkcv181x_tiu_or_int16(&ctx, &p);

    // 打印测试结果（此处应替换为实际的检查/验证逻辑）
    printf("Test passed for cvkcv181x_tiu_or_int16\n");
}

int main() {
    // 运行测试
    test_tiu_or_int8();
    test_tiu_or_int16();
    return 0;
}

