# CV181X测试文件修改文档

## 修改概述

本次修改为CV181X测试文件添加了条件编译支持，使其能够在模拟实现和真实API实现之间切换。具体修改内容如下：

1. 为所有测试文件添加了`CV181X_USE_REAL_IMPL`宏定义条件编译支持
2. 创建了`CMakeLists.txt`文件，支持条件编译的配置选项
3. 提取注释中的真实API实现代码，并通过条件编译整合到文件中
4. 修复了测试文件中的编译错误，包括指针引用和变量声明问题
5. 创建了自动化脚本和验证脚本，用于批量修改和验证测试文件

## 修改文件列表

修改了CV181X目录下的所有TIU测试文件，包括：

- tiu_add_cv181x.c
- tiu_and_cv181x.c
- tiu_conv_cv181x.c
- tiu_ge_cv181x.c
- tiu_lookup_table_cv181x.c
- tiu_mac_cv181x.c
- tiu_max_cv181x.c
- tiu_min_cv181x.c
- tiu_min_pooling_cv181x.c
- tiu_mul_cv181x.c
- tiu_mul_qm_cv181x.c
- tiu_or_cv181x.c
- tiu_shift_cv181x.c
- tiu_sub_cv181x.c
- tiu_xor_cv181x.c

## 修改内容详情

### 条件编译支持

每个测试文件都添加了条件编译支持，格式如下：

```c
#ifndef CV181X_USE_REAL_IMPL
    // 模拟实现的函数和数据结构
#endif // CV181X_USE_REAL_IMPL

void test_tiu_xxx() {
    // 初始化代码...
    
#ifdef CV181X_USE_REAL_IMPL
    // 使用真实TIU API的实现代码
    ctx = cvikernel_register(&reg_info);
    // 创建张量、执行操作等真实代码
#else
    // 使用模拟实现的代码
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    // 打印操作说明和模拟结果
#endif // CV181X_USE_REAL_IMPL

    // 资源释放和收尾代码...
}
```

### CMakeLists.txt文件

为CV181X测试目录创建了CMakeLists.txt文件，内容包括：

```cmake
option(CV181X_USE_REAL_IMPL "Use real hardware API implementation for CV181X tests" OFF)
if(CV181X_USE_REAL_IMPL)
    add_definitions(-DCV181X_USE_REAL_IMPL)
endif()

# 自动查找和编译测试文件
file(GLOB TEST_SOURCES "tiu_*.c")
```

### 脚本工具

创建了以下脚本工具：

1. `modify_cv181x_tests.sh` - 自动为CV181X测试文件添加条件编译支持
2. `fix_cv181x_tests.sh` - 修复修改后文件中的编译错误
3. `final_verify_cv181x.sh` - 验证所有修改后的文件是否能正确编译和运行

## 使用说明

### 默认模式（模拟实现）

默认情况下，测试文件使用模拟实现。直接编译运行即可：

```bash
cc cvikernel/tests/cv181x/<测试文件>.c -o cvikernel/tests/cv181x/<测试文件> -lm
./cvikernel/tests/cv181x/<测试文件>
```

### 真实API实现

要使用真实API实现，需要在编译时定义`CV181X_USE_REAL_IMPL`宏：

```bash
cc -DCV181X_USE_REAL_IMPL cvikernel/tests/cv181x/<测试文件>.c -o cvikernel/tests/cv181x/<测试文件> -lm
./cvikernel/tests/cv181x/<测试文件>
```

或者在CMake构建系统中启用该选项：

```bash
cmake -DCV181X_USE_REAL_IMPL=ON ..
make
```

## 修改过程

修改过程采用自动化和半自动化相结合的方法：

1. 首先运行`modify_cv181x_tests.sh`脚本自动为所有文件添加条件编译支持
2. 然后运行`fix_cv181x_tests.sh`脚本修复修改后可能出现的编译错误
3. 最后运行`final_verify_cv181x.sh`脚本验证所有测试文件能否编译和运行

## 注意事项

1. 真实API模式需要在实际硬件环境中验证
2. 一些特殊的测试文件可能需要额外的修改和调整
3. 默认的模拟实现只是简单的打印操作信息，不执行实际的计算 