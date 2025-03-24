# CV180X测试文件修改文档

## 修改概述

本次修改为CV180X测试文件添加了条件编译支持，使其能够在模拟实现和真实API实现之间切换。具体修改内容如下：

1. 为所有测试文件添加了`CV180X_USE_REAL_IMPL`宏定义条件编译支持
2. 修复了测试文件中的编译错误，包括函数重定义和未声明变量的问题
3. 改进了测试文件的结构，使其更加清晰和易于维护
4. 创建了自动化脚本和验证脚本，用于批量修改和验证测试文件

## 修改文件列表

共修改了15个CV180X测试文件：

- tiu_add_cv180x.c
- tiu_and_cv180x.c
- tiu_avgpool_cv180x.c
- tiu_conv_cv180x.c
- tiu_matmul_cv180x.c
- tiu_max_cv180x.c
- tiu_maxpool_cv180x.c
- tiu_min_cv180x.c
- tiu_mul_cv180x.c
- tiu_normalize_cv180x.c
- tiu_or_cv180x.c
- tiu_quantize_cv180x.c
- tiu_relu_cv180x.c
- tiu_sub_cv180x.c
- tiu_xor_cv180x.c

## 修改内容详情

### 条件编译支持

每个测试文件都添加了条件编译支持，格式如下：

```c
#ifndef CV180X_USE_REAL_IMPL
    // 模拟实现代码
#else
    // 真实API实现代码
#endif
```

### 脚本工具

创建了以下脚本工具：

1. `modify_cv180x_tests.sh` - 自动为CV180X测试文件添加条件编译支持
2. `fix_cv180x_tests.sh` - 修复修改后文件中的编译错误
3. `final_verify_cv180x.sh` - 验证所有修改后的文件是否能正确编译和运行

### CMakeLists.txt修改

在`cvikernel/tests/cv180x/CMakeLists.txt`中添加了对`CV180X_USE_REAL_IMPL`宏的支持：

```cmake
option(CV180X_USE_REAL_IMPL "Use real API implementation for CV180X tests" OFF)
if(CV180X_USE_REAL_IMPL)
    add_definitions(-DCV180X_USE_REAL_IMPL)
endif()
```

## 使用说明

### 默认模式（模拟实现）

默认情况下，测试文件使用模拟实现。直接编译运行即可：

```bash
cc cvikernel/tests/cv180x/<测试文件>.c -o cvikernel/tests/cv180x/<测试文件> -lm
./cvikernel/tests/cv180x/<测试文件>
```

### 真实API实现

要使用真实API实现，需要在编译时定义`CV180X_USE_REAL_IMPL`宏：

```bash
cc -DCV180X_USE_REAL_IMPL cvikernel/tests/cv180x/<测试文件>.c -o cvikernel/tests/cv180x/<测试文件> -lm
./cvikernel/tests/cv180x/<测试文件>
```

或者在CMake构建系统中启用该选项：

```bash
cmake -DCV180X_USE_REAL_IMPL=ON ..
make
```

## 验证结果

所有15个测试文件已经通过了编译和运行测试，可以在默认的模拟模式下正常工作。真实API模式需要在实际硬件环境中进一步验证。 