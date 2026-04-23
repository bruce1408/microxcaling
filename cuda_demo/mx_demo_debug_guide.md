# mx_demo.cpp 调试指南

## 概述

本文档提供 `microxcaling/cuda_demo/mx_demo.cpp` 的完整调试方法。该程序是一个纯 CPU 的 C++ 程序，演示 MX 量化的核心算法，支持多种调试方式。

## 环境准备

```bash
# 检查调试工具
gdb --version          # GDB 调试器
g++ --version          # G++ 编译器
which lldb             # LLDB 调试器（可选）
```

当前环境已安装：
- **GDB 15.0.50** - GNU 调试器
- **G++ 13.3.0** - C++ 编译器

## 调试方法一：GDB 交互式调试（推荐）

### 步骤1：编译带调试符号的版本

```bash
cd /home/workspace/microxcaling/cuda_demo
g++ mx_demo.cpp -o mx_demo_debug.out -std=c++11 -g -O0
```

参数说明：
- `-g`：生成调试符号
- `-O0`：禁止优化，确保代码行号对应

### 步骤2：启动 GDB

```bash
gdb ./mx_demo_debug.out
```

### 步骤3：设置断点

GDB 启动后，在 `(gdb)` 提示符下输入：

```
# 在 main 函数入口设置断点
(gdb) break main

# 在指定行号设置断点（例如第 286 行，for 循环）
(gdb) break 286

# 在指定函数设置断点
(gdb) break mx_get_shared_scale

# 条件断点（当变量满足条件时触发）
(gdb) break 308 if max_biased_exp > 130
```

### 步骤4：运行和调试

```
# 运行程序
(gdb) run

# 单步执行（不进入函数）
(gdb) next

# 单步执行（进入函数）
(gdb) step

# 继续执行到下一个断点
(gdb) continue

# 跳出当前函数
(gdb) finish
```

### 步骤5：检查变量

```
# 打印变量值
(gdb) print max_biased_exp
(gdb) print block[0]
(gdb) print scale

# 以十六进制打印（查看浮点数的位表示）
(gdb) print /x max_biased_exp

# 打印所有局部变量
(gdb) info locals

# 每次暂停时自动打印变量
(gdb) display max_biased_exp
(gdb) display scale
```

### 步骤6：查看调用栈和内存

```
# 查看调用栈
(gdb) backtrace

# 查看当前行号
(gdb) info line

# 查看反汇编
(gdb) disassemble

# 查看内存内容
(gdb) x/4x &block[0]    # 以十六进制查看 block[0] 的 4 个字节
(gdb) x/4f &block[0]    # 以浮点数查看 block[0] 的 4 个字节
```

### 完整 GDB 调试会话示例

```bash
$ gdb ./mx_demo_debug.out

(gdb) break main
Breakpoint 1 at 0x...: file mx_demo.cpp, line 262.

(gdb) run
Starting program: ./mx_demo_debug.out

Breakpoint 1, main () at mx_demo.cpp:262
262     int main() {

(gdb) next
268         std::vector<float> block(32, 1.2f);

(gdb) next
271         block[0] = 300.5f;

(gdb) print block[0]
$1 = 1.20000005

(gdb) next
272         block[1] = -15.25f;

(gdb) print block[0]
$2 = 300.5

(gdb) break 286
Breakpoint 2 at 0x...: file mx_demo.cpp, line 286.

(gdb) continue
Continuing.

Breakpoint 2, main () at mx_demo.cpp:286
286         for (int i = 0; i < 32; i++) {

(gdb) display max_biased_exp
1: max_biased_exp = 0

(gdb) next
287             int exp = get_biased_exponent(block[i]);

(gdb) step
get_biased_exponent (input=300.5) at mx_demo.cpp:80
80      int get_biased_exponent(float input) {

(gdb) print /x input
$3 = 0x43968000    # 300.5 的十六进制表示

(gdb) finish
Run till exit from #0  get_biased_exponent (input=300.5) at mx_demo.cpp:80
0x... in main () at mx_demo.cpp:288
288             if (exp > max_biased_exp) {
Value returned is $4 = 135

(gdb) continue
...
```

## 调试方法二：Printf 调试

### 步骤1：添加调试打印语句

在代码的关键位置添加 `#ifdef DEBUG_PRINT` 包裹的打印语句：

```cpp
// 在 get_biased_exponent 中添加
int get_biased_exponent(float input) {
    u_float_int u;
    u.f = input;
    int result = (u.i & FLOAT32_EXP_MASK) >> FLOAT32_EXP_OFFSET;
#ifdef DEBUG_PRINT
    std::cout << "[DEBUG] get_biased_exponent(" << input 
              << ") -> biased_exp=" << result 
              << " (hex: 0x" << std::hex << u.i << std::dec << ")" << std::endl;
#endif
    return result;
}
```

### 步骤2：编译带调试打印的版本

```bash
g++ mx_demo.cpp -o mx_demo_printf.out -std=c++11 -g -DDEBUG_PRINT
```

### 步骤3：运行查看详细输出

```bash
./mx_demo_printf.out
```

### 关键调试点

以下是建议添加调试打印的关键位置：

| 行号 | 函数 | 调试内容 |
|------|------|----------|
| 80 | `get_biased_exponent` | 输入值和提取的指数 |
| 106 | `get_unbiased_exponent` | 带偏置指数和真实指数 |
| 136 | `construct_float` | 符号、指数、尾数的位组合 |
| 176 | `clamp_shared_exp` | 输入指数、emax、限制结果 |
| 226 | `mx_get_shared_scale` | 各步骤的中间值 |
| 286 | main 循环 | 每个元素的指数 |
| 308 | main | 缩放因子计算结果 |

## 调试方法三：VS Code 可视化调试

### 步骤1：配置 launch.json

在 VS Code 中创建 `.vscode/launch.json`：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug mx_demo",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/microxcaling/cuda_demo/mx_demo_debug.out",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/microxcaling/cuda_demo",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "build_debug",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

### 步骤2：配置 tasks.json

创建 `.vscode/tasks.json`：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build_debug",
            "type": "shell",
            "command": "g++",
            "args": [
                "mx_demo.cpp",
                "-o",
                "mx_demo_debug.out",
                "-std=c++11",
                "-g",
                "-O0"
            ],
            "options": {
                "cwd": "${workspaceFolder}/microxcaling/cuda_demo"
            },
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$gcc"]
        }
    ]
}
```

### 步骤3：使用 VS Code 调试功能

1. 在代码行号左侧点击设置**断点**（红点）
2. 按 `F5` 启动调试
3. 使用调试工具栏：
   - `F10`：单步跳过
   - `F11`：单步进入
   - `Shift+F11`：跳出
   - `F5`：继续
4. 在 **变量** 面板查看变量值
5. 在 **监视** 面板添加表达式

## 调试方法四：AddressSanitizer 内存检查

### 编译和运行

```bash
g++ mx_demo.cpp -o mx_demo_asan.out -std=c++11 -g -fsanitize=address -fno-omit-frame-pointer
./mx_demo_asan.out
```

### 检测内容

- 缓冲区溢出
- 使用已释放的内存
- 内存泄漏
- 栈溢出

## 调试方法五：Valgrind 内存分析

```bash
# 安装 Valgrind
apt-get install valgrind

# 编译带调试符号的版本
g++ mx_demo.cpp -o mx_demo_debug.out -std=c++11 -g -O0

# 运行 Valgrind 内存检查
valgrind --leak-check=full --show-leak-kinds=all ./mx_demo_debug.out

# 运行 Valgrind 并生成调用图
valgrind --tool=callgrind ./mx_demo_debug.out
```

## 常见调试场景

### 场景1：验证浮点数位表示

```cpp
// 在 GDB 中
(gdb) print /x block[0]
$1 = 0x43968000    // 300.5 的 IEEE 754 表示

// 分解：
// 0x43968000 = 0100 0011 1001 0110 1000 0000 0000 0000
// S=0, E=10000111(135), M=00101101000000000000000
// 真实指数 = 135-127 = 8
// 数值 = 1.00101101 × 2^8 = 300.5
```

### 场景2：跟踪共享指数计算

```cpp
// 在 mx_get_shared_scale 设置断点
(gdb) break mx_get_shared_scale
(gdb) run

// 检查参数
(gdb) print shared_exp_biased
$1 = 135
(gdb) print scale_bits
$2 = 8
(gdb) print elem_max_norm
$3 = 240

// 单步执行
(gdb) step  // 进入函数

// 检查中间值
(gdb) print elem_emax
$4 = 7       // 240 的无偏指数

(gdb) next
(gdb) print shared_exp
$5 = 128     // 135 - 7 = 128

(gdb) next
(gdb) print shared_exp
$6 = 128     // clamp_shared_exp(128, 8) -> 在 [-127, 127] 范围内

(gdb) next
(gdb) print scale_mant
$7 = 0       // 正规数，尾数为 0

(gdb) next
(gdb) print construct_float(0, 128, 0)
$8 = 2       // 缩放因子 = 2.0
```

### 场景3：验证缩放结果

```cpp
// 在缩放循环设置断点
(gdb) break 324
(gdb) run

// 检查每个元素的缩放
(gdb) display original
(gdb) display scaled

(gdb) continue
// 输出：
// 1: original = 300.5
// 2: scaled = 150.25

(gdb) continue
// 1: original = -15.25
// 2: scaled = -7.625

(gdb) continue
// 1: original = 9.99995e-41
// 2: scaled = 4.99997e-41
```

## 调试脚本使用

使用 `debug_mx_demo.sh` 脚本可以快速切换调试模式：

```bash
# 添加执行权限
chmod +x debug_mx_demo.sh

# 普通运行
./debug_mx_demo.sh run

# GDB 交互式调试
./debug_mx_demo.sh gdb

# 带详细打印的运行
./debug_mx_demo.sh printf

# AddressSanitizer 检查
./debug_mx_demo.sh asan

# 清理
./debug_mx_demo.sh clean
```

## 调试流程图

```
开始调试
  │
  ├─ 问题类型？
  │   ├─ 逻辑错误 → GDB 交互式调试
  │   ├─ 数值精度 → Printf 调试（打印中间值）
  │   ├─ 内存问题 → AddressSanitizer / Valgrind
  │   └─ 性能问题 → Valgrind Callgrind
  │
  ├─ 设置断点
  │   ├─ main() 入口
  │   ├─ 关键函数（get_biased_exponent, clamp_shared_exp, mx_get_shared_scale）
  │   └─ 循环和条件判断
  │
  ├─ 运行并观察
  │   ├─ 检查变量值
  │   ├─ 检查位表示
  │   └─ 检查调用栈
  │
  └─ 修复问题
      ├─ 修改代码
      ├─ 重新编译
      └─ 重新调试验证
```

## 总结

| 调试方法 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| GDB | 逻辑错误、流程跟踪 | 功能强大、无需修改代码 | 学习曲线较陡 |
| Printf | 数值精度、中间值检查 | 简单直观 | 需要修改代码 |
| VS Code | 可视化调试 | 界面友好、断点管理方便 | 需要配置 |
| ASan | 内存错误 | 自动检测内存问题 | 运行速度较慢 |
| Valgrind | 内存泄漏、性能分析 | 全面分析 | 运行速度很慢 |
