#!/bin/bash
# 设置 PATH，确保能找到 gdb 等调试工具
export PATH="/usr/bin:$PATH"
# ==========================================
# MX 量化演示程序 - 多模式调试脚本
# ==========================================
# 用法:
#   ./debug_mx_demo.sh              # 编译并运行（普通模式）
#   ./debug_mx_demo.sh gdb          # 使用 GDB 交互式调试
#   ./debug_mx_demo.sh printf       # 编译带详细打印的版本并运行
#   ./debug_mx_demo.sh asan         # 编译带 AddressSanitizer 的版本
#   ./debug_mx_demo.sh clean        # 清理编译产物
# ==========================================

set -e

MODE="${1:-run}"
BUILD_DIR="./build_debug"

case "$MODE" in
    run)
        echo "=== [普通模式] 编译并运行 ==="
        g++ mx_demo.cpp -o mx_demo.out -std=c++11 -g
        ./mx_demo.out
        ;;

    gdb)
        echo "=== [GDB 模式] 编译带调试符号的版本 ==="
        g++ mx_demo.cpp -o mx_demo_debug.out -std=c++11 -g -O0
        echo ""
        echo "启动 GDB 交互式调试..."
        echo "GDB 常用命令:"
        echo "  break main          - 在 main 函数设置断点"
        echo "  break 286           - 在第 286 行设置断点"
        echo "  run                 - 运行程序"
        echo "  next                - 单步跳过"
        echo "  step                - 单步进入"
        echo "  print max_biased_exp - 打印变量"
        echo "  display scale       - 每次暂停时自动打印变量"
        echo "  info locals         - 显示所有局部变量"
        echo "  backtrace           - 查看调用栈"
        echo "  quit                - 退出 GDB"
        echo "----------------------------------------"
        gdb ./mx_demo_debug.out
        ;;

    printf)
        echo "=== [Printf 模式] 编译带详细打印的版本 ==="
        # 编译时定义 DEBUG_PRINT 宏，启用代码中的调试打印
        g++ mx_demo.cpp -o mx_demo_printf.out -std=c++11 -g -DDEBUG_PRINT
        echo ""
        echo "运行带详细调试信息的版本:"
        echo "----------------------------------------"
        ./mx_demo_printf.out
        ;;

    asan)
        echo "=== [ASan 模式] 编译带 AddressSanitizer 的版本 ==="
        g++ mx_demo.cpp -o mx_demo_asan.out -std=c++11 -g -fsanitize=address -fno-omit-frame-pointer
        echo ""
        echo "运行带 AddressSanitizer 的版本:"
        echo "----------------------------------------"
        ./mx_demo_asan.out
        ;;

    clean)
        echo "=== 清理编译产物 ==="
        rm -f mx_demo.out mx_demo_debug.out mx_demo_printf.out mx_demo_asan.out
        rm -rf "$BUILD_DIR"
        echo "清理完成"
        ;;

    *)
        echo "用法: $0 [run|gdb|printf|asan|clean]"
        echo "  run     - 普通编译运行（默认）"
        echo "  gdb     - GDB 交互式调试"
        echo "  printf  - 带详细打印的调试版本"
        echo "  asan    - AddressSanitizer 内存检查"
        echo "  clean   - 清理编译产物"
        exit 1
        ;;
esac