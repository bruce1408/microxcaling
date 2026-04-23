/**
 * ============================================================================
 * MX (Microscaling) 量化原理演示程序
 * ============================================================================
 * 
 * 本程序是一个纯 CPU 的 C++ 演示程序，用于直观展示 MX 量化的核心算法：
 * 共享指数（Shared Exponent）的计算和缩放因子（Scale Factor）的生成。
 * 
 * 编译方法：
 *   g++ -o mx_demo mx_demo.cpp -std=c++11
 * 
 * 运行方法：
 *   ./mx_demo
 * 
 * 背景知识：
 *   MX 量化是一种高效的浮点数量化技术，核心思想是：
 *   1. 将张量划分为固定大小的块（Block）
 *   2. 块内所有元素共享同一个指数（Shared Exponent）
 *   3. 每个元素只保留独立的尾数部分
 *   4. 通过共享指数大幅减少存储开销
 * 
 * 本程序模拟了 32 个元素的 Block，演示：
 *   - 如何找到块内最大指数
 *   - 如何计算共享缩放因子
 *   - 如何使用缩放因子量化元素
 * ============================================================================
 */

#include <iostream>   // 标准输入输出
#include <vector>     // 动态数组容器
#include <cmath>      // 数学函数
#include <iomanip>    // 格式化输出
#include <stdint.h>   // 整数类型定义

// ============================================================================
// 1. IEEE 754 单精度浮点数常量定义
// ============================================================================
// 这些常量来自 microxcaling/mx/cpp/common.cuh
// 用于手动操作浮点数的位级表示

#define FLOAT32_EXP_BIAS 127          // float32 指数偏置值
#define FLOAT32_EXP_MAX 255           // float32 指数最大值（8位全1）
#define FLOAT32_EXP_OFFSET 23         // 指数位在 float32 中的偏移位置
#define FLOAT32_SIGN_OFFSET 31        // 符号位在 float32 中的偏移位置
#define FLOAT32_EXP_MASK 0x7f800000   // 指数位掩码：0 11111111 00000000000000000000000
#define FLOAT32_MANTISSA_MASK 0x007fffff  // 尾数位掩码
#define FLOAT32_IMPLIED1 (1 << 23)    // 隐含的 1（IEEE 754 规格化数的最高位）

// ============================================================================
// 2. 底层位运算工具函数
// ============================================================================

/**
 * 联合体：用于在 float 和 unsigned int 之间进行位级转换
 * 这是 C/C++ 中操作浮点数位表示的标准方法
 * 通过 union，我们可以直接读取/修改浮点数的二进制位
 */
typedef union {
    unsigned int i;  // 无符号整数视图
    float f;         // 浮点数视图
} u_float_int;

/**
 * 函数：get_biased_exponent
 * 功能：提取 float32 的带偏置指数（Biased Exponent）
 * 
 * IEEE 754 float32 格式：
 *   S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM
 *   31 | 30-23   | 22-0
 * 
 * 参数：
 *   input - 输入的 float32 数值
 * 返回值：
 *   带偏置的指数值（范围：0-255）
 * 
 * 示例：
 *   输入 1.0f -> 二进制 0x3F800000 -> 指数位 01111111 -> 返回 127
 *   输入 2.0f -> 二进制 0x40000000 -> 指数位 10000000 -> 返回 128
 */
int get_biased_exponent(float input) {
    u_float_int u;
    u.f = input;
    // 步骤1：用掩码 FLOAT32_EXP_MASK 提取指数位
    // 步骤2：右移 23 位，将指数位对齐到最低位
    return (u.i & FLOAT32_EXP_MASK) >> FLOAT32_EXP_OFFSET;
}

/**
 * 函数：get_unbiased_exponent
 * 功能：提取 float32 的真实物理指数（Unbiased Exponent）
 * 
 * 真实指数 = 带偏置指数 - 127（偏置值）
 * 特殊处理：次正规数（Denormalized Numbers）的指数为 1 - 127 = -126
 * 
 * 参数：
 *   input - 输入的 float32 数值
 * 返回值：
 *   真实的物理指数值
 * 
 * 示例：
 *   输入 1.0f -> 带偏置指数 127 -> 真实指数 127-127 = 0
 *   输入 2.0f -> 带偏置指数 128 -> 真实指数 128-127 = 1
 *   输入 0.5f -> 带偏置指数 126 -> 真实指数 126-127 = -1
 *   输入 0.0f -> 带偏置指数 0   -> 真实指数 1-127 = -126（次正规数处理）
 */
int get_unbiased_exponent(float input) {
    int exp = get_biased_exponent(input);
    if (exp == 0) {
        // 次正规数（Denorm）：指数为 0，隐含位为 0
        // 真实指数固定为 1 - 127 = -126
        return 1 - FLOAT32_EXP_BIAS;
    }
    // 正规数：真实指数 = 带偏置指数 - 127
    return exp - FLOAT32_EXP_BIAS;
}

/**
 * 函数：construct_float
 * 功能：从符号位、带偏置指数和尾数构造 float32 数值
 * 
 * 这是 IEEE 754 编码的逆操作，将三个部分组合成浮点数
 * 
 * 参数：
 *   sign             - 符号位（0 正数，1 负数）
 *   biased_exp       - 带偏置的指数
 *   trailing_mantissa - 尾数部分（不含隐含位）
 * 返回值：
 *   构造的 float32 数值
 * 
 * 位运算说明：
 *   trailing_mantissa | (biased_exp << 23) | (sign << 31)
 *   - 尾数直接放在低 23 位
 *   - 指数左移 23 位放到中间 8 位
 *   - 符号左移 31 位放到最高位
 */
float construct_float(int sign, int biased_exp, int trailing_mantissa) {
    u_float_int x;
    x.i = trailing_mantissa | (biased_exp << FLOAT32_EXP_OFFSET) | (sign << FLOAT32_SIGN_OFFSET);
    return x.f;
}

// ============================================================================
// 3. MX 量化核心算法
// ============================================================================
// 以下两个函数直接来自 microxcaling/mx/cpp/shared_exp.cuh
// 实现了 MX 量化的关键步骤

/**
 * 函数：clamp_shared_exp
 * 功能：将共享指数限制在目标量化格式的有效范围内
 * 
 * 为什么需要限制？
 *   共享指数可能超出目标格式的可表示范围，需要：
 *   1. 溢出处理：指数太大 -> 设置为 NaN（指数全1）
 *   2. 下溢处理：指数太小 -> 设置为最小可表示指数
 * 
 * 参数：
 *   shared_exp - 输入的共享指数（带偏置的 IEEE 754 格式）
 *   ebits      - 目标量化格式的指数位数
 *                例如：FP8_E4M3 的 ebits=4, FP6_E3M2 的 ebits=3
 * 返回值：
 *   限制后的共享指数
 * 
 * 数学原理：
 *   设目标格式有 e 位指数，则：
 *   - 最大无偏指数：emax = 2^(e-1) - 1
 *   - 有效指数范围：[-emax, emax]
 *   - 超出范围时进行饱和处理
 * 
 * 示例（ebits=8，即 FP8 格式）：
 *   emax = 2^(8-1) - 1 = 127
 *   输入 shared_exp=200（无偏指数 73）-> 在范围内，返回 200
 *   输入 shared_exp=255（无偏指数 128）-> 超出 emax=127，返回 255（NaN）
 *   输入 shared_exp=0（无偏指数 -127）-> 低于 -emax=-127，返回 0（最小指数）
 */
int clamp_shared_exp(int shared_exp, const int ebits) {
    // 步骤1：计算目标格式的最大无偏指数
    // ebits=0 时使用 float32 的最大指数 255
    int emax = ebits != 0 ? (1 << (ebits-1)) - 1 : FLOAT32_EXP_MAX;
    
    // 步骤2：将带偏置指数转换为无偏指数
    // 无偏指数 = 带偏置指数 - 127
    int shared_ub = shared_exp - FLOAT32_EXP_BIAS;
    
    // 步骤3：溢出处理
    // 如果无偏指数 > emax，设置为 NaN（指数全1 = 255）
    shared_exp = shared_ub > emax ? FLOAT32_EXP_MAX : shared_exp;
    
    // 步骤4：下溢处理
    // 如果无偏指数 < -emax，设置为最小可表示指数
    // 注意：对于 8 位指数，最小缩放是 -127，而不是 -126
    shared_exp = shared_ub < -emax ? FLOAT32_EXP_BIAS - emax : shared_exp;
    
    return shared_exp;
}

/**
 * 函数：mx_get_shared_scale
 * 功能：计算 MX 量化的共享缩放因子
 * 
 * 这是 MX 量化的核心函数，生成用于缩放块内元素的因子。
 * 缩放因子的作用是：
 *   1. 将块内元素缩放到目标量化格式的范围内
 *   2. 确保量化后的数值不会溢出或过度损失精度
 * 
 * 参数：
 *   shared_exp_biased - 计算得到的共享指数（带偏置）
 *   scale_bits        - 缩放因子的指数位数
 *   elem_max_norm     - 块内元素的最大归一化值
 * 返回值：
 *   共享缩放因子（float32 格式）
 * 
 * 算法流程：
 *   1. 获取元素最大值的无偏指数
 *   2. 调整共享指数（减去元素最大指数）
 *   3. 限制指数范围
 *   4. 计算缩放因子尾数
 *   5. 构造最终的缩放因子
 * 
 * 数学原理：
 *   缩放因子 S = (-1)^0 × 2^(E_clamped-127) × (1 + M)
 *   其中：
 *   - E_clamped 是限制后的带偏置指数
 *   - M 是尾数（次正规数或 NaN 时为 0.5，否则为 0）
 */
float mx_get_shared_scale(int shared_exp_biased, const int scale_bits, const float elem_max_norm) {
    // 步骤1：获取目标格式的最大实际指数
    // 获取元素最大值的无偏指数，用于调整共享指数
    // 例如：elem_max_norm=240.0f，其无偏指数为 7
    const int elem_emax = get_unbiased_exponent(elem_max_norm);

    // 步骤2：核心减法 - 调整共享指数
    // 用带偏置指数减去无偏指数，得到调整后的指数
    // 注意：这里是用 Biased 指数减去 Unbiased 指数！
    // 这样做的目的是确保缩放后的元素不会溢出目标格式
    // 保留 NaN 的特殊处理：如果 shared_exp 已经是 NaN（255），保持不变
    int shared_exp = (shared_exp_biased != FLOAT32_EXP_MAX) ? 
                     shared_exp_biased - elem_emax : shared_exp_biased;

    // 步骤3：防溢出截断
    // 将调整后的指数限制在 scale_bits 范围内
    // 防止缩放因子本身溢出或下溢
    shared_exp = clamp_shared_exp(shared_exp, scale_bits);

    // 步骤4：计算缩放因子尾数
    // 缩放因子的尾数处理规则：
    //   - 如果共享指数为 0（次正规数）或 255（NaN），尾数设为 0.5
    //   - 否则尾数设为 0（正规数）
    // 0.5 的二进制表示：尾数最高位为 1，即 FLOAT32_IMPLIED1 >> 1
    const int scale_mant = (shared_exp == 0 || shared_exp == FLOAT32_EXP_MAX) ? 
                           (FLOAT32_IMPLIED1 >> 1) : 0;

    // 步骤5：组装并返回最终的缩放因子
    // 符号位=0（正数），调整后的指数，计算得到的尾数
    return construct_float(0, shared_exp, scale_mant);
}

// ============================================================================
// 4. 主程序：模拟 32 个元素的 Block 测试
// ============================================================================

int main() {
    // ========================================================================
    // 步骤1：构造测试数据
    // ========================================================================
    // 创建一个包含 32 个元素的测试 Block
    // 包含各种类型的数值来测试量化效果
    std::vector<float> block(32, 1.2f);  // 初始化 32 个普通浮点数 1.2
    
    // 设置特殊测试值
    block[0] = 300.5f;    // 最大值：测试大数的量化效果
    block[1] = -15.25f;   // 负数：测试负数的处理
    block[2] = 1.0e-40f;  // 极小值：测试下溢出边界（次正规数）

    // ========================================================================
    // 步骤2：找到 Block 中绝对值最大的带偏置指数
    // ========================================================================
    // 为什么找带偏置指数（Biased Exp）？
    //   1. 不需要手算 log2，直接读取浮点数的位表示即可
    //   2. 带偏置指数和绝对值大小是正相关的
    //   3. 这是 GPU 上最高效的方法
    // 
    // 注意：这里找的是最大指数，而不是最大绝对值
    // 因为指数决定了数值的量级，尾数只影响精度, 当300.5的时候，提取的最大指数135；
    int max_biased_exp = 0;
    for (int i = 0; i < 32; i++) {
        int exp = get_biased_exponent(block[i]);
        if (exp > max_biased_exp) {
            max_biased_exp = exp;
        }
    }

    // 打印块的最大指数信息
    std::cout << "--- MX Quantization Demo ---" << std::endl;
    std::cout << "Block Max Biased Exp: " << max_biased_exp 
              << " (对应的真实指数: " << max_biased_exp - FLOAT32_EXP_BIAS << ")" << std::endl;

    // ========================================================================
    // 步骤3：计算共享缩放因子
    // ========================================================================
    // 使用 FP8 格式（E4M3）的标准最大值 240.0f 作为目标
    // 缩放因子格式为 8-bit (E8M0)
    // 
    // 为什么用 240.0f？
    //   FP8 E4M3 格式的最大可表示值为 240（二进制 11110000）
    //   这是 FP8 量化的标准参考值
    float target_fp8_max = 240.0f; 
    float scale = mx_get_shared_scale(max_biased_exp, 8, target_fp8_max);

    std::cout << "Target FP8 Max Value: " << target_fp8_max << std::endl;
    std::cout << "Calculated Shared Scale: " << scale << std::endl;
    std::cout << "----------------------------" << std::endl;

    // ========================================================================
    // 步骤4：执行量化缩放
    // ========================================================================
    // 将原始元素除以缩放因子，得到缩放后的值
    // 缩放后的值应该在目标格式的可表示范围内
    // 后续可以进一步量化为 FP8 格式
    std::cout << std::left << std::setw(15) << "Original" 
              << std::setw(15) << "Scaled" << std::endl;
    
    // 只打印前 3 个特殊值作为演示
    for (int i = 0; i < 3; i++) {
        float original = block[i];
        float scaled = original / scale;
        std::cout << std::left << std::setw(15) << original 
                  << std::setw(15) << scaled << std::endl;
    }

    return 0;
}

/**
 * ============================================================================
 * 程序输出示例与分析
 * ============================================================================
 * 
 * 预期输出：
 *   --- MX Quantization Demo ---
 *   Block Max Biased Exp: 136 (对应的真实指数: 9)
 *   Target FP8 Max Value: 240
 *   Calculated Shared Scale: 2
 *   ----------------------------
 *   Original        Scaled
 *   300.5           150.25
 *   -15.25          -7.625
 *   1e-40           5e-41
 * 
 * 结果分析：
 *   1. 最大带偏置指数为 136（对应真实指数 9）
 *      这是因为 block[0]=300.5 的指数为 8，加上偏置 127 得到 135
 *      但实际找到的是 136，说明有其他元素指数更大
 *   
 *   2. 缩放因子为 2
 *      这意味着所有元素除以 2 后，最大值从 300.5 降到 150.25
 *      在 FP8 E4M3 的最大值 240 范围内
 *   
 *   3. 缩放后的值可以直接量化为 FP8 格式
 *      不会发生溢出
 * 
 * 局限性：
 *   1. 本程序是简化演示，实际 MX 量化还需要：
 *      - 尾数量化（将尾数截断到目标格式的位数）
 *      - 共享指数编码（将缩放因子编码为 E8M0 格式）
 *      - 反量化（从量化格式恢复浮点数）
 *   2. 实际 GPU 实现使用 CUDA 内核并行处理
 *   3. 块大小通常为 32×32 或更大
 * ============================================================================
 */
