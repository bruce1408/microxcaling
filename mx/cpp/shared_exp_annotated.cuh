#ifndef PYT_MX_SHARED_EXP_CUH
#define PYT_MX_SHARED_EXP_CUH

#include "common.cuh"

// ============================================================================
// MX (Microscaling) 共享指数计算核心函数
// ============================================================================
// 本文件包含 MX 量化中共享指数计算的核心 CUDA 内核函数
// 主要功能：
// 1. clamp_shared_exp: 将共享指数限制在量化格式的有效范围内
// 2. mx_get_shared_scale: 计算 MX 量化的共享缩放因子
// ============================================================================

//-----------------------------------------------------------------------
// 函数：clamp_shared_exp
// 功能：根据指数位数限制共享指数的范围
// 参数：
//   - shared_exp: 输入的共享指数（带偏置的 IEEE 754 格式）
//   - ebits: 目标量化格式的指数位数
// 返回值：限制后的共享指数
// 原理：
//   MX 量化使用共享指数技术，块内所有元素共享同一个指数。
//   该函数确保共享指数不会超出目标格式的表示范围：
//   1. 溢出处理：如果指数超出最大范围，设置为 NaN（指数全1）
//   2. 下溢处理：如果指数低于最小范围，设置为最小可表示指数
//-----------------------------------------------------------------------
__host__ __device__ __forceinline__
int clamp_shared_exp(
    int shared_exp,          // 输入：带偏置的共享指数（IEEE 754 格式，偏置127）
    const int ebits          // 输入：目标量化格式的指数位数
) {
    // 计算目标格式的最大无偏指数
    // 对于 ebits 位指数：
    //   - 最大无偏指数 = 2^(ebits-1) - 1
    //   - 例如：8位指数（ebits=8）时，emax = 2^(8-1) - 1 = 127
    // 特殊处理 ebits=0 的情况（使用 float32 的最大指数）
    int emax = ebits != 0 ? (1 << (ebits-1)) - 1 : FLOAT32_EXP_MAX;
    
    // 将带偏置的指数转换为无偏指数
    // IEEE 754 float32 指数偏置为 127，所以：
    //   无偏指数 = 带偏置指数 - 127
    int shared_ub = shared_exp - FLOAT32_EXP_BIAS;
    
    // 处理溢出：如果无偏指数 > emax，设置为 NaN（指数全1）
    // FLOAT32_EXP_MAX = 255（8位指数全1）
    shared_exp = shared_ub > emax ? FLOAT32_EXP_MAX : shared_exp;
    
    // 处理下溢：如果无偏指数 < -emax，设置为最小可表示指数
    // 最小指数 = 偏置 - emax
    // 注意：对于 8 位指数，最小缩放是 -127，而不是 -126
    shared_exp = shared_ub < -emax ? FLOAT32_EXP_BIAS - emax : shared_exp;
    
    return shared_exp;
}

//-----------------------------------------------------------------------
// 函数：mx_get_shared_scale
// 功能：计算 MX 量化的共享缩放因子
// 参数：
//   - shared_exp: 计算得到的共享指数（带偏置）
//   - scale_bits: 缩放因子的指数位数
//   - elem_max_norm: 块内元素的最大归一化值
// 返回值：共享缩放因子（float32 格式）
// 原理：
//   MX 量化分两步：
//   1. 计算块内元素的共享指数
//   2. 根据共享指数和元素最大归一化值计算缩放因子
//   缩放因子用于将元素缩放到目标量化格式的范围内
//-----------------------------------------------------------------------
__host__ __device__ __forceinline__
float mx_get_shared_scale(
    int shared_exp,          // 输入：计算得到的共享指数（带偏置）
    const int scale_bits,    // 输入：缩放因子的指数位数
    const float elem_max_norm // 输入：块内元素的最大归一化值
) {
    // 步骤1：根据元素最大归一化值调整共享指数
    // 获取 elem_max_norm 的无偏指数（支持次正规数）
    // elem_emax 表示元素本身的最大指数值
    const int elem_emax = get_unbiased_exponent(elem_max_norm);
    
    // 调整共享指数：减去元素的最大指数
    // 这样做的目的是确保缩放后的元素不会溢出目标格式
    // 保留 NaN 的特殊处理：如果 shared_exp 已经是 NaN（255），保持不变
    shared_exp = (shared_exp != FLOAT32_EXP_MAX) ? \
                 shared_exp - elem_emax : shared_exp;
    
    // 步骤2：将调整后的共享指数限制在 scale_bits 范围内
    // 防止缩放因子本身溢出或下溢
    shared_exp = clamp_shared_exp(shared_exp, scale_bits);
    
    // 步骤3：计算缩放因子的尾数部分
    // 缩放因子的尾数处理规则：
    //   - 如果共享指数为 0（次正规数）或 255（NaN），尾数设为 0.5
    //   - 否则尾数设为 0（正规数）
    // 0.5 的二进制表示：尾数最高位为 1，即 FLOAT32_IMPLIED1 >> 1
    const int scale_mant = \
            (shared_exp == 0 || shared_exp == FLOAT32_EXP_MAX) ? \
            (FLOAT32_IMPLIED1 >> 1) : 0;
    
    // 步骤4：构造最终的缩放因子浮点数
    // 使用符号位=0（正数），调整后的共享指数，计算得到的尾数
    return construct_float(0, shared_exp, scale_mant);
}

#endif // PYT_MX_SHARED_EXP_CUH