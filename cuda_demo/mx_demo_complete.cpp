/**
 * ============================================================================
 * MXFP8 (Microscaling FP8) 完整流程演示程序
 * ============================================================================
 *
 * 这是一个不依赖 CUDA 的单文件 C++11 程序，用 CPU 复刻 microxcaling 仓库中
 * MX 量化的关键流程，便于作为“标准且完整”的教学演示：
 *
 *   1. 构造一个 32 元素 Block。
 *   2. 查找 Block 内绝对值最大值。
 *   3. 提取 Block Shared Exponent。
 *   4. 计算 E8M0 Shared Scale。
 *   5. 对每个元素执行 input / scale。
 *   6. 将缩放后的元素量化到 FP8_E4M3。
 *   7. 乘回 scale 得到反量化结果。
 *   8. 打印 FP8 编码、是否饱和以及误差。
 *
 * 对齐的仓库实现：
 *   - mx/cpp/shared_exp.cuh
 *   - mx/cpp/quantize.cuh
 *   - mx/formats.py
 *
 * 编译：
 *   g++ mx_demo_complete.cpp -o mx_demo_complete.out -std=c++11
 *
 * 运行：
 *   ./mx_demo_complete.out
 * ============================================================================
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// --------------- FP32 位域常量 ---------------
// IEEE 754 binary32 格式: 1-bit sign | 8-bit exponent | 23-bit mantissa

const int FLOAT32_EXP_BIAS = 127;        // 指数偏置: 实际指数 = biased_exp - 127
const int FLOAT32_EXP_MAX = 255;         // 有偏指数的上限 (对应无穷大 / NaN)
const int FLOAT32_EXP_OFFSET = 23;       // 指数字段在位表示中的起始偏移
const int FLOAT32_SIGN_OFFSET = 31;      // 符号位的位置 (最高位)
const uint32_t FLOAT32_EXP_MASK = 0x7f800000u;        // 指数字段掩码 (bits 30-23)
const uint32_t FLOAT32_MANTISSA_MASK = 0x007fffffu;   // 尾数字段掩码 (bits 22-0)
const int FLOAT32_IMPLIED1 = 1 << 23;    // 隐含的 leading-1 (正规数时尾数的隐藏位)
const int FLOAT32_FULL_MBITS = 23;       // 尾数存储位数

// 舍入模式枚举
// round_away_from_zero:   向远离零的方向舍入 (即截断时 +0.5, 仓库默认)
// round_to_nearest_even: 向最近的偶数舍入 (IEEE 754 默认)
enum class RoundingMode {
    round_away_from_zero,
    round_to_nearest_even,
};

// 元素格式枚举 —— 对应 mx/formats.py 中定义的所有支持格式
// INT* 为整数格式 (ebits=0), FP* 为浮点格式
enum class ElemFormat {
    int8,
    int4,
    int2,
    fp8_e5m2,
    fp8_e4m3,
    fp6_e3m2,
    fp6_e2m3,
    fp4,
    float16,
    bfloat16,
};

// 格式参数结构体 —— 描述每种元素格式的量化特征
// ebits:   指数位数 (整数格式时为 0)
// mbits:   位数 (包含符号位 + 指定位 + 尾数位，与 mx/formats.py 一致)
// emax:    最大无偏指数
// max_norm:最大正规数
// min_norm:最小正规数
struct FormatParams {
    ElemFormat format;
    std::string name;
    int ebits;
    int mbits;
    int emax;
    float max_norm;
    float min_norm;
};

// 单个元素的量化结果记录 —— 用于教学演示与误差追踪
struct QuantizedElement {
    float original;          // 原始 FP32 值
    float scaled_input;      // 经过 scale 缩放后的值 (input / shared_scale)
    float quantized_scaled;  // 在目标格式中量化后的值
    float dequantized;       // 反量化后的值 (quantized * shared_scale)
    uint8_t fp8_bits;        // 编码后的 FP8 位模式 (原始二进制)
    bool clamped;            // 是否因为超出范围而被截断
};

// --------------- IEEE 754 FP32 位操作工具 ---------------

// 将 float 按位解释为 uint32_t，用于提取 sign/exponent/mantissa
// 使用 memcpy 而非类型转换指针以避免 strict-aliasing 违规
uint32_t float_to_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// 将 uint32_t 位模式重新解释为 float
float bits_to_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// 提取浮点数的符号位: 0 = 正, 1 = 负
int get_sign(float value) {
    return static_cast<int>((float_to_bits(value) >> FLOAT32_SIGN_OFFSET) & 1u);
}

// 提取 FP32 的有偏指数 (0 ~ 255)
int get_biased_exponent(float value) {
    return static_cast<int>((float_to_bits(value) & FLOAT32_EXP_MASK) >> FLOAT32_EXP_OFFSET);
}

// 提取 FP32 的无偏指数
// 注意: 指数为 0 时表示次正规数，此时无偏指数为 1 - bias = -126，而非 0 - 127
int get_unbiased_exponent(float value) {
    const int exp = get_biased_exponent(value);
    return exp == 0 ? 1 - FLOAT32_EXP_BIAS : exp - FLOAT32_EXP_BIAS;
}

// 提取 FP32 的 23-bit 尾数字段 (不含隐含的 leading-1)
int get_trailing_mantissa(float value) {
    return static_cast<int>(float_to_bits(value) & FLOAT32_MANTISSA_MASK);
}

// 根据 sign / biased_exp / trailing_mantissa 重新组装 FP32 浮点数
float construct_float(int sign, int biased_exp, int trailing_mantissa) {
    const uint32_t bits =
        (static_cast<uint32_t>(trailing_mantissa) & FLOAT32_MANTISSA_MASK) |
        ((static_cast<uint32_t>(biased_exp) & 0xffu) << FLOAT32_EXP_OFFSET) |
        ((static_cast<uint32_t>(sign) & 1u) << FLOAT32_SIGN_OFFSET);
    return bits_to_float(bits);
}

// 布尔值转可读文本，用于打印日志
std::string bool_text(bool value) {
    return value ? "yes" : "no";
}

// --------------- 格式参数计算工具 ---------------

// 计算指定指数位数下的最小正规数
// emin = 2 - 2^(ebits - 1)，对应二进制浮点标准中的最小指数
float min_norm_for_ebits(int ebits) {
    if (ebits == 0) {
        return 0.0f;  // 整数格式没有最小正规数概念
    }
    const int emin = 2 - (1 << (ebits - 1));
    return std::pow(2.0f, static_cast<float>(emin));
}

// 根据元素格式枚举生成该格式的关键数值参数。
// 这些参数后续会被量化逻辑使用，用来判断元素格式的指数范围、尾数精度、
// 最大可表示正规数以及最小正规数。
//
// 这里保留了多种格式的参数定义，但 main() 中实际选择的是 FP8_E4M3，
// 这样做是为了让演示代码和 microxcaling 仓库中的通用格式定义保持一致。
FormatParams get_format_params(ElemFormat fmt) {
    FormatParams params;
    params.format = fmt;

    // 根据不同目标格式填入指数位数、尾数相关位数和最大指数。
    // 注意 FP8_E4M3 / FP6 / FP4 的 emax 写法与 IEEE half/bfloat16 不完全相同，
    // 这是为了对齐 MX 仓库中 formats.py 的格式定义。
    switch (fmt) {
        case ElemFormat::int8:
            params.name = "INT8";
            params.ebits = 0;
            params.mbits = 8;
            params.emax = 0;
            break;
        case ElemFormat::int4:
            params.name = "INT4";
            params.ebits = 0;
            params.mbits = 4;
            params.emax = 0;
            break;
        case ElemFormat::int2:
            params.name = "INT2";
            params.ebits = 0;
            params.mbits = 2;
            params.emax = 0;
            break;
        case ElemFormat::fp8_e5m2:
            params.name = "FP8_E5M2";
            params.ebits = 5;
            params.mbits = 4;
            params.emax = (1 << (params.ebits - 1)) - 1;
            break;
        case ElemFormat::fp8_e4m3:
            params.name = "FP8_E4M3";
            params.ebits = 4;
            params.mbits = 5;
            params.emax = 1 << (params.ebits - 1);
            break;
        case ElemFormat::fp6_e3m2:
            params.name = "FP6_E3M2";
            params.ebits = 3;
            params.mbits = 4;
            params.emax = 1 << (params.ebits - 1);
            break;
        case ElemFormat::fp6_e2m3:
            params.name = "FP6_E2M3";
            params.ebits = 2;
            params.mbits = 5;
            params.emax = 1 << (params.ebits - 1);
            break;
        case ElemFormat::fp4:
            params.name = "FP4_E2M1";
            params.ebits = 2;
            params.mbits = 3;
            params.emax = 1 << (params.ebits - 1);
            break;
        case ElemFormat::float16:
            params.name = "FLOAT16";
            params.ebits = 5;
            params.mbits = 12;
            params.emax = (1 << (params.ebits - 1)) - 1;
            break;
        case ElemFormat::bfloat16:
            params.name = "BFLOAT16";
            params.ebits = 8;
            params.mbits = 9;
            params.emax = (1 << (params.ebits - 1)) - 1;
            break;
        default:
            throw std::invalid_argument("unknown element format");
    }

    // 计算目标格式的最大正规数。
    // FP8_E4M3 是特殊格式: 指数全 1 不作为 Inf/NaN 专用，最大有限值为 448。
    // 其他格式使用通用公式: 2^emax * 最大有效尾数比例。
    if (fmt == ElemFormat::fp8_e4m3) {
        params.max_norm = std::pow(2.0f, static_cast<float>(params.emax)) * 1.75f;
    } else {
        params.max_norm = std::pow(2.0f, static_cast<float>(params.emax)) *
                          static_cast<float>((1 << (params.mbits - 1)) - 1) /
                          static_cast<float>(1 << (params.mbits - 2));
    }
    params.min_norm = min_norm_for_ebits(params.ebits);
    return params;
}

// 将共享指数限制到 E{ebits}M0 scale 能表达的范围。
//
// shared_exp 传入的是 FP32 有偏指数，例如实际指数 8 对应 biased exponent 135。
// MX 的 shared scale 只保存指数，不保存普通尾数，因此必须确保指数在 scale 格式范围内。
int clamp_shared_exp(int shared_exp, int ebits) {
    const int emax = ebits != 0 ? (1 << (ebits - 1)) - 1 : FLOAT32_EXP_MAX;
    const int shared_ub = shared_exp - FLOAT32_EXP_BIAS;
    shared_exp = shared_ub > emax ? FLOAT32_EXP_MAX : shared_exp;
    shared_exp = shared_ub < -emax ? FLOAT32_EXP_BIAS - emax : shared_exp;
    return shared_exp;
}

// 根据 block 最大值的指数计算 MX shared scale。
//
// MX scaling 不是直接计算 abs_max / max_norm，而是只利用 block 最大值的指数，
// 再减去目标元素格式最大值的指数，得到一个 2 的幂次 scale。
// 这种设计便于硬件实现: 每个 block 共享一个 E8M0 scale，每个元素只需做指数对齐。
float mx_get_shared_scale(int shared_exp, int scale_bits, float elem_max_norm) {
    const int elem_emax = get_unbiased_exponent(elem_max_norm);
    shared_exp = shared_exp != FLOAT32_EXP_MAX ? shared_exp - elem_emax : shared_exp;
    shared_exp = clamp_shared_exp(shared_exp, scale_bits);

    const int scale_mant =
        (shared_exp == 0 || shared_exp == FLOAT32_EXP_MAX) ? (FLOAT32_IMPLIED1 >> 1) : 0;
    return construct_float(0, shared_exp, scale_mant);
}

// 将 FP32 尾数压缩到低精度格式需要的尾数宽度。
//
// 低精度浮点的尾数位更少，因此需要丢弃低位。丢弃低位时不能简单截断，
// 否则会产生系统性误差；这里根据 rounding_mode 对尾数进行舍入。
void shift_right_round_mantissa(
    int& mantissa,
    bool is_subnorm,
    int mbits,
    int exp_diff,
    RoundingMode rounding_mode,
    bool allow_overflow) {
    mantissa = is_subnorm ? mantissa : mantissa + FLOAT32_IMPLIED1;
    const int fp32_sig_bits = is_subnorm ? 23 : 24;

    // tie/even 只用于 round_to_nearest_even:
    // tie 表示被丢弃部分刚好处于 0.5 的中点；even 表示保留下来的最低位为偶数。
    // 如果同时满足 tie && even，则不进位，从而实现“最近偶数”规则。
    bool tie = false;
    bool even = false;
    if (rounding_mode == RoundingMode::round_to_nearest_even) {
        const int tbits = exp_diff + (fp32_sig_bits - mbits);
        if (tbits > 0) {
            int mask = (1 << (tbits - 1)) - 1;
            tie = !(mantissa & mask);
            mask = 1 << tbits;
            even = !(mantissa & mask);
        }
    }

    // 第一次右移用于处理目标格式指数下溢产生的次正规数对齐。
    mantissa = mantissa >> exp_diff;
    // 第二次右移把 FP32 的有效位数压缩到目标格式有效位数附近，额外保留 1 位用于舍入判断。
    mantissa = mantissa >> (fp32_sig_bits - mbits - 1);

    if ((rounding_mode == RoundingMode::round_away_from_zero ||
         rounding_mode == RoundingMode::round_to_nearest_even) &&
        (allow_overflow || mantissa != ((1 << (mbits + 1)) - 1))) {
        if (!(tie && even)) {
            mantissa += 1;
        }
    }

    // 舍入完成后去掉额外保留的舍入位，得到最终目标尾数字段。
    mantissa = mantissa >> 1;
}

// 将目标格式尾数重新放回 FP32 尾数字段的位置。
//
// quantize_elemwise 的返回值仍然是 float，因此量化后的低精度值需要被重新编码成
// 一个 FP32 可表示的数。这里做的不是恢复精度，而是把“低精度网格点”表示为 FP32。
bool shift_left_mantissa(int& mantissa, bool is_subnorm, int mbits, int exp_diff) {
    const int fp32_sig_bits = is_subnorm ? 23 : 24;
    mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff);
    const bool overflow = mantissa >= (1 << fp32_sig_bits);
    mantissa = (overflow && !is_subnorm) ? mantissa >> 1 : mantissa;
    mantissa = mantissa & (FLOAT32_IMPLIED1 - 1);
    return overflow;
}

// 对单个已经缩放后的元素执行低精度量化。
//
// input 通常是原始值除以 shared scale 后的结果。该函数输出的仍然是 float，
// 但数值已经被限制到目标格式能够表示的离散点上。
float quantize_elemwise(
    float input,
    int bits,
    int exp_bits,
    float max_norm,
    RoundingMode rounding_mode,
    bool saturate_normals,
    bool allow_denorm,
    bool& clamped) {
    clamped = false;
    if (input == 0.0f) {
        return 0.0f;
    }

    // 从 FP32 中拆出符号、指数和尾数，后续直接在位级别模拟低精度格式的舍入过程。
    int biased_exp = get_biased_exponent(input);
    const int sign = get_sign(input);
    int tmant = get_trailing_mantissa(input);

    // 目标格式中除符号位外可用于数值表达的位数。
    const int mbits = bits - 1;
    const bool is_int = exp_bits == 0;
    const int new_bias = is_int ? 1 : (1 << (exp_bits - 1)) - 1;
    const int new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias;

    // 如果目标格式不允许次正规数，且转换后的指数落到正规数范围以下，则直接 flush 为 0。
    if (!is_int && !allow_denorm && new_biased_exp < 1) {
        return 0.0f;
    }

    // exp_diff 表示为了表达目标格式次正规数，需要额外右移尾数的位数。
    int exp_diff = new_biased_exp <= 0 ? 1 - new_biased_exp : 0;
    exp_diff = exp_diff > FLOAT32_FULL_MBITS ? FLOAT32_FULL_MBITS : exp_diff;

    const bool is_subnormal = biased_exp == 0;
    shift_right_round_mantissa(
        tmant, is_subnormal, mbits, exp_diff, rounding_mode, !is_int);

    if (tmant == 0) {
        return 0.0f;
    }

    // 尾数舍入后可能发生进位溢出，例如 1.111 舍入成 10.000，此时指数需要加 1。
    const bool overflow = shift_left_mantissa(tmant, is_subnormal, mbits, exp_diff);
    biased_exp = overflow ? biased_exp + 1 : biased_exp;

    // 将低精度网格点重新组装成 FP32，方便后续打印和误差计算。
    float output = construct_float(sign, biased_exp, tmant);
    if (std::fabs(output) > max_norm) {
        clamped = true;
        if (is_int || saturate_normals) {
            output = sign ? -max_norm : max_norm;
        } else {
            output = construct_float(sign, 0xff, 0);
        }
    }

    return output;
}

// 将量化后的数值编码成 FP8_E4M3 的 8-bit 原始位模式。
//
// quantize_elemwise 返回的是 FP32 数值形式；为了展示真实 FP8 编码，
// 这里把符号位、4-bit 指数字段和 3-bit 尾数字段打包成一个 uint8_t。
uint8_t encode_fp8_e4m3(float value, const FormatParams& fmt) {
    if (value == 0.0f) {
        return static_cast<uint8_t>(get_sign(value) << 7);
    }

    // FP8_E4M3: 1-bit sign | 4-bit exponent | 3-bit mantissa。
    const int sign = get_sign(value);
    float abs_value = std::fabs(value);
    if (abs_value > fmt.max_norm) {
        abs_value = fmt.max_norm;
    }

    const int src_exp = get_unbiased_exponent(abs_value);
    const int target_bias = (1 << (fmt.ebits - 1)) - 1;
    int target_exp = src_exp + target_bias;
    int mantissa_field = 0;

    // target_exp <= 0 表示该值在 FP8_E4M3 下属于次正规数区间。
    if (target_exp <= 0) {
        const float step = std::pow(2.0f, static_cast<float>(1 - target_bias - 3));
        mantissa_field = static_cast<int>(std::floor(abs_value / step + 0.5f));
        if (mantissa_field > 7) {
            mantissa_field = 7;
        }
        target_exp = 0;
    } else {
        // 正规数路径: 先把 abs_value 归一化到 [1, 2)，再提取 3-bit 尾数字段。
        const float scaled = std::ldexp(abs_value, -src_exp);
        mantissa_field = static_cast<int>(std::floor((scaled - 1.0f) * 8.0f + 0.5f));
        if (mantissa_field == 8) {
            mantissa_field = 0;
            target_exp += 1;
        }
        if (target_exp > 15) {
            target_exp = 15;
            mantissa_field = 6; // 448 = 1.75 * 2^8，对应 E4M3 最大有限值。
        }
    }

    return static_cast<uint8_t>((sign << 7) | ((target_exp & 0x0f) << 3) | (mantissa_field & 0x07));
}

// 对单个 block 元素执行完整 MX 量化路径:
//   input -> input / shared scale -> 低精度量化 -> 乘回 shared scale。
//
// flush_tile 用于模拟当整个 tile/block 被判定为 FP32 次正规且需要 flush 时，
// 所有元素直接按 0 处理的行为。
QuantizedElement quantize_mx_elem(
    float input,
    float scale,
    bool flush_tile,
    const FormatParams& elem_format,
    RoundingMode rounding_mode) {
    QuantizedElement result;
    result.original = input;
    // MX 的核心思想: block 内所有元素共用一个 shared scale，元素自身只保存低精度值。
    result.scaled_input = flush_tile ? 0.0f : input / scale;
    result.clamped = false;

    // 对缩放后的值进行 FP8_E4M3 量化。此时量化范围由 elem_format.max_norm 决定。
    result.quantized_scaled = quantize_elemwise(
        result.scaled_input,
        elem_format.mbits,
        elem_format.ebits,
        elem_format.max_norm,
        rounding_mode,
        true,
        true,
        result.clamped);
    // 反量化时乘回同一个 shared scale，得到近似原始值的 FP32 表示。
    result.dequantized = result.quantized_scaled * scale;
    result.fp8_bits = encode_fp8_e4m3(result.quantized_scaled, elem_format);
    return result;
}

// 查找 block 内绝对值最大值。
//
// shared exponent 由 block 最大绝对值决定，因此这个函数对应 MX 量化流程中的
// reduce-absmax 阶段。遇到 NaN 时保留 NaN 传播语义，便于暴露异常输入。
float block_abs_max(const std::vector<float>& block) {
    float abs_max = 0.0f;
    for (size_t i = 0; i < block.size(); ++i) {
        const float abs_value = std::fabs(block[i]);
        if (abs_value > abs_max || std::isnan(abs_value)) {
            abs_max = abs_value;
        }
    }
    return abs_max;
}

// 构造一个 32 元素演示 block。
//
// MX 常以固定 block size 共享一个 scale；这里用 32 个元素模拟一个 block，
// 并刻意放入负数、零、次正规数、大数和普通小数，方便观察不同量化行为。
std::vector<float> make_demo_block() {
    std::vector<float> block(32, 0.0f);
    
    // block[0] = 1000.0f;       // 展示 scale > 1 以及饱和。
    block[0] = 300.5f;        // 展示大数舍入。
    block[1] = -15.25f;       // 展示符号位处理。
    block[2] = 1.0e-40f;      // 展示 FP32 次正规数输入。
    block[3] = -2.7f;         // 展示负数舍入。
    block[4] = -0.03125f;     // 展示小正规数。
    block[5] = 0.0f;          // 展示精确零。
    block[6] = 1.2f;          // 展示普通非精确可表示值。
    for (size_t i = 8; i < block.size(); ++i) {
        block[i] = (i % 2 == 0 ? 1.0f : -1.0f) * (0.125f * static_cast<float>(i - 7));
    }
    return block;
}

// 打印目标元素格式的关键参数，帮助确认当前演示使用的低精度格式。
void print_format_summary(const FormatParams& fmt) {
    std::cout << "Element format: " << fmt.name << "\n";
    std::cout << "  ebits=" << fmt.ebits << ", mbits=" << fmt.mbits
              << ", emax=" << fmt.emax << "\n";
    std::cout << "  max_norm=" << fmt.max_norm << ", min_norm=" << fmt.min_norm << "\n";
}

// 打印 block 级别信息，包括 abs_max、shared exponent 和最终 E8M0 shared scale。
void print_block_summary(const std::vector<float>& block, float abs_max, float scale) {
    const int shared_exp = get_biased_exponent(abs_max);
    const int scale_exp = get_biased_exponent(scale);

    std::cout << "Block size: " << block.size() << "\n";
    std::cout << "Block abs max: " << abs_max << "\n";
    std::cout << "Shared exponent: biased=" << shared_exp
              << ", unbiased=" << get_unbiased_exponent(abs_max) << "\n";
    std::cout << "Shared scale (E8M0): " << scale
              << "  [biased_exp=" << scale_exp
              << ", unbiased_exp=" << get_unbiased_exponent(scale) << "]\n";
}

// 以表格形式打印每个元素的量化结果。
//
// 表格列含义:
// original:  原始 FP32 输入
// x/scale:   除以 shared scale 后送入 FP8 量化器的值
// fp8_value: FP8 网格点对应的数值
// dequant:   乘回 shared scale 后的反量化值
// fp8_hex:   FP8_E4M3 原始 8-bit 编码
// clamped:   是否发生饱和/钳制
// abs_error: 反量化值与原始值的绝对误差
void print_quantized_table(const std::vector<QuantizedElement>& rows) {
    std::cout << std::left
              << std::setw(6) << "idx"
              << std::setw(14) << "original"
              << std::setw(14) << "x/scale"
              << std::setw(14) << "fp8_value"
              << std::setw(14) << "dequant"
              << std::setw(12) << "fp8_hex"
              << std::setw(10) << "clamped"
              << "abs_error" << "\n";

    for (size_t i = 0; i < rows.size(); ++i) {
        const QuantizedElement& r = rows[i];
        const float abs_error = std::fabs(r.dequantized - r.original);
        std::cout << std::left
                  << std::setw(6) << i
                  << std::setw(14) << r.original
                  << std::setw(14) << r.scaled_input
                  << std::setw(14) << r.quantized_scaled
                  << std::setw(14) << r.dequantized;
        std::cout << "0x" << std::hex << std::uppercase << std::setw(2)
                  << std::setfill('0') << static_cast<int>(r.fp8_bits)
                  << std::dec << std::nouppercase << std::setfill(' ');
        std::cout << std::setw(8) << ""
                  << std::setw(10) << bool_text(r.clamped)
                  << abs_error << "\n";
    }
}

} // namespace

// 程序入口: 串联完整 MXFP8 量化演示流程。
int main() {
    // 本演示固定使用 FP8_E4M3 作为元素格式，scale 使用 E8M0。
    const FormatParams elem_format = get_format_params(ElemFormat::fp8_e4m3);
    const int scale_bits = 8;
    const RoundingMode rounding_mode = RoundingMode::round_away_from_zero;
    const bool flush_fp32_subnorms = false;

    // 1. 构造一个 block，并计算 block 级别 abs max。
    const std::vector<float> block = make_demo_block();
    const float abs_max = block_abs_max(block);
    // 2. 从 abs max 中提取 shared exponent，作为计算 shared scale 的基础。
    const int shared_exp = get_biased_exponent(abs_max);
    const bool flush_tile = shared_exp == 0 && flush_fp32_subnorms;
    // 3. 根据 shared exponent 和元素格式最大值，生成 E8M0 shared scale。
    const float scale = mx_get_shared_scale(shared_exp, scale_bits, elem_format.max_norm);

    // 4. 对 block 内每个元素执行共享 scale 下的逐元素量化。
    std::vector<QuantizedElement> quantized;
    quantized.reserve(block.size());
    for (size_t i = 0; i < block.size(); ++i) {
        quantized.push_back(quantize_mx_elem(block[i], scale, flush_tile, elem_format, rounding_mode));
    }

    // 5. 打印格式参数、block 参数和逐元素量化结果。
    std::cout << "=== Complete MXFP8 Quantization Demo ===\n\n";
    print_format_summary(elem_format);
    std::cout << "Scale format: E" << scale_bits << "M0\n";
    std::cout << "Rounding mode: round away from zero (repository default)\n\n";

    print_block_summary(block, abs_max, scale);
    std::cout << "Flush FP32 subnormal tile: " << bool_text(flush_tile) << "\n\n";

    print_quantized_table(quantized);

    std::cout << "\nFlow check:\n";
    std::cout << "  original -> divide by shared scale -> quantize to " << elem_format.name
              << " -> multiply by shared scale\n";
    std::cout << "  MX computes the shared scale from the block max exponent, not from exact max/max_norm division.\n";
    std::cout << "  Here max abs 1000 has exponent 9, FP8_E4M3 max_norm 448 has exponent 8, so scale is 2.\n";
    std::cout << "  1000 / 2 = 500 exceeds FP8_E4M3 max_norm 448, so the element saturates to 448 and dequantizes to 896.\n";

    return 0;
}
