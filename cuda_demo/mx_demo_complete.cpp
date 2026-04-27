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

FormatParams get_format_params(ElemFormat fmt) {
    FormatParams params;
    params.format = fmt;

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

int clamp_shared_exp(int shared_exp, int ebits) {
    const int emax = ebits != 0 ? (1 << (ebits - 1)) - 1 : FLOAT32_EXP_MAX;
    const int shared_ub = shared_exp - FLOAT32_EXP_BIAS;
    shared_exp = shared_ub > emax ? FLOAT32_EXP_MAX : shared_exp;
    shared_exp = shared_ub < -emax ? FLOAT32_EXP_BIAS - emax : shared_exp;
    return shared_exp;
}

float mx_get_shared_scale(int shared_exp, int scale_bits, float elem_max_norm) {
    const int elem_emax = get_unbiased_exponent(elem_max_norm);
    shared_exp = shared_exp != FLOAT32_EXP_MAX ? shared_exp - elem_emax : shared_exp;
    shared_exp = clamp_shared_exp(shared_exp, scale_bits);

    const int scale_mant =
        (shared_exp == 0 || shared_exp == FLOAT32_EXP_MAX) ? (FLOAT32_IMPLIED1 >> 1) : 0;
    return construct_float(0, shared_exp, scale_mant);
}

void shift_right_round_mantissa(
    int& mantissa,
    bool is_subnorm,
    int mbits,
    int exp_diff,
    RoundingMode rounding_mode,
    bool allow_overflow) {
    mantissa = is_subnorm ? mantissa : mantissa + FLOAT32_IMPLIED1;
    const int fp32_sig_bits = is_subnorm ? 23 : 24;

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

    mantissa = mantissa >> exp_diff;
    mantissa = mantissa >> (fp32_sig_bits - mbits - 1);

    if ((rounding_mode == RoundingMode::round_away_from_zero ||
         rounding_mode == RoundingMode::round_to_nearest_even) &&
        (allow_overflow || mantissa != ((1 << (mbits + 1)) - 1))) {
        if (!(tie && even)) {
            mantissa += 1;
        }
    }

    mantissa = mantissa >> 1;
}

bool shift_left_mantissa(int& mantissa, bool is_subnorm, int mbits, int exp_diff) {
    const int fp32_sig_bits = is_subnorm ? 23 : 24;
    mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff);
    const bool overflow = mantissa >= (1 << fp32_sig_bits);
    mantissa = (overflow && !is_subnorm) ? mantissa >> 1 : mantissa;
    mantissa = mantissa & (FLOAT32_IMPLIED1 - 1);
    return overflow;
}

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

    int biased_exp = get_biased_exponent(input);
    const int sign = get_sign(input);
    int tmant = get_trailing_mantissa(input);

    const int mbits = bits - 1;
    const bool is_int = exp_bits == 0;
    const int new_bias = is_int ? 1 : (1 << (exp_bits - 1)) - 1;
    const int new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias;

    if (!is_int && !allow_denorm && new_biased_exp < 1) {
        return 0.0f;
    }

    int exp_diff = new_biased_exp <= 0 ? 1 - new_biased_exp : 0;
    exp_diff = exp_diff > FLOAT32_FULL_MBITS ? FLOAT32_FULL_MBITS : exp_diff;

    const bool is_subnormal = biased_exp == 0;
    shift_right_round_mantissa(
        tmant, is_subnormal, mbits, exp_diff, rounding_mode, !is_int);

    if (tmant == 0) {
        return 0.0f;
    }

    const bool overflow = shift_left_mantissa(tmant, is_subnormal, mbits, exp_diff);
    biased_exp = overflow ? biased_exp + 1 : biased_exp;

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

uint8_t encode_fp8_e4m3(float value, const FormatParams& fmt) {
    if (value == 0.0f) {
        return static_cast<uint8_t>(get_sign(value) << 7);
    }

    const int sign = get_sign(value);
    float abs_value = std::fabs(value);
    if (abs_value > fmt.max_norm) {
        abs_value = fmt.max_norm;
    }

    const int src_exp = get_unbiased_exponent(abs_value);
    const int target_bias = (1 << (fmt.ebits - 1)) - 1;
    int target_exp = src_exp + target_bias;
    int mantissa_field = 0;

    if (target_exp <= 0) {
        const float step = std::pow(2.0f, static_cast<float>(1 - target_bias - 3));
        mantissa_field = static_cast<int>(std::floor(abs_value / step + 0.5f));
        if (mantissa_field > 7) {
            mantissa_field = 7;
        }
        target_exp = 0;
    } else {
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

QuantizedElement quantize_mx_elem(
    float input,
    float scale,
    bool flush_tile,
    const FormatParams& elem_format,
    RoundingMode rounding_mode) {
    QuantizedElement result;
    result.original = input;
    result.scaled_input = flush_tile ? 0.0f : input / scale;
    result.clamped = false;

    result.quantized_scaled = quantize_elemwise(
        result.scaled_input,
        elem_format.mbits,
        elem_format.ebits,
        elem_format.max_norm,
        rounding_mode,
        true,
        true,
        result.clamped);
    result.dequantized = result.quantized_scaled * scale;
    result.fp8_bits = encode_fp8_e4m3(result.quantized_scaled, elem_format);
    return result;
}

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

std::vector<float> make_demo_block() {
    std::vector<float> block(32, 0.0f);
    // block[0] = 1000.0f;       // 展示 scale > 1 以及饱和。
    block[0] = -2.7f;         // 展示负数舍入。
    block[1] = -15.25f;       // 展示符号位处理。
    block[2] = 1.0e-40f;      // 展示 FP32 次正规数输入。
    block[3] = 300.5f;        // 展示大数舍入。
    block[4] = -0.03125f;     // 展示小正规数。
    block[5] = 0.0f;          // 展示精确零。
    block[6] = 1.2f;          // 展示普通非精确可表示值。
    for (size_t i = 8; i < block.size(); ++i) {
        block[i] = (i % 2 == 0 ? 1.0f : -1.0f) * (0.125f * static_cast<float>(i - 7));
    }
    return block;
}

void print_format_summary(const FormatParams& fmt) {
    std::cout << "Element format: " << fmt.name << "\n";
    std::cout << "  ebits=" << fmt.ebits << ", mbits=" << fmt.mbits
              << ", emax=" << fmt.emax << "\n";
    std::cout << "  max_norm=" << fmt.max_norm << ", min_norm=" << fmt.min_norm << "\n";
}

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

int main() {
    const FormatParams elem_format = get_format_params(ElemFormat::fp8_e4m3);
    const int scale_bits = 8;
    const RoundingMode rounding_mode = RoundingMode::round_away_from_zero;
    const bool flush_fp32_subnorms = false;

    const std::vector<float> block = make_demo_block();
    const float abs_max = block_abs_max(block);
    const int shared_exp = get_biased_exponent(abs_max);
    const bool flush_tile = shared_exp == 0 && flush_fp32_subnorms;
    const float scale = mx_get_shared_scale(shared_exp, scale_bits, elem_format.max_norm);

    std::vector<QuantizedElement> quantized;
    quantized.reserve(block.size());
    for (size_t i = 0; i < block.size(); ++i) {
        quantized.push_back(quantize_mx_elem(block[i], scale, flush_tile, elem_format, rounding_mode));
    }

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
