#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

// 这个 demo 演示一种自定义 FP24 浮点格式和 IEEE-754 FP32(float) 之间的互转。
//
// FP24 格式总共 24 bit，按如下方式划分：
//   sign     : 1 bit
//   exponent : 6 bits，bias = 31
//   fraction : 17 bits
//
// 数值规则和 IEEE-754 类似：
//   exp == 0x3F, frac == 0  -> infinity
//   exp == 0x3F, frac != 0  -> NaN
//   exp == 0,    frac == 0  -> signed zero
//   exp == 0,    frac != 0  -> subnormal，value = frac * 2^-47
//   otherwise               -> normal，value = (1.fraction) * 2^(exp - 31)
struct FP24 {
    // 只使用低 24 bit:
    // bit 23      : sign
    // bit 22 - 17 : exponent, 6 bits
    // bit 16 - 0  : fraction, 17 bits
    uint32_t bits = 0;
};

// 直接取出 float 的 IEEE-754 二进制位模式。
// 使用 memcpy 而不是 reinterpret_cast，避免违反 strict aliasing 规则。
static uint32_t float_to_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

// 把 32 bit 位模式重新解释成 float。
// 这个函数当前 demo 暂时没有使用，保留它方便以后做 bit-level 调试。
static float bits_to_float(uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

// 将整数 value 右移 shift 位，并使用 round-to-nearest-even 舍入。
//
// round-to-nearest-even 也叫 bankers rounding：
//   1. remainder > half：向上进 1
//   2. remainder < half：直接截断
//   3. remainder == half：如果截断后的最低位是 1，则进 1；否则保持偶数
//
// 浮点格式转换时会丢掉低位 fraction bit，这个函数负责让丢位过程尽量无偏。
static uint64_t round_shift_right_even(uint64_t value, int shift) {
    // shift <= 0 表示实际不需要右移，反而要左移补足位数。
    if (shift <= 0) {
        return value << (-shift);
    }

    // 右移 64 位或更多会在 C++ 中产生未定义行为；这里提前兜底。
    if (shift >= 64) {
        return 0;
    }

    // truncated 是保留下来的高位；remainder 是被丢弃的低位。
    uint64_t truncated = value >> shift;
    uint64_t remainder = value & ((uint64_t(1) << shift) - 1);
    uint64_t half = uint64_t(1) << (shift - 1);

    if (remainder > half) {
        return truncated + 1;
    }

    if (remainder == half && (truncated & 1)) {
        return truncated + 1;
    }

    return truncated;
}

FP24 fp32_to_fp24(float x) {
    // FP32: exponent 8 bits, bias = 127, fraction = 23 bits.
    // FP24: exponent 6 bits, bias = 31,  fraction = 17 bits.
    constexpr int FP32_BIAS = 127;
    constexpr int FP24_BIAS = 31;
    constexpr int FP24_EXP_MAX = 0x3F;       // 6 bits all ones
    constexpr int FP24_FRAC_BITS = 17;
    constexpr uint32_t FP24_FRAC_MASK = (1u << FP24_FRAC_BITS) - 1; // 17个1也就是说 0x1FFFF 

    // 拆解 FP32 的 sign / exponent / fraction。
    uint32_t u = float_to_bits(x);  // 把float转换成32位二进制，输出就是 int32的整数

    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp32 = (u >> 23) & 0xFF; // 带bias的指数部分也就是真实的指数加上127
    uint32_t frac32 = u & 0x7FFFFF;

    FP24 out;
    uint32_t sign24 = sign << 23; // 这里就是把符号位不管是 0还是1，进行左移23位，hidden_bit + 23位

    // FP32 NaN / Inf
    // FP32 的 exp 全 1 表示 Inf 或 NaN；FP24 也用 exp 全 1 表示这两类特殊值。
    if (exp32 == 0xFF) {
        if (frac32 == 0) {
            // Inf
            out.bits = sign24 | (FP24_EXP_MAX << FP24_FRAC_BITS);
        } else {
            // NaN: 保留一个 quiet NaN pattern
            out.bits = sign24 | (FP24_EXP_MAX << FP24_FRAC_BITS) | 0x1;
        }
        return out;
    }

    // FP32 zero
    // +0 和 -0 都保留符号位，其余位清零。
    if (exp32 == 0 && frac32 == 0) {
        out.bits = sign24;
        return out;
    }

    // FP32 subnormal 对 FP24 来说极小，基本都会下溢为 0
    // 这里采用简化策略：不尝试把 FP32 subnormal 映射到 FP24 subnormal，直接置为 signed zero。
    if (exp32 == 0) {
        out.bits = sign24;
        return out;
    }

    // FP32 normal
    // e 是 unbiased exponent，也就是真实指数。
    // exp24 是把真实指数换成 FP24 bias 后的 exponent 字段。FP32_BIAS = 127
    int e_real = static_cast<int>(exp32) - FP32_BIAS;  // 真实指数

    // 这个是FP24的指数部分，也就是真实指数加上31 
    int exp24 = e_real + FP24_BIAS;

    // FP24 overflow -> Inf
    // 如果换算后的 exponent 超过 FP24 能表达的 normal 范围，就饱和到 Inf。
    if (exp24 >= FP24_EXP_MAX) {
        out.bits = sign24 | (FP24_EXP_MAX << FP24_FRAC_BITS);
        return out;
    }

    // FP32 significand: 1.frac, 共 24 bit
    // 对 normal FP32，最高位 hidden bit 固定是 1，所以 significand 是 1 + 23 fraction bits。
    // 这行代码就是把 FP32 没有显式存储的 hidden bit 补回来，再和 23 位 fraction 拼成 24 位完整尾数。
    uint64_t fp32_significand_with_hidden_bit = (uint64_t(1) << 23) | frac32;

    // FP24 normal
    if (exp24 > 0) {
        // FP32 significand 24 bit -> FP24 significand 18 bit
        // 包含 hidden bit，所以从 23 frac bit 降到 FP24_FRAC_BITS=17 frac bit，需要右移 6
        uint64_t fp24_significand_with_hidden_bit =
            round_shift_right_even(fp32_significand_with_hidden_bit, 23 - FP24_FRAC_BITS);

        // 舍入导致 1.111... -> 10.000...
        // 这种情况下 significand 多出一位，需要右移并让 exponent 加 1。
        if (fp24_significand_with_hidden_bit == (uint64_t(1) << (FP24_FRAC_BITS + 1))) {
            fp24_significand_with_hidden_bit >>= 1;
            exp24 += 1;

            // exponent 加 1 后仍可能溢出，继续按 Inf 处理。
            if (exp24 >= FP24_EXP_MAX) {
                out.bits = sign24 | (FP24_EXP_MAX << FP24_FRAC_BITS);
                return out;
            }
        }

        // normal 数不保存 hidden bit，只保存低 17 位 fraction。
        uint32_t frac24 = static_cast<uint32_t>(fp24_significand_with_hidden_bit) & FP24_FRAC_MASK;
        out.bits = sign24 | (static_cast<uint32_t>(exp24) << FP24_FRAC_BITS) | frac24;
        return out;

        // sign24:
        // 0 000000 00000000000000000

        // exp24 << 17:
        // 0 100010 00000000000000000

        // frac24:
        // 0 000000 01000000000000000

        // OR 后：
        // 0 100010 01000000000000000
    }

    // FP24 subnormal
    // FP24 subnormal value = frac24 * 2^(1 - bias - frac_bits)
    //                       = frac24 * 2^(-47)
    //
    // FP32 normal value = fp32_significand_with_hidden_bit * 2^(e_real - 23)
    //
    // frac24 = round(value / 2^-47)
    //        = round(fp32_significand_with_hidden_bit * 2^(e_real - 23 + 47))
    //        = round(fp32_significand_with_hidden_bit * 2^(e_real + 24))
    //
    // exp24 <= 0 时，FP32 normal 太小，无法作为 FP24 normal 表示；
    // 这里把它换算成 FP24 subnormal 的 fraction 字段。
    int shift = -(e_real + 24);
    uint64_t frac24_rounded = round_shift_right_even(fp32_significand_with_hidden_bit, shift);

    // 舍入后 fraction 仍为 0，说明小到 FP24 subnormal 也无法表示。
    if (frac24_rounded == 0) {
        out.bits = sign24;
        return out;
    }

    // 舍入到最小 normal
    // subnormal 的 fraction 如果进位到 hidden bit 位置，就等价于最小 normal。
    if (frac24_rounded >= (uint64_t(1) << FP24_FRAC_BITS)) {
        out.bits = sign24 | (1u << FP24_FRAC_BITS);
        return out;
    }

    out.bits = sign24 | static_cast<uint32_t>(frac24_rounded);
    return out;
}

float fp24_to_fp32(FP24 h) {
    // FP24 的字段定义。把它还原成 float 时，不需要手动拼 FP32 bit；
    // 直接用 ldexp(value, exponent) 计算数值更直观。
    constexpr int FP24_BIAS = 31;
    constexpr int FP24_FRAC_BITS = 17;
    constexpr uint32_t FP24_EXP_MAX = 0x3F;
    constexpr uint32_t FP24_FRAC_MASK = (1u << FP24_FRAC_BITS) - 1;

    // 拆解 FP24 的 sign / exponent / fraction。
    uint32_t sign = (h.bits >> 23) & 0x1;
    uint32_t exp = (h.bits >> FP24_FRAC_BITS) & 0x3F;
    uint32_t frac = h.bits & FP24_FRAC_MASK;

    float value;

    // exp 全 1：特殊值 Inf / NaN。
    if (exp == FP24_EXP_MAX) {
        if (frac == 0) {
            value = std::numeric_limits<float>::infinity();
        } else {
            value = std::numeric_limits<float>::quiet_NaN();
        }
    } else if (exp == 0) {
        // exp 全 0：zero / subnormal。
        if (frac == 0) {
            value = 0.0f;
        } else {
            // subnormal: frac * 2^-47
            value = std::ldexp(static_cast<float>(frac), -47);
        }
    } else {
        // normal: (1.frac) * 2^(exp - bias) 左移17为
        // sig 包含 hidden bit，实际数值是 sig * 2^(exp - bias - frac_bits)。
        uint32_t sig = (1u << FP24_FRAC_BITS) | frac; // FP24_FRAC_BITS = 17, FP24_BIAS = 31
        value = std::ldexp(static_cast<float>(sig), static_cast<int>(exp) - FP24_BIAS - FP24_FRAC_BITS);
    }

    // 前面统一计算正数部分，最后再恢复符号。
    return sign ? -value : value;
}

// 把 FP24 的 24 bit 打印成 "s eeeeee fffffffffffffffff" 这种分组形式，
// 方便人工检查 sign / exponent / fraction 是否符合预期。
std::string bits24_string(FP24 h) {
    std::string s;
    for (int i = 23; i >= 0; --i) {
        s.push_back((h.bits & (1u << i)) ? '1' : '0');
        if (i == 23 || i == 17) {
            s.push_back(' ');
        }
    }
    return s;
}

// 对单个输入值做一次往返测试：
//   FP32 -> FP24 -> FP32
// 并打印 FP24 的二进制、十六进制表示和回转误差。
void test(float x) {
    FP24 h = fp32_to_fp24(x);
    float y = fp24_to_fp32(h);

    std::cout << std::setprecision(10);
    std::cout << "fp32 input : " << x << "\n";
    std::cout << "fp24 bits  : " << bits24_string(h) << "\n";
    std::cout << "fp24 hex   : 0x" << std::hex << std::setw(6) << std::setfill('0')
              << h.bits << std::dec << std::setfill(' ') << "\n";
    std::cout << "back fp32  : " << y << "\n";
    std::cout << "abs error  : " << std::fabs(x - y) << "\n";
    std::cout << "-----------------------------\n";
}

int main() {
    //   0/-0、普通正负数、圆周率、小数、极小数、极大数、Inf、NaN。
    test(10.0f);
    test(-0.0f);
    test(1.0f);
    test(-2.5f);
    test(3.1415926f);
    test(0.001f);
    test(1e-12f);
    test(1e10f);
    test(std::numeric_limits<float>::infinity());
    test(-std::numeric_limits<float>::infinity());
    test(std::numeric_limits<float>::quiet_NaN());

    return 0;
}
