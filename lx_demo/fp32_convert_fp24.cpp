#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

struct FP24 {
    // 只使用低 24 bit:
    // bit 23      : sign
    // bit 22 - 17 : exponent, 6 bits
    // bit 16 - 0  : fraction, 17 bits
    uint32_t bits = 0;
};

static uint32_t float_to_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static float bits_to_float(uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

static uint64_t round_shift_right_even(uint64_t value, int shift) {
    if (shift <= 0) {
        return value << (-shift);
    }

    if (shift >= 64) {
        return 0;
    }

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
    constexpr int FP32_BIAS = 127;
    constexpr int FP24_BIAS = 31;
    constexpr int FP24_EXP_MAX = 0x3F;       // 6 bits all ones
    constexpr int FP24_FRAC_BITS = 17;
    constexpr uint32_t FP24_FRAC_MASK = (1u << FP24_FRAC_BITS) - 1;

    uint32_t u = float_to_bits(x);

    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp32 = (u >> 23) & 0xFF;
    uint32_t frac32 = u & 0x7FFFFF;

    FP24 out;
    uint32_t sign24 = sign << 23;

    // FP32 NaN / Inf
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
    if (exp32 == 0 && frac32 == 0) {
        out.bits = sign24;
        return out;
    }

    // FP32 subnormal 对 FP24 来说极小，基本都会下溢为 0
    if (exp32 == 0) {
        out.bits = sign24;
        return out;
    }

    // FP32 normal
    int e = static_cast<int>(exp32) - FP32_BIAS;  // 真实指数
    int exp24 = e + FP24_BIAS;

    // FP32 significand: 1.frac, 共 24 bit
    uint64_t sig = (uint64_t(1) << 23) | frac32;

    // FP24 overflow -> Inf
    if (exp24 >= FP24_EXP_MAX) {
        out.bits = sign24 | (FP24_EXP_MAX << FP24_FRAC_BITS);
        return out;
    }

    // FP24 normal
    if (exp24 > 0) {
        // FP32 significand 24 bit -> FP24 significand 18 bit
        // 包含 hidden bit，所以从 23 frac bit 降到 17 frac bit，需要右移 6
        uint64_t rounded_sig = round_shift_right_even(sig, 23 - FP24_FRAC_BITS);

        // 舍入导致 1.111... -> 10.000...
        if (rounded_sig == (uint64_t(1) << (FP24_FRAC_BITS + 1))) {
            rounded_sig >>= 1;
            exp24 += 1;

            if (exp24 >= FP24_EXP_MAX) {
                out.bits = sign24 | (FP24_EXP_MAX << FP24_FRAC_BITS);
                return out;
            }
        }

        uint32_t frac24 = static_cast<uint32_t>(rounded_sig) & FP24_FRAC_MASK;
        out.bits = sign24 | (static_cast<uint32_t>(exp24) << FP24_FRAC_BITS) | frac24;
        return out;
    }

    // FP24 subnormal
    // FP24 subnormal value = frac24 * 2^(1 - bias - frac_bits)
    //                       = frac24 * 2^(-47)
    //
    // FP32 normal value = sig * 2^(e - 23)
    //
    // frac24 = round(value / 2^-47)
    //        = round(sig * 2^(e - 23 + 47))
    //        = round(sig * 2^(e + 24))
    int shift = -(e + 24);
    uint64_t frac24_rounded = round_shift_right_even(sig, shift);

    if (frac24_rounded == 0) {
        out.bits = sign24;
        return out;
    }

    // 舍入到最小 normal
    if (frac24_rounded >= (uint64_t(1) << FP24_FRAC_BITS)) {
        out.bits = sign24 | (1u << FP24_FRAC_BITS);
        return out;
    }

    out.bits = sign24 | static_cast<uint32_t>(frac24_rounded);
    return out;
}

float fp24_to_fp32(FP24 h) {
    constexpr int FP24_BIAS = 31;
    constexpr int FP24_FRAC_BITS = 17;
    constexpr uint32_t FP24_EXP_MAX = 0x3F;
    constexpr uint32_t FP24_FRAC_MASK = (1u << FP24_FRAC_BITS) - 1;

    uint32_t sign = (h.bits >> 23) & 0x1;
    uint32_t exp = (h.bits >> FP24_FRAC_BITS) & 0x3F;
    uint32_t frac = h.bits & FP24_FRAC_MASK;

    float value;

    if (exp == FP24_EXP_MAX) {
        if (frac == 0) {
            value = std::numeric_limits<float>::infinity();
        } else {
            value = std::numeric_limits<float>::quiet_NaN();
        }
    } else if (exp == 0) {
        if (frac == 0) {
            value = 0.0f;
        } else {
            // subnormal: frac * 2^-47
            value = std::ldexp(static_cast<float>(frac), -47);
        }
    } else {
        // normal: (1.frac) * 2^(exp - bias)
        uint32_t sig = (1u << FP24_FRAC_BITS) | frac;
        value = std::ldexp(static_cast<float>(sig), static_cast<int>(exp) - FP24_BIAS - FP24_FRAC_BITS);
    }

    return sign ? -value : value;
}

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
    test(0.0f);
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