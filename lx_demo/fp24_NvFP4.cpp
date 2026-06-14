#include <stdint.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>

typedef uint8_t fp4;
typedef uint8_t fp8; // E4M3

const uint8_t countLeadingZeros8bitsa[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint8_t uint_divide_within_loop(uint32_t &dividend, const uint32_t divisor, unsigned div_loop_num) {
    assert(div_loop_num < 8);
    uint8_t res = 0;
    for (; div_loop_num > 0; --div_loop_num) {
        if (dividend >= divisor) {
            res += 1;
            dividend -= divisor;
        }
        res <<= 1;
        dividend <<= 1;
    }

    res >>= 1;
    return res;
}

// no fp24 subnormal in
fp8 group_fp24_Nvfp4(std::vector<uint32_t> fp24s, std::vector<fp4> &fp4s) {
    assert(fp24s.size() == fp4s.size());
    uint32_t absmax_fp24 = 0x0;
    bool inf_nan = false;
    for (auto fp24 : fp24s) {
        if ((fp24 & 0x7E0000) == 0x7E0000) {
            inf_nan = true;
            break;
        }
        absmax_fp24 = std::max(fp24 & 0x7FFFFF, absmax_fp24);
    }

    // fp24 inf or nan exists
    if (inf_nan) {
        fp4s = std::vector<fp4>(fp24s.size(), 0x0);
        return 0x7f;
    }

    uint32_t dividend = (absmax_fp24 & 0x1FFFF) + 0x20000;
    uint32_t divisor = 0x30000;
    uint8_t exp = (absmax_fp24 & 0x7E0000) >> 17;
    if (dividend < divisor) {
        exp += 7 - 31 - 2 - 1;
        dividend <<= 1;
    }
    else {
        exp += 7 - 31 - 2;
    }

    // exp <= -4, including the case of all zeros
    if ((exp & 0x80) && (exp <= 0xFC)) {
        fp4s = std::vector<fp4>(fp24s.size(), 0x0);
        return 0x0;
    }
    else {
        fp8 scale;
        if ((! exp) || (exp & 0x80)) {
            uint8_t div_loop_num = exp ? 3 - (~exp + 1) : 3;
            scale = uint_divide_within_loop(dividend, divisor, div_loop_num);
            // rounding toward positive
            if (dividend) {
                scale += 1;
            }
        }
        else {
            uint8_t div_loop_num = 4;
            scale = uint_divide_within_loop(dividend, divisor, div_loop_num);
            // rounding toward positive
            if (dividend) {
                scale += 1;
                if (scale & 0x10) {
                    exp += 1;
                    scale >>= 1;
                }
            }

            // overflow clip
            if (exp > 0xf) {
                exp = 0xf;
            }
            scale = (scale & 0x7) + (exp << 3);
            // overflow clip
            if (scale == 0x7f) {
                scale = 0x7E;
            }
        }

        uint8_t scale_exp = (scale & 0x78) >> 3;
        uint32_t scale_mant = scale & 0x7;
        if (scale & 0x78) {
            scale_mant += 0x8;
            scale_mant <<= 14;
        }
        else {
            scale_exp -= countLeadingZeros8bitsa[scale] - 5;
            scale_mant <<= 14 + countLeadingZeros8bitsa[scale] - 5 + 1;
        }

        for (unsigned i = 0; i < fp24s.size(); ++i) {
            uint8_t sign = (fp24s[i] & 0x800000) >> 20;
            uint8_t exp = (fp24s[i] & 0x7e0000) >> 17;
            if (! exp) {
                fp4s[i] = 0x0;
                continue;
            }
            uint32_t mantissa = (fp24s[i] & 0x1ffff) + 0x20000;

            if (mantissa < scale_mant) {
                exp += -31 - (scale_exp - 7) - 1 + 1;
                mantissa <<= 1;
            }
            else {
                exp += -31 - (scale_exp - 7) + 1;
            }

            if ((exp & 0x80) && (exp < 0xFF)) {
                // exp < -1, underflow
                fp4s[i] = 0x0;
            }
            else if ((! exp) || (exp & 0x80)) {
                uint8_t div_loop_num = exp ? 2 - (~exp + 1) : 2;
                auto fp4_mantissa = uint_divide_within_loop(mantissa, scale_mant, div_loop_num);
                // RNE
                fp4_mantissa += (fp4_mantissa & 0x1) ? (mantissa ? 0x2 : (fp4_mantissa & 0x2)) : 0;
                fp4_mantissa >>= 1;
                fp4s[i] = fp4_mantissa ? (sign + fp4_mantissa) : 0x0;
            }
            else {
                uint8_t div_loop_num = 3;
                auto fp4_mantissa = uint_divide_within_loop(mantissa, scale_mant, div_loop_num);
                fp4_mantissa += (fp4_mantissa & 0x1) ? (mantissa ? 0x2 : (fp4_mantissa & 0x2)) : 0;
                if (fp4_mantissa & 0x8) {
                    exp += 1;
                    fp4_mantissa = (fp4_mantissa >> 2) & 0x1;
                }
                else {
                    fp4_mantissa = (fp4_mantissa >> 1) & 0x1;
                }

                // overflow clip
                if (exp > 0x3) {
                    exp = 0x3;
                }

                fp4s[i] = sign + (exp << 1) + fp4_mantissa;
            }
        }
        return scale;
    }
}



static float fp24_to_float(uint32_t fp24) {
    uint8_t sign = (fp24 >> 23) & 1;
    int32_t exp = (fp24 >> 17) & 0x3F;
    uint32_t mant = fp24 & 0x1FFFF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = (float)mant / 0x20000 * std::pow(2.0f, -30);
        return sign ? -val : val;
    }
    if (exp == 0x3F) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    float val = (1.0f + (float)mant / 0x20000) * std::pow(2.0f, (int32_t)(exp - 31));
    return sign ? -val : val;
}

static float fp4_to_float(uint8_t fp4) {
    uint8_t sign = (fp4 >> 3) & 1;
    uint8_t exp  = (fp4 >> 1) & 3;
    uint8_t mant = fp4 & 1;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = (float)mant / 2.0f * 0.5f;
        return sign ? -val : val;
    }
    float val = (1.0f + (float)mant / 2.0f) * std::pow(2.0f, (int32_t)(exp - 1));
    return sign ? -val : val;
}

static float fp8_e4m3_to_float(uint8_t fp8) {
    uint8_t exp  = (fp8 >> 3) & 0xF;
    uint8_t mant = fp8 & 0x7;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        return (float)mant / 8.0f * std::pow(2.0f, -6);
    }
    if (exp == 0xF) {
        if (mant == 0) return INFINITY;
        return NAN;
    }
    return (1.0f + (float)mant / 8.0f) * std::pow(2.0f, (int32_t)(exp - 7));
}

static uint32_t float_to_fp24(float val) {
    if (std::isnan(val))   return 0x7FF000;
    if (std::isinf(val))   return (val > 0 ? 0x7E0000 : 0xFE0000);
    if (val == 0.0f)       return std::signbit(val) ? 0x800000 : 0;
    uint32_t sign = std::signbit(val) ? 0x800000 : 0;
    val = std::fabs(val);
    int exp;
    float frac = std::frexp(val, &exp);
    int biased = exp - 1 + 31;
    if (biased <= 0) {
        float s = std::ldexp(frac, biased + 17);
        uint32_t mant = (uint32_t)(s + 0.5f);
        if (mant > 0x1FFFF) mant = 0x1FFFF;
        return sign | mant;
    }
    if (biased >= 0x3F) {
        biased = 0x3F;
        frac = 0.0f;
    }
    uint32_t mant = (uint32_t)((2.0f * frac - 1.0f) * 0x20000 + 0.5f);
    if (mant >= 0x20000) { mant = 0; biased++; }
    return sign | (biased << 17) | (mant & 0x1FFFF);
}

int main() {
    float test_values[] = {
        6.0f, 3.0f, 1.5f, 1.0f, 0.5f, 0.25f, 0.125f, 0.0f,
        -6.0f, -3.0f, -1.0f, 2.5f, 4.0f, 0.75f, 0.375f, 100.0f
    };
    int N = sizeof(test_values) / sizeof(test_values[0]);

    std::vector<uint32_t> fp24s(N);
    std::vector<fp4> fp4s(N);

    printf("=== fp24 -> NvFP4 Group Quantization ===\n");
    printf("Input values (fp24):\n");
    for (int i = 0; i < N; i++) {
        fp24s[i] = float_to_fp24(test_values[i]);
        printf("  [%2d] %8.4f  -> fp24=0x%06X\n", i, test_values[i], fp24s[i]);
    }

    fp8 scale = group_fp24_Nvfp4(fp24s, fp4s);

    float scale_f = fp8_e4m3_to_float(scale);
    printf("\nShared scale (fp8 E4M3): 0x%02X = %.6f\n", scale, scale_f);

    printf("\nOutput (NvFP4):\n");
    printf("  idx |     fp24 val | fp24 hex | fp4 hex | fp4 val (decoded) | reconstructed\n");
    printf("  ----|-------------|----------|---------|-------------------|---------------\n");
    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float orig = test_values[i];
        float decoded = scale_f * fp4_to_float(fp4s[i]);
        float err = std::fabs(orig - decoded);
        if (err > max_err) max_err = err;
        sum_err += err;
        printf("  %3d | %11.6f |  0x%06X |    0x%01X  | %17.6f | %13.6f\n",
               i, orig, fp24s[i], fp4s[i], fp4_to_float(fp4s[i]), decoded);
    }
    printf("\n  Max error: %.6f\n", max_err);
    printf("  Avg error: %.6f\n", sum_err / N);
    printf("  Compression ratio: 24 bits -> 4 bits (6x smaller)\n");

    printf("\n=== Edge case: single tiny value ===\n");
    {
        std::vector<uint32_t> f24 = {float_to_fp24(0.001f)};
        std::vector<fp4> f4(1);
        fp8 s = group_fp24_Nvfp4(f24, f4);
        printf("  0.001 -> scale=0x%02X fp4=0x%X\n", s, f4[0]);
    }

    printf("\n=== Group 2: tightly clustered values (good fidelity) ===\n");
    {
        float vals[] = {1.0f, 1.2f, 1.5f, 1.8f, 2.0f, 2.25f};
        int M = sizeof(vals) / sizeof(vals[0]);
        std::vector<uint32_t> f24(M);
        std::vector<fp4> f4(M);
        for (int i = 0; i < M; i++) f24[i] = float_to_fp24(vals[i]);
        fp8 s = group_fp24_Nvfp4(f24, f4);
        float sf = fp8_e4m3_to_float(s);
        printf("  scale=0x%02X=%.6f\n", s, sf);
        for (int i = 0; i < M; i++) {
            float dec = sf * fp4_to_float(f4[i]);
            printf("    %.4f -> fp4=0x%X (%.4f) reconstructed=%.4f err=%.4f\n",
                   vals[i], f4[i], fp4_to_float(f4[i]), dec, std::fabs(vals[i] - dec));
        }
    }

    printf("\n=== Edge case: group with inf ===\n");
    {
        std::vector<uint32_t> f24 = {float_to_fp24(1.0f), 0x7E0000};
        std::vector<fp4> f4(2);
        fp8 s = group_fp24_Nvfp4(f24, f4);
        printf("  scale=0x%02X (expected 0x7F=NaN) fp4s=[0x%X, 0x%X] (expected all 0)\n",
               s, f4[0], f4[1]);
    }

    return 0;
}

