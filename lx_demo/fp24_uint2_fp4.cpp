#include <stdint.h>
#include <vector>
#include <cassert>
#include <util.hpp>
#include "fp24_uint2_fp4.hpp"

// use this
// no fp24 subnormal in
std::pair<uint6, uint2> group_fp24_uint2(std::vector<fp24>& fp24s, std::vector<uint2>& uint2s) {
    assert(!fp24s.empty() && fp24s.size() == uint2s.size());
    fp24 minValue = fp24s[0];
    fp24 maxValue = fp24s[0];

    for (int i = 1; i < fp24s.size(); ++i) {
        // inf or NaN, bypass
        if ((fp24s[i].v.u & 0x7E0000) == 0x7E0000) {
            uint2s = std::vector<uint2>(fp24s.size(), 0x0);
            return std::make_pair(0x3F, 0x0);
        }
        maxValue = fp24_max(maxValue, fp24s[i]);
        minValue = fp24_min(minValue, fp24s[i]);
    }

    fp24 neg_5_fp24 = (uint32_t)0xC28000;
    fp24 pos_5_fp24 = (uint32_t)0x428000;

    fp24 neg_5_mul_Fmin = fp24_mul(neg_5_fp24, minValue);
    fp24 neg_Fmin;
    neg_Fmin.v.u = minValue.v.u ^ 0x800000;
    fp24 pos_5_mul_Fmax = fp24_mul(pos_5_fp24, maxValue);

    uint2 zeropoint;
    // Fmax >= -5 * Fmin
    if ((fp24_max(maxValue, neg_5_mul_Fmin).v.u) == (maxValue.v.u)) {
        zeropoint = 0x0;
    }
    // -5 * Fmin > Fmax > -Fmin
    else if ((fp24_max(maxValue, neg_Fmin).v.u) == (maxValue.v.u) && (maxValue.v.u) != (neg_Fmin.v.u)) {
        zeropoint = 0x1; // actually -1
    }
    // -Fmin >= Fmax >= -1/5 * Fmin
    else if ((fp24_max(pos_5_mul_Fmax, neg_Fmin).v.u) == (pos_5_mul_Fmax.v.u)) {
        zeropoint = 0x2; // actually -2;
    }
    // -1/5 * Fmin > Fmax
    else {
        zeropoint = 0x3; // actually -3
    }

    fp24 value = ((maxValue.v.u & 0x7FFFFF) > (minValue.v.u & 0x7FFFFF)) ? maxValue : minValue;

    fp24 temp_scale;

    if (zeropoint == 0x3) {
        fp24 neg_one_of_three = (uint32_t)0xbaaaab;
        temp_scale = fp24_mul(value, neg_one_of_three);
    }
    else if (zeropoint == 0x2) {
        fp24 neg_one_of_two = (uint32_t)0xbc0000;
        temp_scale = fp24_mul(value, neg_one_of_two);
    }
    else if (zeropoint == 0x1) {
        fp24 one_of_two = (uint32_t)0x3c0000;
        temp_scale = fp24_mul(value, one_of_two);
    }
    else if (zeropoint == 0x0) {
        fp24 one_of_three = (uint32_t)0x3aaaab;
        temp_scale = fp24_mul(value, one_of_three);
    }

    uint6 scale = (temp_scale.v.u & 0x1ffff) ? ((temp_scale.v.u & 0x7e0000) >> 17) + 1 : ((temp_scale.v.u & 0x7e0000) >> 17);

    for (int i = 0; i < fp24s.size(); ++i) {
        // zero bypass
        if (fp24s[i].v.u == 0) {
            uint2s[i] = zeropoint;
            continue;
        }

        uint8_t new_exponent = (fp24s[i].v.u & 0x7e0000) >> 17;
        new_exponent += 31 - scale;

        // won't overflow
        if (new_exponent & 0x80 || ! new_exponent) {
            uint2s[i] = zeropoint;
        }
        else {
            fp24 fp24_div_scale;
            fp24_div_scale.v.u = (fp24s[i].v.u & 0x800000) + (new_exponent << 17) + (fp24s[i].v.u & 0x1ffff);
            fp24 fp24_for_convert;

            if (zeropoint == 0x0) {
                fp24_for_convert = fp24_div_scale;
            }
            else if (zeropoint == 0x1) {
                fp24_for_convert = fp24_add(fp24_div_scale, (uint32_t)0x3e0000);
            }
            else if (zeropoint == 0x2) {
                fp24_for_convert = fp24_add(fp24_div_scale, (uint32_t)0x400000);
            }
            else if (zeropoint == 0x3) {
                fp24_for_convert = fp24_add(fp24_div_scale, (uint32_t)0x410000);
            }

            // fp24_for_convert <= 0.5f
            if ((fp24_for_convert.v.u & 0x800000) || (fp24_for_convert.v.u <= 0x3c0000)) {
                uint2s[i] = 0x0;
            }
            // 0.5f < fp24_for_convert < 1.5f
            else if (fp24_for_convert.v.u < 0x3f0000) {
                uint2s[i] = 0x1;
            }
            // 1.5f <= fp24_for_convert <= 2.5f
            else if (fp24_for_convert.v.u <= 0x408000) {
                uint2s[i] = 0x2;
            }
            // 2.5f < fp24_for_convert
            else {
                uint2s[i] = 0x3;
            }
        }
    }

    return std::make_pair(scale, zeropoint);
}
/*
const uint8_t countLeadingZeros8bits[256] = {
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
*/

static uint8_t uint_divide_within_loop(uint32_t &dividend, const uint32_t divisor, unsigned div_loop_num) {
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

// use this
// no fp24 subnormal in
std::pair<fp8, uint2> group_fp24_uint2_fp8scale(std::vector<fp24>& fp24s, std::vector<uint2>& uint2s) {
    assert(!fp24s.empty() && fp24s.size() == uint2s.size());
    fp24 minValue = fp24s[0];
    fp24 maxValue = fp24s[0];


    for (int i = 0; i < fp24s.size(); ++i) {
        // inf or NaN, bypass
        if ((fp24s[i].v.u & 0x7E0000) == 0x7E0000) {
            uint2s = std::vector<uint2>(fp24s.size(), 0x0);
            return std::make_pair(0x7F, 0x0);
        }
        maxValue = fp24_max(maxValue, fp24s[i]);
        minValue = fp24_min(minValue, fp24s[i]);
    }

    fp24 neg_5_fp24 = (uint32_t)0xC28000;
    fp24 pos_5_fp24 = (uint32_t)0x428000;

    fp24 neg_5_mul_Fmin = fp24_mul(neg_5_fp24, minValue);
    fp24 neg_Fmin = minValue.v.u ^ 0x800000;
    fp24 pos_5_mul_Fmax = fp24_mul(pos_5_fp24, maxValue);

    uint2 zeropoint;
    // Fmax >= -5 * Fmin
    if (fp24_max(maxValue, neg_5_mul_Fmin) == maxValue) {
        zeropoint = 0x0;
    }
    // -5 * Fmin > Fmax > -Fmin
    else if (fp24_max(maxValue, neg_Fmin) == maxValue && maxValue != neg_Fmin) {
        zeropoint = 0x1; // actually -1
    }
    // -Fmin >= Fmax >= -1/5 * Fmin
    else if (fp24_max(pos_5_mul_Fmax, neg_Fmin) == pos_5_mul_Fmax) {
        zeropoint = 0x2; // actually -2;
    }
    // -1/5 * Fmin > Fmax
    else {
        zeropoint = 0x3; // actually -3
    }

    fp24 value = (maxValue.v.u & 0x7FFFFF) > (minValue.v.u & 0x7FFFFF) ? maxValue.v.u & 0x7FFFFF : minValue.v.u & 0x7FFFFF;
    uint32_t dividend = (value.v.u & 0x1FFFF) + 0x20000;
    uint32_t divisor = (zeropoint == 0x0 || zeropoint == 0x3) ? 0x30000 : 0x20000;
    uint8_t exp = (value.v.u & 0x7e0000) >> 17;
    if (dividend < divisor) {
        exp += 7 - 31 - 1 - 1;
        dividend <<= 1;
    }
    else {
        exp += 7 - 31 - 1;
    }

    fp8 scale;
    // exp <= -4, including the case of all zeros
    if ((exp & 0x80) && (exp <= 0xFC)) {
        scale = 0x0;
    }
    else if ((! exp) || (exp & 0x80)) {
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

    if (scale == 0x0) {
        uint2s = std::vector<uint2>(uint2s.size(), zeropoint);
    }
    else {
        uint8_t scale_exp = (scale & 0x78) >> 3;
        uint32_t scale_mant = scale & 0x7;
        if (scale & 0x78) {
            scale_mant += 0x8;
            scale_mant <<= 14;
        }
        else {
            scale_exp -= countLeadingZeros8bits[scale] - 5;
            scale_mant <<= 14 + countLeadingZeros8bits[scale] - 5 + 1;
        }

        for (int i = 0; i < fp24s.size(); ++i) {
            // zero bypass
            if (! fp24s[i].v.u) {
                uint2s[i] = zeropoint;
                continue;
            }

            uint8_t new_exponent = (fp24s[i].v.u & 0x7e0000) >> 17;
            uint32_t mantissa = (fp24s[i].v.u & 0x1ffff) + 0x20000;
            if (mantissa < scale_mant) {
                new_exponent += 7 - scale_exp - 1;
                mantissa <<= 1;
            }
            else {
                new_exponent += 7 - scale_exp;
            }

            // won't overflow
            if (new_exponent & 0x80 || ! new_exponent) {
                uint2s[i] = zeropoint;
            }
            else {
                uint8_t div_loop_num = 4;
                auto mantissa_div = uint_divide_within_loop(mantissa, scale_mant, div_loop_num);
                if (mantissa) {
                    mantissa_div |= 0x1;
                }

                fp24 fp24_div_scale = (fp24s[i].v.u & 0x800000) + (new_exponent << 17) + ((mantissa_div & 0x7) << 14);
                fp24 fp24_for_convert;
                if (zeropoint == 0x0) {
                    fp24_for_convert = fp24_div_scale;
                }
                else if (zeropoint == 0x1) {
                    fp24_for_convert = fp24_add(fp24_div_scale, (uint32_t)0x3e0000);
                }
                else if (zeropoint == 0x2) {
                    fp24_for_convert = fp24_add(fp24_div_scale, (uint32_t)0x400000);
                }
                else if (zeropoint == 0x3) {
                    fp24_for_convert = fp24_add(fp24_div_scale, (uint32_t)0x410000);
                }

                // fp24_for_convert <= 0.5f
                if (fp24_for_convert.v.u & 0x800000 || fp24_for_convert.v.u <= 0x3c0000) {
                    uint2s[i] = 0x0;
                }
                // 0.5f < fp24_for_convert < 1.5f
                else if (fp24_for_convert.v.u < 0x3f0000) {
                    uint2s[i] = 0x1;
                }
                // 1.5f <= fp24_for_convert <= 2.5f
                else if (fp24_for_convert.v.u <= 0x408000) {
                    uint2s[i] = 0x2;
                }
                // 2.5f < fp24_for_convert
                else {
                    uint2s[i] = 0x3;
                }
            }
        }
    }
    return std::make_pair(scale, zeropoint);
}

const fp4 uint2_zeropoint_lookup_fp4[4][4] = {
    {0x0, 0xa, 0xc, 0xd},
    {0x2, 0x0, 0xa, 0xc},
    {0x4, 0x2, 0x0, 0xa},
    {0x5, 0x4, 0x2, 0x0}
};

// use this
fp4 uint2_and_zeropoint_convert_to_fp4(uint2 input_uint2, uint2 zeropoint) {
    return uint2_zeropoint_lookup_fp4[input_uint2][zeropoint];
}

