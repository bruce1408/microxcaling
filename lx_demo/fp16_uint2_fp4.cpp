#include <stdint.h>
#include <vector>
#include <cassert>

typedef uint8_t uint2;
typedef uint8_t uint6;
typedef uint16_t fp16;
typedef uint8_t fp4;

// use this
// fp16 subnormal exists
std::pair<uint6, uint2> group_fp16_uint2(std::vector<fp16>& fp16s, std::vector<uint2>& uint2s) {
    assert(!fp16s.empty() && fp16s.size() == uint2s.size());
    fp16 minValue = fp16s[0];
    fp16 maxValue = fp16s[0];

    for (int i = 0; i < fp16s.size(); ++i) {
        // inf or NaN, bypass
        if ((fp16s[i] & 0x7C00) == 0x7C00) {
            uint2s = std::vector<uint2>(fp16s.size(), 0x0);
            return std::make_pair(0x3F, 0x0);
        }
        maxValue = fp16_max(maxValue, fp16s[i]);
        minValue = fp16_min(minValue, fp16s[i]);
    }

    fp16 neg_5_fp16 = 0xC500;
    fp16 pos_5_fp16 = 0x4500;

    fp16 neg_5_mul_Fmin = fp16_mul(neg_5_fp16, minValue);
    fp16 neg_Fmin       = minValue ^ 0x8000;
    fp16 pos_5_mul_Fmax = fp16_mul(pos_5_fp16, maxValue);

    uint2 zeropoint;
    // Fmax >= -5 * Fmin
    if (fp16_max(maxValue, neg_5_mul_Fmin) == maxValue) {
        zeropoint = 0x0;
    }
    // -5 * Fmin > Fmax > -Fmin
    else if (fp16_max(maxValue, neg_Fmin) == maxValue && maxValue != neg_Fmin) {
        zeropoint = 0x1; // actually -1
    }
    // -Fmin >= Fmax >= -1/5 * Fmin
    else if (fp16_max(pos_5_mul_Fmax, neg_Fmin) == pos_5_mul_Fmax) {
        zeropoint = 0x2; // actually -2;
    }
    // -1/5 * Fmin > Fmax
    else {
        zeropoint = 0x3; // actually -3
    }

    fp16 value = (maxValue & 0x7FFF) > (minValue & 0x7FFF) ? maxValue : minValue;

    fp16 temp_scale;
    if (zeropoint == 0x3) {
        fp16 neg_one_of_three = 0xB555;
        temp_scale = fp16_mul(value, neg_one_of_three);
    }
    else if (zeropoint == 0x2) {
        fp16 neg_one_of_two = 0xB800;
        temp_scale = fp16_mul(value, neg_one_of_two);
    }
    else if (zeropoint == 0x1) {
        fp16 one_of_two = 0x3800;
        temp_scale = fp16_mul(value, one_of_two);
    }
    else if (zeropoint == 0x0) {
        fp16 one_of_three = 0x3555;
        temp_scale = fp16_mul(value, one_of_three);
    }

    uint6 scale;
    // 0 or subnormal
    if (! (temp_scale & 0x7C00)) {
        uint16_t mts = temp_scale & 0x3FF;
        if (! mts) {
            scale = 0xF7; // -9
        }
        else {
            scale = 6 - countLeadingZeros16bits(mts);
            if (mts > (1 << (15 - countLeadingZeros16bits(mts)))) {
                ++scale;
            }
        }
    }
    // normal
    else {
        scale = (temp_scale & 0x3FF) ? ((temp_scale & 0x7C00) >> 10) + 1 : ((temp_scale & 0x7C00) >> 10);
    }

    for (int i = 0; i < fp16s.size(); ++i) {
        // zero bypass
        if (! (fp16s[i] & 0x7fff)) {
            uint2s[i] = zeropoint;
            continue;
        }

        uint8_t exp = (fp16s[i] & 0x7C00) >> 10;
        uint16_t mts = fp16s[i] & 0x3FF;
        if (! exp) {
            exp -= countLeadingZeros16bits(mts) - 6;
            mts <<= countLeadingZeros16bits(mts) - 6 + 1;
        }

        exp += 15 - scale;
        // won't overflow
        if (exp & 0x80 || ! exp) {
            uint2s[i] = zeropoint;
        }
        else {
            fp16 fp16_div_scale = (fp16s[i] & 0x8000) + (exp << 10) + (mts & 0x3ff);
            fp16 fp16_for_convert;

            if (zeropoint == 0x0) {
                fp16_for_convert = fp16_div_scale;
            }
            else if (zeropoint == 0x1) {
                fp16_for_convert = fp16_add(fp16_div_scale, 0x3c00);
            }
            else if (zeropoint == 0x2) {
                fp16_for_convert = fp16_add(fp16_div_scale, 0x4000);
            }
            else if (zeropoint == 0x3) {
                fp16_for_convert = fp16_add(fp16_div_scale, 0x4200);
            }

            // fp16_for_convert <= 0.5f
            if (fp16_for_convert & 0x8000 || fp16_for_convert <= 0x3800) {
                uint2s[i] = 0x0;
            }
            // 0.5f < fp16_for_convert < 1.5f
            else if (fp16_for_convert < 0x3e00) {
                uint2s[i] = 0x1;
            }
            // 1.5f <= fp16_for_convert <= 2.5f
            else if (fp16_for_convert <= 0x4100) {
                uint2s[i] = 0x2;
            }
            // 2.5f < fp16_for_convert
            else {
                uint2s[i] = 0x3;
            }
        }
    }

    return std::make_pair(scale + 16, zeropoint);
}

const fp4 uint2_zeropoint_lookup_fp4[4][4] = {{0x0, 0xa, 0xc, 0xd},
                                              {0x2, 0x0, 0xa, 0xc},
                                              {0x4, 0x2, 0x0, 0xa},
                                              {0x5, 0x4, 0x2, 0x0}};

// use this
fp4 uint2_and_zeropoint_convert_to_fp4(uint2 input_uint2, uint2 zeropoint) {
    return uint2_zeropoint_lookup_fp4[input_uint2][zeropoint];
}

