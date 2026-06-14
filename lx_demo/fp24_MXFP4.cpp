#include <stdint.h>
#include <vector>
#include <cassert>

typedef uint8_t fp4;
typedef uint8_t uint6;

// no fp24 subnormal in
uint6 group_fp24_Mxfp4(std::vector<uint32_t> fp24s, std::vector<fp4> &fp4s) {
    assert(fp24s.size() == fp4s.size());
    uint8_t max_exp = 1;
    bool inf_nan = false;
    for (auto fp24 : fp24s) {
        uint8_t exp = (fp24 & 0x7e0000) >> 17;
        if (exp == 0x3f) {
            inf_nan = true;
            break;
        }
        if (exp >= max_exp) {
            uint32_t mantissa = (fp24 & 0x1ffff) + 0x20000;
            mantissa += (mantissa & 0x8000) ? ((mantissa & 0x7fff) ? 0x10000 : mantissa & 0x10000) : 0;
            if (mantissa & 0x40000) {
                exp += 1;
            }
            if (exp == 0x3f) {
                inf_nan = true;
                break;
            }
            max_exp = exp;
        }
    }

    // fp24 inf or nan exists
    if (inf_nan) {
        fp4s = std::vector<fp4>(fp24s.size(), 0x0);
        return 0x3f;
    }

    uint6 mx_scale = max_exp - 2;
    // flush to zero if all in range(0, 2^-30)
    if (mx_scale & 0x80) {
        fp4s = std::vector<fp4>(fp24s.size(), 0x0);
        return 0x0;
    }

    for (unsigned i=0; i<fp24s.size(); ++i) {
        uint8_t sign = (fp24s[i] & 0x800000) >> 20;
        uint8_t exp = (fp24s[i] & 0x7e0000) >> 17;
        if (!exp) {
            fp4s[i] = 0x0;
            continue;
        }
        uint32_t mantissa = (fp24s[i] & 0x1ffff) + 0x20000;
        exp -= mx_scale;

        uint8_t fp4_mantissa;
        uint32_t rounding_bit = 0x8000;
        uint32_t sticky_bit = 0x8000; // get all 1 mask
        if (!(exp & 0x80)) {
            // fp4 normal
            fp4_mantissa = mantissa >> 16;
            fp4_mantissa += (mantissa & rounding_bit) ? ((mantissa & (sticky_bit - 0x1)) ? 0x1 : fp4_mantissa & 0x1) : 0x0;
            // carrying
            if (fp4_mantissa & 0x4) {
                exp += 1;
                fp4_mantissa >>= 1;
            }
            exp += 1;
            fp4s[i] = sign + (exp << 1) + (fp4_mantissa & 0x1);
        }
        else {
            if (exp < 0xfe) {
                // exp <= -3, underflow
                fp4s[i] = 0x0;
            }
            else {
                // convert to fp4 subnormal `probably`
                rounding_bit <<= ~exp + 1;
                sticky_bit <<= ~exp + 1;
                fp4_mantissa = mantissa >> (16 + (~exp + 1));
                fp4_mantissa += (mantissa & rounding_bit) ? ((mantissa & (sticky_bit - 0x1)) ? 0x1 : fp4_mantissa & 0x1) : 0x0;
                fp4s[i] = fp4_mantissa ? (sign + fp4_mantissa) : 0x0;
            }
        }
    }

    return mx_scale;
}

