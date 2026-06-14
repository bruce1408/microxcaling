/*
 * fp24 -> Microscaling (MX) FP8 group quantization
 * fp24 到 MX FP8 的分组量化示例。
 *
 * 这个文件演示如何把一组 fp24 数值压缩成一组 E4M3 fp8 数值。
 * 和普通逐元素 fp8 量化不同，MX 格式会让整组数共享一个指数缩放因子 mx_scale：
 * 每个元素只保存 8-bit fp8，整组额外保存一个 6-bit scale。
 *
 * Data formats / 数据格式:
 *   fp24:
 *     sign(1) + exp(6, bias=31) + mant(17)
 *     bit layout: [sign:23] [exp:22..17] [mant:16..0]
 *
 *   MX FP8 E4M3:
 *     sign(1) + exp(4, bias=7) + mant(3)
 *     bit layout: [sign:7] [exp:6..3] [mant:2..0]
 *
 *   mx_scale:
 *     6-bit unsigned integer shared exponent offset.
 *     代码中 typedef 为 uint8_t，逻辑上只使用低 6 bit。
 *
 * Reconstruction / 重建公式:
 *   value = fp8_to_float(fp8) * 2^(mx_scale - 31)
 *
 * fp8 E4M3 encoding / E4M3 解码规则:
 *   exp=0x0:
 *     subnormal, value = 0.mant * 2^(-6)
 *   exp=0x1..0xF:
 *     normal, value = 1.mant * 2^(exp - 7)
 *   Largest normal:
 *     1.111b * 2^8 = 1.875 * 256 = 480
 *
 * Algorithm overview / 算法分三步:
 *   1. 先扫描整组 fp24，经过一次保守 RNE 舍入后找最大 exponent。
 *      这样可以避免后续 fp8 舍入进位时共享 scale 算得过小。
 *   2. 根据最大 exponent 计算共享缩放因子 mx_scale = max_exp - 8。
 *      这里的 8 来自 E4M3 normal 最大真实指数 15 - bias(7) = 8。
 *   3. 对每个 fp24 元素执行 mantissa 舍入和 exponent 重映射，输出 fp8。
 */

#include <stdint.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>

typedef uint8_t fp8;
typedef uint8_t uint6;


// 将一组 fp24 数值转换为 MX FP8。
// fp24s:
//   输入数组，每个 uint32_t 的低 24 bit 保存一个 fp24。
// fp8s:
//   输出数组，大小必须和 fp24s 相同，每个元素保存一个 E4M3 fp8 编码。
// return:
//   返回该组共享的 mx_scale。
//
// Precondition:
//   注释假设输入中没有 fp24 subnormal；代码中如果遇到 exp==0，仍会按 0 处理。
uint6 group_fp24_Mxfp8(std::vector<uint32_t> fp24s, std::vector<fp8> &fp8s) {
    assert(fp24s.size() == fp8s.size());

    // ============================================================
    // Phase 1: Find the maximum exponent across all fp24 values.
    // 第一阶段：寻找整组 fp24 中最大的 exponent。
    //
    // 这里不是直接使用原始 exponent，而是先做一次“保守 RNE 舍入”。
    // 原因是：最终转 fp8 时 mantissa 可能因为舍入而进位，进而导致 exponent + 1。
    // 如果 scale 只根据原始 exponent 计算，可能会低估组内最大数需要的动态范围。
    //
    // 此处在 bit 14 做舍入，最终 fp8 normal 路径在 bit 13 做舍入。
    // bit 14 比最终舍入位置更粗一位，因此是保守估计。
    // ============================================================
    uint8_t max_exp = 7;          // 初始下界；bias=31 时，exp=7 对应真实指数 -24。
    bool inf_nan = false;
    for (auto fp24 : fp24s) {
        
        // fp24 exponent 位于 bit[22:17]，掩码 0x7e0000 = 0b01111110_00000000_00000000。
        uint8_t exp = (fp24 & 0x7e0000) >> 17;
        if (exp == 0x3f) {                        // fp24 exponent 全 1，表示 Inf 或 NaN。
            inf_nan = true;
            break;
        }
        if (exp >= max_exp) {
            // normal fp24 的有效尾数是 1.mantissa。
            // fp24 显式 mantissa 只有 17 bit，范围是 [0, 0x1ffff]。
            // 加上隐含 leading 1，即加 0x20000，得到 18-bit mantissa，范围是 [0x20000, 0x3ffff]。
            uint32_t mantissa = (fp24 & 0x1ffff) + 0x20000;   // [0x20000, 0x3FFFF]

            // Conservative RNE rounding at bit 14 (0x4000).
            // RNE = round to nearest, ties to even，即“就近舍入，正好一半时舍到偶数”。
            //
            // 判断规则：
            //   1. round bit，也就是 bit 14，为 0：不进位。
            //   2. bit 14 为 1，且更低位 bit[13:0] 有任意 1：说明大于一半，进位。
            //   3. bit 14 为 1，且 bit[13:0] 全 0：说明刚好一半，看 bit 15。
            //      如果 bit 15 为 1，当前保留部分是奇数，进位后变偶数；否则不进位。
            //
            // 这里的 +0x8000 等价于向 bit 15 进位。
            mantissa += (mantissa & 0x4000) ? ((mantissa & 0x3fff) ? 0x8000 : mantissa & 0x8000) : 0;
            if (mantissa & 0x40000) {             // 舍入后出现 bit 18，说明 1.xxx 进位成 10.xxx。
                exp += 1;                         // 尾数溢出需要 exponent 加 1，进入下一个 binade。
            }
            if (exp == 0x3f) {                    // 舍入进位后 exponent 全 1，按 Inf/NaN 处理。
                inf_nan = true;
                break;
            }
            max_exp = exp;                        // 更新当前组内看到的最大 exponent。
        }
    }

    printf("max_exp: %d\n", max_exp);

    // 如果组内任何一个 fp24 是 Inf/NaN，当前实现直接把整组输出标成 0x7f，scale 返回最大值。
    // 注意：文件开头说明 MX E4M3 的 exp=0xF 也可以作为 normal 数使用；这里把 0x7f 当异常标记，
    // 是否符合目标硬件/协议，需要结合实际格式规范确认。
    if (inf_nan) {
        fp8s = std::vector<fp8>(fp24s.size(), 0x7f);   // 0x7F = E4M3 NaN (exp=0xF, mant≠0)
        return 0x3f;                                     // mx_scale = max value
    }

    // ============================================================
    // Phase 2: Compute shared scale mx_scale.
    // 第二阶段：计算共享缩放因子 mx_scale。
    //
    //   mx_scale = max_exp - 8
    //
    // 推导：
    //   fp24 normal 数的真实指数大约是 max_exp - 31。
    //   E4M3 的最大 exponent code 是 15，bias 是 7，因此最大真实指数是 15 - 7 = 8。
    //   重建公式为 fp8_value * 2^(mx_scale - 31)。
    //
    // 希望组内最大值映射到 fp8 最高 exponent 附近，则有：
    //   2^8 * 2^(mx_scale - 31) ~= 2^(max_exp - 31)
    // 所以：
    //   mx_scale = max_exp - 8
    // ============================================================
    uint6 mx_scale = max_exp - 8;

    // 如果 max_exp < 8，则 max_exp - 8 是负数。
    // 由于 mx_scale 是 uint8_t，负数会发生无符号下溢，最高位 bit 7 会变成 1。
    // 这里用 bit 7 检测这种负 scale 情况；当前策略是整组 flush to zero。
    if (mx_scale & 0x80) {                           // bit 7 set => negative offset。
        fp8s = std::vector<fp8>(fp24s.size(), 0x0);
        return 0x0;
    }
    
    // ============================================================
    // Phase 3: Per-element fp24 -> fp8 conversion.
    // 第三阶段：根据共享 mx_scale，把每个 fp24 单独转换成 fp8。
    // ============================================================
    for (unsigned i = 0; i < fp24s.size(); ++i) {
        // fp24 sign bit 在 bit 23；fp8 sign bit 在 bit 7。
        // 右移 16 位后，sign 已经位于 fp8 的符号位位置，结果为 0x00 或 0x80。
        uint8_t sign = (fp24s[i] & 0x800000) >> 16;

        // 取 fp24 的 6-bit exponent。
        uint8_t exp  = (fp24s[i] & 0x7e0000) >> 17;
        if (!exp) {                                    // fp24 zero 或 subnormal，当前实现直接输出 0。
            fp8s[i] = 0x0;
            continue;
        }

        // 组装 normal fp24 的 18-bit 有效尾数：隐含 leading 1 + 17-bit fraction。
        uint32_t mantissa = (fp24s[i] & 0x1ffff) + 0x20000;

        // Adjust exponent: shift fp24's exponent by the shared scale offset.
        // 根据共享 scale 把 fp24 exponent 映射到 fp8 exponent 的工作区间。
        //
        //   exp = fp24_exp + 6 - mx_scale
        //       = fp24_exp - max_exp + 14  (14 = max normal E4M3 exp)
        //
        // 对组内最大 exponent 的元素：
        //   fp24_exp = max_exp => exp = 14
        // 后面 normal path 还会 exp += 1，因此最终 fp8 exponent field 通常会接近 15。
        exp += 6 - mx_scale;

        uint8_t fp8_mantissa;
        if (!(exp & 0x80)) {                    // uint8_t 最高位未置位，表示没有下溢，可走 normal fp8 路径。
            // ----- Normal fp8 path / normal fp8 路径 -----
            // 对 mantissa 做最终 RNE 舍入，只保留 fp8 的 3-bit mantissa。
            // round bit 是 bit 13，即 0x2000；低于 bit 13 的 bit[12:0] 是 sticky bits。
            // 如果正好 tie，则看 bit 14，使结果舍入到偶数。
            mantissa += (mantissa & 0x2000) ?
                ((mantissa & 0x1fff) ? 0x4000 : mantissa & 0x4000) : 0;

            if (mantissa & 0x40000) {           // mantissa 舍入溢出，1.xxx 变成 10.xxx。
                exp += 1;
                mantissa >>= 1;
            }

            // 把前面构造的中间 exponent 转成最终 fp8 exponent field。
            exp += 1;

            // 舍入后取 bit[16:14] 作为 fp8 的 3-bit mantissa。
            fp8_mantissa = (mantissa >> 14) & 0x7;

            // 拼接 fp8 编码：sign 已在 bit 7，exp 左移到 bit[6:3]，mantissa 位于 bit[2:0]。
            fp8s[i] = sign + (exp << 3) + fp8_mantissa;
            // Layout: [sign:7] [exp:6-3] [mant:2-0]
        }
        else {
            // ----- Subnormal fp8 path / subnormal fp8 路径 -----
            // exp 使用 uint8_t 保存。若逻辑上是负数，会发生下溢：
            //   -1 -> 0xff, -2 -> 0xfe, -3 -> 0xfd, -4 -> 0xfc, -5 -> 0xfb
            // E4M3 subnormal 只能覆盖有限范围；如果小于 -4，则直接 flush to zero。
            if (exp < 0xfc) {                   // 等价于逻辑 exponent < -4。
                fp8s[i] = 0x0;                  // 太小，fp8 subnormal 也表示不了，输出 0。
            }
            else {
                // Convert to fp8 subnormal: exp = -1, -2, -3, or -4.
                // 将 normal mantissa 右移成 subnormal mantissa，同时做 RNE 舍入。
                //
                // 对 uint8_t 下溢表示的负数，~exp + 1 等价于取绝对值：
                //   exp=0xff(-1) => 1
                //   exp=0xfe(-2) => 2
                //   exp=0xfd(-3) => 3
                //   exp=0xfc(-4) => 4
                uint32_t rounding_bit = 0x2000 << (~exp + 1);     // 当前 subnormal 右移量对应的 round bit。
                uint32_t sticky_bit = rounding_bit - 1;            // round bit 以下的所有 sticky bits。

                // RNE:
                //   round bit 为 1 且 sticky 非 0：大于一半，进位。
                //   round bit 为 1 且 sticky 为 0：正好一半，看下一保留位是否为奇数。
                mantissa += (mantissa & rounding_bit) ?
                    ((mantissa & sticky_bit) ? rounding_bit << 1 : mantissa & (rounding_bit << 1)) : 0x0;

                // 提取 subnormal mantissa。subnormal 的 exponent field 固定为 0。
                fp8_mantissa = mantissa >> (14 + (~exp + 1));

                // 如果 subnormal mantissa 非 0，保留符号位；如果舍入后仍为 0，则输出 +0。
                fp8s[i] = fp8_mantissa ? (sign + fp8_mantissa) : 0x0;
                // Layout: [sign:7] [exp:6-3 = 0000] [mant:2-0]
            }
        }
    }

    return mx_scale;
}


// 测试辅助函数：把 fp24 编码还原成 float，方便打印和误差分析。
// 这不是量化主流程的一部分，只用于验证 group_fp24_Mxfp8 的输出效果。
static float fp24_to_float(uint32_t fp24) {
    uint8_t sign = (fp24 >> 23) & 1;       // bit 23: 符号位。
    int32_t exp = (fp24 >> 17) & 0x3F;     // bit[22:17]: 6-bit exponent。
    uint32_t mant = fp24 & 0x1FFFF;        // bit[16:0]: 17-bit mantissa。
    if (exp == 0) {
        // exp=0 表示 zero 或 subnormal。
        // subnormal 没有隐含 leading 1，指数按 1-bias=-30 处理。
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = (float)mant / 0x20000 * std::pow(2.0f, -30);
        return sign ? -val : val;
    }
    if (exp == 0x3F) {
        // fp24 exponent 全 1，mant=0 表示 Inf，mant!=0 表示 NaN。
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    // normal fp24:
    //   value = (-1)^sign * (1 + mant / 2^17) * 2^(exp - 31)
    float val = (1.0f + (float)mant / 0x20000) * std::pow(2.0f, (int32_t)(exp - 31));
    return sign ? -val : val;
}

// 测试辅助函数：把 MX FP8 编码按给定 mx_scale 解码回 float。
// 解码分两步：
//   1. 先按 E4M3 得到未缩放的 fp8_value。
//   2. 再乘以共享 scale: 2^(mx_scale - 31)。
static float fp8_e4m3_to_float(uint8_t fp8, uint6 mx_scale) {
    uint8_t sign = (fp8 >> 7) & 1;     // bit 7: 符号位。
    uint8_t exp  = (fp8 >> 3) & 0xF;   // bit[6:3]: 4-bit exponent。
    uint8_t mant = fp8 & 0x7;          // bit[2:0]: 3-bit mantissa。
    float v;
    if (exp == 0) {
        // E4M3 subnormal 或 zero。
        // subnormal value = (mant / 8) * 2^-6。
        if (mant == 0) return sign ? -0.0f : 0.0f;
        v = (float)mant / 8.0f * std::pow(2.0f, -6);
    } else {
        // MX E4M3: exp=0xF 仍作为 normal 使用，不保留 Inf/NaN。
        // normal value = (1 + mant / 8) * 2^(exp - 7)。
        v = (1.0f + (float)mant / 8.0f) * std::pow(2.0f, (int32_t)(exp - 7));
    }
    if (sign) v = -v;

    // 应用 MX 共享缩放因子。
    // reconstruction = fp8_value * 2^(mx_scale - 31)。
    return v * std::pow(2.0f, (int32_t)(mx_scale - 31));
}

// 测试辅助函数：把标准 float 近似编码成 fp24，方便构造输入样例。
// 注意这里是为了 demo 写的简化转换函数，不一定覆盖完整工业级浮点边界语义。
static uint32_t float_to_fp24(float val) {
    // 特殊值处理：NaN、Inf、正负 0。
    if (std::isnan(val))   return 0x7FF000;
    if (std::isinf(val))   return (val > 0 ? 0x7E0000 : 0xFE0000);
    if (val == 0.0f)       return std::signbit(val) ? 0x800000 : 0;

    // 保存符号位，并转成绝对值处理。
    uint32_t sign = std::signbit(val) ? 0x800000 : 0; // 如果是负数，就设置符号位为 1，其他为0即可
    val = std::fabs(val);

    // frexp 返回 frac 和 exp，使得：
    //   val = frac * 2^exp, 其中 frac 在 [0.5, 1) 范围内。

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
        // 超出 fp24 normal 范围，编码为 exponent 全 1。
        biased = 0x3F;
        frac = 0.0f;
    }

    // normal fp24 mantissa：
    //   frac 在 [0.5, 1)，2*frac 在 [1, 2)。
    //   mant = (2*frac - 1) * 2^17。
    // 加 0.5f 是简单的就近舍入。
    uint32_t mant = (uint32_t)((2.0f * frac - 1.0f) * 0x20000 + 0.5f);
    if (mant >= 0x20000) { mant = 0; biased++; }  // mantissa 舍入溢出时，进位到 exponent。
    return sign | (biased << 17) | (mant & 0x1FFFF);
}

int main() {
    // 测试数据：覆盖正数、负数、不同数量级的数据，观察共享 scale 下的重建误差。
    float test_values[] = {6.0f, 3.0f, 1.5f, 1.0f, 0.5f, 0.25f, 100.0f, -3.0f};
    int N = sizeof(test_values) / sizeof(test_values[0]);

    std::vector<uint32_t> fp24s(N);
    std::vector<fp8> fp8s(N);

    printf("=== fp24 -> MX FP8 Group Quantization ===\n");
    printf("Input values (fp24):\n");
    
    for (int i = 0; i < N; i++) {
        // 先把 float 转成 fp24，模拟真实输入已经是 fp24 编码的情况。
        fp24s[i] = float_to_fp24(test_values[i]);
        printf("  [%d] %8.4f -> fp24=0x%06X\n", i, test_values[i], fp24s[i]);
    }

    // 调用核心量化函数，得到每个元素的 fp8 编码和整组共享 mx_scale。
    uint6 mx_scale = group_fp24_Mxfp8(fp24s, fp8s);

    printf("\nShared mx_scale = %d (0x%02X), scale factor = 2^%d = %.6f\n",
           mx_scale, mx_scale, (int)(mx_scale - 31), std::pow(2.0f, (int)(mx_scale - 31)));

    printf("\nOutput (MX FP8):\n");
    printf("  idx |     fp24 val | fp24 hex | fp8 hex |  reconstructed |  error\n");
    printf("  ----|-------------|----------|---------|----------------|--------\n");

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float orig = test_values[i];
        // 用同一个 mx_scale 解码每个 fp8，验证量化误差。
        float decoded = fp8_e4m3_to_float(fp8s[i], mx_scale);
        float err = std::fabs(orig - decoded);
        if (err > max_err) max_err = err;
        printf("  %3d | %11.6f |  0x%06X |    0x%02X | %14.6f | %6.4f\n",
               i, orig, fp24s[i], fp8s[i], decoded, err);
    }

    printf("\n  Max error: %.6f\n", max_err);
    printf("  Compression: 24 bits -> 8 bits (+ 6-bit shared scale per block)\n");

    return 0;
}
