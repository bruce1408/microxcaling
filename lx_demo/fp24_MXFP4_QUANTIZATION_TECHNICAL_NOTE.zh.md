# FP24 到 MXFP4 E2M1 的分组量化技术说明

本文档说明 [`fp24_MXFP4.cpp`](./fp24_MXFP4.cpp) 中实际实现的量化算法。
核心函数为：

```cpp
uint6 group_fp24_Mxfp4(
    std::vector<uint32_t> fp24s,
    std::vector<fp4> &fp4s);
```

它将一组 FP24 位模式转换为一组 FP4 E2M1 位模式，并为整个 group 计算一个共享
指数 `mx_scale`。

---

## 1. 量化目标

输入元素采用自定义 FP24：

```text
FP24 = sign(1 bit) + exponent(6 bits) + fraction(17 bits)
```

输出元素采用 FP4 E2M1：

```text
FP4 = sign(1 bit) + exponent(2 bits) + fraction(1 bit)
```

整个 group 额外共享一个逻辑上为 6 bit 的 `mx_scale`：

```text
{mx_scale, fp4[0], fp4[1], ..., fp4[N-1]}
```

对应的反量化关系是：

```text
reconstructed_value
    = decode_e2m1(fp4)
    * 2^(mx_scale - 31)
```

这是一种 microscaling 或 block floating-point 量化：

- 每个元素只保存 4 bit。
- 同组元素共享一个二进制 scale。
- 组内最大值决定动态范围。
- 较小元素可能进入 subnormal 或被舍入为零。

---

## 2. FP24 输入格式

位布局：

```text
bit 23     bit 22 ........ bit 17     bit 16 ........ bit 0
+--------+--------------------------+-------------------------+
| sign   | exponent, 6 bits         | fraction, 17 bits       |
+--------+--------------------------+-------------------------+
```

对于 normal FP24：

```text
x = (-1)^sign
    * (1 + fraction / 2^17)
    * 2^(exponent - 31)
```

主要掩码：

| 掩码 | 用途 |
|---|---|
| `0x800000` | FP24 sign，bit 23 |
| `0x7E0000` | FP24 exponent，bit[22:17] |
| `0x01FFFF` | FP24 fraction，bit[16:0] |
| `0x020000` | 补回 normal 隐含的 leading 1 |
| `0x040000` | 检测尾数舍入后是否溢出 |

---

## 3. FP4 E2M1 输出格式

代码通过下面的字段拼接生成 FP4：

```cpp
fp4s[i] = sign + (exp << 1) + (fp4_mantissa & 0x1);
```

因此位布局是：

```text
bit 3       bit 2 .... bit 1       bit 0
+----------+----------------------+----------+
| sign     | exponent, 2 bits     | fraction |
+----------+----------------------+----------+
```

该格式使用：

```text
exponent bias = 1
```

normal 数值公式：

```text
value = (-1)^sign
      * (1 + fraction / 2)
      * 2^(exponent - 1)
```

`exponent=0` 时表示 zero 或 subnormal。E2M1 只有一个非零 subnormal：

```text
0.5
```

### 3.1 E2M1 可表示值

正数编码如下：

| FP4 bits | Hex | 数值 |
|---|---:|---:|
| `0 00 0` | `0x0` | 0 |
| `0 00 1` | `0x1` | 0.5 |
| `0 01 0` | `0x2` | 1 |
| `0 01 1` | `0x3` | 1.5 |
| `0 10 0` | `0x4` | 2 |
| `0 10 1` | `0x5` | 3 |
| `0 11 0` | `0x6` | 4 |
| `0 11 1` | `0x7` | 6 |

负数在这些编码上设置 bit 3：

```text
0x8, 0x9, ..., 0xF
```

因此 E2M1：

```text
最大有限值 = 6
最小 normal = 1
非零 subnormal = 0.5
```

仓库中的 FP4 定义不为 Inf/NaN 保留 exponent 编码。

---

## 4. 总体量化流程

核心函数分为三个阶段：

```text
FP24 group
    |
    | Phase 1：寻找舍入后的最大 FP24 exponent
    v
max_exp
    |
    | Phase 2：计算共享 mx_scale
    v
mx_scale
    |
    | Phase 3：使用共享 scale 逐元素量化为 E2M1
    v
FP4 group
```

伪代码：

```text
max_exp = find_max_rounded_exponent(fp24s)

if group contains FP24 Inf or NaN:
    output all zeros
    return 0x3F

mx_scale = max_exp - 2

if mx_scale is logically negative:
    output all zeros
    return 0

for each fp24:
    extract sign, exponent and significand
    align exponent with mx_scale

    if normal:
        round significand to one fraction bit with RNE
        encode E2M1 normal
    else if representable as subnormal:
        shift and round significand with RNE
        encode E2M1 subnormal
    else:
        output zero
```

---

## 5. Phase 1：寻找组内最大 exponent

初始值：

```cpp
uint8_t max_exp = 1;
```

代码遍历整个 group，并提取 FP24 exponent：

```cpp
uint8_t exp = (fp24 & 0x7e0000) >> 17;
```

如果 exponent 全 1：

```cpp
if (exp == 0x3f)
```

则输入为 FP24 Inf 或 NaN，设置 `inf_nan`。

### 5.1 补回隐含 leading 1

对于 normal FP24：

```cpp
uint32_t mantissa = (fp24 & 0x1ffff) + 0x20000;
```

得到的 18-bit 整数表示：

```text
1.fraction * 2^17
```

其范围为：

```text
0x20000 到 0x3FFFF
```

### 5.2 为什么寻找最大值时要先舍入

FP24 有 17-bit fraction，而 E2M1 只有 1-bit fraction。

如果 significand 非常接近 2：

```text
1.111... --RNE--> 10.0
```

舍入会使 exponent 增加一。

如果共享 scale 只根据舍入前的 exponent 计算，可能低估组内最大元素量化后需要的范围。

### 5.3 第一阶段的 RNE

代码：

```cpp
mantissa += (mantissa & 0x8000)
    ? ((mantissa & 0x7fff)
        ? 0x10000
        : mantissa & 0x10000)
    : 0;
```

对应位：

```text
bit 16          bit 15          bit[14:0]
retained LSB    round bit       sticky bits
```

RNE 规则：

| round bit | sticky bits | retained LSB | 操作 |
|---:|---:|---:|---|
| 0 | 任意 | 任意 | 不进位 |
| 1 | 非 0 | 任意 | 大于中点，进位 |
| 1 | 0 | 1 | 正好一半，进位到偶数 |
| 1 | 0 | 0 | 正好一半，保持偶数 |

如果舍入后出现 bit 18：

```cpp
if (mantissa & 0x40000) {
    exp += 1;
}
```

说明：

```text
1.x -> 10.x
```

需要把 exponent 加一。

---

## 6. Phase 2：计算共享 `mx_scale`

代码：

```cpp
uint6 mx_scale = max_exp - 2;
```

### 6.1 为什么减 2

FP24 最大元素的主要数量级是：

```text
2^(max_exp - 31)
```

E2M1 最大 exponent field 为 3，bias 为 1，因此最大真实 exponent 是：

```text
3 - 1 = 2
```

最大 E2M1 值为：

```text
1.5 * 2^2 = 6
```

为了让组内最大值落到 E2M1 的高动态范围：

```text
2^2 * 2^(mx_scale - 31)
    ~= 2^(max_exp - 31)
```

比较指数得到：

```text
mx_scale = max_exp - 2
```

实际共享缩放因子：

```text
scale_factor = 2^(mx_scale - 31)
```

例如：

```text
max_exp = 33
mx_scale = 33 - 2 = 31
scale_factor = 2^(31 - 31) = 1
```

### 6.2 scale 对组内精度的影响

如果同组最大值很大，`mx_scale` 会随之增大：

```text
较大元素：可以利用 FP4 的高值编码
较小元素：可能只能表示为 0.5 * scale，或者变成 0
```

因此，把数量级接近的数据放在同一个 group，通常能获得更好的量化精度。

---

## 7. Phase 3：逐元素量化

### 7.1 符号位映射

```cpp
uint8_t sign = (fp24s[i] & 0x800000) >> 20;
```

FP24 sign 位于 bit 23，FP4 sign 位于 bit 3：

```text
FP24 bit 23 -> 右移 20 位 -> FP4 bit 3
```

因此 `sign` 为：

```text
正数：0x0
负数：0x8
```

### 7.2 提取 exponent 和 significand

```cpp
uint8_t exp = (fp24s[i] & 0x7e0000) >> 17;
uint32_t mantissa = (fp24s[i] & 0x1ffff) + 0x20000;
```

如果 FP24 exponent 为零：

```cpp
if (!exp) {
    fp4s[i] = 0x0;
    continue;
}
```

代码将 FP24 zero 和 FP24 subnormal 全部清零，并且不保留负零。

### 7.3 exponent 对齐

代码：

```cpp
exp -= mx_scale;
```

这里得到的是中间 exponent：

```text
intermediate_exp = fp24_exp - mx_scale
```

normal 路径后面还会执行：

```cpp
exp += 1;
```

所以最终 FP4 exponent field 为：

```text
fp4_exp = fp24_exp - mx_scale + 1
```

也可以从数值等价关系推导：

```text
FP24 real exponent
    = FP4 real exponent + shared scale exponent

fp24_exp - 31
    = (fp4_exp - 1) + (mx_scale - 31)
```

整理：

```text
fp4_exp = fp24_exp - mx_scale + 1
```

与代码一致。

---

## 8. FP4 normal 路径

代码通过：

```cpp
if (!(exp & 0x80))
```

判断中间 exponent 是否发生 `uint8_t` 下溢。

如果没有下溢，则进入 normal 路径。

### 8.1 截取 1-bit fraction

先取 significand 的高两位：

```cpp
fp4_mantissa = mantissa >> 16;
```

这里得到：

```text
bit 17：隐含 leading 1
bit 16：目标 FP4 fraction
```

初始结果可能为：

```text
binary 10 -> 1.0
binary 11 -> 1.5
```

### 8.2 RNE 舍入

```cpp
uint32_t rounding_bit = 0x8000;
uint32_t sticky_bit = 0x8000;
```

对应：

```text
保留 fraction：bit 16
round bit：     bit 15
sticky bits：  bit[14:0]
```

舍入代码：

```cpp
fp4_mantissa += (mantissa & rounding_bit)
    ? ((mantissa & (sticky_bit - 1))
        ? 1
        : fp4_mantissa & 1)
    : 0;
```

含义仍然是 RNE：

- 大于中点时进位。
- 小于中点时不进位。
- 正好中点时，选择最低保留位为偶数的结果。

### 8.3 舍入进位

如果：

```cpp
if (fp4_mantissa & 0x4)
```

表示两位 significand 从：

```text
11 + rounding -> 100
```

也就是：

```text
1.5 附近的值舍入到了 2.0
```

此时需要：

```cpp
exp += 1;
fp4_mantissa >>= 1;
```

把 significand 重新归一化，并增加 exponent。

### 8.4 组装 FP4

```cpp
exp += 1;
fp4s[i] = sign + (exp << 1) + (fp4_mantissa & 0x1);
```

其中：

```text
sign                 -> bit 3
exp << 1             -> bit[2:1]
fp4_mantissa & 0x1   -> bit 0
```

由于字段互不重叠，加法等价于按位或：

```cpp
sign | (exp << 1) | (fp4_mantissa & 1)
```

---

## 9. FP4 subnormal 路径

如果 `exp -= mx_scale` 的逻辑结果为负，`uint8_t` 会发生下溢：

```text
逻辑值     uint8_t
-1         0xFF
-2         0xFE
-3         0xFD
```

代码只处理 `-1` 和 `-2`：

```cpp
if (exp < 0xfe) {
    fp4s[i] = 0x0;
}
```

也就是说：

```text
intermediate_exp <= -3 -> 直接清零
```

### 9.1 计算额外右移量

对于 `0xFF` 和 `0xFE`：

```cpp
~exp + 1
```

利用补码关系得到：

```text
0xFF -> 1
0xFE -> 2
```

这表示 significand 还需要额外右移 1 或 2 位。

代码相应移动 round bit：

```cpp
rounding_bit <<= ~exp + 1;
sticky_bit <<= ~exp + 1;
```

然后提取 subnormal mantissa：

```cpp
fp4_mantissa = mantissa >> (16 + (~exp + 1));
```

再执行 RNE：

```cpp
fp4_mantissa += (mantissa & rounding_bit)
    ? ((mantissa & (sticky_bit - 1))
        ? 1
        : fp4_mantissa & 1)
    : 0;
```

### 9.2 输出唯一的非零 subnormal

E2M1 只有一个 fraction bit，因此非零 subnormal 只有：

```text
FP4 0x1 -> 0.5
FP4 0x9 -> -0.5
```

代码：

```cpp
fp4s[i] = fp4_mantissa
    ? (sign + fp4_mantissa)
    : 0x0;
```

如果舍入结果为零，则输出 `+0`。

### 9.3 为什么还处理 `intermediate_exp = -2`

当 `intermediate_exp = -1` 时，缩放后的绝对值约位于：

```text
[0.5, 1.0)
```

它会舍入到 `0.5` 或 `1.0` 附近。

当 `intermediate_exp = -2` 时，缩放后的绝对值约位于：

```text
[0.25, 0.5)
```

虽然没有对应的 normal 编码，但大于中点的值仍可能舍入为 `0.5`。正好为 `0.25`
时处于 `0` 和 `0.5` 的中点，RNE 会选择偶数端，也就是 `0`。

---

## 10. 异常和极小值处理

### 10.1 group 中包含 FP24 Inf 或 NaN

代码检测：

```cpp
if (exp == 0x3f)
```

一旦发现任何异常元素，整个 group 被处理为：

```cpp
fp4s = std::vector<fp4>(fp24s.size(), 0x0);
return 0x3f;
```

即：

```text
所有 FP4 元素 = 0
mx_scale       = 0x3F
```

因为 E2M1 不保留 Inf/NaN 编码，所以这里使用特殊 scale `0x3F` 表示组异常。
是否能正确传播异常值，取决于调用方是否把该 scale 解释为异常标记。

### 10.2 共享 scale 发生无符号下溢

`uint6` 实际定义为：

```cpp
typedef uint8_t uint6;
```

如果：

```text
max_exp - 2 < 0
```

会在 `uint8_t` 中发生下溢。代码通过：

```cpp
if (mx_scale & 0x80)
```

检测这种情况，并把整个 group 清零。

---

## 11. 完整示例：scale 为 1 的一组数据

考虑一个 group：

```text
{6.0, 3.0, 1.5, 1.0, 0.5, 0.25, -3.0}
```

最大值 `6.0` 的 FP24 exponent 为 33，因此：

```text
max_exp = 33
mx_scale = 33 - 2 = 31
scale_factor = 2^(31 - 31) = 1
```

此时重建值就是：

```text
decode_e2m1(fp4) * 1
```

预期量化结果：

| 原值 | FP4 | E2M1 值 | 重建值 | 说明 |
|---:|---:|---:|---:|---|
| 6.0 | `0x7` | 6.0 | 6.0 | 精确 |
| 3.0 | `0x5` | 3.0 | 3.0 | 精确 |
| 1.5 | `0x3` | 1.5 | 1.5 | 精确 |
| 1.0 | `0x2` | 1.0 | 1.0 | 精确 |
| 0.5 | `0x1` | 0.5 | 0.5 | subnormal，精确 |
| 0.25 | `0x0` | 0.0 | 0.0 | 中点，RNE 舍向偶数 0 |
| -3.0 | `0xD` | -3.0 | -3.0 | 精确 |

### 11.1 `6.0 -> 0x7`

`6.0` 可写作：

```text
1.5 * 2^2
```

E2M1：

```text
sign = 0
exp = 2 + bias(1) = 3 = binary 11
mantissa = 1
```

编码：

```text
0 11 1 = 0x7
```

### 11.2 `0.25 -> 0`

E2M1 最小非零值是 `0.5`。

`0.25` 正好位于：

```text
0 和 0.5 的中点
```

RNE 在中点选择最低保留位为偶数的一侧，因此选择：

```text
0
```

### 11.3 `0.375 -> 0.5`

`0.375` 距离：

```text
到 0：   0.375
到 0.5： 0.125
```

因此会舍入为：

```text
0.5 -> FP4 0x1
```

---

## 12. 与 MXFP8 版本的主要区别

| 项目 | MXFP8 E4M3 | MXFP4 E2M1 |
|---|---:|---:|
| 每元素位数 | 8 | 4 |
| exponent bits | 4 | 2 |
| fraction bits | 3 | 1 |
| 最大元素值 | 480 | 6 |
| 非零表示密度 | 较高 | 很低 |
| 共享 scale | `max_exp - 8` | `max_exp - 2` |
| 最终 fraction 保留位 | 3 bit | 1 bit |
| 量化误差 | 较小 | 通常更大 |

FP4 的压缩率更高，但同一个 scale 下只有很少的可表示值，因此：

- 分组策略更加重要。
- 离群值造成的影响更明显。
- 小值更容易被量化为零。

---

## 13. RNE 舍入的作用

代码在以下位置使用 RNE：

1. 寻找组内最大 exponent 时，预测尾数舍入进位。
2. FP4 normal 路径中，把 17-bit fraction 压缩为 1 bit。
3. FP4 subnormal 路径中，移动 significand 后进行舍入。

RNE 的目标是减少大量量化操作中的系统性偏差：

```text
小于中点 -> 向下
大于中点 -> 向上
正好中点 -> 舍向最低保留位为偶数的一侧
```

---

## 14. 实现限制和注意事项

1. `fp4` 实际是 `uint8_t`，只有低 4 bit 有效。
2. `uint6` 实际也是 `uint8_t`，不会自动限制为 6 bit。
3. 输入 vector 按值传递，会复制整个 FP24 group。
4. FP24 subnormal 在核心函数中直接清零。
5. 量化为零时不会保留负零。
6. 中间 exponent 依赖 `uint8_t` 的模 256 下溢行为。
7. `exp & 0x80` 被用作逻辑负 exponent 的检测。
8. 源码将 subnormal 分支标注为 `probably`，说明作者也把该部分视为待验证实现。
9. 代码没有提供配套的 E2M1 解码函数或 `main()` 测试程序。
10. 一个 Inf/NaN 会使整个 group 的 FP4 元素清零，并返回特殊 scale `0x3F`。
11. 调用方必须同时保存 FP4 元素和 `mx_scale`，否则无法恢复原数量级。
12. 该文件没有系统覆盖 exponent 溢出、所有 tie、subnormal 边界和异常传播测试。

---

## 15. 建议的解码函数

为了验证量化结果，可以使用：

```cpp
static float fp4_e2m1_to_float(uint8_t fp4, uint6 mx_scale) {
    uint8_t sign = (fp4 >> 3) & 0x1;
    uint8_t exp = (fp4 >> 1) & 0x3;
    uint8_t mant = fp4 & 0x1;

    float value;
    if (exp == 0) {
        value = mant ? 0.5f : 0.0f;
    } else {
        value = (1.0f + mant / 2.0f)
              * std::pow(2.0f, static_cast<int>(exp) - 1);
    }

    if (sign) {
        value = -value;
    }

    return value
         * std::pow(2.0f, static_cast<int>(mx_scale) - 31);
}
```

如果 `mx_scale == 0x3F` 被定义为异常标记，解码器还应在正常数值解码前单独处理它。

---

## 16. 核心公式汇总

FP24 normal：

```text
x = (-1)^S
  * (1 + M24 / 2^17)
  * 2^(E24 - 31)
```

E2M1 normal：

```text
q = (-1)^S
  * (1 + M4 / 2)
  * 2^(E4 - 1)
```

E2M1 subnormal：

```text
q = (-1)^S * M4 * 0.5
```

MX 重建：

```text
x_reconstructed = q * 2^(mx_scale - 31)
```

共享 scale：

```text
mx_scale = max_exp - 2
```

逐元素 exponent 映射：

```text
E4 = E24 - mx_scale + 1
```

最大 E2M1 值：

```text
1.5 * 2^2 = 6
```

---

## 17. 一句话理解这段代码

这段代码先用组内最大 FP24 exponent 选择一个共享的二进制 scale，再把每个 FP24
压缩成只有 1-bit sign、2-bit exponent 和 1-bit fraction 的 E2M1；它用 RNE 控制
舍入偏差，但由于每个元素只有 4 bit，组内小值和离群值带来的精度损失会非常明显。
