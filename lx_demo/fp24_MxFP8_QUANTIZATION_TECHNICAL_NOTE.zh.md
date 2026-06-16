# FP24 到 MXFP8 E4M3 的分组量化技术说明

本文档说明 [`fp24_MxFP8.cpp`](./fp24_MxFP8.cpp) 中实际实现的量化方法。
重点是核心函数：

```cpp
uint6 group_fp24_Mxfp8(
    std::vector<uint32_t> fp24s,
    std::vector<fp8> &fp8s);
```

该函数将一组 FP24 位模式转换为一组 FP8 E4M3 位模式，并为整组数据计算一个共享
指数 `mx_scale`。

---

## 1. 量化目标

输入组中的每个元素使用 24 bit：

```text
FP24 = sign(1 bit) + exponent(6 bits) + fraction(17 bits)
```

输出组中的每个元素使用 8 bit：

```text
FP8 E4M3 = sign(1 bit) + exponent(4 bits) + fraction(3 bits)
```

此外，整个 group 共享一个逻辑上为 6 bit 的 `mx_scale`。

所以一个 group 的完整量化结果是：

```text
{mx_scale, fp8[0], fp8[1], ..., fp8[N-1]}
```

反量化公式为：

```text
reconstructed_value
    = decode_e4m3(fp8)
    * 2^(mx_scale - 31)
```

这是一种 block floating-point 思路：每个元素只保存较短的 FP8，整组数据共享一个
二进制缩放因子。

---

## 2. 数据格式

### 2.1 FP24

位布局：

```text
bit 23     bit 22 ........ bit 17     bit 16 ........ bit 0
+--------+--------------------------+-------------------------+
| sign   | exponent, 6 bits         | fraction, 17 bits       |
+--------+--------------------------+-------------------------+
```

对于 FP24 normal：

```text
x = (-1)^sign
    * (1 + fraction / 2^17)
    * 2^(exponent - 31)
```

其中 exponent bias 为 31。

代码使用的主要掩码：

| 掩码 | 含义 |
|---|---|
| `0x800000` | FP24 sign，bit 23 |
| `0x7E0000` | FP24 exponent，bit[22:17] |
| `0x01FFFF` | FP24 fraction，bit[16:0] |
| `0x020000` | normal 隐含的 leading 1，即 `1 << 17` |

### 2.2 FP8 E4M3

位布局：

```text
bit 7      bit 6 .... bit 3      bit 2 .... bit 0
+--------+---------------------+-------------------+
| sign   | exponent, 4 bits    | fraction, 3 bits  |
+--------+---------------------+-------------------+
```

代码中的解码规则是：

```text
exp == 0:
    value = (-1)^sign * (fraction / 8) * 2^-6

exp != 0:
    value = (-1)^sign * (1 + fraction / 8) * 2^(exp - 7)
```

此实现把 `exp=15` 也当作 normal，因此最大有限 E4M3 数为：

```text
(1 + 7/8) * 2^(15 - 7)
= 1.875 * 256
= 480
```

---

## 3. 总体流程

核心量化分为三个阶段：

```text
FP24 group
   |
   | Phase 1: 扫描整组，寻找经过保守舍入后的最大 exponent
   v
max_exp
   |
   | Phase 2: 根据 max_exp 计算共享 mx_scale
   v
mx_scale
   |
   | Phase 3: 使用同一个 mx_scale 逐元素转换为 E4M3
   v
FP8 group
```

伪代码如下：

```text
max_exp = find_group_max_rounded_exponent(fp24s)
mx_scale = max_exp - 8

for each fp24:
    sign, exponent, mantissa = unpack(fp24)
    exponent = align_exponent(exponent, mx_scale)

    if aligned exponent can be represented as FP8 normal:
        round mantissa to 3 bits with RNE
        encode FP8 normal
    else if it can be represented as FP8 subnormal:
        shift and round mantissa with RNE
        encode FP8 subnormal
    else:
        output zero
```

---

## 4. Phase 1：确定组内最大指数

代码首先提取每个 FP24 的 exponent：

```cpp
uint8_t exp = (fp24 & 0x7e0000) >> 17;
```

对于 normal，补回隐含的 leading 1：

```cpp
uint32_t mantissa = (fp24 & 0x1ffff) + 0x20000;
```

此时 `mantissa` 是一个 18-bit 整数，表示：

```text
1.fraction * 2^17
```

### 4.1 为什么不能只比较原始 exponent

如果尾数非常接近 `2.0`，后续从 17-bit fraction 压缩到 3-bit fraction 时可能发生：

```text
1.111... --RNE--> 10.000...
```

这会令 exponent 加一。

如果共享 scale 只根据舍入前的 exponent 计算，可能低估该元素最终需要的动态范围。
因此，代码在寻找 `max_exp` 时先进行一次保守 RNE。

### 4.2 第一阶段的 RNE

代码：

```cpp
mantissa += (mantissa & 0x4000)
    ? ((mantissa & 0x3fff) ? 0x8000 : mantissa & 0x8000)
    : 0;
```

参与判断的位为：

```text
bit 15       bit 14       bit 13 ........ bit 0
retained LSB round bit    sticky bits
```

规则：

| round bit | sticky bits | retained LSB | 结果 |
|---:|---:|---:|---|
| 0 | 任意 | 任意 | 不进位 |
| 1 | 非 0 | 任意 | 大于中点，进位 |
| 1 | 0 | 1 | 正好一半，进位到偶数 |
| 1 | 0 | 0 | 正好一半，保持偶数 |

这就是 Round to Nearest, Ties to Even，简称 RNE。

如果舍入后 bit 18 被置位：

```cpp
if (mantissa & 0x40000) {
    exp += 1;
}
```

说明尾数已从 `1.x` 进位为 `10.x`，因此 exponent 需要加一。

最终得到整个 group 的 `max_exp`。

---

## 5. Phase 2：计算共享 `mx_scale`

代码使用：

```cpp
uint6 mx_scale = max_exp - 8;
```

推导如下。

FP24 最大元素的主要数量级为：

```text
2^(max_exp - 31)
```

E4M3 最大 exponent field 是 15，bias 是 7，所以最大真实 exponent 是：

```text
15 - 7 = 8
```

量化希望把组内最大元素对齐到 E4M3 的高 exponent 区间：

```text
2^8 * 2^(mx_scale - 31)
    ~= 2^(max_exp - 31)
```

比较指数：

```text
8 + mx_scale - 31 = max_exp - 31
```

得到：

```text
mx_scale = max_exp - 8
```

注意，`mx_scale` 本身采用 bias=31 的编码。实际缩放因子是：

```text
scale_factor = 2^(mx_scale - 31)
```

例如：

```text
max_exp = 37
mx_scale = 37 - 8 = 29
scale_factor = 2^(29 - 31) = 2^-2 = 0.25
```

### 5.1 共享 scale 的意义

共享 scale 越大，整组表示的数值范围越大，但较小元素越容易落入 FP8 subnormal 或被
舍入为零。

因此，MX 分组量化存在典型权衡：

```text
组内动态范围较集中：
    小值和大值都能较准确表示

组内动态范围很大：
    scale 被最大值决定，小值精度下降，甚至变成零
```

---

## 6. Phase 3：逐元素量化

### 6.1 提取并移动符号位

```cpp
uint8_t sign = (fp24s[i] & 0x800000) >> 16;
```

FP24 sign 位于 bit 23，FP8 sign 位于 bit 7。右移 16 位后，符号直接落到 FP8 的目标位置：

```text
FP24 bit 23 -> FP8 bit 7
```

### 6.2 提取 exponent 和 significand

```cpp
uint8_t exp = (fp24s[i] & 0x7e0000) >> 17;
uint32_t mantissa = (fp24s[i] & 0x1ffff) + 0x20000;
```

当前实现遇到 `exp == 0` 时直接输出零：

```cpp
if (!exp) {
    fp8s[i] = 0x0;
    continue;
}
```

这表示 FP24 zero 和 FP24 subnormal 都被 flush to zero，并且负零也不会保留。

### 6.3 exponent 对齐

代码：

```cpp
exp += 6 - mx_scale;
```

如果把原始 exponent 写作 `fp24_exp`，中间值为：

```text
intermediate_exp = fp24_exp + 6 - mx_scale
```

normal 路径之后还会执行：

```cpp
exp += 1;
```

因此最终 FP8 exponent field 为：

```text
fp8_exp = fp24_exp + 7 - mx_scale
```

它也可以从真实指数相等关系推导：

```text
fp24_exp - 31
    = (fp8_exp - 7) + (mx_scale - 31)
```

整理得到：

```text
fp8_exp = fp24_exp + 7 - mx_scale
```

两种推导结果一致。

---

## 7. FP8 normal 路径

当中间 exponent 没有发生 `uint8_t` 下溢时，代码进入 normal 路径：

```cpp
if (!(exp & 0x80)) {
    ...
}
```

这里的实现依赖无符号 8-bit 的模 256 算术。逻辑上的负数会表现为：

```text
-1 -> 0xFF
-2 -> 0xFE
```

所以 bit 7 可用于判断中间 exponent 是否是下溢后的负值。

### 7.1 尾数从 17 bit 压缩到 3 bit

最终保留：

```text
mantissa bit[16:14]
```

舍入位为：

```text
round bit  = bit 13
sticky     = bit[12:0]
tie LSB    = bit 14
```

对应代码：

```cpp
mantissa += (mantissa & 0x2000)
    ? ((mantissa & 0x1fff) ? 0x4000 : mantissa & 0x4000)
    : 0;
```

这仍然是 RNE。

如果尾数舍入溢出：

```cpp
if (mantissa & 0x40000) {
    exp += 1;
    mantissa >>= 1;
}
```

代码增加 exponent，并将尾数重新归一化。

随后提取 3-bit fraction：

```cpp
fp8_mantissa = (mantissa >> 14) & 0x7;
```

最后拼接 FP8：

```cpp
fp8s[i] = sign + (exp << 3) + fp8_mantissa;
```

其位布局为：

```text
sign             exponent           fraction
bit 7            bit[6:3]           bit[2:0]
```

由于三个字段互不重叠，这里的加法等价于：

```cpp
sign | (exp << 3) | fp8_mantissa
```

---

## 8. FP8 subnormal 路径

如果 exponent 对齐后逻辑上小于 normal 可表示范围，`uint8_t` 会发生下溢：

```text
逻辑 exponent    uint8_t 位模式
-1               0xFF
-2               0xFE
-3               0xFD
-4               0xFC
-5               0xFB
```

代码只处理 `-1` 到 `-4`：

```cpp
if (exp < 0xfc) {
    fp8s[i] = 0x0;
}
```

逻辑 exponent 小于 `-4` 时，数值太小，直接输出零。

对于 `-1` 到 `-4`，需要把 significand 额外右移：

```cpp
uint32_t shift = ~exp + 1;
```

在低 8 bit 补码意义下：

```text
~0xFF + 1 -> 1
~0xFE + 1 -> 2
~0xFD + 1 -> 3
~0xFC + 1 -> 4
```

代码根据右移量动态确定 round bit：

```cpp
uint32_t rounding_bit = 0x2000 << (~exp + 1);
uint32_t sticky_bit = rounding_bit - 1;
```

然后执行 RNE，并提取 subnormal fraction：

```cpp
fp8_mantissa = mantissa >> (14 + (~exp + 1));
```

subnormal 的 exponent field 固定为零，所以只需要组合 sign 和 fraction：

```cpp
fp8s[i] = fp8_mantissa ? (sign + fp8_mantissa) : 0x0;
```

如果最终 fraction 为零，代码输出 `+0`，不保留负零。

---

## 9. 异常值处理

### 9.1 输入组包含 FP24 Inf 或 NaN

FP24 exponent 全 1 时：

```cpp
if (exp == 0x3f) {
    inf_nan = true;
}
```

当前策略不是逐元素传播异常值，而是把整个 group 写成：

```cpp
fp8s = std::vector<fp8>(fp24s.size(), 0x7f);
return 0x3f;
```

即：

```text
所有 FP8 元素 = 0x7F
mx_scale       = 0x3F
```

这是该 demo 自定义的整组异常标记策略。它与解码函数“把 `exp=15` 作为 normal”的规则并不
完全一致，因此实际工程中必须根据目标 MXFP8 规范统一异常语义。

### 9.2 共享 scale 为负

`mx_scale` 实际类型为 `uint8_t`：

```cpp
typedef uint8_t uint6;
```

如果 `max_exp - 8` 为负，会发生无符号下溢。代码通过 bit 7 检测：

```cpp
if (mx_scale & 0x80) {
    fp8s = std::vector<fp8>(fp24s.size(), 0x0);
    return 0x0;
}
```

因此，这种情况下整个 group 被清零。

---

## 10. 当前测试组的完整示例

`main()` 使用：

```text
{6.0, 3.0, 1.5, 1.0, 0.5, 0.25, 100.0, -3.0}
```

组内最大值是 `100.0`，其 FP24 biased exponent 为 37，因此：

```text
max_exp = 37
mx_scale = 37 - 8 = 29
scale_factor = 2^(29 - 31) = 0.25
```

量化相当于先从数值意义上除以共享 scale：

```text
E4M3 element value ~= original value / 0.25
                   ~= original value * 4
```

然后将结果表示为 E4M3。

### 10.1 `6.0` 的量化

程序生成的 FP24：

```text
6.0 -> 0x430000
```

字段为：

```text
sign     = 0
exponent = 33
fraction = 0x10000
```

FP24 数值：

```text
(1 + 0x10000 / 2^17) * 2^(33 - 31)
= 1.5 * 4
= 6
```

共享 scale 为 0.25，所以 E4M3 内部要表示：

```text
6 / 0.25 = 24
```

输出 FP8：

```text
0x5C = 0 1011 100
       S exp  mant
```

E4M3 解码：

```text
(1 + 4/8) * 2^(11 - 7)
= 1.5 * 16
= 24
```

乘回共享 scale：

```text
24 * 0.25 = 6
```

因此该元素可以精确重建。

### 10.2 `100.0` 的量化误差

`100.0` 除以共享 scale 后需要 E4M3 表示：

```text
100 / 0.25 = 400
```

当前代码将其量化为：

```text
FP8 = 0x7C = 0 1111 100
```

E4M3 解码：

```text
(1 + 4/8) * 2^(15 - 7)
= 1.5 * 256
= 384
```

乘回共享 scale：

```text
384 * 0.25 = 96
```

所以：

```text
original      = 100
reconstructed = 96
absolute error = 4
```

误差来自 E4M3 只有 3-bit fraction，而不是共享 scale 计算错误。

---

## 11. RNE 为什么重要

如果始终截断尾数，量化误差会产生明显的单向偏差。

RNE 的目标是：

```text
距离较近者优先；
正好在两个可表示值中间时，选择最低保留位为偶数的一侧。
```

它可以降低大量量化操作累积后的统计偏差。

本代码在三个位置使用 RNE 思路：

1. Phase 1 中保守估计舍入后最大 exponent。
2. FP8 normal 路径中把 FP24 fraction 压缩到 3 bit。
3. FP8 subnormal 路径中右移 significand。

---

## 12. 辅助函数与核心量化的区别

### `float_to_fp24`

用途：

```text
普通 float -> 构造 demo 所需的 FP24 输入位模式
```

它不是核心 MXFP8 量化算法，只是测试输入生成器。

该函数使用 `+0.5f` 后转整数的简化舍入，不是完整的 RNE。此外，当前 exponent 溢出分支
先设置：

```cpp
biased = 0x3F;
frac = 0.0f;
```

随后仍继续执行 normal mantissa 计算。若要用于生产代码，更清晰的处理方式是直接返回
带符号的 infinity 编码：

```cpp
return sign | 0x7E0000;
```

### `fp24_to_float`

用途：

```text
FP24 位模式 -> float
```

它只用于理解和验证 FP24，不参与 `group_fp24_Mxfp8` 的量化。

### `fp8_e4m3_to_float`

用途：

```text
FP8 位模式 + mx_scale -> 重建 float
```

它先解码 E4M3，再乘以：

```text
2^(mx_scale - 31)
```

---

## 13. 实现限制和工程注意事项

1. `uint6` 实际是 `uint8_t`，类型系统不会自动限制为 6 bit。
2. 中间 exponent 依赖 `uint8_t` 下溢和补码位模式，可读性较低。
3. FP24 subnormal 在核心量化函数中直接清零。
4. 舍入为零时不保留负零。
5. 组内出现一个 FP24 Inf/NaN，会使整个 group 输出异常标记。
6. `exp=15` 同时被解码器视为 E4M3 normal，又被 `0x7F` 路径用于异常标记，语义需统一。
7. scale 由组内最大 exponent 决定，离群大值会显著降低小值的精度。
8. `group_fp24_Mxfp8` 的输入 vector 按值传递，会复制整个输入组。
9. demo 没有对所有边界、tie、subnormal 和异常组合进行系统测试。
10. `float_to_fp24` 是简化测试辅助函数，不应直接作为工业级 FP24 编码器使用。

---

## 14. 核心公式汇总

FP24 normal：

```text
x = (-1)^S * (1 + M24 / 2^17) * 2^(E24 - 31)
```

E4M3 normal：

```text
q = (-1)^S * (1 + M8 / 2^3) * 2^(E8 - 7)
```

MX 重建：

```text
x_reconstructed = q * 2^(mx_scale - 31)
```

共享 scale：

```text
mx_scale = max_exp - 8
```

逐元素 exponent 映射：

```text
E8 = E24 + 7 - mx_scale
```

量化误差：

```text
error = original - reconstructed
absolute_error = |error|
```

---

## 15. 一句话理解这段量化代码

这段代码先用组内最大 FP24 exponent 选择一个共享的二进制 scale，再把所有 FP24 元素
在该 scale 下压缩为 1-bit sign、4-bit exponent 和 3-bit fraction 的 E4M3，并使用
RNE 处理尾数精度损失；大值决定范围，小值承担精度代价。
