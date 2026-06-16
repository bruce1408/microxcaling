# FP24 到 NVFP4 的分组量化技术说明

本文档说明 [`fp24_NvFP4.cpp`](./fp24_NvFP4.cpp) 中实际实现的量化方法。

核心函数为：

```cpp
fp8 group_fp24_Nvfp4(
    std::vector<uint32_t> fp24s,
    std::vector<fp4> &fp4s);
```

它将一组 FP24 数值量化为：

```text
一个共享 FP8 E4M3 scale + 一组 FP4 E2M1 元素
```

与 `fp24_MXFP4.cpp` 不同，本文件的共享 scale 不是纯二次幂指数，而是一个带
3-bit fraction 的 FP8 E4M3 数值。

---

## 1. 整体量化模型

一个 group 的完整结果为：

```text
{scale_e4m3, fp4[0], fp4[1], ..., fp4[N-1]}
```

反量化公式：

```text
reconstructed_value
    = decode_e4m3(scale)
    * decode_e2m1(fp4)
```

量化过程可以从数值角度理解为：

```text
1. 找到组内绝对值最大值 absmax
2. 计算理想共享 scale：

       ideal_scale = absmax / 6

3. 把 ideal_scale 向正方向量化为 FP8 E4M3：

       scale = ceil_to_e4m3(ideal_scale)

4. 对每个元素计算：

       normalized = input / scale

5. 把 normalized 用 RNE 量化为 FP4 E2M1
```

之所以除以 6，是因为该文件使用的 E2M1 最大有限值为 6。

---

## 2. 三种数据格式

### 2.1 FP24 输入

位布局：

```text
bit 23     bit 22 ........ bit 17     bit 16 ........ bit 0
+--------+--------------------------+-------------------------+
| sign   | exponent, 6 bits         | fraction, 17 bits       |
+--------+--------------------------+-------------------------+
```

normal FP24 的值：

```text
x = (-1)^sign
    * (1 + fraction / 2^17)
    * 2^(exponent - 31)
```

主要掩码：

| 掩码 | 含义 |
|---|---|
| `0x800000` | sign |
| `0x7E0000` | 6-bit exponent |
| `0x01FFFF` | 17-bit fraction |
| `0x7FFFFF` | 清除 sign，得到绝对值位模式 |
| `0x020000` | 补回 normal 的隐含 leading 1 |

### 2.2 FP4 E2M1 元素

位布局：

```text
bit 3       bit 2 .... bit 1       bit 0
+----------+----------------------+----------+
| sign     | exponent, 2 bits     | fraction |
+----------+----------------------+----------+
```

normal 解码公式：

```text
value = (-1)^sign
      * (1 + fraction / 2)
      * 2^(exponent - 1)
```

本文件的辅助解码函数把非零 subnormal 定义为：

```cpp
(mant / 2) * 0.5
```

因此：

```text
FP4 0x1 = 0.25
FP4 0x9 = -0.25
```

正数码表：

| FP4 | 数值 |
|---:|---:|
| `0x0` | 0 |
| `0x1` | 0.25 |
| `0x2` | 1 |
| `0x3` | 1.5 |
| `0x4` | 2 |
| `0x5` | 3 |
| `0x6` | 4 |
| `0x7` | 6 |

注意，`0.25` 和最小 normal `1.0` 之间没有其他可表示值。

### 2.3 FP8 E4M3 共享 scale

位布局：

```text
bit 7       bit 6 ........ bit 3       bit 2 .... bit 0
+----------+--------------------------+-------------------+
| sign     | exponent, 4 bits         | fraction, 3 bits  |
+----------+--------------------------+-------------------+
```

本文件只生成非负 scale，所以 sign 固定为 0。

normal scale：

```text
scale = (1 + fraction / 8) * 2^(exponent - 7)
```

subnormal scale：

```text
scale = (fraction / 8) * 2^-6
```

代码将：

```text
0x7F
```

留作 group 含 Inf/NaN 时的异常标记；正常 scale 最大截断到 `0x7E`。

---

## 3. 总体执行流程

```text
FP24 group
    |
    | 1. 查找绝对值最大元素
    v
absmax_fp24
    |
    | 2. 计算 absmax / 6
    | 3. 向正方向量化为 FP8 E4M3 scale
    v
shared scale
    |
    | 4. 把 scale 展开为整数 significand 和 exponent
    v
scale_exp + scale_mant
    |
    | 5. 对每个元素计算 input / scale
    | 6. 使用 RNE 编码为 FP4 E2M1
    v
FP4 group
```

---

## 4. 查找组内绝对值最大值

代码：

```cpp
uint32_t absmax_fp24 = 0x0;

for (auto fp24 : fp24s) {
    if ((fp24 & 0x7E0000) == 0x7E0000) {
        inf_nan = true;
        break;
    }

    absmax_fp24 = std::max(
        fp24 & 0x7FFFFF,
        absmax_fp24);
}
```

`fp24 & 0x7FFFFF` 清除 bit 23 的 sign，所以正负相同数值会得到相同绝对值位模式。

对于相同格式的非负 normal 浮点位模式：

```text
exponent 越大 -> 数值越大
exponent 相同 -> fraction 越大，数值越大
```

因此可以直接比较清除 sign 后的整数位模式。

### 4.1 异常输入

如果任意 FP24 的 exponent 全 1：

```cpp
if ((fp24 & 0x7E0000) == 0x7E0000)
```

代码将整个 group 处理为：

```cpp
fp4s = std::vector<fp4>(fp24s.size(), 0x0);
return 0x7f;
```

即：

```text
所有 FP4 = 0
scale = 0x7F
```

调用方必须把 `scale=0x7F` 解释为异常标记，而不能继续进行普通乘法重建。

---

## 5. 共享 scale 为什么是 `absmax / 6`

E2M1 最大编码：

```text
0x7 = 0 11 1
```

解码：

```text
(1 + 1/2) * 2^(3 - 1)
= 1.5 * 4
= 6
```

为了让最大元素映射到 FP4 最大值附近：

```text
absmax / scale ~= 6
```

因此：

```text
ideal_scale = absmax / 6
```

源码没有使用浮点除法，而是用 FP24 exponent、significand 和整数长除法构造 E4M3 scale。

---

## 6. 构造 `absmax / 6`

### 6.1 提取最大值 significand

```cpp
uint32_t dividend =
    (absmax_fp24 & 0x1FFFF) + 0x20000;
```

`dividend` 表示 FP24 normal significand：

```text
1.fraction * 2^17
```

### 6.2 除数为什么是 `0x30000`

```cpp
uint32_t divisor = 0x30000;
```

因为：

```text
0x30000 / 2^17 = 1.5
```

而：

```text
6 = 1.5 * 2^2
```

所以除以 6 被拆成：

```text
significand 除以 1.5
exponent 再减 2
```

这就是代码中：

```text
dividend / divisor
```

以及 exponent 表达式中 `-2` 的来源。

### 6.3 exponent 转换

代码：

```cpp
uint8_t exp =
    (absmax_fp24 & 0x7E0000) >> 17;

if (dividend < divisor) {
    exp += 7 - 31 - 2 - 1;
    dividend <<= 1;
}
else {
    exp += 7 - 31 - 2;
}
```

常数含义：

```text
+7   ：转换到 E4M3 exponent bias
-31  ：移除 FP24 exponent bias
-2   ：除以 2^2
-1   ：当 significand / 1.5 小于 1 时进行归一化补偿
```

如果：

```text
dividend < divisor
```

说明：

```text
FP24 significand / 1.5 < 1
```

需要把 dividend 左移一位，使商重新落入 normal significand 区间，同时 exponent 减一。

---

## 7. `uint_divide_within_loop`：整数二进制长除法

辅助函数：

```cpp
uint8_t uint_divide_within_loop(
    uint32_t &dividend,
    const uint32_t divisor,
    unsigned div_loop_num);
```

它逐位计算二进制商：

```cpp
for (; div_loop_num > 0; --div_loop_num) {
    if (dividend >= divisor) {
        res += 1;
        dividend -= divisor;
    }

    res <<= 1;
    dividend <<= 1;
}

res >>= 1;
```

每轮执行：

1. 比较当前余数和 divisor。
2. 如果余数足够大，当前商位写 1，并减去 divisor。
3. 商左移，为下一位腾出位置。
4. 余数左移，相当于继续展开二进制小数。

函数返回商的若干高位，并通过引用参数 `dividend` 留下最终余数。

这个余数随后被用于判断：

- scale 是否需要向上舍入。
- FP4 元素是否大于中点。
- RNE tie 时应该选择哪一侧。

---

## 8. 把理想 scale 量化为 E4M3

### 8.1 scale 太小

代码注释把逻辑 exponent `<= -4` 的情况整体清零：

```cpp
if ((exp & 0x80) && (exp <= 0xFC)) {
    fp4s = std::vector<fp4>(fp24s.size(), 0x0);
    return 0x0;
}
```

这里依赖 `uint8_t` 的模 256 表示：

```text
-1 -> 0xFF
-2 -> 0xFE
-3 -> 0xFD
-4 -> 0xFC
```

全零 group 也会走到该路径。

### 8.2 E4M3 subnormal scale

如果 exponent 为 0 或逻辑负数，但仍在可表示范围：

```cpp
if ((!exp) || (exp & 0x80))
```

代码根据 exponent 决定需要生成多少个 scale fraction bit：

```cpp
uint8_t div_loop_num =
    exp ? 3 - (~exp + 1) : 3;
```

之后执行整数长除法。

### 8.3 E4M3 normal scale

normal scale 使用 4 个商位：

```cpp
uint8_t div_loop_num = 4;
scale = uint_divide_within_loop(
    dividend,
    divisor,
    div_loop_num);
```

这些商位包括：

```text
1-bit leading significand + 3-bit E4M3 fraction
```

### 8.4 scale 使用向正方向舍入

无论 normal 还是 subnormal，scale 都使用：

```cpp
if (dividend) {
    scale += 1;
}
```

这里的 `dividend` 是长除法后的余数。

只要余数非零，就把 scale 增加一个最低单位，相当于：

```text
scale = ceil_to_e4m3(ideal_scale)
```

它不是 RNE。

这样做的目的，是避免 scale 小于 `absmax / 6`：

```text
scale >= absmax / 6
```

于是：

```text
absmax / scale <= 6
```

最大元素不会因为 scale 向下舍入而超出 E2M1 最大值 6。

### 8.5 scale 舍入进位

如果向上舍入产生第 5 bit：

```cpp
if (scale & 0x10) {
    exp += 1;
    scale >>= 1;
}
```

说明 significand 从：

```text
1.111 + rounding -> 10.000
```

需要右移重新归一化，并增加 exponent。

### 8.6 scale 溢出截断

```cpp
if (exp > 0xf) {
    exp = 0xf;
}
```

拼接 E4M3：

```cpp
scale = (scale & 0x7) + (exp << 3);
```

如果结果是异常标记 `0x7F`：

```cpp
if (scale == 0x7f) {
    scale = 0x7E;
}
```

即把正常数值 scale 的上限限制为 `0x7E`，避免和 group 异常标记冲突。

---

## 9. 展开 scale 以便逐元素相除

计算得到 FP8 scale 后，代码重新提取：

```cpp
uint8_t scale_exp = (scale & 0x78) >> 3;
uint32_t scale_mant = scale & 0x7;
```

### 9.1 normal scale

```cpp
if (scale & 0x78) {
    scale_mant += 0x8;
    scale_mant <<= 14;
}
```

`scale_mant += 0x8` 补回 E4M3 normal 的 leading 1：

```text
1.xxx
```

左移 14 位后，scale significand 与 FP24 的 18-bit `mantissa` 对齐，便于整数除法。

### 9.2 subnormal scale

```cpp
else {
    scale_exp -= countLeadingZeros8bitsa[scale] - 5;
    scale_mant <<= 14
        + countLeadingZeros8bitsa[scale]
        - 5
        + 1;
}
```

E4M3 subnormal 没有隐含 leading 1，所以需要：

1. 通过前导零数量找到最高有效位。
2. 调整 `scale_exp`。
3. 左移 `scale_mant` 完成归一化。

`countLeadingZeros8bitsa` 是一个 256 项查表，用于快速获得 8-bit 值的前导零数量。

---

## 10. 逐元素计算 `input / scale`

### 10.1 提取符号

```cpp
uint8_t sign =
    (fp24s[i] & 0x800000) >> 20;
```

FP24 sign 从 bit 23 移到 FP4 bit 3：

```text
正数 -> 0x0
负数 -> 0x8
```

### 10.2 FP24 zero 和 subnormal

```cpp
if (!exp) {
    fp4s[i] = 0x0;
    continue;
}
```

该实现把 FP24 zero 和 FP24 subnormal 都清零，并且不保留负零。

### 10.3 significand 和 exponent 对齐

```cpp
uint32_t mantissa =
    (fp24s[i] & 0x1ffff) + 0x20000;
```

随后根据输入 significand 与 scale significand 的大小决定是否归一化：

```cpp
if (mantissa < scale_mant) {
    exp += -31 - (scale_exp - 7) - 1 + 1;
    mantissa <<= 1;
}
else {
    exp += -31 - (scale_exp - 7) + 1;
}
```

从数值意义看，这一步在计算：

```text
normalized = input / scale
```

其中：

```text
normalized exponent
    = input_real_exp - scale_real_exp
```

附加的 `+1` 用于转换到 E2M1 bias=1 的 exponent field。

如果输入 significand 小于 scale significand，商的 significand 小于 1，需要：

- 输入 mantissa 左移一位。
- exponent 再减一。

---

## 11. FP4 下溢和 subnormal 路径

如果对齐后的 exponent 逻辑上小于 `-1`：

```cpp
if ((exp & 0x80) && (exp < 0xFF)) {
    fp4s[i] = 0x0;
}
```

代码直接清零。

如果 exponent 为 0 或逻辑值为 `-1`：

```cpp
else if ((!exp) || (exp & 0x80))
```

进入 FP4 subnormal/normal 边界路径。

商需要 2 个 bit：

```cpp
uint8_t div_loop_num =
    exp ? 2 - (~exp + 1) : 2;

auto fp4_mantissa =
    uint_divide_within_loop(
        mantissa,
        scale_mant,
        div_loop_num);
```

### 11.1 RNE 舍入

长除法返回值的最低 bit 是 round bit；引用参数 `mantissa` 保存余数，承担 sticky 信息。

```cpp
fp4_mantissa +=
    (fp4_mantissa & 0x1)
        ? (mantissa
            ? 0x2
            : (fp4_mantissa & 0x2))
        : 0;

fp4_mantissa >>= 1;
```

含义：

```text
round bit = 0:
    不进位

round bit = 1 且余数非零:
    大于中点，进位

round bit = 1 且余数为零:
    正好中点，由上一保留位决定 ties-to-even
```

如果结果非零：

```cpp
fp4s[i] = sign + fp4_mantissa;
```

否则统一输出 `+0`。

---

## 12. FP4 normal 路径

normal 路径固定提取 3 个商位：

```cpp
uint8_t div_loop_num = 3;

auto fp4_mantissa =
    uint_divide_within_loop(
        mantissa,
        scale_mant,
        div_loop_num);
```

这些位可以理解为：

```text
leading significand bit
FP4 fraction bit
round bit
```

随后执行与 subnormal 路径相同的 RNE：

```cpp
fp4_mantissa +=
    (fp4_mantissa & 0x1)
        ? (mantissa
            ? 0x2
            : (fp4_mantissa & 0x2))
        : 0;
```

### 12.1 舍入进位

```cpp
if (fp4_mantissa & 0x8) {
    exp += 1;
    fp4_mantissa =
        (fp4_mantissa >> 2) & 0x1;
}
else {
    fp4_mantissa =
        (fp4_mantissa >> 1) & 0x1;
}
```

如果出现 bit 3，说明：

```text
1.1 + rounding -> 10.0
```

需要 exponent 增加一，并重新提取 fraction。

### 12.2 exponent 溢出截断

```cpp
if (exp > 0x3) {
    exp = 0x3;
}
```

E2M1 exponent 只有 2 bit，所以最大为 3。

最终拼接：

```cpp
fp4s[i] =
    sign
    + (exp << 1)
    + fp4_mantissa;
```

---

## 13. 完整示例：组内最大值为 100

当前 `main()` 的 group 包含：

```text
{..., 6, 3, 1.5, ..., -6, ..., 100}
```

因此：

```text
absmax = 100
ideal_scale = 100 / 6
            = 16.666666...
```

代码将 scale 向正方向量化到 E4M3：

```text
scale = 18
```

E4M3 编码：

```text
18 = 1.125 * 2^4

sign     = 0
exponent = 4 + 7 = 11 = 1011
fraction = 0.125 * 8 = 1 = 001

scale bits = 0 1011 001 = 0x59
```

程序实际输出：

```text
Shared scale = 0x59 = 18
```

### 13.1 `100 -> FP4 0x7`

归一化值：

```text
100 / 18 = 5.555...
```

E2M1 相邻高值为：

```text
4 和 6
```

`5.555...` 更接近 6，因此：

```text
FP4 = 0x7
decode_e2m1(0x7) = 6
```

重建：

```text
18 * 6 = 108
```

误差：

```text
108 - 100 = 8
```

这里 scale 采用向上舍入，使最大值不会溢出 FP4，但也可能导致重建结果大于原值。

### 13.2 `6 -> FP4 0x1`

归一化：

```text
6 / 18 = 0.333...
```

本文件 E2M1 的非零 subnormal 是：

```text
0.25
```

因此量化为：

```text
FP4 = 0x1
```

重建：

```text
18 * 0.25 = 4.5
```

绝对误差：

```text
|6 - 4.5| = 1.5
```

### 13.3 `3 -> 0`

归一化：

```text
3 / 18 = 0.1667
```

距离：

```text
到 0：    0.1667
到 0.25： 0.0833
```

从理想最近值角度看它更接近 0.25，但当前位级实现实际输出 `0x0`。这说明该 demo 的
FP4 subnormal 边界和指数分支具有实现特定行为，不能仅根据抽象 E2M1 码表推断所有结果。

程序实际输出应作为该文件行为的最终依据。

---

## 14. 紧密分布 group 的示例

第二组测试：

```text
{1.0, 1.2, 1.5, 1.8, 2.0, 2.25}
```

最大值：

```text
absmax = 2.25
ideal_scale = 2.25 / 6 = 0.375
```

`0.375` 可被 E4M3 精确表示：

```text
scale = 0x2C = 0.375
```

实际输出：

| 原值 | FP4 | FP4 值 | 重建值 | 绝对误差 |
|---:|---:|---:|---:|---:|
| 1.0 | `0x5` | 3 | 1.125 | 0.125 |
| 1.2 | `0x5` | 3 | 1.125 | 0.075 |
| 1.5 | `0x6` | 4 | 1.5 | 0 |
| 1.8 | `0x6` | 4 | 1.5 | 0.3 |
| 2.0 | `0x7` | 6 | 2.25 | 0.25 |
| 2.25 | `0x7` | 6 | 2.25 | 0 |

这说明：

```text
group 内数值越集中，scale 越合适，整体有效精度通常越高。
```

---

## 15. 与 MXFP4 的主要区别

| 项目 | MXFP4 | 本文件 NVFP4 |
|---|---|---|
| 元素格式 | FP4 E2M1 | FP4 E2M1 |
| 共享 scale | 6-bit 纯指数 | FP8 E4M3 数值 |
| scale 精度 | 仅二次幂 | 有 3-bit fraction |
| scale 计算 | 根据最大 exponent | 近似 `absmax / 6` |
| scale 舍入 | exponent 对齐 | 向正方向量化 |
| 元素舍入 | RNE | RNE |
| 重建 | `fp4 * 2^k` | `fp4 * E4M3_scale` |

NVFP4 的 E4M3 scale 可以表示：

```text
1.0、1.125、1.25、1.5、1.75 等非二次幂比例
```

因此它比纯指数 scale 更灵活，但 scale 计算和硬件实现也更复杂。

---

## 16. 为什么 scale 向上舍入，而元素使用 RNE

两种舍入承担不同目标。

### scale 向上舍入

目标：

```text
避免 absmax / scale > 6
```

从而尽量避免最大元素超过 FP4 动态范围。

### FP4 元素使用 RNE

目标：

```text
在已有 scale 下，选择最接近的 E2M1 值，
并减少大量量化操作中的统计偏差。
```

简而言之：

```text
scale 舍入优先保证范围
元素舍入优先保证局部精度
```

---

## 17. 辅助函数

### `fp24_to_float`

把 FP24 位模式解码为普通 `float`，用于测试和理解格式。

它不参与核心量化流程，而且当前文件中没有实际调用，因此编译时会产生未使用函数警告。

### `fp4_to_float`

把 FP4 E2M1 解码为 `float`：

```text
normal:
    (1 + mant/2) * 2^(exp-1)

subnormal:
    (mant/2) * 0.5
```

### `fp8_e4m3_to_float`

把共享 E4M3 scale 解码为 `float`。

### `float_to_fp24`

把测试用 FP32 值近似编码为 FP24。

它是 demo 输入生成器，不是 NVFP4 核心算法的一部分。

---

## 18. 实现限制和注意事项

1. `fp4` 和 `fp8` 都只是 `uint8_t` 类型别名。
2. FP4 只有低 4 bit 有效。
3. 输入 vector 按值传递，会复制整个 group。
4. FP24 subnormal 在核心量化中直接清零。
5. 量化为零时不保留负零。
6. 多处 exponent 计算依赖 `uint8_t` 下溢和补码位模式。
7. `scale=0x7F` 被当作异常标记，调用方必须单独识别。
8. 正常 scale 被限制到 `0x7E`，避免与异常标记冲突。
9. scale 使用向正方向舍入，不是 RNE。
10. FP4 元素使用 RNE，但 subnormal 边界行为具有代码特定性。
11. `countLeadingZeros8bitsa` 只服务于 E4M3 subnormal scale 的归一化。
12. `uint_divide_within_loop` 会修改传入的 dividend，使其变成缩放后的余数。
13. `float_to_fp24` 是简化测试函数，不是工业级 FP24 编码器。
14. 压缩率输出只计算每元素 24 bit 到 4 bit，没有把每组共享的 8-bit scale 计入。
15. 代码缺少系统化单元测试来覆盖所有 scale、tie、subnormal 和溢出边界。

---

## 19. 实际存储成本

假设 group 大小为 `N`：

```text
FP24 输入：24N bits
NVFP4 输出：4N + 8 bits
```

实际压缩比：

```text
compression ratio = 24N / (4N + 8)
```

例如：

```text
N = 16:
输入 = 384 bits
输出 = 64 + 8 = 72 bits
压缩比 = 384 / 72 = 5.33
```

所以程序打印的“6x smaller”是忽略共享 scale 开销后的渐近值。当 group 足够大时，压缩比
才逐渐接近 6。

---

## 20. 核心公式汇总

FP24 normal：

```text
x = (-1)^S
  * (1 + M24 / 2^17)
  * 2^(E24 - 31)
```

理想 NVFP4 scale：

```text
ideal_scale = absmax / 6
```

实际 scale：

```text
scale = ceil_to_e4m3(ideal_scale)
```

逐元素归一化：

```text
normalized = x / scale
```

FP4 E2M1 normal：

```text
q = (-1)^S
  * (1 + M4 / 2)
  * 2^(E4 - 1)
```

本文件的 E2M1 subnormal：

```text
q = (-1)^S * M4 * 0.25
```

重建：

```text
x_reconstructed = scale * q
```

---

## 21. 一句话理解这段代码

这段代码先取组内绝对值最大值，把 `absmax / 6` 向上量化成一个 FP8 E4M3 共享 scale，
再用整数二进制长除法计算每个 `input / scale`，最后通过 RNE 将结果压缩成 FP4 E2M1；
E4M3 scale 提供了比纯二次幂 scale 更细的缩放精度，但实现和边界处理也明显更复杂。
