# `fp24_MxFP8.cpp` 位操作详解

## 1. 代码解决什么问题

这个程序把一组自定义 FP24 数值量化为一组 MXFP8 E4M3 数值。

普通 FP8 为每个数独立保存符号、指数和尾数；MXFP8 额外让整个 group 共享一个 scale：

```text
原始值 ~= E4M3 元素值 * 2^(mx_scale - 31)
```

因此，一个 group 的完整结果不是只有 `fp8s`，而是：

```text
{mx_scale, fp8s[0], fp8s[1], ..., fp8s[N-1]}
```

## 2. 两种位布局

### 2.1 FP24

```text
23 22                 17 16                               0
+--+--------------------+----------------------------------+
|S | exponent, 6 bits   | fraction, 17 bits                |
+--+--------------------+----------------------------------+
```

正规数公式：

```text
(-1)^S * (1 + fraction / 2^17) * 2^(exponent - 31)
```

字段掩码：

```cpp
0x800000  // bit 23，sign
0x7E0000  // bit[22:17]，exponent
0x01FFFF  // bit[16:0]，fraction
```

### 2.2 FP8 E4M3

```text
7 6             3 2             0
+-+---------------+---------------+
|S| exponent 4 bit| fraction 3 bit|
+-+---------------+---------------+
```

字段掩码：

```cpp
0x80  // bit 7，sign
0x78  // bit[6:3]，exponent
0x07  // bit[2:0]，fraction
```

## 3. 必须先掌握的位运算

### 3.1 按位与 `&`

`&` 用来清除不需要的位。

```cpp
uint8_t exp = (fp24 & 0x7E0000) >> 17;
```

假设：

```text
fp24       = S EEEEEE FFFFFFFFFFFFFFFFF
0x7E0000   = 0 111111 00000000000000000
按位与结果 = 0 EEEEEE 00000000000000000
```

然后右移 17 位：

```text
00000000 00000000 00EEEEEE
```

最终得到普通整数形式的 exponent。

### 3.2 左移 `<<`

左移用于把字段放到目标位置，或乘以 2 的幂：

```cpp
exp << 3
```

把 4-bit exponent 从 bit[3:0] 移到 FP8 的 bit[6:3]。

数值上，在没有溢出的情况下：

```text
x << n = x * 2^n
```

### 3.3 右移 `>>`

右移用于提取高位字段，或除以 2 的幂：

```cpp
(fp24 & 0x800000) >> 16
```

FP24 sign 原来位于 bit23，右移 16 位后落到 bit7，正好成为 FP8 sign。

### 3.4 按位或 `|`

`|` 用于拼接互不重叠的字段：

```cpp
sign | (exp << 3) | mantissa
```

代码部分位置使用加法：

```cpp
sign + (exp << 3) + mantissa
```

因为字段互不重叠，两者结果相同；`|` 更能表达“拼接位字段”的意图。

### 3.5 按位取反 `~`

代码使用：

```cpp
~exp + 1
```

这是补码求负公式：

```text
-x = ~x + 1
```

当 `exp` 的低 8 bit 是：

```text
0xFF -> 逻辑 -1 -> ~0xFF + 1 = 1
0xFE -> 逻辑 -2 -> ~0xFE + 1 = 2
0xFD -> 逻辑 -3 -> ~0xFD + 1 = 3
```

程序借此计算 subnormal 还需要额外右移多少位。

## 4. 掩码速查表

| 掩码 | 二进制意义 | 用途 |
|---|---|---|
| `0x800000` | FP24 bit23 | 提取符号 |
| `0x7E0000` | FP24 bit[22:17] | 提取 6-bit exponent |
| `0x1FFFF` | FP24 bit[16:0] | 提取 17-bit fraction |
| `0x20000` | `1 << 17` | 补回 normal 的 implied leading 1 |
| `0x40000` | `1 << 18` | 检测尾数舍入溢出 |
| `0x4000` | `1 << 14` | 第一阶段 round bit |
| `0x3FFF` | bit[13:0] | 第一阶段 sticky bits |
| `0x8000` | `1 << 15` | 第一阶段 retained LSB/进位值 |
| `0x2000` | `1 << 13` | FP8 最终舍入的 round bit |
| `0x1FFF` | bit[12:0] | FP8 最终舍入的 sticky bits |
| `0x4000` | `1 << 14` | 最终 retained LSB/进位值 |
| `0x7` | `0b111` | 只保留 3-bit FP8 fraction |
| `0x3F` | `0b111111` | 6-bit exponent 全 1 |
| `0x80` | `0b10000000` | uint8_t 最高位/FP8 sign |

## 5. `group_fp24_Mxfp8`

### 用法

```cpp
std::vector<uint32_t> fp24s = {/* FP24 位模式 */};
std::vector<fp8> fp8s(fp24s.size());

uint6 scale = group_fp24_Mxfp8(fp24s, fp8s);
```

输出：

- `fp8s`：每个元素的 E4M3 编码
- 返回值：整个 group 的共享 `mx_scale`

### 第一阶段：寻找最大 exponent

```cpp
uint8_t max_exp = 7;
```

为最大 exponent 设置下界。后续 scale 由它计算。

```cpp
for (auto fp24 : fp24s)
```

遍历 group 中的所有 FP24 位模式。

```cpp
uint8_t exp = (fp24 & 0x7E0000) >> 17;
```

先屏蔽无关位，再把 exponent 移到最低位。

```cpp
if (exp == 0x3F)
```

6-bit exponent 全 1，FP24 是 Inf 或 NaN。

```cpp
uint32_t mantissa = (fp24 & 0x1FFFF) + 0x20000;
```

提取 17-bit fraction，并补回正规数隐含的 leading 1。

得到的整数 `mantissa` 表示：

```text
1.fraction * 2^17
```

### 第一阶段的 RNE

原始代码的核心表达式为：

```cpp
mantissa += (mantissa & 0x4000)
    ? ((mantissa & 0x3fff) ? 0x8000 : (mantissa & 0x8000))
    : 0;
```

等价于：

```cpp
bool round_bit = mantissa & (1 << 14);
bool sticky = mantissa & ((1 << 14) - 1);
bool retained_lsb = mantissa & (1 << 15);

if (round_bit && (sticky || retained_lsb)) {
    mantissa += 1 << 15;
}
```

四种情况：

| round | sticky | retained LSB | 操作 |
|---:|---:|---:|---|
| 0 | 任意 | 任意 | 不进位 |
| 1 | 1 | 任意 | 大于中点，进位 |
| 1 | 0 | 1 | 正好中点，奇数进位成偶数 |
| 1 | 0 | 0 | 正好中点，已经为偶数 |

这就是 Round to Nearest, Ties to Even。

```cpp
if (mantissa & 0x40000) exp += 1;
```

如果舍入后出现 bit18，表示：

```text
1.111... + rounding -> 10.000...
```

尾数跨越了一个 2 的幂区间，因此 exponent 必须加一。

### 第二阶段：计算共享 scale

```cpp
uint6 mx_scale = max_exp - 8;
```

FP24 的真实指数：

```text
max_exp - 31
```

代码采用的 E4M3 最大真实指数是：

```text
15 - 7 = 8
```

因此共享 scale 的真实指数约为：

```text
(max_exp - 31) - 8
```

编码成 bias=31 后：

```text
mx_scale = max_exp - 8
```

例如最大元素是 100：

```text
FP24 exponent = 37
mx_scale = 37 - 8 = 29
真实 scale exponent = 29 - 31 = -2
scale factor = 2^-2 = 0.25
```

### 第三阶段：逐元素转换

```cpp
uint8_t sign = (fp24s[i] & 0x800000) >> 16;
```

FP24 sign 从 bit23 移到 FP8 bit7。

```cpp
uint8_t exp = (fp24s[i] & 0x7E0000) >> 17;
```

提取 FP24 exponent。

```cpp
if (!exp)
```

`!exp` 等价于 `exp == 0`。该实现把 zero 和 FP24 subnormal 都输出为零。

```cpp
exp += 6 - mx_scale;
```

将 FP24 exponent 对齐到共享 scale 下的 FP8 指数工作区间。

这里依赖 `uint8_t` 的模 256 算术。如果逻辑结果为负：

```text
-1 -> 255 -> 0xFF
-2 -> 254 -> 0xFE
```

因此：

```cpp
if (!(exp & 0x80))
```

检查 bit7 是否为 0，用于判断是否进入 normal 路径。

这是一种紧凑但可读性较弱的硬件风格写法。教学代码通常更适合用 `int` 保存中间 exponent。

### Normal 路径

最终只保留原有效尾数的 bit[16:14]：

```text
保留：bit[16:14]
round：bit13
sticky：bit[12:0]
```

RNE 后：

```cpp
fp8_mantissa = (mantissa >> 14) & 0x7;
```

步骤：

1. `>>14` 把 bit[16:14] 移到 bit[2:0]
2. `&0x7` 清除更高位

字段拼接：

```cpp
fp8s[i] = sign + (exp << 3) + fp8_mantissa;
```

等价位布局：

```text
sign             exp << 3          mantissa
10000000       0EEEE000           00000MMM
```

### Subnormal 路径

当中间 exponent 逻辑上为负时，它在 `uint8_t` 中表现为：

```text
-1 = 0xFF
-2 = 0xFE
-3 = 0xFD
-4 = 0xFC
-5 = 0xFB
```

小于 -4 的数直接清零。

对于 -1 到 -4：

```cpp
const uint8_t subnormal_shift = ~exp + 1;
```

得到绝对值 1 到 4，表示还需额外右移多少位。

subnormal 的 exponent field 固定为 0，因此最后只拼接：

```text
sign + subnormal mantissa
```

## 6. `fp24_to_float`

### 用法

```cpp
float value = fp24_to_float(fp24_bits);
```

用途是把 FP24 位模式解码成普通 `float`，便于测试和打印。

这个函数分三种情况：

```text
exp=0，mant=0       -> 正负零
exp=0，mant!=0      -> subnormal
exp=0x3F            -> Inf/NaN
其他                -> normal
```

它不是核心量化流程的一部分。

## 7. `fp8_e4m3_to_float`

### 用法

```cpp
float reconstructed = fp8_e4m3_to_float(fp8_bits, mx_scale);
```

它先解码 E4M3 元素值：

```text
normal = (-1)^S * (1 + mant/8) * 2^(exp-7)
```

再乘回 group scale：

```text
2^(mx_scale-31)
```

没有 `mx_scale` 时，无法恢复原始数量级。

## 8. `float_to_fp24`

### 用法

```cpp
uint32_t fp24_bits = float_to_fp24(6.0f);
```

它是测试数据生成器，不是量化主函数。

```cpp
float frac = std::frexp(val, &exp);
```

会得到：

```text
val = frac * 2^exp
0.5 <= frac < 1
```

FP24 normal 使用 `1.x` 表示，因此：

```cpp
biased = exp - 1 + 31;
```

`exp-1` 将 `0.5..1` 表示转换为 `1..2` 表示，加 31 是 FP24 exponent bias。

字段拼接：

```cpp
return sign | (biased << 17) | (mant & 0x1FFFF);
```

## 9. `main`

`main()` 展示完整调用流程：

```text
FP32 测试值
  -> float_to_fp24
  -> group_fp24_Mxfp8
  -> fp8_e4m3_to_float
  -> 计算绝对误差
```

编译运行：

```bash
c++ -std=c++17 -O2 lx_demo/fp24_MxFP8.cpp -o /tmp/fp24_mxfp8_demo
/tmp/fp24_mxfp8_demo
```

当前测试中：

```text
mx_scale = 29
scale factor = 0.25
100 -> 96
最大绝对误差 = 4
```

## 10. 一个完整位操作示例：FP24 的 `6.0`

程序生成：

```text
6.0 -> fp24 0x430000
```

转成二进制字段：

```text
0x430000 = 0 100001 10000000000000000
           S exponent fraction
```

提取 exponent：

```cpp
(0x430000 & 0x7E0000) >> 17
= 0x420000 >> 17
= 33
```

真实指数：

```text
33 - 31 = 2
```

提取 fraction：

```cpp
0x430000 & 0x1FFFF = 0x10000
```

有效尾数：

```text
1 + 0x10000 / 0x20000
= 1 + 0.5
= 1.5
```

数值：

```text
1.5 * 2^2 = 6
```

当 group scale 为 0.25 时：

```text
E4M3 内部元素值 = 6 / 0.25 = 24
```

24 的 E4M3 编码为 `0x5C`：

```text
0x5C = 0 1011 100
       S exp  mant
```

解码：

```text
(1 + 4/8) * 2^(11-7)
= 1.5 * 16
= 24
```

乘回 scale：

```text
24 * 0.25 = 6
```

## 11. 阅读这份代码时需要特别注意

1. `fp8` 和 `uint6` 都只是整数别名，不是原生浮点类型。
2. `uint6` 实际占 8 bit，不会自动限制高两位。
3. 中间 exponent 使用 `uint8_t`，代码依赖无符号下溢和补码位模式。
4. `exp & 0x80` 在这里被当作“逻辑 exponent 为负”的检测。
5. RNE 表达式本质上是在检查 round、sticky 和 retained LSB。
6. `fp8s` 必须和 `mx_scale` 配套保存。
7. FP24 subnormal 在核心量化函数中被直接清零。
8. 代码对 E4M3 最大值和异常值的处理具有实验性质，应与目标硬件格式规范再次核对。
