# MX Quantization CUDA 实现伪代码与流程解析

## 文件依赖关系

```
mx.cu (CUDA host wrapper, 选择 kernel 启动策略)
  └─ mx.cuh (3 种 CUDA kernel 模板)
       ├─ shared_exp.cuh (共享指数/缩放系数计算)
       └─ quantize.cuh (元素级量化/反量化)
            └─ common.cuh (FP32 位操作 + 辅助工具)

funcs.cpp (PyTorch PYBIND11 绑定入口)
  └─ mx.cu (调用上述 CUDA 实现)
```

---

## 1. 入口与调度层 — `funcs.cpp` / `mx.cu`

```
// =====================================================
// Python 调用入口 (funcs.cpp)
// =====================================================
PYBIND11_MODULE(m, kwargs):
    m.def("quantize_mx_by_tile_func_cuda", quantize_mx_by_tile_func_cuda)

FUNCTION quantize_mx_by_tile_func_cuda(A, scale_bits, elem_ebits, elem_mbits,
                                        elem_max_norm, tile_size, axis,
                                        flush_fp32_subnorms, rounding_mode):
    CHECK_INPUT(A)
    return quantize_mx_by_tile_cuda(A, scale_bits, elem_ebits, elem_mbits,
                                     elem_max_norm, tile_size, axis,
                                     flush_fp32_subnorms, rounding_mode)


// =====================================================
// CUDA Host Wrapper — 选择最优 kernel (mx.cu)
// =====================================================
FUNCTION quantize_mx_by_tile_cuda(input, scale_bits, elem_ebits, elem_mbits,
                                   elem_max_norm, tile_size, axis,
                                   flush_fp32_subnorms, rounding_mode):

    // 1. 解析 tensor 维度
    ndim = input.dim()
    axis_size = input_sizes[axis]
    tsize     = tile_size > 0 ? tile_size : axis_size

    // 计算 pre/post axis 维度
    pre_axis_size  = product(input_sizes[0:axis])
    post_axis_size = product(input_sizes[axis+1:ndim])
    num_tiles      = ceil(axis_size / tsize)

    // 2. 选择量化策略

    // CASE A: 快速路径 — innermost axis + tile 对齐 + tile≤32 + tile 是 2 的幂
    IF axis == ndim-1
       AND axis_size % tsize == 0
       AND tsize ≤ WARP_SIZE(=32)
       AND is_power_of_two(tsize):

        total_size = input.numel()
        blocks  = get_blocks(total_size)    // ceil(total_size / MAX_THREADS)
        threads = get_threads(total_size)   // min(total_size, MAX_THREADS)

        LAUNCH quantize_mx_innermost_cuda_kernel<<<blocks, threads>>>(
                   input, ..., tsize, ..., output)

    // CASE B: 通用路径 — 任意 axis/tile_size
    ELSE:
        total_tiles = pre_axis_size × num_tiles × post_axis_size
        blocks  = get_blocks(total_tiles)    // 1 线程 / tile
        threads = get_threads(total_tiles)

        LAUNCH quantize_mx_by_tile_cuda_kernel<<<blocks, threads>>>(
                   input, ..., tsize, num_tiles, axis_size, post_axis_size, ..., output)

    RETURN output
```

---

## 2. Kernel 伪代码

### 2.1 快速路径 — `quantize_mx_innermost_cuda_kernel`

**适用条件**: 共享指数 axis 是 innermost axis，tile_size ≤ 32 且为 2 的幂，整除 axis_size

```
// =====================================================
// 每个线程处理 1 个元素，整个 warp 共享同一 tile
// 使用 __shfl_xor_sync 做 warp 级 allreduce 求 max
// =====================================================
KERNEL quantize_mx_innermost_cuda_kernel(in[], scale_bits, elem_ebits, elem_mbits,
                                          elem_max_norm, total_size, tile_size,
                                          flush_fp32_subnorms, rounding_mode,
                                          out[]):

    idx = blockDim.x × blockIdx.x + threadIdx.x
    IF idx ≥ total_size: RETURN

    // Step 1: 每个线程加载自己的元素
    elem = in[idx]
    abs_elem = |elem|

    // Step 2: Warp 内 AllReduce — 求 tile 内最大绝对值
    // 蝴蝶网络: 5 轮 (tile_size=32 → mask=16,8,4,2,1)
    // 所有 warp thread 最后持有相同的最大值
    FOR mask = tile_size/2 DOWN TO 1, HALF EACH ITERATION:
        tmp = __shfl_xor_sync(0xFFFFFFFF, abs_elem, mask)  // 从相距 mask 的 thread 取值
        abs_elem = max(abs_elem, tmp)                      // 保留较大者

    // Step 3: 提取共享指数 (block 内所有元素共享一个指数)
    shared_exp  = get_biased_exponent(abs_elem)   // IEEE 754 取出指数部分
    flush_tile  = (shared_exp == 0) AND flush_fp32_subnorms  // 次正规 flush 标志

    // Step 4: 计算共享缩放系数
    //   shared_exp 减去 elem_max_norm 的无偏指数
    //   夹紧到 scale_bits 范围内
    //   构造 2 的幂浮点数作为 scale
    scale = mx_get_shared_scale(shared_exp, scale_bits, elem_max_norm)

    // Step 5: MX 元素级量化 (含 除scale → 量化 → 乘scale)
    out[idx] = quantize_mx_elem(elem, scale, flush_tile,
                                 elem_ebits, elem_mbits,
                                 elem_max_norm, rounding_mode)
```

### 2.2 通用路径 — `quantize_mx_by_tile_cuda_kernel`

**适用条件**: 任意 axis/tile_size，1 线程负责整个 tile

```
// =====================================================
// 每个线程管理 1 个 tile（含多个元素）
// 两次遍历 tile: 第 1 轮求 max, 第 2 轮量化
// =====================================================
KERNEL quantize_mx_by_tile_cuda_kernel(in[], scale_bits, elem_ebits, elem_mbits,
                                        elem_max_norm, total_tiles, tile_size,
                                        num_tiles, axis_size, post_axis_size,
                                        flush_fp32_subnorms, rounding_mode,
                                        out[]):

    idx = blockDim.x × blockIdx.x + threadIdx.x
    IF idx ≥ total_tiles: RETURN

    // 计算 tile 在多维 tensor 中的位置
    post_axis_i   = idx % post_axis_size
    num_tiles_i   = (idx / post_axis_size) % num_tiles
    pre_axis_i    = idx / (num_tiles × post_axis_size)

    // 处理不完整 tile（边界情况）
    adjusted_tsize = (num_tiles_i+1) × tile_size > axis_size
                     ? axis_size % tile_size
                     : tile_size

    // ---- 第 1 轮: 遍历 tile 求最大绝对值 ----
    abs_max = 0
    FOR i = 0 TO adjusted_tsize-1:
        elem_i = pre_axis_i × axis_size × post_axis_size
               + (num_tiles_i × tile_size + i) × post_axis_size
               + post_axis_i
        abs_max = max(abs_max, |in[elem_i]|)

    // ---- 计算共享指数与缩放系数 (同快速路径) ----
    shared_exp = get_biased_exponent(abs_max)
    flush_tile = (shared_exp == 0) AND flush_fp32_subnorms
    scale      = mx_get_shared_scale(shared_exp, scale_bits, elem_max_norm)

    // ---- 第 2 轮: 再次遍历 tile, 逐元素量化 ----
    FOR i = 0 TO adjusted_tsize-1:
        elem_i = 同上索引公式
        out[elem_i] = quantize_mx_elem(in[elem_i], scale, flush_tile,
                                        elem_ebits, elem_mbits,
                                        elem_max_norm, rounding_mode)
```

### 2.3 预计算 max 路径 — `quantize_mx_cuda_kernel`

**适用条件**: 用户已通过 `reduce_max_inner_dim` 算好每个 block 的 max_value

```
// =====================================================
// 1 线程 / 元素, max_values 由外部 kernel 预先计算
// =====================================================
KERNEL quantize_mx_cuda_kernel(in[], scale_bits, elem_ebits, elem_mbits,
                                elem_max_norm, max_values[],
                                total_size, axis_size, post_axis_size,
                                flush_fp32_subnorms, rounding_mode,
                                out[]):

    idx = blockDim.x × blockIdx.x + threadIdx.x
    IF idx ≥ total_size: RETURN

    // 根据 idx 反算 block 索引 (哪个 tile 的 max 适用)
    post_axis_i = idx % post_axis_size
    pre_axis_i  = idx / (post_axis_size × axis_size)

    // 查找预计算的 max 值
    max_idx     = pre_axis_i × post_axis_size + post_axis_i
    shared_exp  = get_biased_exponent(max_values[max_idx])
    flush_tile  = (shared_exp == 0) AND flush_fp32_subnorms
    scale       = mx_get_shared_scale(shared_exp, scale_bits, elem_max_norm)

    out[idx] = quantize_mx_elem(in[idx], scale, flush_tile,
                                 elem_ebits, elem_mbits,
                                 elem_max_norm, rounding_mode)
```

---

## 3. 共享指数与缩放系数 — `shared_exp.cuh`

```
// =====================================================
// 核心概念:
//   一个 block/tile 内的所有元素共享一个指数
//   scale = 2^(shared_exp_adjusted) × mantissa
//   量化公式: quantized = round(input/scale) × scale
// =====================================================

// ----- 3a: 将共享指数夹紧到 E{ebits}M0 格式 -----
FUNCTION clamp_shared_exp(shared_exp, ebits):

    // E{ebits}M0 的最大可表示无偏指数
    emax = ebits ≠ 0 ? 2^(ebits-1) - 1 : 255

    // 转无偏指数并夹紧
    unbiased = shared_exp - 127
    IF unbiased > emax:    shared_exp = 255        // NaN (指数全 1)
    IF unbiased < -emax:   shared_exp = 127 - emax  // 最小可表示值

    RETURN shared_exp


// ----- 3b: 计算共享缩放系数 -----
FUNCTION mx_get_shared_scale(shared_exp, scale_bits, elem_max_norm):

    // Step 1: 获取 elem_max_norm 的无偏指数
    // 例如: FP8_E4M3 的 max_norm=448.0 → emax = 8
    emax = get_unbiased_exponent(elem_max_norm)

    // Step 2: 偏移 shared_exp
    // 目标是 max(input/scale) ≈ max_norm
    // 即 scale = 2^(exp_of_block_max - exp_of_max_norm)
    IF shared_exp ≠ 255:      // 保留 NaN
        shared_exp = shared_exp - emax

    // Step 3: 夹紧到 scale 格式范围
    shared_exp = clamp_shared_exp(shared_exp, scale_bits)

    // Step 4: 构造 scale 浮点数
    // 次正规或 NaN 时尾数 = 0.5; 正规时尾数 = 0
    mantissa = (shared_exp==0 OR shared_exp==255) ? 0.5 : 0

    RETURN construct_float(sign=0, shared_exp, mantissa)
    // 结果: 一个 2 的幂浮点数
```

---

## 4. 元素级量化 — `quantize.cuh`

```
// =====================================================
// 将任意 FP32 量化到指定格式 (bits 个有符号位)
// 支持: FP8_E5M2, FP8_E4M3, FP6, FP4, INT8, ...
// =====================================================
FUNCTION quantize_elemwise(input_fp32, total_bits, exp_bits,
                            max_norm, rounding_mode,
                            saturate_normals, allow_denorm):

    IF input_fp32 == 0: RETURN 0

    // Step 1: 分解 FP32 为 3 部分
    f32_union  = reinterpret(input_fp32, uint32)
    sign       = f32_union >> 31                      // 符号位 (1 bit)
    biased_exp = (f32_union & 0x7F800000) >> 23       // 指数 (8 bits)
    tmant      = f32_union & 0x007FFFFF               // 尾数 (23 bits)

    // Step 2: 计算目标格式的参数
    mbits   = total_bits - 1              // 去除符号位的比特数
    is_int  = (exp_bits == 0)             // 整数格式?
    new_bias = is_int ? 1 : 2^(exp_bits-1) - 1   // 目标格式的指数偏置

    // 将 FP32 指数映射到目标格式指数
    new_biased_exp = biased_exp - 127 + new_bias

    // Step 3: 处理次正规
    // 若 new_biased_exp ≤ 0, 需要额外右移尾数
    exp_diff = max(1 - new_biased_exp, 0)   // 额外移位数
    exp_diff = min(exp_diff, 24)            // 上限 24 (尾数总位数)
    is_subnorm_input = (biased_exp == 0)    // 原始值本身是次正规?

    // Step 4: 尾数截断与舍入
    (tmant, biased_exp) = shift_right_round_mantissa(
        tmant, is_subnorm_input, mbits, exp_diff, rounding_mode, allow_overflow=True)

    // 若舍入后为 0, 直接返回
    IF tmant == 0: RETURN 0

    // Step 5: 尾数左移恢复 + overflow 处理
    (tmant, overflow) = shift_left_mantissa(tmant, is_subnorm, mbits, exp_diff)
    IF overflow:
        biased_exp = biased_exp + 1   // 舍入导致进位, 指数 +1

    // Step 6: 重构 FP32
    output = construct_float(sign, biased_exp, tmant)

    // Step 7: 溢出/饱和处理
    IF |output| > max_norm:
        IF is_int OR saturate_normals:
            output = ±max_norm   // 饱和
        ELSE:
            output = ±Inf        // 上溢出到无穷

    RETURN output


// ----- 4a: 尾数右移 + 舍入 (核心) -----
FUNCTION shift_right_round_mantissa(inout_mantissa, is_subnorm, mbits,
                                     exp_diff, rounding_mode, allow_overflow):

    // 扩展为完整尾数 (含 implied 1)
    mantissa = is_subnorm ? inout_mantissa : inout_mantissa + 2^23
    fp32_sig_bits = is_subnorm ? 23 : 24

    // 计算需要截断的总位数
    tbits = exp_diff + (fp32_sig_bits - mbits)

    // RNE (round-to-nearest-even) 逻辑:
    tie  = (被截断的位除最高位外全为 0)
    even = (保留位中最后一位是 0)

    // 右移 exp_diff 位 (次正规调整)
    mantissa = mantissa >> exp_diff

    // 右移到 (mbits+1) 位 (保留 1 个额外位用于舍入判断)
    mantissa = mantissa >> (fp32_sig_bits - mbits - 1)

    // 舍入: 除了 tie-and-even 情况外都 +1
    IF rounding_mode == rd_away OR
       (rounding_mode == rd_even AND NOT (tie AND even)):
        IF allow_overflow OR mantissa ≠ (2^(mbits+1) - 1):
            mantissa = mantissa + 1

    // 去掉额外位, 得到最终 mbits 位尾数
    mantissa = mantissa >> 1
    RETURN mantissa


// ----- 4b: 尾数左移恢复 -----
FUNCTION shift_left_mantissa(inout_mantissa, is_subnorm, mbits, exp_diff):

    fp32_sig_bits = is_subnorm ? 23 : 24
    mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff)

    // 检测 overflow: 舍入进位导致 mantissa ≥ 2^24
    overflow = (mantissa ≥ 2^24)
    IF overflow AND NOT is_subnorm:
        mantissa = mantissa >> 1   // 右移归一化, 由调用方负责指数 +1

    // 去掉 implied 1, 恢复 23-bit trailing mantissa
    mantissa = mantissa & (2^23 - 1)
    RETURN (mantissa, overflow)


// =====================================================
// MX 元素量化入口 (quantize_mx_elem)
// =====================================================
FUNCTION quantize_mx_elem(input, scale, flush_tile,
                           elem_ebits, elem_mbits, elem_max_norm,
                           rounding_mode):

    // 若 tile 被 flush (次正规 flush), 直接输出 0
    scaled_input = flush_tile ? 0 : input / scale

    // 量化缩放到目标格式
    scaled_output = quantize_elemwise(
        scaled_input,
        bits     = elem_mbits,   // 已包含符号位
        exp_bits = elem_ebits,
        max_norm = elem_max_norm,
        rounding_mode,
        saturate_normals = true,  // 溢出时饱和而非 Inf
        allow_denorm = true)

    // 反量化: 乘回 scale
    RETURN scaled_output × scale
```

---

## 5. 辅助工具 — `common.cuh`

```
// ----- FP32 位操作 (union 实现零开销转换) -----
UNION FloatUint:
    uint32 i
    float  f

FUNCTION get_biased_exponent(f):
    union = reinterpret(f, FloatUint)
    RETURN (union.i & 0x7F800000) >> 23

FUNCTION get_sign(f):
    union = reinterpret(f, FloatUint)
    RETURN union.i >> 31

FUNCTION get_trailing_mantissa(f):
    union = reinterpret(f, FloatUint)
    RETURN union.i & 0x007FFFFF

FUNCTION get_unbiased_exponent(f):
    biased = get_biased_exponent(f)
    IF biased == 0:                // 次正规
        RETURN 1 - 127             // = -126
    ELSE:
        RETURN biased - 127

FUNCTION construct_float(sign, biased_exp, trailing_mantissa):
    union = FloatUint()
    union.i = trailing_mantissa
            | (biased_exp << 23)
            | (sign << 31)
    RETURN union.f
```

```
// ----- GPU 线程/块辅助 -----
FUNCTION get_blocks(size, max_threads=1024):
    // 返回需要的 block 数量
    RETURN ceil(size / max_threads)

FUNCTION get_threads(size, max_threads=1024):
    // 返回每个 block 的线程数
    RETURN min(size, max_threads)
```

---

## 6. 规约核 — `reduce.cuh` (用于预计算 max_values)

```
// =====================================================
// 沿 innermost dim 的 max 规约
// reduce_max_inner_dim → reduce_max_kernel
// =====================================================
KERNEL reduce_max_kernel(in[], total_size, inner_dim_size, out[]):

    idx = blockDim.x × blockIdx.x + threadIdx.x
    IF idx ≥ total_size: RETURN

    wid  = threadIdx.x / WARP_SIZE     // 哪个 warp
    lane = threadIdx.x % WARP_SIZE     // warp 内第几个线程

    // ---- 第一层: warp 级规约 (32 个线程 → 1 个值) ----
    val = in[idx]
    FOR offset = WARP_SIZE/2 DOWN TO 1:
        tmp = __shfl_down_sync(0xFFFFFFFF, val, offset)
        val = max(val, tmp)

    // 每个 warp 的 thread 0 写入 shared memory
    IF lane == 0:
        shared[wid] = val
    __syncthreads()

    // ---- 第二层: block 级规约 ----
    rows_per_block  = blockDim.x / inner_dim_size
    warps_per_block = blockDim.x / WARP_SIZE
    warps_per_row   = inner_dim_size / WARP_SIZE

    // warp 0 从 shared memory 读取其他 warp 的结果
    IF wid == 0 AND lane < warps_per_block:
        val = shared[lane]    // 32 个 warp 的部分结果
    // 再次 warp 规约
    FOR offset = WARP_SIZE/2 DOWN TO 1:
        tmp = __shfl_down_sync(0xFFFFFFFF, val, offset)
        val = max(val, tmp)

    // ---- 原子写入输出 ----
    IF lane == 0:
        IF wid < rows_per_block:
            out[blockIdx.x × rows_per_block + wid] = val
        ELIF threadIdx.x == 0:
            // 每个 block 只有 1 行时
            adj = offset / inner_dim_size
            atomicMax(out + adj, val)    // 浮点用 __float_as_int 转换
```

---

## 7. 32 元素量化完整流程示例

```
输入: [3.14, -2.71, 1.0, -0.5, 0.0, 0.001, ..., 500.0, ..., -256.0]
      共 32 个 FP32, tile_size=32, axis=0 (innermost)
      目标格式: FP8_E4M3 (elem_max_norm=448.0)

┌─────────────────────────────────────────────────────────────────┐
│  CUDA Kernel Launch: <<<1, 32>>>                               │
│  1 block, 32 threads (1 warp), SM 调度 32 个 core 并行执行     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  Thread 0               Thread 19             Thread 31
  elem=3.14              elem=500.0            elem=-256.0
  abs=3.14               abs=500.0             abs=256.0
        │                     │                     │
        └──────────┬──────────┴──────────┬──────────┘
                   ▼                     ▼
          ┌─────────────────────────────────────┐
          │  Warp AllReduce (__shfl_xor_sync)    │
          │  mask=16 → 8 → 4 → 2 → 1            │
          │  所有线程最终 abs_elem = 500.0       │
          └─────────────────────────────────────┘
                   │
                   ▼
          ┌─────────────────────────────────────┐
          │  shared_exp = 135 (500.0 的指数)    │
          │  flush_tile = false                 │
          │  scale = 1.0 (emax=8, 135-8=127)    │
          └─────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
  每个线程独立:          每个线程独立:
  3.14/1.0 → round      500.0/1.0 → round
  → to FP8 grid         → to FP8 grid
  → 3.125               → 448.0 (饱和!)
         │                    │
         └──────────┬─────────┘
                    ▼
           ┌────────────────┐
           │ Store result[] │  32 个并行写回 global memory
           └────────────────┘

最终输出:
  [3.125, -3.125, 1.0, -0.5, 0.0, ~0.000992, ..., 448.0, ..., -256.0]
    ↑饱和 ↑     ↑准确     ↑准确  ↑准确  ↑微小误差      ↑饱和
```

---

## 8. 关键优化总结


| 技术                         | 目的                           | 实现                                                   |
| -------------------------- | ---------------------------- | ---------------------------------------------------- |
| `**__shfl_xor_sync` 蝴蝶网络** | Warp 内零开销 allreduce 求 max    | 5 轮寄存器交换 (tile=32)                                   |
| **预计算 max_values**         | 分离 max 求值 + 量化，提高 occupancy  | `reduce_max_inner_dim` kernel                        |
| **1 线程 / tile**            | 通用路径，任意 tile_size/axis       | `quantize_mx_by_tile_cuda_kernel`                    |
| **尾数 bitwise 操作**          | 无需查表或 FPU，所有格式通用             | `shift_right_round_mantissa` + `shift_left_mantissa` |
| **rd_away / rd_even**      | 两种舍入模式                       | tie/even 位检测后决定是否 +1                                 |
| `**__host__ __device__`**  | 同一份代码同时支持 GPU 和 CPU fallback | `quantize_elemwise` / `mx_get_shared_scale` 等        |


## 9. 核心概念：三维分解 (内存布局)

代码把任意 N 维 tensor 在逻辑上分解为三个区域：

```
输入 shape: [D₀, D₁, ..., Dₙ₋₁]
指定共享指数 axis = a

pre_axis_size  = D₀ × D₁ × ... × Dₐ₋₁       // axis 之前的维度乘积
axis_size      = Dₐ                            // 共享指数所在维度
post_axis_size = Dₐ₊₁ × ... × Dₙ₋₁            // axis 之后的维度乘积
```

这三个量决定了**哪个元素和哪个元素共享同一个指数**：

> **共享规则**: 索引 `(pre_axis_i, post_axis_i)` 相同的所有元素共享一个指数
> 即固定 axis 之前和之后的所有坐标，沿 axis 方向的所有元素属于一个 block

**线性内存索引公式** (row-major):

```
linear_idx = pre_axis_i × (axis_size × post_axis_size)
           + k × post_axis_size          // k = 0..axis_size-1, 沿 axis 方向
           + post_axis_i
```

---

## 10. 具体例子可视化

### 例 1: 2D Tensor — `shape=[4, 12]`, `axis=1`, `tile_size=4`

```
pre_axis_size = 4       (dim 0)
axis_size     = 12      (dim 1)
post_axis_size = 1      (无后续维度)
num_tiles     = 12/4 = 3
```

**内存布局与共享指数分配**:

```
                  axis=1 (dim 1, 12 列)
               ├── tile 0 ──┤├── tile 1 ──┤├── tile 2 ──┤
               k=0  1  2  3  4  5  6  7  8  9 10 11
      ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
i=0   │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │  ← pre_axis_i=0, post_axis_i=0
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
i=1   │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │  ← pre_axis_i=1, post_axis_i=0
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
i=2   │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │  ← pre_axis_i=2, post_axis_i=0
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
i=3   │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │  ← pre_axis_i=3, post_axis_i=0
      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

- 同一行 (pre_axis_i 相同) 具有相同的 `post_axis_i=0`
- **同一行内每 4 个元素 (一个 tile) 共享一个指数**
- `total_tiles = 4 × 3 = 12`，kernel 启动 12 个线程，每个线程处理一行中的一个 tile

**线性内存** (row-major, FP32 每个 4 字节):

```
offset: 0   1   2   3 | 4   5   6   7 | 8   9   10  11 | 12  13  14  15 | ...
       [0,0][0,1][0,2][0,3] [0,4][0,5][0,6][0,7] [0,8][0,9][0,10][0,11]
                        [1,0][1,1][1,2][1,3] ...
       ├── tile 0 ──────┤ ├── tile 1 ──────┤ ├── tile 2 ────────┤
```

**关键观察**: 当 `post_axis_size=1` (axis=1, ndim=2) 时，tile 内的元素**在内存中是连续排列的**。

### 例 2: 3D Tensor — `shape=[3, 128, 256]`, `axis=1`, `tile_size=32`

```
pre_axis_size     = 3          (dim 0)
axis_size         = 128        (dim 1)
post_axis_size    = 256        (dim 2)
num_tiles         = 128/32 = 4
total_tiles       = 3 × 4 × 256 = 3072
```

**可视化**:

```
                       axis=1 (dim 1, 共享指数方向, 128 个元素)
             ├────── tile 0 (i1=0..31) ──────┤├── tile 1 ──┤...

post_axis (dim 2, 256) ──────┐
                             │
                  ┌──────────┬──────────┬──────────┬──────
                  │[0,0,0]   │[0,0,1]   │[0,0,2]   │...  ← pre_axis=0, i1=0
                  │[0,1,0]   │[0,1,1]   │[0,1,2]   │...  ← pre_axis=0, i1=1
                  │...       │...       │...       │
                  │[0,31,0]  │[0,31,1]  │[0,31,2]  │...  ← pre_axis=0, i1=31
                  ├──────────┼──────────┼──────────┼──────  ← tile 0 结束
                  │[0,32,0]  │[0,32,1]  │[0,32,2]  │...  ← tile 1 开始
                  └──────────┴──────────┴──────────┴──────
```

**线性内存** (row-major):

```
offset 0:      [0,0,0]
offset 1:      [0,0,1]
...
offset 255:    [0,0,255]    ← 完成 pre_axis=0, axis=0, 所有 post_axis
offset 256:    [0,1,0]      ← axis 方向走一步
offset 257:    [0,1,1]
...
offset 511:    [0,1,255]
...
offset 7936:   [0,31,0]     ← tile 0 的最后一行开始
...
offset 8191:   [0,31,255]   ← tile 0 结束
offset 8192:   [0,32,0]     ← tile 1 开始
```

**关键观察**: 同一个 tile 内，沿 axis 方向相邻的元素 `[0,k,0]` 和 `[0,k+1,0]` 在内存中相距 `post_axis_size=256` 个元素，即 **tile 内元素在内存中是步长为 `post_axis_size` 的非连续访问**。

---

## 11. 每个 kernel 的线程-数据映射

### Kernel A: `quantize_mx_innermost_cuda_kernel` (快速路径)

**适用**: `axis == ndim-1` (innermost dim)，即 `post_axis_size == 1`

```
例: shape=[4, 12], axis=1, tile_size=4

线程映射:
  block 0, warp 0:  thread 0..3  →  offset 0..3  → [0,0..3] (tile 0, 行 0)
                     thread 4..7  →  offset 4..7  → [0,4..7] (tile 1, 行 0)
                     thread 8..11 →  offset 8..11 → [0,8..11](tile 2, 行 0)
  block 0, warp 1:  thread 0..3  →  offset 12..15→ [1,0..3] (tile 0, 行 1)
                     ...

内存访问模式:
  warp 0 的 thread 0..3 读取 offset 0,1,2,3 → 连续 16 字节 → 合并访问 (coalesced) ✓
  warp 0 的 thread 4..7 读取 offset 4,5,6,7 → 也是连续 ✓
```

`__shfl_xor_sync` 在 warp 内做 allreduce，因为 `tile_size=4`，warp 内 4 个线程 (0-3) 一起归约求 tile 最大，后 4 个线程 (4-7) 做另一个 tile 的归约。

**内存优势**: `post_axis_size=1` → tile 内元素**连续排列** → GPU 全局内存**合并访问**

### Kernel B: `quantize_mx_by_tile_cuda_kernel` (通用路径)

**适用**: `axis` 非 innermost 或 tile_size > 32 等

```
例: shape=[3, 128, 256], axis=1, tile_size=32

线程映射: 1 线程 / tile
  total_tiles = 3 × 4 × 256 = 3072

  线程 0:  pre_axis_i=0, num_tiles_i=0, post_axis_i=0
           → 管理 [0, 0..31, 0] 共 32 个元素
  线程 1:  pre_axis_i=0, num_tiles_i=0, post_axis_i=1
           → 管理 [0, 0..31, 1] 共 32 个元素
  ...
  线程 255: pre_axis_i=0, num_tiles_i=0, post_axis_i=255
           → 管理 [0, 0..31, 255]
  线程 256: pre_axis_i=0, num_tiles_i=1, post_axis_i=0
           → 管理 [0, 32..63, 0]
```

**线程索引到 tile 的映射公式** (`mx.cuh:118-120`):

```cpp
post_axis_i   = offset % post_axis_size           // = offset % 256
num_tiles_i   = (offset / post_axis_size) % num_tiles  // = (offset/256) % 4
pre_axis_i    = offset / (num_tiles × post_axis_size)  // = offset / 1024
```

**tile 内元素的内存地址** (`mx.cuh:133-135`):

```cpp
in_i = pre_axis_i × axis_size × post_axis_size     // 行基址
     + (num_tiles_i × tile_size + i) × post_axis_size  // tile 内第 i 个元素沿 axis 的偏移
     + post_axis_i                                    // post_axis 偏移
```

对于 `[0, 0..31, 0]` (pre=0, tile=0, post=0):

```
i=0: 0 × 32768 + (0 × 32 + 0) × 256 + 0 = 0         → [0,0,0]
i=1: 0 × 32768 + (0 × 32 + 1) × 256 + 0 = 256       → [0,1,0]
i=2: 0 × 32768 + (0 × 32 + 2) × 256 + 0 = 512       → [0,2,0]
...
i=31: 0 × 32768 + (0 × 32 + 31) × 256 + 0 = 7936    → [0,31,0]
```

第 0 个元素到第 31 个元素，地址增量是 256 (post_axis_size)，所以**步长为 256 个元素**。这意味着每次加载 `i` 和 `i+1` 之间有 `256 × 4 = 1024 字节` 的跨度 — **非合并访问**。

### Kernel C: `quantize_mx_cuda_kernel` (预计算 max 路径)

```
例: shape=[3, 128, 256], axis=1

线程映射: 1 线程 / 元素
  total_size = 3 × 128 × 256 = 98304
  blocks = ceil(98304/1024) = 96
  threads = 1024

  线程 offset=0:     idx=0      → [0,0,0]
  线程 offset=256:   idx=256    → [0,1,0]
  线程 offset=32768: idx=32768  → [1,0,0]

  max_values 查找:
    post_axis_i = offset % 256
    pre_axis_i  = offset / (256 × 128)
    m_i = pre_axis_i × 256 + post_axis_i

    例如 offset=0:
      post_axis_i=0, pre_axis_i=0, m_i=0 → max_values[0,0] = tile [0,:,0] 的 max
    例如 offset=256:
      post_axis_i=0, pre_axis_i=0, m_i=0 → 同一个 tile! 相同 max
    例如 offset=1:
      post_axis_i=1, pre_axis_i=0, m_i=1 → tile [0,:,1] 的 max
```

每个线程独立查找对应的 max_value。同属一个 tile 的所有 `axis_size` 个元素 (此处 128 个) 映射到同一个 m_i。

---

## 12. max_values 内存布局

`quantize_mx_func_cuda` 路径需要在 Python 侧预计算 max_values:

```python
# mx_ops.py:247
max_values = A.abs().max(dim=axis, keepdim=True).values
```

对 `shape=[3, 128, 256]`, `axis=1`:

```
max_values shape: [3, 1, 256]   (keepdim=True, dim 1 被压缩为 size 1)

内存布局 (row-major):
  [0,0,0] [0,0,1] ... [0,0,255] [1,0,0] ... [1,0,255] [2,0,0] ... [2,0,255]
  ↑ tile [0,:,0] 的 max          ↑ tile [1,:,0] 的 max
```

kernel 中 `m_i = pre_axis_i × post_axis_size + post_axis_i`:

```
m_i=0:   pre=0, post=0 → max_values flat offset 0 → max_values[0,0,0] = tile [0,:,0] max
m_i=255: pre=0, post=255 → offset 255 → max_values[0,0,255] = tile [0,:,255] max
m_i=256: pre=1, post=0 → offset 256 → max_values[1,0,0] = tile [1,:,0] max
```

`keepdim=True` 产生的 size-1 维度在 flat 索引中不产生任何影响，kernel 的 flat 索引逻辑完全正确。

**形状关系总结**:

```
input shape:   [pre, axis, post]  (逻辑上的三维分解)
max_values shape (逻辑): [pre, 1, post]  即 [pre_axis_size, 1, post_axis_size]
max_values shape (flat): [pre_axis_size × post_axis_size]
```

---

## 13. 三种路径的内存模式对比


|                | 快速路径 (innermost)    | 通用路径 (by_tile)      | 预计算 max 路径        |
| -------------- | ------------------- | ------------------- | ----------------- |
| **线程/元素比**     | 1:1 (1 线程/元素)       | 1:tile_size         | 1:1               |
| **max 求值方式**   | warp allreduce 寄存器  | 线程内串行循环             | 预先 kernel 算好      |
| **tile 内内存访问** | **连续 (coalesced)**  | 步长 `post_axis_size` | 连续 (1 个元素)        |
| **额外内存**       | 无                   | 无                   | max_values tensor |
| **tile 大小限制**  | ≤ 32, 2 的幂          | 任意                  | 任意                |
| **适用场景**       | axis=-1, small tile | 任意 axis             | 需要复用 max_values 时 |


**快速路径在 PCIe 上的优势图解** (shape=[4, 12], tile=4):

```
warp 0 的一次内存事务:
  thread 0..3 请求 offset 0,1,2,3
  GPU 内存控制器: 一次 128 字节事务读取 cacheline → 包含所有 4 个 float ✓

通用路径:
  线程 0 请求 offset 0, 256, 512, 768  (步长 256)
  GPU 内存控制器: 需要 4 次独立 cacheline 读取 ✗
```

---

## 14. 完整数据流 (Python → CUDA)

```
Python: mx_ops.py:_quantize_mx
  │
  ├─ 路径 A: 单 axis + (block_size≤32 或 axis 非 innermost)
  │    │  A.contiguous() 确保连续
  │    │
  │    └─ quantize_mx_by_tile_func_cuda (funcs.cpp)
  │         │  传入原始 tensor, 不 reshape
  │         │
  │         └─ quantize_mx_by_tile_cuda (mx.cu)
  │              │  pre_axis_size, axis_size, post_axis_size, num_tiles 计算
  │              │
  │              ├─ 快速路径: quantize_mx_innermost_cuda_kernel (mx.cuh)
  │              │   1 线程/元素, warp allreduce, 连续内存
  │              │
  │              └─ 通用路径: quantize_mx_by_tile_cuda_kernel (mx.cuh)
  │                  1 线程/tile, 串行遍历, 步长 post_axis_size
  │
  └─ 路径 B: Python 侧 reshape + 预计算 max
       │  _reshape_to_blocks 添加 tile 维度 + padding
       │  max_values = A.abs().max(dim=axis, keepdim=True)
       │
       └─ quantize_mx_func_cuda (mx.cu)
            └─ quantize_mx_cuda_kernel (mx.cuh)
                1 线程/元素, 从 max_values 查共用指数
```

