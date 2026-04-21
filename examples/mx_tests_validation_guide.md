# MX 量化库测试验证指南

## 概述

本文档详细说明 `microxcaling/mx/tests` 目录下的代码如何进行验证。MX 量化库包含完整的测试套件，用于验证各种量化格式、操作和硬件的正确性。

## 测试目录结构

```
microxcaling/mx/tests/
├── __init__.py              # 测试包初始化
├── common_lib.py            # 测试工具函数库
├── test_activations.py      # 激活函数测试
├── test_adaptive_avg_pooling.py
├── test_batchnorm.py        # 批归一化测试
├── test_bmm.py              # 批量矩阵乘法测试
├── test_conv.py             # 卷积测试
├── test_corners_elemwise.py # 逐元素操作边界测试
├── test_corners_mx.py       # MX 量化边界测试
├── test_e5m0_scale.py       # E5M0 格式缩放测试
├── test_finalize_mxfp.py    # MXFP 格式最终化测试
├── test_formats.py          # 量化格式测试
├── test_fp8_e4m3_fix.py     # FP8 E4M3 修复测试
├── test_gradcheck.py        # 梯度检查测试
├── test_groupnorm.py        # 组归一化测试
├── test_layernorm.py        # 层归一化测试
├── test_linear.py           # 线性层测试
├── test_matmul.py           # 矩阵乘法测试
├── test_mxfp_none.py        # 无量化测试
├── test_quantize_elemwise.py # 逐元素量化测试
├── test_quantize_mx.py      # MX 量化测试
├── test_reduce.py           # 归约操作测试
├── test_rnn.py              # RNN 测试
├── test_simd.py             # SIMD 操作测试
└── test_softmax.py          # Softmax 测试
```

## 测试框架

### 1. 测试工具库 (common_lib.py)

`common_lib.py` 提供了核心验证工具：

#### 主要函数：

1. **`check_diff(y1, y2, tol=0, ntol=0, handle_infs=False)`**
   - 比较两个张量的差异
   - `tol`: 允许的最大绝对误差
   - `ntol`: 允许的误差点数量
   - `handle_infs`: 是否处理无穷大值

2. **`check_diff_quantize(x, y1, y2, tol=0, ntol=0, handle_infs=False)`**
   - 专门用于量化操作的差异检查
   - 显示原始输入和量化输出的位级信息

3. **`all_encodings(_e, _m, encodes_infs=True, device="cpu")`**
   - 生成特定浮点格式的所有编码值
   - 用于测试量化格式的完整性

#### 验证原理：

```python
# 验证流程示例
def test_linear():
    # 1. 创建相同的输入数据
    m1 = torch.tensor(m_, requires_grad=True)
    m2 = torch.tensor(m_, requires_grad=True)
    
    # 2. 运行基准实现（PyTorch 原生）
    q1 = F.linear(m1, w1, b1)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    
    # 3. 运行 MX 量化实现
    q2 = linear(m2, w2, b2, mx_specs)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    
    # 4. 比较结果
    check_diff(q1, q2, tol=tolf)          # 前向传播结果
    check_diff(m1.grad, m2.grad, tol=tolb) # 输入梯度
    check_diff(w1.grad, w2.grad, tol=tolb) # 权重梯度
```

### 2. 测试配置

测试使用 pytest 框架，支持参数化测试：

```python
DEVICE__CUSTOM_CUDA = [
    ('cpu',  False),      # CPU 测试
    ('cuda', False),      # CUDA 测试（使用 PyTorch 实现）
    ('cuda', True)        # CUDA 测试（使用自定义 CUDA 内核）
]

@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
@pytest.mark.parametrize("shape", [(1, 32, 5, 7), (8, 13, 491, 511)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_linear(f1, f2, shape, use_bias, device, custom_cuda):
    # 测试逻辑
```

## 验证方法

### 1. 功能正确性验证

#### 前向传播验证：
- 比较 MX 量化操作与 PyTorch 原生操作的结果
- 支持多种量化格式：FP6_E3M2、FP8_E4M3、BF16 等
- 验证量化误差在可接受范围内

#### 反向传播验证：
- 比较梯度计算的正确性
- 支持量化感知训练（quantize_backprop）
- 验证梯度传播的数值稳定性

### 2. 硬件兼容性验证

#### 多设备测试：
- **CPU**: 验证纯 Python 实现
- **CUDA (PyTorch)**: 验证 PyTorch CUDA 实现
- **CUDA (Custom)**: 验证自定义 CUDA 内核

#### 量化配置测试：
```python
mx_specs = {
    'bfloat': 16,                    # BF16 量化
    'round': 'even',                 # 舍入模式
    'bfloat_subnorms': True,         # 支持次正规数
    'w_elem_format': 'fp8_e4m3',     # 权重量化格式
    'a_elem_format': 'fp8_e4m3',     # 激活量化格式
    'block_size': 32,                # 块大小
    'quantize_backprop': True,       # 反向传播量化
    'custom_cuda': custom_cuda       # 自定义 CUDA 内核
}
```

### 3. 边界条件验证

#### 特殊值测试：
- 零值、无穷大、NaN
- 次正规数（subnormals）
- 溢出和下溢

#### 极端形状测试：
- 小批量大小（batch=1）
- 大维度（>500）
- 非对齐内存访问

### 4. 性能验证

#### 预量化权重测试：
```python
# 标准量化
y1 = linear(x, w, bias=b, mx_specs=mx_specs)

# 预量化权重（减少运行时开销）
w = _quantize_bfloat(w, 16, round=mx_specs["round_weight"]).to(torch.bfloat16)
b = _quantize_bfloat(b, 16, round=mx_specs["round_weight"]).to(torch.bfloat16)
y2 = linear(x, w, bias=b, mx_specs=mx_specs, prequantized_weights=True)

# 验证结果一致性
check_diff(y1.to(torch.float32), y2.to(torch.float32), tol=0)
```

## 运行测试

### 1. 环境要求

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install pytest packaging numpy

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. 运行完整测试套件

```bash
# 进入项目目录
cd microxcaling

# 运行所有测试
python -m pytest mx/tests/ -v

# 运行特定测试模块
python -m pytest mx/tests/test_linear.py -v

# 运行特定测试函数
python -m pytest mx/tests/test_linear.py::test_linear -v

# 运行带标记的测试
python -m pytest mx/tests/ -m "not slow" -v
```

### 3. 测试输出示例

```
============================= test session starts ==============================
platform linux -- Python 3.13.9, pytest-9.0.3, pluggy-1.5.0
rootdir: /home/bruce/workspace/microxcaling
configfile: pytest.ini
plugins: anyio-4.11.0
collected 25 items

mx/tests/test_linear.py::test_linear[F.linear-linear-(1, 32, 5, 7)-False-cpu-False] PASSED
mx/tests/test_linear.py::test_linear[F.linear-linear-(1, 32, 5, 7)-False-cuda-False] PASSED
mx/tests/test_linear.py::test_linear[F.linear-linear-(1, 32, 5, 7)-False-cuda-True] PASSED
...
============================== 25 passed in 15.32s ==============================
```

### 4. 调试失败的测试

```bash
# 显示详细错误信息
python -m pytest mx/tests/test_linear.py -v --tb=short

# 在第一个失败时停止
python -m pytest mx/tests/ -x

# 显示打印输出
python -m pytest mx/tests/ -s

# 运行特定设备测试
python -m pytest mx/tests/test_linear.py -k "cpu" -v
```

## 测试覆盖范围

### 1. 量化格式测试
- **FP6_E3M2**: 6位浮点（3指数位 + 2尾数位）
- **FP8_E4M3**: 8位浮点（4指数位 + 3尾数位）
- **FP8_E5M2**: 8位浮点（5指数位 + 2尾数位）
- **BF16**: 脑浮点16位
- **自定义格式**: 支持任意 eXmY 格式

### 2. 操作类型测试
- **线性代数**: matmul, linear, bmm
- **归一化**: layernorm, batchnorm, groupnorm
- **激活函数**: relu, gelu, sigmoid, tanh, silu
- **卷积**: conv1d, conv2d, conv3d
- **池化**: adaptive_avg_pooling
- **RNN**: LSTM, GRU

### 3. 量化配置测试
- **块大小**: 16, 32, 64, 128
- **舍入模式**: even, stochastic, truncate
- **共享指数方法**: max, rms, mean
- **反向传播量化**: 启用/禁用

## 验证最佳实践

### 1. 新增测试的指导原则

```python
def test_new_feature():
    # 1. 设置随机种子确保可重复性
    torch.manual_seed(0xdeadbeef)
    np.random.seed(0xdeadbeef)
    
    # 2. 定义测试参数
    mx_specs = {
        'bfloat': 16,
        'quantize_backprop': True,
        'custom_cuda': custom_cuda
    }
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)
    
    # 3. 创建测试数据
    x = torch.randn(shape, device=device, requires_grad=True)
    
    # 4. 运行基准和MX实现
    y_baseline = torch_implementation(x)
    y_mx = mx_implementation(x, mx_specs=mx_specs)
    
    # 5. 比较结果（前向和反向）
    check_diff(y_baseline, y_mx, tol=1e-5)
    
    # 6. 验证梯度
    loss_baseline = y_baseline.sum()
    loss_baseline.backward()
    grad_baseline = x.grad.clone()
    
    x.grad = None
    loss_mx = y_mx.sum()
    loss_mx.backward()
    grad_mx = x.grad
    
    check_diff(grad_baseline, grad_mx, tol=1e-4)
```

### 2. 性能基准测试

```python
import time

def benchmark_mx_operation():
    # 预热
    for _ in range(10):
        _ = mx_operation(x, mx_specs)
    
    # 计时
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = mx_operation(x, mx_specs)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Average time: {(end - start) / 100 * 1000:.2f} ms")
```

### 3. 内存使用验证

```python
def check_memory_usage():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 运行操作
    y = mx_operation(x, mx_specs)
    
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory: {peak_memory / 1024**2:.2f} MB")
```

## 常见问题排查

### 1. 测试失败原因

#### 数值精度问题：
- 调整 `tol` 和 `ntol` 参数
- 检查量化配置是否合理
- 验证输入数据范围

#### CUDA 内核问题：
- 检查 CUDA 版本兼容性
- 验证自定义内核编译
- 检查设备内存限制

#### 梯度检查失败：
- 验证 `quantize_backprop` 配置
- 检查梯度裁剪和缩放
- 验证自定义 autograd Function

### 2. 调试技巧

```python
# 在测试中添加调试输出
def test_with_debug():
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)
    print(f"Finalized specs: {mx_specs}")
    
    # 检查中间结果
    y_mx = mx_operation(x, mx_specs=mx_specs)
    print(f"Output range: [{y_mx.min():.6f}, {y_mx.max():.6f}]")
    print(f"Output mean: {y_mx.mean():.6f}, std: {y_mx.std():.6f}")
    
    # 比较差异详情
    diff = torch.abs(y_baseline - y_mx)
    print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
```

## 结论

`microxcaling/mx/tests` 目录提供了全面的验证框架，确保 MX 量化库在各种配置和设备上的正确性。通过运行测试套件，可以：

1. **验证功能正确性**：确保量化操作与浮点基准一致
2. **确保数值稳定性**：验证梯度传播和边界条件
3. **测试硬件兼容性**：支持 CPU 和多种 CUDA 配置
4. **性能基准测试**：比较不同实现的效率

建议在修改 MX 量化库代码后运行相关测试，确保不会引入回归问题。对于新功能开发，应添加相应的测试用例以保持测试覆盖率。