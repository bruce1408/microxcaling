MXFP6 自动量化示例 - 使用 microxcaling 库

这个示例展示了如何使用 microxcaling 库自动将 PyTorch 模型转换为 MXFP6 量化格式。
MX (Microscaling) 是一种高效的量化技术，可以在保持模型精度的同时显著减少内存占用和计算开销。

主要功能:
1. 定义一个残差多层感知机 (Residual MLP) 模型
2. 配置 MXFP6 量化参数
3. 自动注入 MX 量化操作到 PyTorch 模块
4. 运行量化后的模型

作者: microxcaling 团队

================================================================================
代码执行说明:
================================================================================

1. 环境要求:
   - Python 3.8+
   - PyTorch 1.12+
   - CUDA 11.0+ (如果使用 GPU)
   - microxcaling 库已安装

2. 安装 microxcaling:
   ```bash
   pip install microxcaling
   ```
   或者从源码安装:
   ```bash
   git clone https://github.com/microsoft/microxcaling
   cd microxcaling
   pip install -e .
   ```

3. 运行示例:
   ```bash
   # 使用默认参数运行 (隐藏层维度128, GPU)
   python ffn_mx_auto.py
   
   # 指定隐藏层维度为256
   python ffn_mx_auto.py --hidden_size 256
   
   # 在CPU上运行
   python ffn_mx_auto.py --device cpu
   
   # 查看所有可用参数
   python ffn_mx_auto.py --help
   ```

4. 输出说明:
   - 程序会显示输入输出张量的形状
   - 显示 MX 量化配置参数
   - 显示运行设备信息
   - 如果一切正常，最后会显示 "DONE!"

5. 代码结构:
   - ResidualMLP 类: 定义残差 MLP 模型架构
   - MX 配置部分: 设置量化参数 (FP6_E3M2, 块大小32等)
   - 自动注入: 使用 mx_mapping.inject_pyt_ops() 替换 PyTorch 操作
   - 模型运行: 创建随机输入，运行量化模型

6. 注意事项:
   - mx_mapping.inject_pyt_ops() 是全局操作，会影响之后创建的所有相关模块
   - 量化配置中的 quantize_backprop=False 表示反向传播使用 FP32
   - 如果需要训练量化模型，建议设置 quantize_backprop=True 并进行量化感知训练

7. 扩展使用:
   - 修改 mx_specs 字典可以尝试不同的量化格式
   - 可以替换为其他模型架构进行测试
   - 可以添加性能测试和精度评估代码

================================================================================
