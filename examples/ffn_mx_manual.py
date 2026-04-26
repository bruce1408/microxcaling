"""
这是一个使用 Microscaling (MX) 量化技术的 FFN (Feed-Forward Network) 手动示例脚本。
与 ffn_mx_auto.py 不同，此脚本手动导入并使用 MX 库的特定模块（如 Linear、LayerNorm、gelu 等），
并显式传递 mx_specs 参数，实现 FP6 等低精度量化。
支持 SIMD 操作（如 simd_split、simd_add）优化残差连接。
通过命令行参数配置 MX 规格，支持自定义 CUDA 等。
不修改原有代码逻辑，仅添加详细注释解释每个部分。
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse

# 从 MX 库导入量化优化后的 Linear 层，支持 MX 格式计算
from mx import Linear, LayerNorm

# 从 MX 库导入优化函数：gelu（激活）、simd_split（SIMD 分割）、simd_add（SIMD 加法）
from mx import gelu, simd_split, simd_add

# 从 MX 库导入工具函数：add_mx_args（添加 MX 参数到解析器）、get_mx_specs（从参数获取 MX 规格）
from mx import add_mx_args, get_mx_specs

"""
定义 ResidualMLP 类：
这是一个残差多层感知机 (Residual MLP) 模块，手动集成 MX 量化。
与 auto 版本不同：
- 显式使用 MX 的 Linear、LayerNorm、gelu 等模块，并传递 mx_specs。
- forward 中使用 simd_split 分离输入用于残差，simd_add 进行残差加法。
结构：
- LayerNorm：输入归一化 (MX 版本)
- Linear (h -> 4h)：维度扩展 (MX Linear)
- GELU 激活：非线性变换 (MX gelu)
- Linear (4h -> h)：维度压缩 (MX Linear)
- SIMD 残差连接：residual + MLP 输出
"""
class ResidualMLP(torch.nn.Module):
    def __init__(self, hidden_size, mx_specs):
        """
        初始化方法：
        - 保存 mx_specs 用于后续模块和函数调用。
        - 创建 MX LayerNorm：对 hidden_size 维输入进行归一化，支持 MX 量化。
        - dense_4h：MX Linear，将 hidden_size 扩展到 4 * hidden_size。
        - dense_h：MX Linear，将 4 * hidden_size 压缩回 hidden_size。
        """
        super(ResidualMLP, self).__init__()

        # 保存 MX 规格，用于 forward 中的函数调用
        self.mx_specs = mx_specs

        # MX LayerNorm 层：标准化输入特征，支持 MX 量化格式
        self.layernorm = LayerNorm(
            hidden_size,
            mx_specs=mx_specs  # 显式传递 MX 规格
        )

        # MX 第一个线性变换层：MLP 扩展阶段 (hidden_size -> 4 * hidden_size)
        self.dense_4h = Linear(
            hidden_size,
            4 * hidden_size,
            mx_specs=mx_specs  # 显式传递 MX 规格，实现权重量化
        )

        # MX 第二个线性变换层：MLP 压缩阶段 (4 * hidden_size -> hidden_size)
        self.dense_h = Linear(
            4 * hidden_size,
            hidden_size,
            mx_specs=mx_specs  # 显式传递 MX 规格
        )

    def forward(self, inputs):
        """
        前向传播方法（手动 MX 版本）：
        1. simd_split：使用 SIMD 优化将输入分割为 MLP 输入和残差。
        2. LayerNorm 归一化 MLP 输入。
        3. MLP：MX Linear1 -> MX gelu -> MX Linear2。
        4. simd_add：SIMD 优化残差连接（残差 + MLP 输出）。
        返回：经过 MX 量化残差 MLP 处理的输出张量。
        """
        # 第一步：SIMD 分割输入，将 inputs 分离为 MLP 处理部分和残差部分
        # simd_split 优化用于高效的张量分割，支持 MX 格式
        inputs, residual = simd_split(inputs)

        # 第二步：应用 MX LayerNorm 归一化 MLP 输入
        norm_outputs = self.layernorm(inputs)

        # MLP 核心计算块开始
        # MLP
        # 扩展投影：MX dense_4h，维度 hidden_size -> 4 * hidden_size
        proj_outputs = self.dense_4h(norm_outputs)
        # MX GELU 激活：非线性变换，显式传递 mx_specs
        proj_outputs = gelu(proj_outputs,
                            mx_specs=self.mx_specs)
        # 压缩投影：MX dense_h，维度 4 * hidden_size -> hidden_size
        mlp_outputs = self.dense_h(proj_outputs)

        # 残差连接：使用 SIMD 优化加法，残差 + MLP 输出
        # Residual Connection
        # simd_add 高效融合加法，支持 MX 量化激活
        outputs = simd_add(residual, mlp_outputs,
                           mx_specs=self.mx_specs)

        # 返回最终输出
        return outputs


# 脚本主入口：仅在直接运行此文件时执行 (python ffn_mx_manual.py)
if __name__ == '__main__':
    """
    主程序逻辑（手动 MX 版本）：
    1. 解析命令行参数：hidden_size, device，并通过 add_mx_args 添加 MX 相关参数（如 fp6_e3m2 等）。
    2. 从参数获取 mx_specs：get_mx_specs 处理 args 生成规格。
    3. 验证 mx_specs 有效。
    4. 生成随机输入，实例化手动 MX ResidualMLP，进行前向推理测试。
    5. 打印完成信息。
    此示例演示手动集成 MX，无需自动注入，但需显式使用 MX 模块。
    """
    # 添加基础命令行参数
    # Add config arguments
    parser = argparse.ArgumentParser()
    # --hidden_size：模型隐藏层维度，默认128
    parser.add_argument("--hidden_size", default=128)
    # --device：计算设备，默认'cuda'
    parser.add_argument("--device", default='cuda')
    # Add MX arguments：添加 MX 特定参数到解析器（如量化格式、块大小等）
    parser = add_mx_args(parser)
    # 解析所有参数
    args = parser.parse_args()

    # 从解析的参数生成 MX 规格字典
    # Process args to obtain mx_specs
    mx_specs = get_mx_specs(args)
    # 断言 mx_specs 有效（非 None）
    assert(mx_specs != None)

    # Run MLP：运行手动 MX MLP 测试
    # 生成随机输入：形状 (16, hidden_size)，batch_size=16
    x = np.random.randn(16, args.hidden_size)
    # 转换为 PyTorch 张量：FP32，移至设备
    x = torch.tensor(x, dtype=torch.float32, device=args.device)

    # 实例化手动 MX ResidualMLP，传递 hidden_size 和 mx_specs
    mlp = ResidualMLP(args.hidden_size, mx_specs)
    # 模型移至设备
    mlp.to(args.device)

    # 前向推理：使用手动 MX 操作进行量化计算
    y = mlp(x)

    # 测试完成
    print("DONE!")
