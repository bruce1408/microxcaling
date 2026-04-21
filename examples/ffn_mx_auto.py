import torch
import torch.nn.functional as F
import numpy as np
import argparse

# 从 microxcaling 库导入 MX 相关的模块
from mx import finalize_mx_specs
from mx import mx_mapping

class ResidualMLP(torch.nn.Module):
    """
    残差多层感知机 (Residual MLP) 模块
    
    这是 Transformer 架构中常用的前馈网络结构，包含:
    1. Layer Normalization (层归一化)
    2. 扩展线性层 (4倍隐藏层维度)
    3. GELU 激活函数
    4. 投影线性层 (回到原始隐藏层维度)
    5. 残差连接
    
    这种结构在 BERT、GPT 等现代 Transformer 模型中广泛使用。
    """
    def __init__(self, hidden_size):
        """
        初始化 ResidualMLP 模块
        
        Args:
            hidden_size: 隐藏层维度大小，也称为 d_model
        """
        super(ResidualMLP, self).__init__()

        # 层归一化: 对输入进行归一化处理，有助于稳定训练
        # 公式: output = (input - mean) / sqrt(var + eps) * gamma + beta
        self.layernorm = torch.nn.LayerNorm(
            hidden_size
        )

        # 第一个线性层: 将隐藏维度扩展到 4 倍
        # 这是 Transformer 中 FFN 的典型设计 (d_model -> 4*d_model)
        # 扩展维度有助于模型学习更复杂的特征表示
        self.dense_4h = torch.nn.Linear(
            hidden_size,
            4 * hidden_size
        )

        # 第二个线性层: 将扩展后的维度投影回原始隐藏维度
        # (4*d_model -> d_model)
        # 这个投影层将特征映射回原始空间，与残差连接结合
        self.dense_h = torch.nn.Linear(
            4 * hidden_size,
            hidden_size
        )

    def forward(self, inputs):
        """
        前向传播函数
        
        Args:
            inputs: 输入张量，形状为 (batch_size, hidden_size)
            
        Returns:
            outputs: 输出张量，形状与输入相同 (batch_size, hidden_size)
        """
        # 1. 层归一化: 对输入进行归一化，稳定训练过程
        norm_outputs = self.layernorm(inputs)

        # 2. MLP 部分 (前馈网络)
        # 2.1 扩展维度: hidden_size -> 4*hidden_size
        proj_outputs = self.dense_4h(norm_outputs)
        # 2.2 GELU 激活函数: 引入非线性
        # GELU(x) = x * Φ(x)，其中 Φ(x) 是标准正态分布的累积分布函数
        # GELU 比 ReLU 更平滑，在 Transformer 中表现更好
        proj_outputs = F.gelu(proj_outputs)
        # 2.3 投影回原始维度: 4*hidden_size -> hidden_size
        mlp_outputs = self.dense_h(proj_outputs)

        # 3. 残差连接 (Residual Connection)
        # 将输入直接加到 MLP 输出上，有助于梯度传播和训练深层网络
        # 这是 ResNet 和 Transformer 中的关键技术
        outputs = inputs + mlp_outputs

        return outputs


if __name__ == '__main__':
    # ============================================================
    # 命令行参数配置
    # ============================================================
    parser = argparse.ArgumentParser(
        description='MXFP6 自动量化示例 - 使用 microxcaling 库'
    )
    parser.add_argument(
        "--hidden_size",
        default=128,
        type=int,
        help='隐藏层维度大小 (默认: 128)'
    )
    parser.add_argument(
        "--device",
        default='cuda',
        help='运行设备，cuda 或 cpu (默认: cuda)'
    )
    args = parser.parse_args()

    # ============================================================
    # MX (Microscaling) 量化配置
    # ============================================================
    # MXFP6 格式说明:
    # - FP6_E3M2 表示 6 位浮点数: 1 位符号 + 3 位指数 + 2 位尾数
    # - 相比 FP32 (32位)，FP6 可以显著减少内存占用和计算量
    # - 使用 block_size=32 表示每 32 个元素共享一个指数
    
    mx_specs = {
        # w_elem_format: 权重(Weight)的元素格式
        # 'fp6_e3m2' = 1位符号 + 3位指数 + 2位尾数 = 6位浮点
        'w_elem_format': 'fp6_e3m2',
        
        # a_elem_format: 激活值(Activation)的元素格式
        # 同样使用 FP6_E3M2 格式
        'a_elem_format': 'fp6_e3m2',
        
        # block_size: 共享指数的块大小
        # MX 量化中，每个块内的元素共享同一个指数
        # 较小的 block_size 量化更精细，但开销更大
        'block_size': 32,
        
        # bfloat: 内部计算的 bfloat 格式位数
        # 16 表示使用 bfloat16 进行中间计算
        'bfloat': 16,
        
        # custom_cuda: 是否启用自定义 CUDA 内核
        # True 表示使用优化的 CUDA 量化内核以获得更好的性能
        'custom_cuda': True,
        
        # quantize_backprop: 是否在反向传播中进行量化
        # False 表示反向传播使用 FP32，不进行量化
        # 这在量化感知微调时很有用，可以保持梯度精度
        'quantize_backprop': False,
    }
    
    # finalize_mx_specs: 验证并完善 MX 配置
    # 它会设置默认值、验证配置的有效性，并处理配置间的依赖关系
    mx_specs = finalize_mx_specs(mx_specs)

    # ============================================================
    # 自动注入 MX 量化的核心操作
    # ============================================================
    # inject_pyt_ops 函数会自动替换 PyTorch 的部分模块和函数，
    # 使它们支持 MX 量化。
    #
    # 被替换的模块包括:
    # - torch.nn.Linear -> mx.linear.Linear (支持量化权重的线性层)
    # - torch.nn.LayerNorm -> mx.layernorm.LayerNorm (支持量化的层归一化)
    # - torch.nn.GELU -> mx.activations.GELU (支持量化的激活函数)
    #
    # 被替换的函数包括:
    # - torch.nn.functional.gelu -> mx.activations.gelu
    # - torch.matmul / torch.mm -> mx.matmul.matmul
    #
    # 注意: 这种注入是全局的，会影响之后创建的所有相关模块！
    mx_mapping.inject_pyt_ops(mx_specs)

    # ============================================================
    # 运行带有 MX 量化的 MLP
    # ============================================================
    
    # 1. 创建随机输入数据
    # 形状: (batch_size=16, hidden_size=args.hidden_size)
    x = np.random.randn(16, args.hidden_size)
    
    # 2. 转换为 PyTorch 张量并移动到指定设备
    # dtype=torch.float32: 输入数据保持 FP32 精度
    # device=args.device: 数据移动到 GPU (如果可用)
    x = torch.tensor(x, dtype=torch.float32, device=args.device)

    # 3. 创建 MLP 模型实例
    # 由于之前调用了 inject_pyt_ops，这里的 Linear、LayerNorm、GELU
    # 都会被自动替换为支持 MX 量化的版本
    mlp = ResidualMLP(args.hidden_size)
    
    # 将模型移动到指定设备 (GPU)
    mlp.to(args.device)

    # 4. 前向传播
    # 此时所有的线性运算、层归一化、激活函数都会自动应用 MXFP6 量化
    # 量化过程对用户是透明的，无需修改模型代码
    y = mlp(x)

    # 5. 输出结果信息
    print("=" * 50)
    print("MXFP6 自动量化示例运行完成!")
    print("=" * 50)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"隐藏层维度: {args.hidden_size}")
    print(f"扩展维度: {4 * args.hidden_size}")
    print(f"运行设备: {args.device}")
    print(f"权重格式: {mx_specs['w_elem_format']}")
    print(f"激活格式: {mx_specs['a_elem_format']}")
    print(f"块大小: {mx_specs['block_size']}")
    print("=" * 50)
    print("DONE!")

# ============================================================
# MX 量化技术原理详解
# ============================================================
"""
MX (Microscaling) 量化技术原理:

1. 什么是 MX 量化?
   MX 是一种高效的浮点数量化技术，通过减少浮点数的位数来降低内存占用和计算开销。
   传统的 FP32 使用 32 位(1符号位 + 8指数位 + 23尾数位)，而 MXFP6 只使用 6 位。

2. MXFP6 格式详解:
   - FP6_E3M2: 1位符号 + 3位指数 + 2位尾数 = 6位
   - 动态范围: 指数位决定数值范围，3位指数提供约 8 个不同的指数值
   - 精度: 2位尾数提供约 4 个不同的精度级别
   - 相比 FP32: 内存占用减少约 5.3 倍 (32/6 ≈ 5.33)

3. 共享指数 (Shared Exponent) 技术:
   - 核心思想: 一组数值共享同一个指数
   - block_size=32: 每 32 个元素共享一个指数
   - 优势: 减少指数存储开销，提高内存效率
   - 计算: 每个元素 = 符号位 * 尾数 * 2^(共享指数)

4. 量化过程:
   a) 前向传播:
      - 权重和激活值被量化为 MXFP6 格式
      - 线性运算在量化后的数值上进行
      - 结果可能被重新量化或保持高精度
   
   b) 反向传播 (quantize_backprop=False):
      - 梯度计算使用 FP32 精度
      - 权重更新使用 FP32 精度
      - 这有助于保持训练稳定性

5. 性能优势:
   - 内存带宽减少: 6位 vs 32位，减少约 81% 的内存传输
   - 计算加速: 更小的数据位宽可以在硬件上实现更高的计算吞吐量
   - 能耗降低: 减少的数据移动和计算降低能耗

6. 精度保持策略:
   - 块内归一化: 每个块内的数值被归一化到合适的范围
   - 动态范围适应: 共享指数根据块内最大绝对值动态调整
   - 舍入模式控制: 支持多种舍入模式(最近舍入、向下舍入等)

7. 应用场景:
   - 推理加速: 在部署时使用 MX 量化减少模型大小和加速推理
   - 训练加速: 在训练中使用 MX 量化减少内存占用，支持更大批次
   - 边缘设备: 在资源受限的设备上部署大型模型

8. 与其他量化技术的比较:
   - 与 INT8 量化相比: MXFP6 保持浮点特性，对异常值更鲁棒
   - 与 FP16 相比: MXFP6 进一步减少位宽，但需要共享指数技术
   - 与二值化相比: MXFP6 保持更多精度信息，适用性更广

注意: MX 量化是一种有损压缩技术，可能会引入一定的精度损失。
在实际应用中，通常需要通过量化感知训练(QAT)来微调模型以恢复精度。
"""
