"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

"""
MicroXcaling 格式定义模块

本模块定义了量化/低精度计算中使用的各种数据格式，包括：
1. 舍入模式 (RoundingMode)
2. 元素格式 (ElemFormat) - 各种整数和浮点格式
3. 格式参数计算函数

该模块是 MicroXcaling 库的核心组件，用于支持混合精度计算、
量化操作和低精度浮点格式（如 FP8、FP6、FP4 等）。
"""

from enum import Enum, IntEnum

# FP32 格式的常量定义
FP32_EXPONENT_BIAS = 127  # FP32 指数偏置 (bias)，即 FP32 有偏指数的偏移量
                          # 有偏指数 = 无偏指数 + 127
                          # 无偏指数范围: [-126, 127]
                          # 有偏指数范围: [1, 254]（0 和 255 保留给特殊值）
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)  # FP32 最小正规数 (最小正规格化数)
                                                   # = 2^(-126) ≈ 1.1755e-38
                                                   # 这是 FP32 能表示的最小正规格化数
                                                   # 小于此值的数为非规格化数 (subnormals)

# Enum for rounding modes
class RoundingMode(IntEnum):
    """
    舍入模式枚举
    
    定义了量化操作中使用的舍入方法。这些模式决定了如何将浮点数
    舍入到目标格式（如整数或低精度浮点）。
    
    成员:
        nearest (0): 最近舍入 (round to nearest, ties away from zero)
        floor (1):   向下舍入 (round toward negative infinity)
        even (2):    向偶数舍入 (round to nearest, ties to even)
    """
    nearest = 0  # 最近舍入：四舍五入，0.5 时远离零舍入
                 # 例如: 2.5 → 3, -2.5 → -3
    floor = 1    # 向下舍入：向负无穷方向舍入（截断）
                 # 例如: 2.5 → 2, -2.5 → -3
    even = 2     # 向偶数舍入：四舍五入，0.5 时向最近的偶数舍入
                 # 例如: 2.5 → 2, 3.5 → 4（避免统计偏差）
                 # 这是 IEEE 754 默认的舍入模式

    @staticmethod
    def string_enums():
        """
        返回所有舍入模式名称的列表
        
        返回:
            list[str]: 包含所有舍入模式名称的列表，例如 ['nearest', 'floor', 'even']
        """
        return [s.name for s in list(RoundingMode)]

# Enum for scalar data formats
class ElemFormat(Enum):
    """
    元素格式枚举
    
    定义了支持的各种标量数据格式，包括整数格式和浮点格式。
    这些格式用于低精度计算、量化和混合精度操作。
    
    成员:
        int8 (1):       8位有符号整数
        int4 (2):       4位有符号整数
        int2 (3):       2位有符号整数
        fp8_e5m2 (4):   FP8 格式 (5位指数, 2位尾数) - IEEE 标准
        fp8_e4m3 (5):   FP8 格式 (4位指数, 3位尾数) - NVIDIA H100 支持
        fp6_e3m2 (6):   FP6 格式 (3位指数, 2位尾数)
        fp6_e2m3 (7):   FP6 格式 (2位指数, 3位尾数)
        fp4 (8):        FP4 格式 (2位指数, 1位尾数)
        fp4_e2m1 (8):   FP4 格式别名 (与 fp4 相同)
        float16 (9):    半精度浮点 (FP16) - IEEE 754
        fp16 (9):       FP16 别名
        bfloat16 (10):  脑浮点16 (BF16) - Google Brain 格式
        bf16 (10):      BF16 别名
        
    注意:
        - fp4 和 fp4_e2m1 共享相同的值，是同一格式的两种名称
        - float16 和 fp16 是同一格式
        - bfloat16 和 bf16 是同一格式
    """
    int8 = 1        # 8位有符号整数（符号-幅度表示）
    int4 = 2        # 4位有符号整数（符号-幅度表示）
    int2 = 3        # 2位有符号整数（符号-幅度表示）
    fp8_e5m2 = 4    # FP8 (5指数位, 2尾数位) - IEEE P3109 标准
                    # 动态范围大，精度低
                    # emax = 15, max_norm = 57344.0
    fp8_e4m3 = 5    # FP8 (4指数位, 3尾数位) - NVIDIA H100 GPU 支持
                    # 动态范围小，精度高
                    # emax = 8, max_norm = 448.0
                    # 注意：fp8_e4m3 不使用 NaN/Inf，而是用指数 0b1111 表示 NaN
    fp6_e3m2 = 6    # FP6 (3指数位, 2尾数位)
                    # 无 NaN/Inf，指数全1保留
                    # emax = 4, max_norm = 28.0
    fp6_e2m3 = 7    # FP6 (2指数位, 3尾数位)
                    # 无 NaN/Inf，指数全1保留
                    # emax = 2, max_norm = 7.5
    fp4 = 8         # FP4 (2指数位, 1尾数位)
                    # 无 NaN/Inf，指数全1保留
                    # emax = 2, max_norm = 6.0
    fp4_e2m1 = 8    # FP4 别名 (与 fp4 相同)
    float16 = 9     # 半精度浮点 (FP16) - IEEE 754 标准
                    # emax = 15, max_norm = 65504.0
    fp16 = 9        # FP16 别名
    bfloat16 = 10   # 脑浮点16 (BF16) - Google Brain 格式
                    # 与 FP32 相同的 8 位指数范围，但只有 7 位尾数
                    # emax = 127, max_norm = 3.3895e+38
    bf16 = 10       # BF16 别名

    @staticmethod
    def from_str(s):
        """
        从字符串转换为 ElemFormat 枚举
        
        参数:
            s (str): 格式名称字符串，不区分大小写
            
        返回:
            ElemFormat: 对应的枚举值
            
        异常:
            AssertionError: 如果输入为 None
            Exception: 如果字符串不对应任何已知格式
        """
        assert(s != None), "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)


def _get_min_norm(ebits):
    """
    计算浮点格式的最小正规数 (minimum normal number)
    
    对于给定的指数位数，计算该浮点格式能表示的最小正规格化数。
    这个值对应于指数为 emin (最小正规指数) 且尾数为 1.0 的情况。
    
    浮点格式的指数分布：
      - 指数位数为 ebits 时，有偏指数范围为 [1, 2^ebits - 2]
      - 最小正规指数 emin = 1 - bias
      - bias = 2^(ebits-1) - 1（标准 IEEE 格式）
      - 但本函数使用 emin = 2 - 2^(ebits-1) 的简化公式
    
    公式:
        emin = 2 - (2^(ebits-1))
        min_norm = 2^emin   (当 ebits > 0)
        min_norm = 0        (当 ebits = 0，即整数格式)
    
    参数:
        ebits (int): 指数位数
        
    返回:
        float: 最小正规数。如果 ebits=0（整数格式），返回 0。
        
    示例:
        _get_min_norm(5)  → FP8_E5M2:  emin = 2 - 16 = -14,  min_norm = 2^(-14) ≈ 6.10e-5
        _get_min_norm(4)  → FP8_E4M3:  emin = 2 - 8  = -6,   min_norm = 2^(-6)  ≈ 1.56e-2
        _get_min_norm(3)  → FP6_E3M2:  emin = 2 - 4  = -2,   min_norm = 2^(-2)  = 0.25
        _get_min_norm(2)  → FP4:       emin = 2 - 2  = 0,    min_norm = 2^0     = 1.0
        _get_min_norm(0)  → 整数格式:  min_norm = 0
    """
    # emin = 2 - 2^(ebits-1)
    # 推导：emin = 1 - bias = 1 - (2^(ebits-1) - 1) = 2 - 2^(ebits-1)
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin


def _get_max_norm(ebits, mbits):
    """
    计算浮点格式的最大正规数 (maximum normal number)
    
    仅适用于定义了 NaN/Inf 的浮点格式（即指数位数 >= 5）。
    对于不支持 NaN/Inf 的格式（如 FP4、FP6），使用不同的计算方法。
    
    公式:
        emax = 2^(ebits-1) - 1   (最大正规指数)
        max_norm = 2^emax * (2^(mbits-1) - 1) / 2^(mbits-2)
        
    其中 mbits 包括符号位和隐式位。
    
    推导说明:
        - 浮点数的值为: (-1)^sign × 2^(exp-bias) × (1 + mantissa/2^(mbits-1))
        - 最大正规数: 指数 = emax, 尾数全1
        - 尾数全1的值 = (2^(mbits-1) - 1) / 2^(mbits-2)
          （分母 2^(mbits-2) 是因为隐式1在最高位，尾数部分占 mbits-1 位）
        - 所以 max_norm = 2^emax × (2^(mbits-1) - 1) / 2^(mbits-2)
    
    参数:
        ebits (int): 指数位数
        mbits (int): 尾数位数（包括符号位和隐式位）
                    例如 FP8_E5M2: mbits=4 表示 [sign][implied 1][2 mantissa bits]
        
    返回:
        float: 最大正规数
        
    异常:
        AssertionError: 如果 ebits < 5（格式不支持 NaN/Inf）
        
    示例:
        _get_max_norm(5, 4)  → FP8_E5M2:  emax=15, max_norm=2^15×1.75=57344.0
        _get_max_norm(5, 12) → FP16:      emax=15, max_norm=2^15×1.9921875=65504.0
    """
    assert(ebits >= 5), "invalid for floats that don't define NaN"
    # emax = 2^(ebits-1) - 1
    # 例如 ebits=5: emax = 2^4 - 1 = 15
    # 例如 ebits=8: emax = 2^7 - 1 = 127
    emax = 0 if ebits==0 else 2**(ebits - 1) - 1
    # max_norm = 2^emax × (2^(mbits-1) - 1) / 2^(mbits-2)
    # (2^(mbits-1) - 1) / 2^(mbits-2) 是尾数全1时的值
    # 例如 mbits=4: (2^3 - 1) / 2^2 = 7/4 = 1.75
    return 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)


_FORMAT_CACHE = {}
def _get_format_params(fmt):
    """ Allowed formats:
        - intX:         2 <= X <= 32, assume sign-magnitude, 1.xxx representation
        - floatX/fpX:   16 <= X <= 28, assume top exp is used for NaN/Inf
        - bfloatX/bfX:  9 <= X <= 32
        - fp4,                  no NaN/Inf
        - fp6_e3m2/e2m3,        no NaN/Inf 
        - fp8_e4m3/e5m2,        e5m2 normal NaN/Inf, e4m3 special behavior

        Returns:
          ebits: exponent bits
          mbits: mantissa bits: includes sign and implicit bits
          emax: max normal exponent
          max_norm: max normal number
          min_norm: min normal number
    """
    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)

    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]

    # ============================================================
    # 整数格式：ebits=0，无指数，直接符号-幅度表示
    # ============================================================
    if fmt == ElemFormat.int8:
        ebits, mbits = 0, 8
        emax = 0
    elif fmt == ElemFormat.int4:
        ebits, mbits = 0, 4
        emax = 0
    elif fmt == ElemFormat.int2:
        ebits, mbits = 0, 2
        emax = 0

    # ============================================================
    # FP8_E5M2 (5指数位, 2尾数位)
    # IEEE P3109 标准格式，与 FP16 相同的指数范围
    # 总位数 = 1(符号) + 5(指数) + 2(尾数) = 8
    # mbits = 1(符号) + 1(隐式1) + 2(尾数) = 4
    # emax = 2^(5-1) - 1 = 15
    # max_norm = 2^15 * (2^3 - 1) / 2^2 = 32768 * 1.75 = 57344.0
    # min_norm = 2^(2 - 2^4) = 2^(-14) ≈ 6.10e-5
    # 有 NaN/Inf（指数全1时）
    # ============================================================
    elif fmt == ElemFormat.fp8_e5m2:
        ebits, mbits = 5, 4
        emax = 2**(ebits - 1) - 1

    # ============================================================
    # FP8_E4M3 (4指数位, 3尾数位)
    # NVIDIA H100 支持的格式，精度高但范围小
    # 总位数 = 1(符号) + 4(指数) + 3(尾数) = 8
    # mbits = 1(符号) + 1(隐式1) + 3(尾数) = 5
    # emax = 2^(4-1) = 8（注意：这里不是 -1，因为无 NaN/Inf）
    # max_norm = 2^8 * 1.75 = 448.0（特殊处理）
    # min_norm = 2^(2 - 2^3) = 2^(-6) ≈ 1.56e-2
    # 无 NaN/Inf，指数 0b1111 保留用于特殊用途
    # ============================================================
    elif fmt == ElemFormat.fp8_e4m3:
        ebits, mbits = 4, 5
        emax = 2**(ebits - 1)

    # ============================================================
    # FP6_E3M2 (3指数位, 2尾数位)
    # 总位数 = 1(符号) + 3(指数) + 2(尾数) = 6
    # mbits = 1(符号) + 1(隐式1) + 2(尾数) = 4
    # emax = 2^(3-1) = 4（无 NaN/Inf）
    # max_norm = 2^4 * (2^3 - 1) / 2^2 = 16 * 1.75 = 28.0
    # min_norm = 2^(2 - 2^2) = 2^(-2) = 0.25
    # ============================================================
    elif fmt == ElemFormat.fp6_e3m2:
        ebits, mbits = 3, 4
        emax = 2**(ebits - 1)

    # ============================================================
    # FP6_E2M3 (2指数位, 3尾数位)
    # 总位数 = 1(符号) + 2(指数) + 3(尾数) = 6
    # mbits = 1(符号) + 1(隐式1) + 3(尾数) = 5
    # emax = 2^(2-1) = 2（无 NaN/Inf）
    # max_norm = 2^2 * (2^4 - 1) / 2^3 = 4 * 1.875 = 7.5
    # min_norm = 2^(2 - 2^1) = 2^0 = 1.0
    # ============================================================
    elif fmt == ElemFormat.fp6_e2m3:
        ebits, mbits = 2, 5
        emax = 2**(ebits - 1)

    # ============================================================
    # FP4 (2指数位, 1尾数位)
    # 总位数 = 1(符号) + 2(指数) + 1(尾数) = 4
    # mbits = 1(符号) + 1(隐式1) + 1(尾数) = 3
    # emax = 2^(2-1) = 2（无 NaN/Inf）
    # max_norm = 2^2 * (2^2 - 1) / 2^1 = 4 * 1.5 = 6.0
    # min_norm = 2^(2 - 2^1) = 2^0 = 1.0
    # ============================================================
    elif fmt == ElemFormat.fp4:
        ebits, mbits = 2, 3
        emax = 2**(ebits - 1)

    # ============================================================
    # FP16 / float16 (5指数位, 10尾数位)
    # IEEE 754 半精度浮点标准
    # 总位数 = 1(符号) + 5(指数) + 10(尾数) = 16
    # mbits = 1(符号) + 1(隐式1) + 10(尾数) = 12
    # emax = 2^(5-1) - 1 = 15
    # max_norm = 2^15 * (2^11 - 1) / 2^10 = 32768 * 1.9990234375 = 65504.0
    # min_norm = 2^(2 - 2^4) = 2^(-14) ≈ 6.10e-5
    # 有 NaN/Inf（指数全1时）
    # ============================================================
    elif fmt == ElemFormat.float16:
        ebits, mbits = 5, 12
        emax = 2**(ebits - 1) - 1

    # ============================================================
    # BF16 / bfloat16 (8指数位, 7尾数位)
    # Google Brain 的脑浮点格式，与 FP32 相同的指数范围
    # 总位数 = 1(符号) + 8(指数) + 7(尾数) = 16
    # mbits = 1(符号) + 1(隐式1) + 7(尾数) = 9
    # emax = 2^(8-1) - 1 = 127（与 FP32 相同）
    # max_norm = 2^127 * (2^8 - 1) / 2^7 ≈ 3.3895e+38（与 FP32 相同量级）
    # min_norm = 2^(2 - 2^7) = 2^(-126) ≈ 1.1755e-38（与 FP32 相同）
    # 有 NaN/Inf（指数全1时）
    # ============================================================
    elif fmt == ElemFormat.bfloat16:
        ebits, mbits = 8, 9
        emax = 2**(ebits - 1) - 1

    else:
        raise Exception("Unknown element format %s" % fmt)
    
    # ============================================================
    # 计算 max_norm（最大正规数值）
    # 通用公式：max_norm = 2^emax × (2^(mbits-1) - 1) / 2^(mbits-2)
    #
    # 推导：
    #   浮点数值 = (-1)^sign × 2^(exp-bias) × (1 + mantissa/2^(mbits-1))
    #   其中 mantissa 是 mbits-1 位的尾数（不含隐式1）
    #   最大正规数：exp = emax, mantissa = 全1 = 2^(mbits-1) - 1
    #   尾数部分 = 1 + (2^(mbits-1) - 1) / 2^(mbits-1)
    #            = (2^(mbits-1) + 2^(mbits-1) - 1) / 2^(mbits-1)
    #            = (2^mbits - 1) / 2^(mbits-1)
    #            = (2^(mbits-1) - 1) / 2^(mbits-2)  （分子分母同除以2）
    #
    # 特殊处理 fp8_e4m3：
    #   fp8_e4m3 的 max_norm 不是用通用公式计算
    #   而是直接使用 2^emax * 1.75
    #   这是因为 fp8_e4m3 的指数编码特殊（emax=8 而非 7）
    #   且尾数全1时值为 1.75（与 fp8_e5m2 相同）
    # ============================================================
    if fmt != ElemFormat.fp8_e4m3:
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
    else:
        max_norm = 2**emax * 1.75  # FP8 has custom max_norm
                                   # fp8_e4m3: max_norm = 2^8 * 1.75 = 448.0

    # 计算 min_norm（最小正规数）
    min_norm = _get_min_norm(ebits)
    
    # 缓存结果，避免重复计算
    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)

    return ebits, mbits, emax, max_norm, min_norm
