---
name: tests
description: "Skill for the Tests area of microxcaling. 90 symbols across 26 files."
---

# Tests

90 symbols | 26 files | Cohesion: 63%

## When to Use

- Working with code in `mx/`
- Understanding how get_s_e_m, float_to_bits, check_diff_quantize work
- Modifying tests-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `mx/tests/test_corners_elemwise.py` | test_fp16_max, test_custom_shift, test_float16_subnorms, test_bfloat16_limits, test_bfloat16_subnorms (+4) |
| `mx/tests/test_activations.py` | torch_gelu, torch_quick_gelu, test_gelu, test_activation, test_activation_class (+2) |
| `mx/tests/test_quantize_elemwise.py` | test_exponents, test_bfloat_random, test_empty_quantization, test_bfloat16_RNE, test_torch_bfloat16 (+2) |
| `mx/tests/common_lib.py` | get_s_e_m, float_to_bits, check_diff_quantize, torch_version_ge, check_diff (+1) |
| `mx/tests/test_fp8_e4m3_fix.py` | test_fp8_e4m3_fix_pytorch, test_fp8_e4m3_fix_cpp, test_fp8_e4m3_fix_innermost_cuda, test_fp8_e4m3_fix_by_tile_cuda, test_fp8_e4m3_fix_func_cuda (+1) |
| `mx/mx_ops.py` | _shared_exponents, _reshape_to_blocks, _reshape, _undo_reshape_to_blocks, _quantize_mx |
| `mx/tests/test_e5m0_scale.py` | test_e5m0_scale_pytorch, test_e5m0_scale_cpp, test_e5m0_scale_innermost_cuda, test_e5m0_scale_by_tile_cuda, test_e5m0_scale_func_cuda |
| `mx/tests/test_simd.py` | test_simd1, test_const, test_simd_reduce, test_simd_broadcast |
| `mx/tests/test_corners_mx.py` | test_mx_nans, test_mx_rounding, test_mx_hw_test |
| `mx/tests/test_quantize_mx.py` | test_empty_quantization, test_mx_encoding, test_mx_encoding_cpu_cuda |

## Entry Points

Start here when exploring this area:

- **`get_s_e_m`** (Function) — `mx/tests/common_lib.py:30`
- **`float_to_bits`** (Function) — `mx/tests/common_lib.py:31`
- **`check_diff_quantize`** (Function) — `mx/tests/common_lib.py:90`
- **`test_mx_nans`** (Function) — `mx/tests/test_corners_mx.py:37`
- **`test_mx_rounding`** (Function) — `mx/tests/test_corners_mx.py:60`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `get_s_e_m` | Function | `mx/tests/common_lib.py` | 30 |
| `float_to_bits` | Function | `mx/tests/common_lib.py` | 31 |
| `check_diff_quantize` | Function | `mx/tests/common_lib.py` | 90 |
| `test_mx_nans` | Function | `mx/tests/test_corners_mx.py` | 37 |
| `test_mx_rounding` | Function | `mx/tests/test_corners_mx.py` | 60 |
| `test_mx_hw_test` | Function | `mx/tests/test_corners_mx.py` | 82 |
| `test_e5m0_scale_pytorch` | Function | `mx/tests/test_e5m0_scale.py` | 28 |
| `test_e5m0_scale_cpp` | Function | `mx/tests/test_e5m0_scale.py` | 40 |
| `test_e5m0_scale_innermost_cuda` | Function | `mx/tests/test_e5m0_scale.py` | 52 |
| `test_e5m0_scale_by_tile_cuda` | Function | `mx/tests/test_e5m0_scale.py` | 65 |
| `test_e5m0_scale_func_cuda` | Function | `mx/tests/test_e5m0_scale.py` | 78 |
| `test_fp8_e4m3_fix_pytorch` | Function | `mx/tests/test_fp8_e4m3_fix.py` | 21 |
| `test_fp8_e4m3_fix_cpp` | Function | `mx/tests/test_fp8_e4m3_fix.py` | 32 |
| `test_fp8_e4m3_fix_innermost_cuda` | Function | `mx/tests/test_fp8_e4m3_fix.py` | 44 |
| `test_fp8_e4m3_fix_by_tile_cuda` | Function | `mx/tests/test_fp8_e4m3_fix.py` | 57 |
| `test_fp8_e4m3_fix_func_cuda` | Function | `mx/tests/test_fp8_e4m3_fix.py` | 72 |
| `test_mxfp8_e4m3_round` | Function | `mx/tests/test_fp8_e4m3_fix.py` | 90 |
| `test_empty_quantization` | Function | `mx/tests/test_quantize_mx.py` | 37 |
| `test_mx_encoding` | Function | `mx/tests/test_quantize_mx.py` | 52 |
| `test_mx_encoding_cpu_cuda` | Function | `mx/tests/test_quantize_mx.py` | 80 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Vec_reduce_mean → String_enums` | cross_community | 6 |
| `Vec_reduce_mean → _get_min_norm` | cross_community | 6 |
| `Vec_reduce_mean → _safe_lshift` | cross_community | 6 |
| `Vec_reduce_mean → _round_mantissa` | cross_community | 6 |
| `Forward → String_enums` | cross_community | 5 |
| `Forward → _get_min_norm` | cross_community | 5 |
| `Forward → _safe_lshift` | cross_community | 5 |
| `Forward → _round_mantissa` | cross_community | 5 |
| `Forward → String_enums` | cross_community | 5 |
| `Forward → _get_min_norm` | cross_community | 5 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Mx | 9 calls |

## How to Explore

1. `gitnexus_context({name: "get_s_e_m"})` — see callers and callees
2. `gitnexus_query({query: "tests"})` — find related execution flows
3. Read key files listed above for implementation details
