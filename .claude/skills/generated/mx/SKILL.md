---
name: mx
description: "Skill for the Mx area of microxcaling. 162 symbols across 24 files."
---

# Mx

162 symbols | 24 files | Cohesion: 84%

## When to Use

- Working with code in `mx/`
- Understanding how sigmoid, tanh, relu work
- Modifying mx-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `mx/simd_ops.py` | simd_add, simd_sub, simd_mul, simd_div, simd_split (+22) |
| `mx/activations.py` | sigmoid, tanh, relu, relu6, leaky_relu (+16) |
| `mx/convolution.py` | __init__, apply_mx_specs, __init__, apply_mx_specs, forward (+7) |
| `mx/vector_ops.py` | vec_quantize, vec_add, vec_sub, vec_mul, vec_div (+7) |
| `mx/batchnorm.py` | batch_norm, __init__, forward, backward, forward (+4) |
| `mx/layernorm.py` | __init__, apply_mx_specs, __init__, apply_mx_specs, layer_norm (+3) |
| `mx/rnn.py` | __init__, _cell, _proj_input, _hx_slice, _hx_cat (+3) |
| `mx/specs.py` | apply_mx_specs, mx_assert_test, get_backwards_mx_specs, get_default_mx_specs, add_mx_args (+3) |
| `mx/linear.py` | linear, __init__, apply_mx_specs, forward, prequantize_weights (+2) |
| `mx/adaptive_avg_pooling.py` | adaptive_avg_pool2d, __init__, start_index, end_index, forward (+1) |

## Entry Points

Start here when exploring this area:

- **`sigmoid`** (Function) — `mx/activations.py:25`
- **`tanh`** (Function) — `mx/activations.py:34`
- **`relu`** (Function) — `mx/activations.py:43`
- **`relu6`** (Function) — `mx/activations.py:52`
- **`leaky_relu`** (Function) — `mx/activations.py:61`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `BatchNorm1d` | Class | `mx/batchnorm.py` | 218 |
| `BatchNorm2d` | Class | `mx/batchnorm.py` | 225 |
| `BatchNorm3d` | Class | `mx/batchnorm.py` | 232 |
| `sigmoid` | Function | `mx/activations.py` | 25 |
| `tanh` | Function | `mx/activations.py` | 34 |
| `relu` | Function | `mx/activations.py` | 43 |
| `relu6` | Function | `mx/activations.py` | 52 |
| `leaky_relu` | Function | `mx/activations.py` | 61 |
| `silu` | Function | `mx/activations.py` | 73 |
| `gelu` | Function | `mx/activations.py` | 82 |
| `adaptive_avg_pool2d` | Function | `mx/adaptive_avg_pooling.py` | 19 |
| `batch_norm` | Function | `mx/batchnorm.py` | 131 |
| `bmm` | Function | `mx/bmm.py` | 137 |
| `group_norm` | Function | `mx/groupnorm.py` | 80 |
| `layer_norm` | Function | `mx/layernorm.py` | 202 |
| `linear` | Function | `mx/linear.py` | 202 |
| `matmul` | Function | `mx/matmul.py` | 205 |
| `addmm_mx` | Function | `mx/mx_mapping.py` | 59 |
| `quantize_bfloat` | Function | `mx/quantize.py` | 13 |
| `simd_add` | Function | `mx/simd_ops.py` | 426 |

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
| Tests | 10 calls |

## How to Explore

1. `gitnexus_context({name: "sigmoid"})` — see callers and callees
2. `gitnexus_query({query: "mx"})` — find related execution flows
3. Read key files listed above for implementation details
