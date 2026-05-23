---
name: cuda-demo
description: "Skill for the Cuda_demo area of microxcaling. 27 symbols across 3 files."
---

# Cuda_demo

27 symbols | 3 files | Cohesion: 89%

## When to Use

- Working with code in `cuda_demo/`
- Understanding how quantize_mx_func_cpp, quantize_tensor, quantize_elemwise_func_cpp work
- Modifying cuda_demo-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `cuda_demo/mx_demo_complete.cpp` | float_to_bits, bits_to_float, get_sign, get_biased_exponent, get_unbiased_exponent (+18) |
| `mx/cpp/funcs.cpp` | quantize_mx_func_cpp, quantize_tensor, quantize_elemwise_func_cpp |
| `mx/cpp/funcs.h` | quantize_mx_cpp |

## Entry Points

Start here when exploring this area:

- **`quantize_mx_func_cpp`** (Function) — `mx/cpp/funcs.cpp:22`
- **`quantize_tensor`** (Function) — `mx/cpp/funcs.cpp:100`
- **`quantize_elemwise_func_cpp`** (Function) — `mx/cpp/funcs.cpp:116`
- **`quantize_mx_cpp`** (Function) — `mx/cpp/funcs.h:61`
- **`main`** (Function) — `cuda_demo/mx_demo_complete.cpp:588`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `quantize_mx_func_cpp` | Function | `mx/cpp/funcs.cpp` | 22 |
| `quantize_tensor` | Function | `mx/cpp/funcs.cpp` | 100 |
| `quantize_elemwise_func_cpp` | Function | `mx/cpp/funcs.cpp` | 116 |
| `quantize_mx_cpp` | Function | `mx/cpp/funcs.h` | 61 |
| `main` | Function | `cuda_demo/mx_demo_complete.cpp` | 588 |
| `float_to_bits` | Function | `cuda_demo/mx_demo_complete.cpp` | 106 |
| `bits_to_float` | Function | `cuda_demo/mx_demo_complete.cpp` | 113 |
| `get_sign` | Function | `cuda_demo/mx_demo_complete.cpp` | 120 |
| `get_biased_exponent` | Function | `cuda_demo/mx_demo_complete.cpp` | 125 |
| `get_unbiased_exponent` | Function | `cuda_demo/mx_demo_complete.cpp` | 131 |
| `get_trailing_mantissa` | Function | `cuda_demo/mx_demo_complete.cpp` | 137 |
| `construct_float` | Function | `cuda_demo/mx_demo_complete.cpp` | 142 |
| `clamp_shared_exp` | Function | `cuda_demo/mx_demo_complete.cpp` | 263 |
| `mx_get_shared_scale` | Function | `cuda_demo/mx_demo_complete.cpp` | 276 |
| `shift_right_round_mantissa` | Function | `cuda_demo/mx_demo_complete.cpp` | 290 |
| `shift_left_mantissa` | Function | `cuda_demo/mx_demo_complete.cpp` | 336 |
| `quantize_elemwise` | Function | `cuda_demo/mx_demo_complete.cpp` | 349 |
| `encode_fp8_e4m3` | Function | `cuda_demo/mx_demo_complete.cpp` | 413 |
| `quantize_mx_elem` | Function | `cuda_demo/mx_demo_complete.cpp` | 460 |
| `print_block_summary` | Function | `cuda_demo/mx_demo_complete.cpp` | 533 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Quantize_mx_func_cpp → Float_to_bits` | intra_community | 7 |
| `Quantize_mx_func_cpp → Bits_to_float` | intra_community | 5 |
| `Quantize_mx_func_cpp → Shift_right_round_mantissa` | intra_community | 5 |
| `Quantize_elemwise_func_cpp → Float_to_bits` | intra_community | 5 |
| `Quantize_mx_func_cpp → Clamp_shared_exp` | intra_community | 4 |
| `Quantize_elemwise_func_cpp → Shift_right_round_mantissa` | intra_community | 4 |
| `Main → Min_norm_for_ebits` | intra_community | 3 |
| `Main → Float_to_bits` | cross_community | 3 |

## How to Explore

1. `gitnexus_context({name: "quantize_mx_func_cpp"})` — see callers and callees
2. `gitnexus_query({query: "cuda_demo"})` — find related execution flows
3. Read key files listed above for implementation details
