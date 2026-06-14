---
name: cpp
description: "Skill for the Cpp area of microxcaling. 10 symbols across 2 files."
---

# Cpp

10 symbols | 2 files | Cohesion: 100%

## When to Use

- Working with code in `mx/`
- Understanding how quantize_mx_func_cuda, quantize_mx_cuda, quantize_mx_by_tile_func_cuda work
- Modifying cpp-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `mx/cpp/funcs.cpp` | quantize_mx_func_cuda, quantize_mx_by_tile_func_cuda, quantize_elemwise_func_cuda, reduce_sum_inner_dim_cuda, reduce_max_inner_dim_cuda |
| `mx/cpp/funcs.h` | quantize_mx_cuda, quantize_mx_by_tile_cuda, quantize_elemwise_cuda, reduce_sum_inner_dim, reduce_max_inner_dim |

## Entry Points

Start here when exploring this area:

- **`quantize_mx_func_cuda`** (Function) — `mx/cpp/funcs.cpp:153`
- **`quantize_mx_cuda`** (Function) — `mx/cpp/funcs.h:15`
- **`quantize_mx_by_tile_func_cuda`** (Function) — `mx/cpp/funcs.cpp:176`
- **`quantize_mx_by_tile_cuda`** (Function) — `mx/cpp/funcs.h:27`
- **`quantize_elemwise_func_cuda`** (Function) — `mx/cpp/funcs.cpp:198`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `quantize_mx_func_cuda` | Function | `mx/cpp/funcs.cpp` | 153 |
| `quantize_mx_cuda` | Function | `mx/cpp/funcs.h` | 15 |
| `quantize_mx_by_tile_func_cuda` | Function | `mx/cpp/funcs.cpp` | 176 |
| `quantize_mx_by_tile_cuda` | Function | `mx/cpp/funcs.h` | 27 |
| `quantize_elemwise_func_cuda` | Function | `mx/cpp/funcs.cpp` | 198 |
| `quantize_elemwise_cuda` | Function | `mx/cpp/funcs.h` | 39 |
| `reduce_sum_inner_dim_cuda` | Function | `mx/cpp/funcs.cpp` | 218 |
| `reduce_sum_inner_dim` | Function | `mx/cpp/funcs.h` | 49 |
| `reduce_max_inner_dim_cuda` | Function | `mx/cpp/funcs.cpp` | 225 |
| `reduce_max_inner_dim` | Function | `mx/cpp/funcs.h` | 53 |

## How to Explore

1. `gitnexus_context({name: "quantize_mx_func_cuda"})` — see callers and callees
2. `gitnexus_query({query: "cpp"})` — find related execution flows
3. Read key files listed above for implementation details
