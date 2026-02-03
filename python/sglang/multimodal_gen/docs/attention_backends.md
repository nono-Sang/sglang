# Attention Backends

This document describes the attention backends available in sglang diffusion (`sglang.multimodal_gen`) and how to select them.

## Overview

Attention backends are defined by `AttentionBackendEnum` (`sglang.multimodal_gen.runtime.platforms.interface.AttentionBackendEnum`) and selected via the CLI flag `--attention-backend`.

Backend selection is performed by the shared attention layers (e.g. `LocalAttention` / `USPAttention` / `UlyssesAttention` in `sglang.multimodal_gen.runtime.layers.attention.layer`) and therefore applies to any model component using these layers (e.g. diffusion transformer / DiT and encoders).

- **CUDA**: prefers FlashAttention (FA3/FA4) when supported; otherwise falls back to PyTorch SDPA.
- **ROCm**: uses FlashAttention when available; otherwise falls back to PyTorch SDPA.
- **MPS**: always uses PyTorch SDPA.

## Backend options

The CLI accepts the lowercase names of `AttentionBackendEnum`. The table below lists the backends implemented by the built-in platforms. `fa3`/`fa4` are accepted as aliases for `fa`.

| CLI value | Enum value | Notes |
|---|---|---|
| `fa` / `fa3` / `fa4` | `FA` | FlashAttention. `fa3/fa4` are normalized to `fa` during argument parsing (`ServerArgs.__post_init__`). |
| `lite_attn` | `LITE_ATTN` | LiteAttention (FA3 wrapper with skip optimization). Requires `lite_attention` (Hopper build). |
| `torch_sdpa` | `TORCH_SDPA` | PyTorch `scaled_dot_product_attention`. |
| `sliding_tile_attn` | `SLIDING_TILE_ATTN` | Sliding Tile Attention (STA). Requires `st_attn` and a mask-strategy config file set via the `SGLANG_DIFFUSION_ATTENTION_CONFIG` environment variable. |
| `sage_attn` | `SAGE_ATTN` | Requires `sageattention`. Upstream SageAttention CUDA extensions target SM80/SM86/SM89/SM90/SM120 (compute capability 8.0/8.6/8.9/9.0/12.0); see upstream `setup.py`: https://github.com/thu-ml/SageAttention/blob/main/setup.py. |
| `sage_attn_3` | `SAGE_ATTN_3` | Requires SageAttention3 installed per upstream instructions. |
| `video_sparse_attn` | `VIDEO_SPARSE_ATTN` | Requires `vsa`. |
| `vmoba_attn` | `VMOBA_ATTN` | Requires `kernel.attn.vmoba_attn.vmoba`. |
| `aiter` | `AITER` | Requires `aiter`. |

## Selection priority

The selection order in `runtime/layers/attention/selector.py` is:

1. `global_force_attn_backend(...)` / `global_force_attn_backend_context_manager(...)`
2. CLI `--attention-backend` (`ServerArgs.attention_backend`)
3. Auto selection (platform capability, dtype, and installed packages)

## Platform support matrix

| Backend | CUDA | ROCm | MPS | Notes |
|---|---:|---:|---:|---|
| `fa` | ✅ | ✅ | ❌ | CUDA requires SM80+ and fp16/bf16. FlashAttention is only used when the required runtime is installed; otherwise it falls back to `torch_sdpa`. |
| `lite_attn` | ✅ | ❌ | ❌ | CUDA-only (Hopper/H100 recommended). Requires `lite_attention`. |
| `torch_sdpa` | ✅ | ✅ | ✅ | Most compatible option across platforms. |
| `sliding_tile_attn` | ✅ | ❌ | ❌ | CUDA-only. Requires `st_attn` and `SGLANG_DIFFUSION_ATTENTION_CONFIG`. |
| `sage_attn` | ✅ | ❌ | ❌ | CUDA-only (optional dependency). |
| `sage_attn_3` | ✅ | ❌ | ❌ | CUDA-only (optional dependency). |
| `video_sparse_attn` | ✅ | ❌ | ❌ | CUDA-only. Requires `vsa`. |
| `vmoba_attn` | ✅ | ❌ | ❌ | CUDA-only. Requires `kernel.attn.vmoba_attn.vmoba`. |
| `aiter` | ✅ | ❌ | ❌ | Requires `aiter`. |

## Usage

### Select a backend via CLI

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend fa
```

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend torch_sdpa
```

### Using Sliding Tile Attention (STA)

```bash
export SGLANG_DIFFUSION_ATTENTION_CONFIG=/abs/path/to/mask_strategy.json

sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend sliding_tile_attn
```

### Using LiteAttention

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend lite_attn
```

Optional env vars for LiteAttention:

```bash
export SGLANG_DIFFUSION_LITE_ATTENTION_ENABLE_SKIPPING=true
export SGLANG_DIFFUSION_LITE_ATTENTION_SKIP_ONLY_SELF_ATTN=true
export SGLANG_DIFFUSION_LITE_ATTENTION_THRESHOLD=-10.0
export SGLANG_DIFFUSION_LITE_ATTENTION_MAX_BATCH_SIZE=2
export SGLANG_DIFFUSION_LITE_ATTENTION_REVERSE_SKIP_LIST=true
export SGLANG_DIFFUSION_LITE_ATTENTION_USE_INT8=false
```

### Notes for ROCm / MPS

- ROCm: use `--attention-backend torch_sdpa` or `fa` depending on what is available in your environment.
- MPS: the platform implementation always uses `torch_sdpa`.
