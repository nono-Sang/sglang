from __future__ import annotations

import torch

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from lite_attention import LiteAttention  # type: ignore

    _lite_attn_available = True
except Exception:
    LiteAttention = None  # type: ignore[assignment]
    _lite_attn_available = False


class LiteAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.LITE_ATTN

    @staticmethod
    def get_impl_cls() -> type["LiteAttentionImpl"]:
        return LiteAttentionImpl


class LiteAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if not _lite_attn_available:
            raise ImportError(
                "LiteAttention backend is not installed. "
                "Install LiteAttention (Hopper build) and ensure `lite_attention` "
                "is importable."
            )
        if causal:
            logger.warning(
                "LiteAttention backend does not implement causal masking; "
                "ensure it is used only for non-causal attention."
            )
        if not current_platform.is_cuda():
            raise ValueError("LiteAttention backend requires CUDA.")
        if not current_platform.is_hopper():
            logger.warning(
                "LiteAttention is optimized for Hopper (H100/H200); "
                "continuing on non-Hopper hardware."
            )

        self.softmax_scale = softmax_scale
        self.enable_skipping = envs.SGLANG_DIFFUSION_LITE_ATTENTION_ENABLE_SKIPPING
        self.skip_only_self_attn = (
            envs.SGLANG_DIFFUSION_LITE_ATTENTION_SKIP_ONLY_SELF_ATTN
        )
        self.threshold = envs.SGLANG_DIFFUSION_LITE_ATTENTION_THRESHOLD
        self.max_batch_size = envs.SGLANG_DIFFUSION_LITE_ATTENTION_MAX_BATCH_SIZE
        self.reverse_skip_list = envs.SGLANG_DIFFUSION_LITE_ATTENTION_REVERSE_SKIP_LIST
        self.use_int8 = envs.SGLANG_DIFFUSION_LITE_ATTENTION_USE_INT8

        self._attn_skip = LiteAttention(
            enable_skipping=True,
            threshold=self.threshold,
            max_batch_size=self.max_batch_size,
            reverse_skip_list=self.reverse_skip_list,
            use_int8=self.use_int8,
        )
        self._attn_noskip = LiteAttention(
            enable_skipping=False,
            threshold=self.threshold,
            max_batch_size=self.max_batch_size,
            reverse_skip_list=self.reverse_skip_list,
            use_int8=self.use_int8,
        )
        self._logged_fa3_fallback = False

    def _use_noskip(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[bool, str]:
        if not self.enable_skipping:
            return True, "skipping disabled by config"
        if self.skip_only_self_attn:
            if (
                query.shape[0] != key.shape[0]
                or query.shape[1] != key.shape[1]
                or key.shape[1] != value.shape[1]
            ):
                return True, "non-self-attention shapes"
        return False, ""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ):
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "LiteAttention backend only supports float16/bfloat16 inputs."
            )

        use_noskip, reason = self._use_noskip(query, key, value)
        if use_noskip and not self._logged_fa3_fallback:
            logger.info(
                "LiteAttention fallback to FA3 (no skipping) due to %s.",
                reason,
            )
            self._logged_fa3_fallback = True
        attn_impl = self._attn_noskip if use_noskip else self._attn_skip
        output = attn_impl(
            query,
            key,
            value,
            scale=self.softmax_scale,
            return_softmax_lse=return_softmax_lse,
        )
        return output
