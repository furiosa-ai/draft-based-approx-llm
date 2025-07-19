import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


def vertical_slash_sparse_attention_forward(
    module: Optional[LlamaAttention | Qwen2Attention],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    v_idx: Tensor,
    s_idx: Tensor,
) -> tuple[torch.Tensor, None]:
    from minference import vertical_slash_sparse_attention

    if module is not None and hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        v_idx = repeat_kv(v_idx[..., None], module.num_key_value_groups)[..., 0]
        s_idx = repeat_kv(s_idx[..., None], module.num_key_value_groups)[..., 0]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    attention_output = vertical_slash_sparse_attention(
        query, key, value, v_idx.clone(), s_idx.clone()
    )
    attention_output = attention_output.transpose(1, 2).contiguous()

    return attention_output, None


def compress_kv(
    key_states: Tensor,
    query_states: Tensor,
    value_states: Tensor,
    window_size: int,
    max_capacity_prompt: int,
    prefill_vertical_size: int,
    reduction_type: Literal["mean", "max"],
    pool_type: Literal["mean", "max"],
    kernel_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _, _, seq_len, head_dim = query_states.shape
    assert key_states.shape[-2] == seq_len
    assert max_capacity_prompt is None or seq_len > max_capacity_prompt

    num_key_value_heads = key_states.shape[1]
    num_key_value_groups = query_states.shape[1] // num_key_value_heads

    attention_scores = torch.matmul(
        query_states[..., -window_size:, :],
        repeat_kv(key_states, num_key_value_groups).transpose(2, 3),
    ) / math.sqrt(head_dim)

    attention_mask = torch.full(
        (window_size, window_size),
        torch.finfo(attention_scores.dtype).min,
        device=attention_scores.device,
    ).triu(1)

    attention_scores[:, :, -window_size:, -window_size:] += attention_mask
    attention_scores = attention_scores.softmax(dim=-1, dtype=torch.float32)
    attention_scores = attention_scores[:, :, -window_size:, :-window_size]

    attention_scores = attention_scores.view(
        attention_scores.shape[0],
        num_key_value_heads,
        num_key_value_groups * attention_scores.shape[2],
        attention_scores.shape[3],
    )
    if reduction_type == "mean":
        attention_scores = attention_scores.mean(dim=2)
    elif reduction_type == "max":
        attention_scores = attention_scores.max(dim=2).values
    else:
        raise ValueError(f"{reduction_type=} not supported.")

    if pool_type == "mean":
        attention_scores = F.avg_pool1d(
            attention_scores,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    elif pool_type == "max":
        attention_scores = F.max_pool1d(
            attention_scores,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    else:
        raise ValueError(f"{pool_type=} not supported.")

    indices = attention_scores.topk(
        min(
            max(max_capacity_prompt - window_size, prefill_vertical_size) if max_capacity_prompt is not None else prefill_vertical_size,
            attention_scores.shape[-1],
        ),
        dim=-1,
    ).indices

    if max_capacity_prompt is None:
        return key_states, value_states, indices

    topk_indices = indices[..., : max_capacity_prompt - window_size]
    gather_indices = topk_indices[..., None].expand(-1, -1, -1, head_dim)

    key_past_compress = key_states[:, :, :-window_size, :].gather(
        dim=2,
        index=gather_indices,
    )
    value_past_compress = value_states[:, :, :-window_size, :].gather(
        dim=2,
        index=gather_indices,
    )
    key_window = key_states[:, :, -window_size:, :]
    value_window = value_states[:, :, -window_size:, :]
    key_states = torch.cat([key_past_compress, key_window], dim=2)
    value_states = torch.cat([value_past_compress, value_window], dim=2)

    return key_states, value_states, indices
