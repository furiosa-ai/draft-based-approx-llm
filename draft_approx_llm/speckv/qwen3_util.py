from typing import Callable, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)

from .util import compress_kv, vertical_slash_sparse_attention_forward



def update_qwen3_model_for_speckv(model):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    class Qwen3AttentionSpecKV(Qwen3Attention):
        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            indices = None
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

                if query_states.shape[2] > 1 and (self.config.max_capacity_prompt is None or (
                    past_key_value.get_seq_length(self.layer_idx) == 0
                    and query_states.shape[2] > self.config.max_capacity_prompt
                )):
                    key_states_compress, value_states_compress, indices = compress_kv(
                        key_states,
                        query_states,
                        value_states,
                        window_size=self.config.window_size,
                        max_capacity_prompt=self.config.max_capacity_prompt,
                        prefill_vertical_size=self.config.prefill_vertical_size,
                        reduction_type=self.config.reduction_type,
                        pool_type=self.config.pool_type,
                        kernel_size=self.config.kernel_size,
                    )

                    past_key_value.update(
                        key_states_compress,
                        value_states_compress,
                        self.layer_idx,
                        cache_kwargs,
                    )
                else:
                    key_states, value_states = past_key_value.update(
                        key_states,
                        value_states,
                        self.layer_idx,
                        cache_kwargs,
                    )

            prefill_vertical_size = self.config.prefill_vertical_size
            prefill_window_size = self.config.prefill_window_size
            if indices is not None and input_shape[-1] > prefill_vertical_size + prefill_window_size:
                v_idx = indices[..., :prefill_vertical_size].int()
                s_idx = torch.arange(
                    prefill_window_size,
                    -1,
                    -64,
                    dtype=v_idx.dtype,
                    device=v_idx.device,
                )[None, None].expand(*v_idx.shape[:-1], -1)

                attn_output, attn_weights = vertical_slash_sparse_attention_forward(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    v_idx,
                    s_idx,
                )
            else:
                sliding_window = None
                if (
                    self.config.use_sliding_window
                    and getattr(self.config, "sliding_window", None) is not None
                    and self.layer_idx >= self.config.max_window_layers
                ):
                    # sliding_window = self.config.sliding_window
                    logger.warning_once(  # type: ignore
                        "Dynamic sparse attention is not compatible with sliding window."
                    )

                attention_interface: Callable = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and kwargs.get(
                        "output_attentions", False
                    ):
                        logger.warning_once(  # type: ignore
                            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                            'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                        )
                    else:
                        attention_interface = ALL_ATTENTION_FUNCTIONS[  # type: ignore
                            self.config._attn_implementation
                        ]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=sliding_window,
                    **kwargs,
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            return attn_output, attn_weights  # type: ignore

    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.forward = Qwen3AttentionSpecKV.forward.__get__(
            model.model.layers[i].self_attn,
            type(model.model.layers[i].self_attn),
        )

    return model


def reset_qwen2_model(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = Qwen2Attention.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model
