import contextlib
from typing import Optional
from einops import reduce
from transformers.models.llama.modeling_llama import repeat_kv
import torch
from torch import nn
import math


class AttentionStoreFunction(nn.Module):
    def __init__(self, attn_func, attn_store_func, window_size):
        super().__init__()

        self.attn_func = attn_func
        self.attn_store_func = attn_store_func
        self.window_size = window_size

    def compute_window_attn_weights(self, module, query_states, key_states, window_size):
        num_queries, num_keys = query_states.shape[2], key_states.shape[2]

        assert query_states.shape[0] == key_states.shape[0], f"Batch size mismatch"
        assert query_states.shape[3] == key_states.shape[3], f"Dim mismatch"
        
        if query_states.shape[1] != key_states.shape[1]:
            key_states = repeat_kv(key_states, module.num_key_value_groups)

        assert query_states.shape[1] == key_states.shape[1], "head number mismatch"

        
        if num_queries != num_keys:
            if "gemma3" in module.config.model_type:
                key_states, key_states_pad = key_states[:, :, :num_queries], key_states[:, :, num_queries:]
                assert key_states_pad.sum() == 0
        
        if num_queries != num_keys:
            raise Exception("num_queries != num_keys")

        attn_weights = torch.matmul(query_states[..., -window_size:, :], key_states.transpose(2, 3)) / math.sqrt(key_states.shape[3])
        
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        mask = mask[None, None, :, :]
        attn_weights[:, :, -min(window_size, attn_weights.shape[2]):, -window_size:] += mask[:, :, -min(window_size, attn_weights.shape[2]):, :]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return attn_weights

    # copied from transformers/models/llama/modeling_llama.py
    def eager_attention_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs):

        is_prefill = query.shape[2] > 1
        attn_kwargs = dict(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            **kwargs
        )

        if is_prefill:
            # compute window attention weights for specpc in eager mode
            # compuate full attention weights with flash attention

            # skip vision encoder layers
            attn_output, _attn_weights = self.attn_func(**attn_kwargs)
            assert _attn_weights is None, "attn_weights should be None"

            if not any(m in module.__class__.__name__ for m in ["SiglipAttention", "Qwen2_5_VLVision"]):
                self.attn_store_func(module.layer_idx, self.compute_window_attn_weights(module, query, key, self.window_size))
        else:
            # just use eager attention since q=1
            attn_output, attn_weights = self.eager_attention_forward(**attn_kwargs)
            assert attn_weights is not None

            self.attn_store_func(module.layer_idx, attn_weights)

        return attn_output, None


class AttentionCache:
    def __init__(self, config):
        self.starting_layer_index = config.starting_layer_index
        self.reduction_type = config.reduction_type
        self.weighted_query = config.weighted_query
        
        if self.reduction_type == "mean":
            self.reduction_type = "sum"  # same as mean in topk

        self.device = None
        self.cache = []

    def aggregate_attn_weights(self, attn_weights, is_prefill):
        batch_size, num_heads, q, k = attn_weights.shape
        
        assert attn_weights.ndim == 4
        # assert q == 1 or (is_prefill and self.num_past_tokens > 0)
        assert batch_size == 1

        # is_prefill = q > 1

        if q > 1 and self.weighted_query:
            weights = torch.linspace(0, 1, q, device=attn_weights.device, dtype=attn_weights.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, q, 1)
            attn_weights = attn_weights * weights

        return reduce(attn_weights, "b h q k -> b k", self.reduction_type)

    def aggregate_attn_weights_over_layers(self, attn_weights): 
        attn_weights = [[attn.to(self.device) for attn in attns if attn is not None] for attns in attn_weights]

        out = [reduce(torch.stack(attns, 0), "l b k -> b k", self.reduction_type) for attns in attn_weights]

        return out

    def add(self, layer_idx, attn_weights):
        if self.device is None:
            self.device = attn_weights.device

        if len(self.cache) <= layer_idx:
            self.cache.append([])
            is_prefill = True
        else:
            is_prefill = False

        if layer_idx >= self.starting_layer_index:
            self.cache[layer_idx].append(self.aggregate_attn_weights(attn_weights, is_prefill=is_prefill))

    def get_aggreated_attention(self):
        """
        Returns list of aggregated attention activations per key for all prefill tokens.
        """

        attn_weights = self.cache
        attn_weights = [m for m in attn_weights if len(m) > 0]  # remove unstored layers
        attn_weights_per_gen_token = list(map(list, zip(*attn_weights)))  # transpose -> steps, layers, keys
        attn_weights_per_gen_token = self.aggregate_attn_weights_over_layers(attn_weights_per_gen_token)

        seq_len = attn_weights_per_gen_token[0].shape[1]

        # cut off generated keys
        attn_weights_per_gen_token = [a[:, :seq_len] for a in attn_weights_per_gen_token]

        # aggregate over all input and output tokens (k=1 for output tokens)
        attn_weights_agg = reduce(torch.stack(attn_weights_per_gen_token, 0), "n b k -> b k", self.reduction_type)

        self.cache = []
        return attn_weights_agg


@contextlib.contextmanager
def patch_model_attn_cache(model, config):
    attn_cache = AttentionCache(config)

    if model.config.model_type == "qwen2_5_vl":
        raise NotImplementedError("Attention cache is not implemented for model type 'qwen2_5_vl'.")
    else:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # assert max_gen_len <= 1, "not implemented yet for new transformers"

        attn_key = model.config._attn_implementation

        attn_func_original = ALL_ATTENTION_FUNCTIONS[attn_key]

        ALL_ATTENTION_FUNCTIONS[attn_key] = AttentionStoreFunction(
            ALL_ATTENTION_FUNCTIONS[attn_key], 
            lambda layer_idx, attn_weights: attn_cache.add(layer_idx, attn_weights), 
            config.window_size
        ).forward

        yield attn_cache

        ALL_ATTENTION_FUNCTIONS[attn_key] = attn_func_original


def generate_aggregated_attention(model, config, input_ids, max_new_tokens=None, **kwargs):
    """
    Generates tokens and returns aggregated attention activations per key for all prefill tokens and generated tokens.
    """

    if max_new_tokens is None:
        max_new_tokens = config.lookahead_tokens

    with patch_model_attn_cache(model, config) as attn_cache:
        input_output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=max_new_tokens is None or max_new_tokens > 1,
            **kwargs,
        )
        output_ids = input_output_ids[:, input_ids.shape[1]:]

        return attn_cache.get_aggreated_attention(), output_ids