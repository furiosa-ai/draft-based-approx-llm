import functools
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from draft_approx_llm import DraftApproxLLMModelOutput, SpecPCConfig
from .specpc_utils import generate_aggregated_attention


def specpc_patch_model(model, draft_model, config: SpecPCConfig):
    """
    Patch the model and tokenizer for SpecPC.
    """

    model.generate = types.MethodType(functools.partial(specpc_generate, specpc_draft_model=draft_model, specpc_config=config), model)
    return model


def _unique(tensor, sort):
    if sort:
        return tensor.unique(dim=-1)
    else:
        # pandas.unique does not sort, unlike torch.unique
        return torch.tensor(np.stack([pd.unique(t) for t in tensor.cpu().numpy()]), device=tensor.device)


def specpc_select_input_token_indices(attn_per_key: torch.Tensor, specpc_config: SpecPCConfig):
    kernel_size = specpc_config.kernel_size
    neighbor_tokens = specpc_config.neighbor_tokens
    batch_size, seq_len = attn_per_key.shape

    # smooth
    if specpc_config.pool_type is not None:
        pool_func = {
            "avg": F.avg_pool1d, 
            "mean": F.avg_pool1d, 
            "max": F.max_pool1d,
        }[specpc_config.pool_type]

        # center pool
        pad_mode = "constant"
        pad = ((kernel_size - 1) // 2, kernel_size - 1 - (kernel_size - 1) // 2)
        attn_per_key_pad = F.pad(attn_per_key, pad, mode=pad_mode)
        attn_per_key = pool_func(attn_per_key_pad, kernel_size=kernel_size, stride=1)

    # sorted by importance descending
    ind = attn_per_key.topk(specpc_config.max_capacity_prompt, sorted=True, dim=1).indices

    if neighbor_tokens is None:
        neighbor_tokens = kernel_size

    # max pool
    if neighbor_tokens > 0:
        # store future tokens
        adj_ind = torch.arange(-((neighbor_tokens-1) // 2), neighbor_tokens - 1 - (neighbor_tokens - 1) // 2 + 1, device=ind.device)

        # sort by distance so most adjacent tokens are kept in the end
        adj_ind = adj_ind[adj_ind.abs().argsort()]
        
        ind_with_adj = (ind[..., None] + adj_ind)
        ind_with_adj = torch.clamp(ind_with_adj.reshape(batch_size, -1), 0, seq_len - 1)

        # keep topk order 
        ind = _unique(ind_with_adj, sort=False)

    # keep window
    if specpc_config.window_size > seq_len:
        raise ValueError(f"window_size ({specpc_config.window_size}) cannot be greater than seq_len ({seq_len}).")
    
    ind_wnd = torch.arange(seq_len - specpc_config.window_size, seq_len, device=ind.device).unsqueeze(0)

    # keep topk order
    ind = _unique(torch.cat([ind_wnd, ind], 1), sort=False)

    # cut off to max capacity
    ind = ind[:, :specpc_config.max_capacity_prompt]

    ind = ind.sort(-1).values

    return ind


def specpc_compress_prompt(specpc_draft_model, specpc_config, input_ids, attention_mask=None, **kwargs):
    if input_ids.shape[1] > specpc_config.max_capacity_prompt:
        attn_per_key = generate_aggregated_attention(specpc_draft_model, specpc_config, input_ids=input_ids, attention_mask=attention_mask, **kwargs)  # b k
        important_token_ind = specpc_select_input_token_indices(attn_per_key, specpc_config)
        input_ids = torch.gather(input_ids, 1, important_token_ind)

        if attention_mask is not None:
            attention_mask = torch.gather(attention_mask, 1, important_token_ind)
    
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,  # e.g., sampling params
    )


def specpc_generate(self, *args, max_new_tokens, specpc_draft_model, specpc_config, return_dict_in_generate=True, **kwargs):
    # draft generate to compress prompt
    gen_kwargs = specpc_compress_prompt(
        specpc_draft_model, specpc_config, *args, **kwargs
    )
    
    # target generate
    out = self.__class__.generate(
        self,
        **gen_kwargs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
    )

    return DraftApproxLLMModelOutput(
        sequences=out.sequences,
        input_size=gen_kwargs["input_ids"].size(1)
    )