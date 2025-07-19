import functools
import types
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from ..draft_approx_llm import DraftApproxLLMModelOutput, SpecKVConfig

from .llama_util import update_llama_model_for_speckv
from .qwen2_util import update_qwen2_model_for_speckv
from .qwen3_util import update_qwen3_model_for_speckv


# TODO: add sampling support
def speckv_generate_from_lookahead_ids(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    speckv_config: SpecKVConfig,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lookahead_ids: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    generate_fn =  model.__class__.generate
    max_capacity_prompt = speckv_config.max_capacity_prompt

    assert max_capacity_prompt is None or max_capacity_prompt <= speckv_config.prefill_window_size + speckv_config.prefill_vertical_size
    assert speckv_config.prefill_window_size % 64 == 0 and speckv_config.kernel_size % 2 == 1

    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_speckv(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_speckv(model)
    elif model.config.model_type == "qwen3":
        update_qwen3_model_for_speckv(model)
    else:
        raise NotImplementedError()

    sparse_prefill_only = max_capacity_prompt is None

    if not sparse_prefill_only and input_ids.shape[1] <= max_capacity_prompt:
        return generate_fn(  # type: ignore
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

    lookahead_size = lookahead_ids.shape[1] - input_ids.shape[1]

    model.config.window_size = speckv_config.window_size + lookahead_size + 1
    model.config.max_capacity_prompt = (max_capacity_prompt + lookahead_size + 1) if max_capacity_prompt is not None else None
    model.config.prefill_window_size = speckv_config.prefill_window_size
    model.config.prefill_vertical_size = speckv_config.prefill_vertical_size
    model.config.reduction_type = speckv_config.reduction_type
    model.config.pool_type = speckv_config.pool_type
    model.config.kernel_size = speckv_config.kernel_size

    extended_attention_mask = F.pad(attention_mask, (0, lookahead_size), value=1)

    past_key_values = DynamicCache()

    # prefill
    model.model(
        input_ids=lookahead_ids,
        attention_mask=extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )

    if max_capacity_prompt is None:
        max_capacity_prompt = input_ids.shape[1] -1  # to crop the last token

    past_key_values.crop(max_capacity_prompt)
    del extended_attention_mask

    # decode
    generated_ids = generate_fn(
        model,
        input_ids=input_ids[:, -max_capacity_prompt - 1 :],
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_key_values,
        return_dict_in_generate=False,
        **kwargs,
    )

    return torch.cat((input_ids[:, : -max_capacity_prompt - 1], generated_ids), dim=1)  # type: ignore



def speckv_generate(self, input_ids, *args, speckv_draft_model, speckv_config, max_new_tokens, attention_mask=None, return_dict_in_generate=False, **kwargs):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    # draft generate
    lookahead_tokens = speckv_config.lookahead_tokens if speckv_config.lookahead_tokens is not None else max_new_tokens
    lookahead_ids = speckv_draft_model.generate(input_ids, *args, attention_mask=attention_mask, max_new_tokens=lookahead_tokens, return_dict_in_generate=False, **kwargs)

    # target generate
    out = speckv_generate_from_lookahead_ids(
        self,
        speckv_config=speckv_config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        lookahead_ids=lookahead_ids,
        **kwargs,
        max_new_tokens=max_new_tokens,
    )

    return DraftApproxLLMModelOutput(
        sequences=out,
        input_size=input_ids.size(1)
    )


def speckv_patch_model(model, draft_model, config: SpecKVConfig):
    """
    Patch the model and tokenizer for SpecPC.
    """

    model.generate = types.MethodType(functools.partial(speckv_generate, speckv_draft_model=draft_model, speckv_config=config), model)
    return model