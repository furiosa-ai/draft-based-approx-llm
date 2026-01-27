import functools
import types
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


from ..specpc.specpc import specpc_compress_prompt
from ..speckv.speckv import speckv_generate_from_lookahead_ids

from ..draft_approx_llm import DraftApproxLLMModelOutput, SpecKVPCConfig


def speckvpc_generate(self, input_ids, *args, speckvpc_draft_model, speckvpc_config: SpecKVPCConfig, max_new_tokens, attention_mask=None, return_dict_in_generate=False, **kwargs):
    # 1. compress prompt with SpecPC
    gen_kwargs = specpc_compress_prompt(
        input_ids=input_ids, attention_mask=attention_mask, specpc_draft_model=speckvpc_draft_model, specpc_config=speckvpc_config.specpc_config, force_lookahead=True, *args, **kwargs
    )

    # 2. decode target with SpecKV
    input_ids = gen_kwargs.pop("input_ids")
    attention_mask = gen_kwargs.pop("attention_mask")
    lookahead_ids = gen_kwargs.pop("lookahead_ids")
    lookahead_ids = torch.cat([input_ids, lookahead_ids], dim=1)  # prepend input_ids to lookahead_ids for speckv

    out = speckv_generate_from_lookahead_ids(
        self,
        input_ids=input_ids,
        lookahead_ids=lookahead_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
        speckv_config=speckvpc_config.speckv_config,
        max_new_tokens=max_new_tokens,
    )

    return DraftApproxLLMModelOutput(
        sequences=out,
        input_size=input_ids.size(1)
    )


def speckvpc_patch_model(model, draft_model, config: SpecKVPCConfig):
    """
    Patch the model and tokenizer for SpecPC.
    """

    model.generate = types.MethodType(functools.partial(speckvpc_generate, speckvpc_draft_model=draft_model, speckvpc_config=config), model)
    return model