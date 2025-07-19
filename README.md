# Draft-based Approximate Inference for LLMs

<p align="center">
      <a href="https://scholar.google.com/citations?user=G1EpeWYAAAAJ&hl=en" target="_blank">Kevin Galim</a><sup>1*</sup>, 
      <a href="https://scholar.google.com/citations?user=H1hMLnkAAAAJ&hl=en" target="_blank">Ethan Ewer</a><sup>2*</sup>, 
      <a href="https://scholar.google.com/citations?user=Q-ARWkwAAAAJ&hl=eh" target="_blank">Wonjun Kang</a><sup>1,3</sup>, 
      <a href="https://scholar.google.com/citations?user=XJXKp60AAAAJ&hl=en" target="_blank">Minjae Lee</a><sup>1</sup>, 
      <a href="http://cvml.ajou.ac.kr/wiki/index.php/Professor" target="_blank">Hyung Il Koo</a><sup>1,4</sup>, 
      <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a><sup>2</sup>
  </p>
  <p  align="center">
    <sup>1</sup>FuriosaAI, <sup>2</sup>UW-Madison, <sup>3</sup>Seoul National University, <sup>4</sup>Ajou University
   </p>
<p align="center">
    <a href="https://arxiv.org/abs/2506.08373">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2506.08373-b31b1b.svg">
    </a>
</p>

<p align="center">
<img src = "docs/framework.png" width="70%" height="auto">
</p>

## 🚀 Overview
**Draft-based Approximate Inference for LLMs** leverages small draft models to more sharply distinguish important tokens and key-value (KV) pairs in long-context large language models. Our core contributions, **SpecKV** and **SpecPC**, enable smarter KV cache eviction and prompt compression, delivering more precise, efficient approximate inference than existing techniques.

## 📝 Abstract 
Optimizing inference for long-context Large Language Models (LLMs) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, such as key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on rough predictions of token or KV pair importance. We propose a novel framework for approximate LLM inference that leverages small draft models to more accurately predict the importance of tokens and KV pairs. Specifically, we introduce two instantiations of our proposed framework: 
1. **SpecKV**, which leverages a draft output to accurately assess the importance of each KV pair for more effective KV cache dropping
2. **SpecPC**, which uses the draft model's attention activations to identify and discard unimportant prompt tokens. 

To the best of our knowledge, this is the first work to use draft models for approximate LLM inference acceleration, extending their utility beyond traditional lossless speculative decoding. We motivate our methods with theoretical and empirical analyses, and show a strong correlation between the attention patterns of draft and target models. Extensive experiments on long-context benchmarks show that our methods consistently achieve higher accuracy than existing baselines, while preserving the same improvements in memory usage, latency, and throughput.

## 🌟 Features

- **Plug & Play**: Add to any HuggingFace-compatible LLM with just a few lines.
- **Higher Retained Accuracy**: SpecKV/SpecPC preserve more model accuracy vs previous methods.
- **Flexible**: Supports Qwen2.5, Llama-3, and more.

## 🛠️ Installation

**1. Clone repository:**
```bash
git clone https://github.com/furiosa-ai/draft-based-approx-llm
```

**2. Install PyTorch (example for CUDA 12.4):**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**3. Install other dependencies:**
```bash
pip install -r requirements.txt --no-build-isolation
```

**4. Install FlashAttention:**
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

**5. Prepare the RULER benchmark:**
```bash
python scripts/create_data.py \
    --data ruler \
    --seq_len 4096 8192 16384 32768 65536 \
    --model \
        meta-llama/Llama-3.2-1B-Instruct \
        Qwen/Qwen2.5-0.5B-Instruct
```

## 🧩 Example Usage

<details> <summary><strong>SpecKV</strong> </summary>

```python
from draft_approx_llm import SpecKVConfig, patch_model
from transformers import AutoModelForCausalLM

# Load base and draft models
model_kwargs = {
    "torch_dtype": "auto",
    "attn_implementation": "flash_attention_2",
    "device_map": "auto"
}

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", **model_kwargs)
draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", **model_kwargs)

# Configure SpecKV
speckv_config = SpecKVConfig(
    max_capacity_prompt=256,
    window_size=32,
    pool_type="max",
    kernel_size=7,
    reduction_type="max",
    lookahead_tokens=None,
    prefill_window_size=2048,
    prefill_vertical_size=2048
)

# Patch target model with the draft model to use SpecKV
model = patch_model(model, draft_model, speckv_config)

# Generate output
model.generate(inputs, max_new_tokens=32, return_dict_in_generate=True)
```

*See more in [notebooks/example_usage_speckv.ipynb](notebooks/example_usage_speckv.ipynb).*

</details>

<details> <summary><strong>SpecPC</strong> </summary>

```python
from draft_approx_llm import SpecPCConfig, patch_model
from transformers import AutoModelForCausalLM

# Load base and draft models
model_kwargs = {
    "torch_dtype": "auto",
    "attn_implementation": "flash_attention_2",
    "device_map": "auto"
}

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", **model_kwargs)
draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", **model_kwargs)

# Configure SpecPC
specpc_config = SpecPCConfig(
    max_capacity_prompt=1024,
    window_size=64,
    pool_type="max",
    kernel_size=64,
    reduction_type="max",
    lookahead_tokens=1,
    neighbor_tokens=64,
    starting_layer_index=8,
    weighted_query=True
)

# Patch target model with the draft model to use SpecKV
model = patch_model(model, draft_model, specpc_config)

# Generate output
model.generate(inputs, max_new_tokens=32, return_dict_in_generate=True)
```

*See more in [notebooks/example_usage_specpc.ipynb](notebooks/example_usage_specpc.ipynb).*

</details>

## Reproducing Paper Results

Run evaluation (results logged to Weights & Biases):

### SpecKV
```bash
python eval.py --cfg cfg/paper/speckv/longbench/llama3_1b_8b/cmax_*/*.yaml
python eval.py --cfg cfg/paper/speckv/longbench/qwen25_05b_14b/cmax_*/*.yaml
python eval.py --cfg cfg/paper/speckv/ruler/*/llama3_1b_8b/cmax_*/*.yaml
python eval.py --cfg cfg/paper/speckv/ruler/*/qwen25_05b_14b/cmax_*/*.yaml
```
### SpecPC
```bash
python eval.py --cfg cfg/paper/specpc/longbench/llama3_1b_8b/cmax_*/*.yaml
python eval.py --cfg cfg/paper/specpc/longbench/qwen25_05b_14b/cmax_*/*.yaml
python eval.py --cfg cfg/paper/specpc/ruler/*/llama3_1b_8b/cmax_*/*.yaml
python eval.py --cfg cfg/paper/specpc/ruler/*/qwen25_05b_14b/cmax_*/*.yaml
```


## ⏳ Roadmap
- [x] Release codebase for SpecKV and SpecPC
- [ ] Enable vLLM compatibility (SpecKV draft, SpecPC target)
- [ ] Release Ada-SpecKV
- [ ] Release Qwen2.5-VL support

## 📖 Citation

If you find this useful, please cite:

```bibtex
@article{galim2025draft,
  title={Draft-based Approximate Inference for LLMs},
  author={Galim, Kevin and Ewer, Ethan and Kang, Wonjun and Lee, Minjae and Koo, Hyung Il and Lee, Kangwook},
  journal={arXiv preprint arXiv:2506.08373},
  year={2025}
}
```


## 🤝 Contributions

Pull requests, issues, and feedback are welcome!

