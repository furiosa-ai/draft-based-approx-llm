import argparse
import os

from pathlib import Path
import torch
from tqdm import tqdm
import yaml
from dataset import load_dataset
from draft_approx_llm import DraftApproxLLMConfig, patch_model
from utils.logger import NoLogger
from utils.tokenize_utils import InputProcessor
from utils.utils import create_logger

from transformers import AutoTokenizer, AutoModelForCausalLM


def run_sample(model, input_processor, input_txt, max_new_tokens):
    inputs = input_processor.encode(input_txt, device=model.device)

    model_output = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    model_output_txt = input_processor.decode(model_output.output_ids.squeeze())

    return model_output_txt


def load_model(cfg, tokenizer):
    cfg = {**cfg}
    model_name_or_path = cfg.pop("model_name_or_path")
    dtype = cfg.pop("dtype")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map="auto", **cfg, trust_remote_code=True)
    model.generation_config.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    return model


def run_eval(cfg, args):
    logger = create_logger(cfg) if not args.get("no_log", False) else NoLogger()

    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"]["model_name_or_path"], trust_remote_code=True)
    draft_model = load_model(cfg["draft_model"], tokenizer)
    target_model = load_model(cfg["target_model"], tokenizer)
    sparse_config = DraftApproxLLMConfig.from_dict(cfg["sparse_config"])

    data = load_dataset(tokenizer, **cfg["dataset"], model_cfg=cfg["target_model"])
    max_new_tokens = data.max_new_tokens

    model = patch_model(target_model, draft_model, sparse_config)

    input_processor = InputProcessor(model, tokenizer, data)

    inputs_outputs_labels = []
    for sample in tqdm(data, desc="Running samples"):
        input_txt = sample["input"]
        label = sample["label"]

        output_txt = run_sample(model, input_processor, input_txt, max_new_tokens)
        inputs_outputs_labels.append((input_txt, output_txt, label))

    inputs, outputs, labels = zip(*inputs_outputs_labels)
    metrics, metrics_all = data.compute_metrics(predictions=outputs, references=labels, inputs=inputs)
    
    logger.log_metrics(metrics)

    logger.finish()


def load_cfg(cfg_file):
    if not Path(cfg_file).exists():
        return None

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["cfg_file"] = cfg_file
    cfg["device"] = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    return cfg

 
@torch.inference_mode()
def main(cfg_files, args):
    for cfg_file in cfg_files:
        cfg = load_cfg(cfg_file)

        if cfg is None:
            print(f"Error loading cfg file {cfg_file}")
            continue

        run_eval(cfg, args)


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, nargs="+", help="Path(s) to configuration file(s)")
    parser.add_argument("--no_log", action="store_true", help="Disable logging of metrics and evaluation results")
    args = parser.parse_args()

    return args.cfg, vars(args)


if __name__ == "__main__":
    cfg_files, args = parse_cfg()
    main(cfg_files, args)
