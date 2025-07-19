import importlib
from pathlib import Path
import re
import subprocess
from tqdm import tqdm
from transformers.models.auto import AutoTokenizer
import yaml


from ..base import DatasetBase
import numpy as np
import pandas as pd    


def postprocess_pred(predict_str: str, task_config: dict):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str



def get_pred_and_ref(
    results,
    task_config: dict,
    input_field: str = 'input',
    references_field: str = 'outputs',
    prediction_field: str = 'pred',
    metadata_field: str = 'others',
):
    # lines = read_manifest(predictions_file)

    inputs = []
    predicts = []
    references = []
    indices = []

    for line in tqdm(results):
        input = line[input_field]
        predict = line[prediction_field]
        predict = postprocess_pred(predict, task_config)
        reference = line.get(references_field, [line.get('output', '')])
        index = line[metadata_field].get('id', line['index'])
        
        inputs.append(input)
        predicts.append(predict)
        references.append(reference)
        indices.append(index)
        
    return inputs, predicts, references, indices

def run_evaluation_per_task(task_config: dict, results, verbose: int = 0):
    inputs, predicts, references, indices = get_pred_and_ref(
        results=results,
        task_config=task_config,
    )

    task_nulls = f'{sum([len(x)==0 for x in predicts])}/{len(predicts)}'

    if len(references) > 0 and references[0][0] is not None:
        task_score = task_config['metric_fn'](predicts, references)
    else:
        task_score = 0.0

    if verbose != 0:
        print('=' * 40)
        for i, (input, reference, predict) in enumerate(zip(inputs, references, predicts)):
            print(f'Input     : {input}')
            print(f'Reference : {reference}')
            print(f'Prediction: {predict}')
            print('=' * 40)
            if i > verbose:
                break

    return task_score, task_nulls, predicts, indices


def get_task_configs(benchmark):
    try:
        module = importlib.import_module(f"dataset.ruler.eval.{benchmark}.constants")
    except ImportError:
        raise Exception(f"Module dataset.ruler.eval.{benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(Path(__file__).parent / f"{benchmark}.yaml", "r") as f:
        tasks_customized = yaml.safe_load(f)
        
    tasks_customized
    for _, config in tasks_customized.items():
        config.update(tasks_base[config['task']])

    return tasks_customized


ruler_tasks = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


# from dataset/ruler/data/synthetic/constants.py
ruler_gen_lengths = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 64,  # 30,
    "cwe": 120,
    "fwe": 64,  # 50,
    "qa_1": 64,  # 32,
    "qa_2": 64,  # 32,
}


ruler_categories = {
    "NIAH": [
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
    ],
    "Multi-hop Tracing": [
        "vt",
    ],
    "Aggregation": [
        "cwe",
        "fwe",
    ],
    "QA": [
        "qa_1",
        "qa_2",
    ],
}


ruler_dft_num_samples = 500
ruler_dft_random_seed = 42

class RulerDataset(DatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, task, seq_len, model_cfg, benchmark="synthetic", benchmark_config="synthetic", split="val", 
                 random_seed=None, overwrite=False, shuffle_seed=42, num_samples=None):
        # assert name in longbench_datasets

        assert task in ruler_tasks

        path = None
        self.task = task.lower()
        self.dataset = None
        split = "val"  # only train
        self.dataset_name = "ruler"
        self.benchmark_config = benchmark_config

        self.model_name_or_path = model_cfg["model_name_or_path"]

        self.seq_len = seq_len
        self.overwrite = overwrite
        self.shuffle_seed = shuffle_seed
        self.num_samples = num_samples

        # if num_samples is None:
        #     num_samples = ruler_dft_num_samples
        # else:
        #     assert num_samples != ruler_dft_num_samples

        if random_seed is None:
            random_seed = ruler_dft_random_seed
        else:
            assert random_seed != ruler_dft_random_seed

        # self.num_samples = num_samples
        self.random_seed = random_seed

        self.model_name_to_type = {
            "meta-llama/Llama-3.1-8B-Instruct": "meta-llama3",
            "meta-llama/Llama-3.2-1B-Instruct": "meta-llama3",
            "meta-llama/Llama-3.2-3B-Instruct": "meta-llama3",
            "meta-llama/Llama-3.1-70B-Instruct": "meta-llama3",
            "llamafactory/tiny-random-Llama-3": "meta-llama3",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-r1-qwen",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-r1-qwen",
            "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5",
            "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5",
            "Qwen/Qwen2.5-3B-Instruct": "qwen2.5",
            "Qwen/Qwen2.5-7B-Instruct": "qwen2.5",
            "Qwen/Qwen2.5-14B-Instruct": "qwen2.5",
            "Qwen/Qwen2.5-32B-Instruct": "qwen2.5",
            "Qwen/Qwen2.5-72B-Instruct": "qwen2.5",
            "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": "qwen2.5",
            "Qwen/Qwen3-0.6B": "qwen3",
            "Qwen/Qwen3-32B": "qwen3",
        }
        self.task_cfgs = get_task_configs(benchmark)


        data_path = [
            "data",
            "ruler",
            self.model_name_to_type[self.model_name_or_path],
            self.benchmark_config,
            str(seq_len),
            "data"
        ]

        self.data_path = Path(*data_path)

        self.task_path = self.data_path / task

        super().__init__()

    @property
    def max_new_tokens(self):
        return ruler_gen_lengths[self.task]

    def __len__(self):
        return len(self.get_dataset())

    @property
    def add_special_tokens(self):
        return False

    @classmethod
    def download_dataset(cls):
        data_dir = Path(__file__).parent / "data/synthetic/json"

        if not (data_dir / "squad.json").exists() or not (data_dir / "hotpotqa.json").exists():
            print("Downloading QA datasets")
            subprocess.run(["bash", "download_qa_dataset.sh"], cwd=str(data_dir))
            print("Done")

        if not (data_dir / "PaulGrahamEssays.json").exists():
            print("Downloading Paul Graham Essays")
            subprocess.run(["python", "download_paulgraham_essay.py"], cwd=str(data_dir))
            print("Done")

    def create_dataset(self):
        self.download_dataset()

        cmd = [
            "python",
            Path(__file__).parent / "data" / "prepare.py",
            "--save_dir", self.data_path, 
            "--max_seq_length", self.seq_len, 
            "--benchmark", "synthetic", 
            "--benchmark_file", self.benchmark_config,
            "--task", self.task,
            "--tokenizer_path", self.model_name_or_path, 
            "--num_samples", ruler_dft_num_samples,  # self.num_samples,
            "--random_seed", self.random_seed,
            "--model_template_type", self.model_name_to_type[self.model_name_or_path], 
            "--tokenizer_type", "hf",
        ]

        cmd = [str(c) for c in cmd]
        print(" ".join(cmd).replace("--", "\\\n--"))
        subprocess.run(cmd)

    def get_dataset(self):
        if self.dataset is None:
            task_file = self.task_path / "validation.jsonl"

            if self.overwrite:
                if task_file.exists():
                    task_file.unlink()

            if not task_file.exists():
                self.create_dataset()

            self.dataset = pd.read_json(path_or_buf=task_file, lines=True)

            if self.num_samples is not None:
                if self.shuffle_seed is not None:
                    ind = np.random.RandomState(self.shuffle_seed).permutation(len(self.dataset))
                    ind = ind[:self.num_samples]
                else:
                    ind = list(range(self.num_samples))

                self.dataset = self.dataset.iloc[ind].reset_index(drop=True)

        return self.dataset

    def get_input_label(self, idx):
        self.get_dataset()

        sample = self.dataset.loc[idx]
        input = sample["input"]

        return input, {
            "outputs": sample["outputs"],
            "index": sample["index"],
            "length": sample.get("length", -1),
            "truncation": sample.get("truncation", -1),
            "others": sample.get("others", {}),
        }
    
    @property
    def use_chat_template(self):
        return False  # TODO: check if needed
    
    def format_prediction(self, pred):
        tags = [
            "<think/>",
            "Answer:",
            "Summary:",
            "The final answer is:",
            "The answer is:",
            "Next line of code:",
        ]

        for tag in tags:
            pred = pred.split(tag)[-1]

        pred = pred.strip()
        return pred
    
    def compute_metrics(self, predictions, references, inputs):
        assert len(predictions) == len(references)
        assert len(references) == len(inputs)

        # for r, p in zip(references, predictions):
        #     assert r["index"] == p["index"]

        results = [{
            "input": i,
            "pred": self.format_prediction(p),
            "outputs": r["outputs"],
            "index": r["index"],
            "truncation": r["truncation"],
            "others": r["others"],
        } for r, p, i in zip(references, predictions, inputs)]

        task_score, task_nulls, predicts, indices = run_evaluation_per_task(
            task_config=self.task_cfgs[self.task],
            results=results,
            verbose=0,
        )

        metrics = {
            "score": task_score,
            "nulls": task_nulls,
        }

        extra = {}

        return metrics, extra
    
    @staticmethod
    def get_task_category(name):
        for category, datasets in ruler_categories.items():
            if name in datasets:
                return category
        return None


    