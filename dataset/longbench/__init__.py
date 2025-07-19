import json
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.longbench.metrics import qa_f1_score, rouge_score, classification_score, retrieval_score, count_score, code_sim_score

from ..base import DatasetBase
import numpy as np


longbench_categories_datasets = {
    "Single-document QA": [
        "hotpotqa", 
        "2wikimqa", 
        "musique",
    ],

    "Multi-document QA": [
        "narrativeqa", 
        "qasper", 
        "multifieldqa_en", 
    ],

    "Summarization": [
        "gov_report", 
        "qmsum", 
        "multi_news", 
    ],

    "Few-shot learning": [
        "trec", 
        "triviaqa", 
        "samsum", 
    ],

    "Synthetic Tasks": [
        "passage_count", 
        "passage_retrieval_en", 
    ],

    "Code Completion": [
        "lcc", 
        "repobench-p"
    ]
}



longbench_datasets = [
    # Single-document QA
    "narrativeqa", 
    "qasper", 
    "multifieldqa_en", 
    # "multifieldqa_zh",  # chinese

    # Multi-document QA
    "hotpotqa", 
    "2wikimqa", 
    "musique",
    # "dureader",  # chinese

    # Summarization
    "gov_report", 
    "qmsum", 
    "multi_news", 
    # "vcsum",   # chinese

    # Few-shot learning
    "trec", 
    "triviaqa", 
    "samsum", 
    # "lsht",   # chinese

    # Synthetic Tasks
    # "passage_count", 
    # "passage_retrieval_en", 
    # "passage_retrieval_zh",   # chinese

    # Code Completion
    "lcc", 
    "repobench-p"
]


longbench_datasets_with_ch = [
    # Single-document QA
    "narrativeqa", 
    "qasper", 
    "multifieldqa_en", 
    "multifieldqa_zh",  # chinese

    # Multi-document QA
    "hotpotqa", 
    "2wikimqa", 
    "musique",
    "dureader",  # chinese

    # Summarization
    "gov_report", 
    "qmsum", 
    "multi_news", 
    "vcsum",   # chinese

    # Few-shot learning
    "trec", 
    "triviaqa", 
    "samsum", 
    "lsht",   # chinese

    # Synthetic Tasks
    "passage_count", 
    "passage_retrieval_en", 
    "passage_retrieval_zh",   # chinese

    # Code Completion
    "lcc", 
    "repobench-p"
]


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    # "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    # "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    # "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    # "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


class LongBenchDataset(DatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, name, split="test", model_cfg=None, num_samples=None):
        assert name in longbench_datasets_with_ch

        self.name = name.lower()
        self.task = name
        self.dataset = None
        self.split = "test"  # only train
        self.dataset_name = "longbench"
        self.model_cfg = model_cfg
        self.num_samples = num_samples

        # assert name in ("scbench_choice_eng",)

        # self.choice_labels = ["A", "B", "C", "D", "E"]

        with open("dataset/longbench/dataset2prompt.json", "r") as f:
            self.dataset2prompt = json.load(f)

        with open("dataset/longbench/dataset2maxlen.json", "r") as f:
            self.max_new_tokens = json.load(f)[name]

        super().__init__()

    def __len__(self):
        return len(self.get_dataset())

    def get_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset("THUDM/LongBench", self.name, split=self.split, trust_remote_code=True)

            if self.num_samples is not None:
                self.dataset = self.dataset.shuffle(seed=42).select(range(self.num_samples))

        return self.dataset

    def get_input_label(self, idx):
        self.get_dataset()

        sample = self.dataset[idx]

        input = self.dataset2prompt[self.name].format(**sample)

        return input, {"answers": sample["answers"], "all_classes": sample["all_classes"]}
    
    @property
    def use_chat_template(self):
        if "Qwen3" in self.model_cfg["model_name_or_path"]:
            # always use chat template for Qwen3 to disable thinking
            return True

        return self.name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]

    def scorer(self, predictions, answers, all_classes):
        dataset = self.name

        total_score = 0.
        scores = []
        for (prediction, ground_truths) in zip(predictions, answers):
            score = 0.
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            scores.append(score)
            total_score += score

        mean_score = round(100 * total_score / len(predictions), 2)
        return mean_score, scores

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

    def compute_metrics(self, predictions, references=None, inputs=None):
        answers = [r["answers"] for r in references]
        # all_classes = [r["all_classes"] for r in references]
        all_classes = references[0]["all_classes"]

        predictions = [self.format_prediction(pred) for pred in predictions]

        score, scores_all = self.scorer(predictions, answers, all_classes)

        metrics = {
            "score": score,
        }

        # return all scores
        extra = [{
            "score": s
        } for s in scores_all]

        return metrics, extra
    
    @staticmethod
    def get_task_category(name):
        for category, datasets in longbench_categories_datasets.items():
            if name in datasets:
                return category
        return None
