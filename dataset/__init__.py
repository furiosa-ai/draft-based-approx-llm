

from dataset.longbench import LongBenchDataset
from dataset.multi_dataset import MultiDataset
from dataset.ruler import RulerDataset


# prompts based on: https://github.com/open-compass/opencompass
def load_dataset(tokenizer, dataset_name, split=None, task=None, num_samples=None, model_cfg=None, **kwargs):
    dataset_name = dataset_name.lower()

    if dataset_name == "thudm/longbench" or dataset_name == "longbench":
        data = LongBenchDataset(tokenizer, task, split=split, num_samples=num_samples, model_cfg=model_cfg, **kwargs)
    elif dataset_name == "ruler":
        if isinstance(task, list):
            data = MultiDataset([RulerDataset(tokenizer, t, split=split, num_samples=num_samples, model_cfg=model_cfg, **kwargs) for t in task])
        else:
            data = RulerDataset(tokenizer, task, split=split, num_samples=num_samples, model_cfg=model_cfg, **kwargs)
    else:     
        raise ValueError(f"got {dataset_name}")

    return data