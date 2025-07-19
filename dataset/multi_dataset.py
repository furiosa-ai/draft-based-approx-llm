
import random
import numpy as np


class MultiDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        
        data_lens = [len(dataset) for dataset in datasets]
        self.data_ends = np.cumsum(data_lens)

        use_chat_template = set(getattr(dataset, "use_chat_template", True) for dataset in datasets)
        add_special_tokens = set(getattr(dataset, "add_special_tokens", True) for dataset in datasets)
        assert len(use_chat_template) == 1
        assert len(add_special_tokens) == 1

        self.use_chat_template = use_chat_template.pop()
        self.add_special_tokens = add_special_tokens.pop()

        self.indices = list(range(self.data_ends[-1]))

    def shuffle(self, seed=None):
        random.Random(seed).shuffle(self.indices)
        return self
    
    def select(self, indices):
        self.indices = indices
        return self

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]

        dataset_idx = np.searchsorted(self.data_ends, idx, side="right")
        if dataset_idx > 0:
            idx = idx - self.data_ends[dataset_idx - 1]
        
        return self.datasets[dataset_idx][idx]

    def get_all(self):
        return [self.__getitem__(i) for i in range(len(self))]
