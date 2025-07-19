from abc import abstractmethod, ABC


class DatasetBase(ABC):
    shuffle_seeds = [
        123,
        321,
        532,
        523,
    ]
    
    def __init__(self):
        super().__init__()

        self.it = None

    @property
    def use_chat_template(self):
        return True
    
    @property
    def add_special_tokens(self):
        return True

    def __iter__(self):
        self.it = iter(self.__getitem__(i) for i in range(len(self)))
        return self

    def __next__(self):
        return next(self.it)

    def __getitem__(self, idx):
        input, label = self.get_input_label(idx)
        return dict(input=input, label=label)

    @abstractmethod
    def get_input_label(self, idx):
        pass

    @abstractmethod
    def compute_metrics(self, eval_preds):
        pass