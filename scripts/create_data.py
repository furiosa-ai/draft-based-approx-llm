import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from tqdm import tqdm
from dataset.ruler import RulerDataset, ruler_tasks
from dataset.longbench import LongBenchDataset, longbench_datasets

import nltk


def create_dataset(args):
    dataset_name, seq_len, task, benchmark_config, model = args

    try:
        if dataset_name == "ruler":
            sample = RulerDataset(None, task, seq_len, {"model_name_or_path": model}, benchmark_config=benchmark_config)[0]
        elif dataset_name == "longbench":
            sample = LongBenchDataset(None, task)[0]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    except Exception as e:
        print(task, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--seq_len", nargs="+", type=int)
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--task", nargs="+")
    parser.add_argument("--benchmark_config", nargs="+", default=["synthetic"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    nltk.download('punkt_tab')

    if args.task is not None:
        tasks = args.task
    else:
        if args.data == "ruler":
            tasks = ruler_tasks
        elif args.data == "longbench":
            tasks = longbench_datasets
            args.seq_len = [None]
            args.benchmark_config = [None]
        else:
            raise ValueError(f"Unknown dataset name: {args.data}")

    args.data = [args.data]
    args.model = args.model

    if "ruler" in args.data:
        RulerDataset.download_dataset()

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        _ = list(
            tqdm(executor.map(
                create_dataset,
                list(product(args.data, args.seq_len, tasks, args.benchmark_config, args.model)),
            ), total=len(tasks))
        )


if __name__ == "__main__":
    main()
