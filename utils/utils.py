
import gc
import torch

from utils.constants import WANDB_PROJECT_NAME, WANDB_TEAM_NAME
from utils.logger import WandbLogger


def create_logger(cfg, **kwargs):
    run_name = cfg["cfg_file"].replace("cfg/", "").replace(".yaml", "")

    logger = WandbLogger(project=WANDB_PROJECT_NAME, entity=WANDB_TEAM_NAME, name=run_name, config=cfg, **kwargs)
    return logger


def free_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()
