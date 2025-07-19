import os

WANDB_TEAM_NAME = None
WANDB_PROJECT_NAME = "draft_approx_llm"

def initialize_wandb_settings():
    global WANDB_TEAM_NAME
    global WANDB_PROJECT_NAME

    WANDB_TEAM_NAME = os.getenv("WANDB_TEAM_NAME", WANDB_TEAM_NAME)
    WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME", WANDB_PROJECT_NAME)

    print(f"WANDB_TEAM_NAME: {WANDB_TEAM_NAME}")
    print(f"WANDB_PROJECT_NAME: {WANDB_PROJECT_NAME}")

initialize_wandb_settings()