import wandb


class Logger:
    def __init__(self):
        pass

    def log(self, data, step):
        raise NotImplementedError

    def log_metrics(self, metrics):
        raise NotImplementedError
    
    def finish(self):
        raise NotImplementedError


class WandbLogger(Logger):
    def __init__(self, project, entity, name, config, resume=False, exist_ok=True):
        super().__init__()

        cfg_file = config["cfg_file"]
        self.last_ts = None
        self.step = -1

        if not exist_ok:
            api = wandb.Api()
            runs = list(api.runs(project, {"config.cfg_file": cfg_file}))
            if len(runs) > 0:
                raise FileExistsError(f"Found {len(runs)} runs with cfg_file {cfg_file}")

        if resume:
            api = wandb.Api()
            runs = list(api.runs(project, {"config.cfg_file": cfg_file}))
            assert len(runs) == 1, f"Found {len(runs)} runs with cfg_file {cfg_file}"
            wandb.init(project=project, entity=entity, id=runs[0].id, resume="must", config=config, settings=wandb.Settings(init_timeout=1200))
        else:
            wandb.init(project=project, entity=entity, name=name, config=config, settings=wandb.Settings(init_timeout=1200))

    def log_metrics(self, metrics):
        wandb.log(metrics)

    def log(self, data, step=None):
        data = {**data}

        if step is None:
            self.step += 1
            step = self.step
        else:
            self.step = step

        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()


class NoLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def log_metrics(self, metrics):
        pass

    def log(self, data, step):
        pass

    def finish(self):
        pass
