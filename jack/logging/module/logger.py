import logging
import os
import pathlib
from typing import Optional

import random_name
import wandb
from icecream import ic
from jack.logging.iface import Logger
from loguru import logger as jogger

_logger = logging.getLogger(__name__)


class WANDBLogger(Logger):
    """
    Weights and biases logger. See <a href="https://docs.wandb.ai">here</a> for more details.
    """

    experiment = None
    save_dir = str(pathlib.Path(os.getcwd()) / ".wandb")
    offset_step = 0
    sync_step = True
    prefix = ""
    log_checkpoint = False

    @classmethod
    def init_experiment(
        cls,
        experiment_name,
        project_name,
        api: Optional[str] = None,
        notes=None,
        tags=None,
        entity=None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        _id: Optional[str] = None,
        log_checkpoint: Optional[bool] = False,
        sync_step: Optional[bool] = True,
        prefix: Optional[str] = "",
        notebook: Optional[str] = None,
        **kwargs,
    ):
        if offline:
            os.environ["WANDB_MODE"] = "dryrun"
        if api is not None:
            os.environ["WANDB_API_KEY"] = api
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = wandb.util.generate_id() if _id is None else _id
        os.environ["WANDB_NOTEBOOK_NAME"] = notebook if notebook else "atomicai/jack"

        if wandb.run is not None:
            cls.end_run()

        cls.experiment = wandb.init(
            resume=sync_step,
            name=experiment_name,
            dir=save_dir,
            project=project_name,
            notes=notes,
            tags=tags,
            entity=entity,
            **kwargs,
        )

        cls.offset_step = cls.experiment.step
        cls.prefix = prefix
        cls.sync_step = sync_step
        cls.log_checkpoint = log_checkpoint

        return cls(tracking_uri=cls.experiment.url)

    @classmethod
    def end_run(cls):
        if cls.experiment is not None:
            # Global step saving for future resuming
            cls.offset_step = cls.experiment.step
            # Send all checkpoints to WB server
            if cls.log_checkpoint:
                wandb.save(os.path.join(cls.save_dir, "*ckpt"))
            cls.experiment.finish()

    def log_metrics(self, metrics, step, **kwargs):
        assert self.experiment is not None, "Initialize experiment first by calling `WANDBLogger.init_experiment(...)`"
        metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
        if self.sync_step and step + self.offset_step < self.experiment.step:
            logger.warning("Trying to log at a previous step. Use `sync_step=False`")
        if self.sync_step:
            self.experiment.log(metrics, step=(step + self.offset_step) if step is not None else None)
        elif step is not None:
            self.experiment.log({**metrics, 'x': step + self.offset_step}, sync=False, **kwargs)
        else:
            self.experiment.log(metrics)

    def log_params(self, params, **kwargs):
        assert self.experiment is not None, "Initialize experiment first by calling `WANDBLogger.init_experiment(...)`"
        self.experiment.config.update(params, allow_val_change=True)

    def log_artifacts(self, artifacts):
        raise NotImplementedError()


class JUSTLogger(Logger):

    offset_step = 0

    login: bool = False

    @classmethod
    def init_experiment(
        cls,
        experiment_name,
        project_name,
        api: Optional[str] = None,
        notes=None,
        tags=None,
        entity=None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        _id: Optional[str] = None,
        log_checkpoint: Optional[bool] = False,
        sync_step: Optional[bool] = True,
        prefix: Optional[str] = "",
        notebook: Optional[str] = None,
        **kwargs,
    ):
        if save_dir is None:
            save_dir = pathlib.Path.home() / "Logging" / f"{project_name}"
            save_dir.mkdir(exist_ok=True, parents=True)

        logging_pth = save_dir / f"{experiment_name}.log"

        if logging_pth.exists():
            logging_pth = save_dir / f"{experiment_name}_{random_name.generate_name()}.log"

        cls.experiment_name = experiment_name
        cls.project_name = project_name
        jogger.add(
            str(save_dir / f"{experiment_name}.log"),
            enqueue=True,
            colorize=False,
            format="{extra[user]} ¬ <level>{message}</level>",
        )

        cls.login = True
        cls.prefix = prefix

        cls.logger = jogger.bind(user=f"{prefix}")

        return cls(tracking_uri=str(logging_pth))

    def log_metrics(self, metrics, step, prefix: str = None):
        assert self.login is True, "Initialize experiment first by calling `JUSTLogger.init_experiment(...)`"
        if prefix is not None:
            self.prefix = prefix
            self.logger = self.logger.bind(user=f"{prefix}")
        metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
        if "x" not in metrics.keys():
            metrics["x"] = step
        # if self.sync_step and step + self.offset_step < self.experiment.step:
        #     logger.warning("Trying to log at a previous step. Use `sync_step=False`")
        # if self.sync_step:
        #     self.experiment.log(metrics, step=(step + self.offset_step) if step is not None else None)
        # elif step is not None:
        #     self.experiment.log({**metrics, 'x': step + self.offset_step}, sync=False, **kwargs)
        # else:
        self.logger.info(metrics)

    def end_run(self):
        self.login = False
        # self.logger.disable() name param is missing

    def log_params(self, params, **kwargs):
        pass

    def log_artifacts(self, artifacts):
        raise NotImplementedError()
