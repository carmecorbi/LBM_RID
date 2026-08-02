import importlib
import logging
import re
import time
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from ..models.base.base_model import BaseModel
from .training_config import TrainingConfig

logging.basicConfig(level=logging.INFO)


class TrainingPipeline(pl.LightningModule):
    """
    Main Training Pipeline class

    Args:

        model (BaseModel): The model to train
        pipeline_config (TrainingConfig): The configuration for the training pipeline
        verbose (bool): Whether to print logs in the console. Default is False.
    """

    def __init__(
        self,
        model: BaseModel,
        pipeline_config: TrainingConfig,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.pipeline_config = pipeline_config
        self.log_samples_model_kwargs = pipeline_config.log_samples_model_kwargs

        # save hyperparameters.
        self.save_hyperparameters(ignore="model")
        self.save_hyperparameters({"model_config": model.config.to_dict()})

        # logger.
        self.verbose = verbose

    def on_fit_start(self) -> None:
        self.model.on_fit_start(device=self.device)
        if self.global_rank == 0:
            self.timer = time.perf_counter()

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        self.model.on_train_batch_end(batch)

        average_time_frequency = 10
        if self.global_rank == 0 and batch_idx % average_time_frequency == 0:
            delta = time.perf_counter() - self.timer
            logging.info(
                f"Average time per batch {batch_idx} took {delta / (batch_idx + 1)} seconds"
            )

    def configure_optimizers(self):
        lr = self.pipeline_config.learning_rate

        param_list = {"params": []}
        n_params = 0

        for name, param in self.model.named_parameters():
            for regex in self.pipeline_config.trainable_params:
                if re.search(regex, name):
                    param.requires_grad = True
                    param_list["params"].append(param)
                    n_params += param.numel()

        logging.info(f"Number of trainable parameters: {n_params}")

        optimizer_cls = getattr(
            importlib.import_module("torch.optim"),
            self.pipeline_config.optimizer_name,
        )

        optimizer = optimizer_cls(
            [param_list],
            lr=lr,
            **self.pipeline_config.optimizer_kwargs,
        )

        if self.pipeline_config.lr_scheduler_name is None:
            return optimizer

        scheduler_cls = getattr(
            importlib.import_module("torch.optim.lr_scheduler"),
            self.pipeline_config.lr_scheduler_name,
        )

        scheduler = scheduler_cls(
            optimizer,
            **self.pipeline_config.lr_scheduler_kwargs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.pipeline_config.lr_scheduler_interval,
                "monitor": "val/loss",
            },
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch)

        loss = outputs["loss"]
        latent_loss = outputs["latent_recon_loss"]
        #pixel_loss = outputs["pixel_recon_loss"]
        #reconstruction_loss = outputs["reconstruction_loss"]

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/latent_recon_loss", latent_loss, on_step=True, on_epoch=True, logger=True)
        #self.log("train/pixel_recon_loss", pixel_loss, on_step=True, on_epoch=True, logger=True)
        #self.log("train/reconstruction_loss", reconstruction_loss, on_step=True, on_epoch=True, logger=True)

        

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.model(batch)["loss"]

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def log_samples(self, batch: Dict[str, Any]):
        logging.debug("log_samples")
        logs = self.model.log_samples(
            batch,
            **self.log_samples_model_kwargs,
        )

        if logs is not None:
            N = min([logs[keys].shape[0] for keys in logs])
        else:
            N = 0

        # Log inputs
        if self.log_keys is not None:
            for key in self.log_keys:
                if key in batch:
                    if N > 0:
                        logs[key] = batch[key][:N]
                    else:
                        logs[key] = batch[key]

        return logs
