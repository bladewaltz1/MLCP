import logging
import os

import torch


class Checkpointer(object):

    def __init__(self, model, optimizer=None, scheduler=None, loss_scaler=None,
                 save_dir="", save_to_disk=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_scaler = loss_scaler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger("Checkpointer")
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        if not self.save_to_disk:
            return

        data = {}
        if self.model is not None:
            data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        if self.loss_scaler is not None:
            data["loss_scaler"] = self.loss_scaler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, path, model_only=False, is_strict=True):
        if os.path.exists(path):
            self.logger.info("Loading checkpoint from {}".format(path))
            checkpoint = torch.load(path, map_location=torch.device("cpu"))

            if "model" in checkpoint and self.model:
                self.model.load_state_dict(checkpoint.pop("model"),
                                            strict=is_strict)
            if model_only:
                checkpoint.pop("optimizer", None)
                checkpoint.pop("scheduler", None)
                checkpoint.pop("loss_scaler", None)
                return checkpoint

            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(path))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))

            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(path))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

            if "loss_scaler" in checkpoint and self.loss_scaler:
                self.logger.info("Loading loss_scaler from {}".format(path))
                self.loss_scaler.load_state_dict(checkpoint.pop("loss_scaler"))

            return checkpoint
        else:
            self.logger.info("No checkpoint found.")
        return {}
