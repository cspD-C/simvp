import os

import torch

from core.losses import build_loss
from core.utils import (
    dump_args,
    ensure_dir,
    format_namespace,
    print_log,
    resolve_device,
    resolve_output_dir,
    set_seed,
    setup_logging,
)
from datasets import build_dataloaders
from engine import fit, test_model, validate_one_epoch
from models import build_model


class Exp:
    """
    Backward-compatible experiment wrapper.
    New code path uses builder-style modules (datasets/models/engine).
    """

    def __init__(self, args):
        self.args = args
        self.config = vars(args)
        self.device = resolve_device(args)

        self._prepare()
        self._build()

    def _prepare(self):
        set_seed(self.args.seed)
        self.path = resolve_output_dir(self.args)
        ensure_dir(self.path)

        self.checkpoints_path = os.path.join(self.path, "checkpoints")
        ensure_dir(self.checkpoints_path)

        dump_args(self.args, os.path.join(self.path, "model_param.json"))
        setup_logging(os.path.join(self.path, "log.log"))
        print_log(format_namespace(self.args))

    def _build(self):
        self.train_loader, self.vali_loader, self.test_loader = build_dataloaders(self.args)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader

        # Keep model input shape aligned with dataset settings.
        sample_x, _ = self.train_loader.dataset[0]
        self.args.in_shape = [int(v) for v in sample_x.shape]
        dump_args(self.args, os.path.join(self.path, "model_param.json"))

        self.model = build_model(self.args).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        schedule_epochs = self.args.epochs if self.args.epochs > 0 else self.args.max_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.lr,
            steps_per_epoch=len(self.train_loader),
            epochs=schedule_epochs,
        )
        self.criterion = build_loss(self.args.loss)

    def train(self, args=None):
        fit(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.vali_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            device=self.device,
            args=self.args,
            checkpoints_dir=self.checkpoints_path,
        )
        return self.model

    def vali(self, vali_loader):
        val_loss, val_stats = validate_one_epoch(
            model=self.model,
            data_loader=vali_loader,
            criterion=self.criterion,
            device=self.device,
            max_samples=self.args.max_val_samples,
        )
        print_log(
            "vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}".format(
                val_stats["mse"], val_stats["mae"], val_stats["ssim"], val_stats["psnr"]
            )
        )
        return val_loss

    def test(self, args=None):
        save_dir = os.path.join(self.path, "results", self.args.ex_name, "sv")
        metrics = test_model(
            model=self.model,
            test_loader=self.test_loader,
            device=self.device,
            save_dir=save_dir,
        )
        print_log(
            "mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}".format(
                metrics["mse"], metrics["mae"], metrics["ssim"], metrics["psnr"]
            )
        )
        return metrics["mse"]
