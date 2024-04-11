#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import wandb
import warnings

# warnings.filterwarnings("ignore")

from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from src.utils.parser_args import parser
from lightning.pytorch.loggers import WandbLogger

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.utils.train_utils import (
    train_load,
    test_load,
)
from src.utils.import_tools import import_module
import wandb

from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.profilers import AdvancedProfiler
from src.utils.train_utils import get_samples_steps_per_epoch, model_setup, get_gpu_dev


def main():
    args = parser.parse_args()
    args = get_samples_steps_per_epoch(args)
    args.local_rank = 0
    training_mode = not args.predict
    if training_mode:
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    model = model_setup(args, data_config)
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
    else:
        print("No GPUs flag provided - Setting GPUs to [0]")
        gpus = [0]
    wandb_logger = WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
    )

    if args.export_onnx:
        print("exporting to onnx")
        filepath = args.model_prefix + "model.onnx"
        input_sample = torch.randn((1, 7))
        onnx_program = torch.onnx.dynamo_export(model, input_sample)
        onnx_program.save(filepath)
        # model.to_onnx(filepath, input_sample, export_params=True, verbose=True)

    elif training_mode:
        if args.load_model_weights is not None:
            from src.models.Gatr import ExampleWrapper

            model = ExampleWrapper.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0
            )
        val_every_n_epochs = 1

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.model_prefix,  # checkpoints_path, # <--- specify this on the trainer itself for version control
            filename="_{epoch}",
            every_n_epochs=val_every_n_epochs,
            save_top_k=-1,  # <--- this is important!
            save_weights_only=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            lr_monitor,
        ]
        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=[0, 1, 2, 3],
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            max_epochs=1,
            strategy="ddp",
            # limit_train_batches=20,
            limit_train_batches=20,
            limit_val_batches=20,
        )
        args.local_rank = trainer.global_rank
        train_loader, val_loader, data_config, train_input_names = train_load(args)

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            # ckpt_path=args.load_model_weights,
        )

    if args.data_test:
        trainer = L.Trainer(
            callbacks=[TQDMProgressBar(refresh_rate=1)],
            accelerator="gpu",
            devices=[0],
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            limit_val_batches=5,
        )
        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            trainer.validate(
                model=model,
                ckpt_path=args.load_model_weights,
                dataloaders=test_loader,
            )


if __name__ == "__main__":
    main()
