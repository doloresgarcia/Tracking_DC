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

from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from src.utils.parser_args import parser
from src.utils.import_tools import import_module


from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.profilers import AdvancedProfiler

from src.utils.train_utils import (
    train_load,
    test_load,
)
from src.utils.train_utils import get_samples_steps_per_epoch, model_setup, get_gpu_dev
from src.models.Build_graphs import FreezeEFDeepSet

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


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
        # entity=args.wandb_entity,
        name=args.wandb_displayname,
    )
    if args.export_onnx:
        print("exporting to onnx")
        filepath = args.model_prefix + "model_multivector_input_011124_v2.onnx"
        # args1 = (torch.randn((10, 3)), torch.randn((10, 1)), torch.randn((10, 3)))
        torch._dynamo.config.verbose = True
        if args.load_model_weights is not None:
            from src.models.Gatr_v_onnx import ExampleWrapper

            print("adding weights")
            model = ExampleWrapper.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0
            )
        model.eval()
        model.ScaledGooeyBatchNorm2_1.momentum = 0
        args1 = torch.randn((10, 7))
        torch.onnx.export(model, 
                        args1,
                        filepath, 
                        dynamo=True, 
                        # report=True, 
                        # verify=True,       
                        input_names=["input"],
                        output_names=["output"], 
                         dynamic_axes={
                            "input": [0]}) 



        
        # export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        # onnx_program = torch.onnx.dynamo_export(
        #     model, args1, export_options=export_options
        # )
        # onnx_program.save(filepath)
       

    elif training_mode:
        print("USING TRAINING MODE")
        # if args.load_model_weights is not None:
        #     # from src.models.Gatr_v import ExampleWrapper

        #     # model = ExampleWrapper.load_from_checkpoint(
        #     #     args.load_model_weights, args=args, dev=0
        #     # )
        #     from src.models.Edge_filtering import EFDeepSet
        #     EFDeepSet_model = EFDeepSet.load_from_checkpoint(
        #         "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_CLD/models/baseline_gnn/edge_filtering_v1_geo_cuts/_epoch=6_step=25000.ckpt", args=args,
        #         dev=0, strict=False, map_location=torch.device("cuda:0"))  # Load the good clustering
        #     model.EFDeepSet.fcnn = EFDeepSet_model.fcnn

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.model_prefix,  # checkpoints_path, # <--- specify this on the trainer itself for version control
            filename="_{epoch}_{step}",
            every_n_train_steps=1000,
            save_top_k=-1,  # <--- this is important!
            save_weights_only=True,
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            lr_monitor,
        ]
        # only needed for the GNN baseline
        # callbacks.append(FreezeEFDeepSet())
        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=[3],
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            # max_epochs=5,
            strategy="ddp",
            # limit_train_batches=20,
            # limit_train_batches=890,
            limit_val_batches=5,
        )
        args.local_rank = trainer.global_rank
        # train_loader, val_loader, data_config, train_input_names = train_load(args)

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            # ckpt_path=args.load_model_weights,
        )

    elif args.data_test:
        trainer = L.Trainer(
            callbacks=[TQDMProgressBar(refresh_rate=1)],
            accelerator="gpu",
            devices=[0],
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            # limit_val_batches=5,
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
