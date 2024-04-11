import base64
from dataclasses import dataclass
from io import BytesIO
from os import path
from typing import Dict, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.serve import ServableModule, ServableModuleValidator
from lightning.pytorch.utilities.model_helpers import get_torchvision_model
from PIL import Image as PILImage
from src.models.Gatr_v import ExampleWrapper as LitModule
from src.utils.train_utils import (
    train_load,
    test_load,
)

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


class CIFAR10DataModule(LightningDataModule):
    def train_dataloader(self, *args, **kwargs):
        train_loader, val_loader, data_config, train_input_names = train_load(args)
        return train_loader

    def val_dataloader(self, *args, **kwargs):
        train_loader, val_loader, data_config, train_input_names = train_load(args)
        return val_loader


@dataclass(unsafe_hash=True)
class Image:
    height: Optional[int] = None
    width: Optional[int] = None
    extension: str = "JPEG"
    mode: str = "RGB"
    channel_first: bool = False

    def deserialize(self, data: str) -> torch.Tensor:
        encoded_with_padding = (data + "===").encode("UTF-8")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = PILImage.open(buffer, mode="r")
        if self.height and self.width:
            img = img.resize((self.width, self.height))
        arr = np.array(img)
        return T.ToTensor()(arr).unsqueeze(0)


class Top1:
    def serialize(self, tensor: torch.Tensor) -> int:
        return torch.nn.functional.softmax(tensor).argmax().item()


class ProductionReadyModel(LitModule, ServableModule):
    def configure_payload(self):
        # 1: Access the train dataloader and load a single sample.
        image, _ = self.trainer.train_dataloader.dataset[0]

        # 2: Convert the image into a PIL Image to bytes and encode it with base64
        pil_image = T.ToPILImage()(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("UTF-8")

        return {"body": {"x": img_str}}

    def configure_serialization(self):
        return {"x": Image(224, 224).deserialize}, {"output": Top1().serialize}

    def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"output": self.model(x)}

    def configure_response(self):
        return {"output": 7}


def cli_main():
    cli = LightningCLI(
        ProductionReadyModel,
        CIFAR10DataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        trainer_defaults={
            "accelerator": "cpu",
            "callbacks": [ServableModuleValidator()],
            "max_epochs": 1,
            "limit_train_batches": 5,
            "limit_val_batches": 5,
        },
    )
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
