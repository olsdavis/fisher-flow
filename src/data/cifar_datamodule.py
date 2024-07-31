from typing import Any
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning import LightningDataModule


class CIFARDataModule(LightningDataModule):
    """
    CIFAR-10 data module.
    """

    def __init__(
        self, 
        data_dir: str = "data/cifar10",
        one_hot: bool = True,
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def prepare_data(self) -> None:
        """Download dataset."""
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def _discretize_transform(self, x: Any) -> Tensor:
        """Discretize the input tensor.

        :param x: The input tensor.
        :return: The discretized tensor.
        """
        x = T.ToTensor()(x)
        x = (x * 255).long()
        x = F.one_hot(x, num_classes=256).float()
        return x

    def setup(self, stage: str | None = None):
        """Split the dataset into train, validation and test sets.

        :param stage: The stage being setup. Either `"fit"`, `"validate"`, `"test"`, or `None`.
        """
        self.data_train = CIFAR10(self.hparams.data_dir, train=True, transform=self._discretize_transform)
        self.data_val = CIFAR10(self.hparams.data_dir, train=False, transform=self._discretize_transform)
        self.data_test = CIFAR10(self.hparams.data_dir, train=False, transform=self._discretize_transform)
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert self.data_val
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.data_test
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def get_all_test_set(self) -> Tensor:
        """Get the entire test set, encoded with floats between 0 and 1.

        :return: The entire test set.
        """
        assert self.data_test
        return torch.stack([x.reshape(256, 3, 32, 32).argmax(dim=0).float() / 255.0 for x, _ in self.data_test])

    def teardown(self, stage: str | None = None):
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """


if __name__ == "__main__":
    dm = CIFARDataModule()
    dm.prepare_data()
    dm.setup()
    print(dm.train_dataloader())
    print(dm.val_dataloader())
    print(dm.test_dataloader())
