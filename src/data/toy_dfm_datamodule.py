from typing import Any

import torch
from torch import Tensor, nn
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


from src.sfm import NSphere, manifold_from_name


class ToyDataset(torch.utils.data.IterableDataset):
    """
    Adapted from `https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/dataset.py`.
    """
    def __init__(self, manifold, probs: Tensor, toy_seq_len: int, toy_simplex_dim: int, sz: int = 100_000):
        super().__init__()
        self.m = manifold
        self.sz = sz
        self.seq_len = toy_seq_len
        self.alphabet_size = toy_simplex_dim
        self.probs = probs

    def __len__(self) -> int:
        return self.sz

    def __iter__(self):
        while True:
            sample = torch.multinomial(replacement=True, num_samples=1, input=self.probs)
            one_hot = nn.functional.one_hot(sample, self.alphabet_size).float()
            sample = self.m.smooth_labels(one_hot, 0.99 if isinstance(self.m, NSphere) else 0.9)
            yield sample.squeeze()


class ToyDFMDataModule(LightningDataModule):
    """
    Toy DFM data module.
    """

    def __init__(
        self,
        k: int = 4,
        dim: int = 100,
        data_dir: str = "data/",
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.k = k
        self.dim = dim
        self.probs = torch.softmax(torch.rand(k, dim), dim=-1)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        """Nothing to download."""

    def setup(self, stage: str | None = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            manifold = manifold_from_name(self.hparams.get("manifold", "sphere"))
            self.data_train, self.data_val, self.data_test = (
                ToyDataset(
                    manifold,
                    self.probs,
                    self.k,
                    self.dim,
                    sz,
                ) for sz in self.hparams.train_val_test_split
            )

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
    _ = ToyDFMDataModule()
