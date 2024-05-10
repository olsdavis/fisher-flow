from typing import Any
import pickle
import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from lightning import LightningDataModule


class DNAEnhancerDataModule(LightningDataModule):
    """
    DNA Enhancer data module.
    """

    def __init__(
        self,
        dataset: str = "MEL2",
        data_dir: str = "data/enhancer/the_code/General/data/Deep",
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize a `DNAEnhancerDataModule`.

        :param dataset: The dataset to use, choices: "MEL2", "FlyBrain". Defaults to `"MEL2"`.
        :param data_dir: The data directory. Defaults to `"data/text8"`.
        :param train_val_test_split: Not used. The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        assert dataset in ["MEL2", "FlyBrain"], f"Invalid dataset '{dataset}'. Choose from 'MEL2', 'FlyBrain'."
        self.dataset = dataset

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        """Prepare data."""

    def setup(self, stage: str | None = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        all_data = pickle.load(
            open(f"{self.hparams.data_dir}{self.dataset}_data.pkl", "rb")
        )
        for split in ["train", "valid", "test"]:
            data = torch.nn.functional.one_hot(
                torch.from_numpy(all_data[f"{split}_data"]).argmax(dim=-1),
                num_classes=4,
            ).float()
            clss = torch.from_numpy(all_data[f"y_{split}"]).argmax(dim=-1).float()
            print(data.shape, clss.shape)
            dataset = TensorDataset(data, clss)
            setattr(
                self,
                f"data_{split}",
                dataset
            )
        # Divide batch size by the number of devices.
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
        assert self.data_valid
        return DataLoader(
            dataset=self.data_valid,
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
    _ = DNAEnhancerDataModule().prepare_data()
