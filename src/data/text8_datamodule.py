from typing import Any
import pickle
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from src.sfm import manifold_from_name


class Text8Dataset(torch.utils.data.Dataset):
    """
    Adapted from `https://github.com/andrew-cr/discrete_flow_models/blob/main/train.py`
    """
    def __init__(self, manifold, dataset, vocab_size: int, block_size: int, split: str = 'train'):
        super().__init__()
        self.m = manifold
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.dataset[idx + self.block_size].astype(np.int64))
        one_hot = nn.functional.one_hot(sample, self.vocab_size).float()
        # if there is a need to smooth labels, it is done in the model's training step
        yield one_hot.squeeze()


class Text8DataModule(LightningDataModule):
    """
    Text8 data module.
    """

    def __init__(
        self,
        k: int = 4,
        data_dir: str = "data/text8",
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize a `Text8DataModule`.

        :param data_dir: The data directory. Defaults to `"data/text8"`.
        :param train_val_test_split: Not used. The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # corresponds to window size
        self.k = k

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        """Nothing to download."""
        data_dir = "./data"
        meta_path = os.path.join(data_dir, 'meta_text8.pkl')
        assert os.path.exists(meta_path)
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
        self.meta_vocab_size = self.meta['vocab_size']
        print(f"found vocab_size = {self.meta_vocab_size} (inside {meta_path})")

        self.stoi = self.meta['stoi']
        self.itos = self.meta['itos']

        # increase vocab size by 1 to include a mask token
        self.meta_vocab_size += 1
        self.mask_token_id = self.meta_vocab_size - 1
        self.stoi['X'] = self.mask_token_id
        self.itos[self.mask_token_id] = 'X'
        self.data_train_base = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.data_val_base = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

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
            self.data_train = Text8Dataset(
                manifold,
                self.data_train_base,
                self.meta_vocab_size,
                self.k,
                split = 'train'
            )
            self.data_val = Text8Dataset(
                manifold,
                self.data_val_base,
                self.meta_vocab_size,
                self.k,
                split = 'val'
            )

            # There is no test set so we set it to val
            self.data_test = self.data_val

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
    _ = Text8DataModule().prepare_data()
