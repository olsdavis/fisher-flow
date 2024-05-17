from typing import Any
import pickle
import os
import copy
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from lightning import LightningDataModule

"""
test datamodule

python -m src.data.dna_enhancer_datamodule
"""

class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset, split='train'):
        assert dataset in ["MEL2", "FlyBrain"], f"Invalid dataset '{dataset}'. Choose from 'MEL2', 'FlyBrain'."
        all_data = pickle.load(open(f'{data_dir}{dataset}_data.pkl', 'rb'))
        self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1) # NOT one-hot encoded sequences
        self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'y_{split}'])), dim=-1) # NOT one-hot encoded classes
        self.num_cls = all_data[f'y_{split}'].shape[-1]
        self.alphabet_size = 4

        print(f"alphabet_size {self.alphabet_size}")
        print(f"num_cls {self.num_cls}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx] # MEL2 [B, 500, 4], [B, num_cls]

class DNAEnhancerDataModule(LightningDataModule):
    """
    DNA Enhancer data module.
    """

    def __init__(
        self,
        dataset: str = "MEL2", # choices: "MEL2", "FlyBrain"
        data_dir: str = "data/the_code/General/data/Deep",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        subset_train_as_val: bool = False,
    ):
        """Initialize a `DNAEnhancerDataModule`.

        :param dataset: The dataset to use, choices: "MEL2", "FlyBrain". Defaults to `"MEL2"`.
        :param data_dir: The data directory. Defaults to `"data/enhancer/"`.
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
        self.subset_train_as_val = subset_train_as_val # use a subset of the training set as validation set

    def prepare_data(self):
        """Prepare data."""
        splits = ["train", "valid", "test"]
        names = ["train", "val", "test"]
        for name, split in zip(names, splits):
            dataset = EnhancerDataset(self.hparams.data_dir, self.dataset, split=split)
            setattr(
                self,
                f"data_{name}",
                dataset
            )
            print(f"{name} len {len(dataset)}")
        
        if self.subset_train_as_val:
            val_set_size = len(self.data_val)
            self.data_val = Subset(self.data_train, torch.randperm(len(self.data_train))[:val_set_size])

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
            shuffle=True,
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

    # def teardown(self, stage: str | None = None):
    #     """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
    #     `trainer.test()`, and `trainer.predict()`.

    #     :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
    #         Defaults to ``None``.
    #     """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    # def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    #     """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
    #     `state_dict()`.

    #     :param state_dict: The datamodule state returned by `self.state_dict()`.
    #     """


if __name__ == "__main__":
    data_module = DNAEnhancerDataModule()
    data_module.prepare_data()
    train_dataloader = data_module.train_dataloader()
    batch = next(iter(train_dataloader))
    print(f"xs: {batch[0].shape}")
    print(f"ys: {batch[1].shape}")

