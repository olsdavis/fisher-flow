from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
import dgl

from src.data.components import MoleculeDataset

"""
test module loading:

python -m src.data.qm9_datamodule
"""

class MoleculeDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_config: dict,
        dm_prior_config: dict,
        batch_size: int,
        num_workers: int = 0,
        distributed: bool = False,
        max_num_edges: int = 40000,
    ):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prior_config = dm_prior_config
        self.max_num_edges = max_num_edges
        self.save_hyperparameters()

    def prepare_data(self) -> None:

        self.train_dataset = MoleculeDataset(
            'train', 
            self.dataset_config, 
            prior_config=self.prior_config,
        )

        self.val_dataset = MoleculeDataset(
            'val', 
            self.dataset_config, 
            prior_config=self.prior_config,
        )

        self.test_dataset = MoleculeDataset(
            'test', 
            self.dataset_config, 
            prior_config=self.prior_config,
        )
    
    def setup(self, stage: str | None = None) -> None:
        self.prepare_data()

    def train_dataloader(self):
        assert self.train_dataset
        dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=dgl.batch, 
            num_workers=self.num_workers,
        )
        return dataloader
    
        # # i wrote the following code under the assumption that we had to align predictions to the target, but i don't think this is true
        # if self.x_subspace == 'se3-quotient':
        #     # if we are using the se3-quotient subspace, then we need to do same-size sampling so that we can efficiently compute rigid aligments during training
        #     if self.distributed:
        #         batch_sampler = SameSizeDistributedMoleculeSampler(self.train_dataset, batch_size=self.batch_size, max_num_edges=self.max_num_edges)
        #     else:
        #         batch_sampler = SameSizeMoleculeSampler(self.train_dataset, batch_size=self.batch_size, max_num_edges=self.max_num_edges)

        #     dataloader = DataLoader(dataset=self.train_dataset, batch_sampler=batch_sampler, collate_fn=dgl.batch, num_workers=self.num_workers)

        # elif self.x_subspace == 'com-free':
        #     # if we are using the com-free subspace, then we don't need to do same-size sampling - and life is easier!
        #     dataloader = DataLoader(self.train_dataset, 
        #                             batch_size=self.batch_size, 
        #                             shuffle=True, 
        #                             collate_fn=dgl.batch, 
        #                             num_workers=self.num_workers)

                
        # return dataloader

    def test_dataloader(self):
        assert self.test_dataset
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size*2, 
            shuffle=True,
            collate_fn=dgl.batch,
            num_workers=self.num_workers,
        )
        return dataloader
    
    def val_dataloader(self):
        assert self.val_dataset
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size*2, 
            shuffle=True,
            collate_fn=dgl.batch,
            num_workers=self.num_workers,
        )
        return dataloader

        # if self.x_subspace == 'se3-quotient':
        #     # if we are using the se3-quotient subspace, then we need to do same-size sampling so that we can efficiently compute rigid aligments during training
        #     if self.distributed:
        #         batch_sampler = SameSizeDistributedMoleculeSampler(self.train_dataset, batch_size=self.batch_size*2)
        #     else:
        #         batch_sampler = SameSizeMoleculeSampler(self.train_dataset, batch_size=self.batch_size*2)

        #     dataloader = DataLoader(dataset=self.train_dataset, batch_sampler=batch_sampler, collate_fn=dgl.batch, num_workers=self.num_workers)

        # elif self.x_subspace == 'com-free':
        #     # if we are using the com-free subspace, then we don't need to do same-size sampling - and life is easier!
        #     dataloader = DataLoader(self.train_dataset, 
        #                             batch_size=self.batch_size*2, 
        #                             shuffle=True, 
        #                             collate_fn=dgl.batch, 
        #                             num_workers=self.num_workers)
        # return dataloader

if __name__ == "__main__":
    dataset_config = {
        'processed_data_dir': 'data/qm9',
        'raw_data_dir': 'data/qm9_raw',
        'dataset_name': 'qm9',
    }
    prior_config = {
        'a': {
            'align': False,
            'kwargs': {},
            'type': 'gaussian',
        },
        'c': {
            'align': False,
            'kwargs': {},
            'type': 'gaussian',
        },
        'e': {
            'align': False,
            'kwargs': {},
            'type': 'gaussian',
        },
        'x': {
            'align': True,
            'kwargs': {},
            'std': 1.0,
            'type': 'centered-normal',
        },
    }
    mod = MoleculeDataModule(dataset_config, prior_config, 32, 0, False, 40000)
    mod.prepare_data()
    mod.setup()
    data_loader = mod.train_dataloader()
    x = next(iter(data_loader))
    print(type(x))
    print(x)