import os
from typing import Any, Sequence
import subprocess
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning import LightningDataModule

from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from src.data import retrobridge_utils


from src.sfm import manifold_from_name

DOWNLOAD_URL_TEMPLATE = 'https://zenodo.org/record/8114657/files/{fname}?download=1'
USPTO_MIT_DOWNLOAD_URL = 'https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip'

"""
Test to see whether we can load the data:

python -m src.data.retrobridge_datamodule

"""

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

class RetroBridgeDatasetInfos:
    atom_encoder = {
        'N': 0, 'C': 1, 'O': 2, 'S': 3, 'Cl': 4, 'F': 5, 'B': 6, 'Br': 7, 'P': 8,
        'Si': 9, 'I': 10, 'Sn': 11, 'Mg': 12, 'Cu': 13, 'Zn': 14, 'Se': 15, '*': 16,
    }
    atom_decoder = ['N', 'C', 'O', 'S', 'Cl', 'F', 'B', 'Br', 'P', 'Si', 'I', 'Sn', 'Mg', 'Cu', 'Zn', 'Se', '*']
    max_n_dummy_nodes = 10

    def __init__(self, datamodule):
        self.name = 'USPTO50K-RetroBridge'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True
        self.max_weight = 1000
        self.possible_num_dummy_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.valencies = None
        self.atom_weights = None

        self.dummy_nodes_dist = None
        self.n_nodes = None
        self.max_n_nodes = None
        self.node_types = None
        self.edge_types = None
        self.valency_distribution = None
        self.nodes_dist = None

        self.init_attributes(datamodule)

    def init_attributes(self, datamodule):
        self.valencies = [5, 4, 6, 6, 7, 1, 3, 7, 5, 4, 7, 4, 2, 4, 2, 6, 0]
        self.atom_weights = {
            1: 14.01, 2: 12.01, 3: 16., 4: 32.06, 5: 35.45, 6: 19., 7: 10.81, 8: 79.91, 9: 30.98,
            10: 28.01, 11: 126.9, 12: 118.71, 13: 24.31, 14: 63.55, 15: 65.38, 16: 78.97, 17: 0.0
        }

        if datamodule.extra_nodes:
            info_dir = f'{datamodule.data_root}/info_retrobridge_extra_nodes'
        else:
            info_dir = f'{datamodule.data_root}/info_retrobridge'

        os.makedirs(info_dir, exist_ok=True)

        if datamodule.evaluation and os.path.exists(f'{info_dir}/dummy_nodes_dist.txt'):
            self.dummy_nodes_dist = torch.tensor(np.loadtxt(f'{info_dir}/dummy_nodes_dist.txt'))
            self.n_nodes = torch.tensor(np.loadtxt(f'{info_dir}/n_counts.txt'))
            self.max_n_nodes = len(self.n_nodes) - 1
            self.node_types = torch.tensor(np.loadtxt(f'{info_dir}/atom_types.txt'))
            self.edge_types = torch.tensor(np.loadtxt(f'{info_dir}/edge_types.txt'))
            self.valency_distribution = torch.tensor(np.loadtxt(f'{info_dir}/valencies.txt'))
            self.nodes_dist = retrobridge_utils.DistributionNodes(self.n_nodes)
        else:
            self.dummy_nodes_dist = datamodule.dummy_atoms_counts(self.max_n_dummy_nodes)
            print("Distribution of number of dummy nodes", self.dummy_nodes_dist)
            np.savetxt(f'{info_dir}/dummy_nodes_dist.txt', self.dummy_nodes_dist.numpy())

            self.n_nodes = datamodule.node_counts()
            self.max_n_nodes = len(self.n_nodes) - 1
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(f'{info_dir}/n_counts.txt', self.n_nodes.numpy())

            self.node_types = datamodule.node_types()
            print("Distribution of node types", self.node_types)
            np.savetxt(f'{info_dir}/atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(f'{info_dir}/edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(f'{info_dir}/valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
            self.nodes_dist = retrobridge_utils.DistributionNodes(self.n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features, use_context):
        example_batch = next(iter(datamodule.train_dataloader()))
        r_ex_dense, r_node_mask = retrobridge_utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch
        )
        p_ex_dense, p_node_mask = retrobridge_utils.to_dense(
            example_batch.p_x,
            example_batch.p_edge_index,
            example_batch.p_edge_attr,
            example_batch.batch
        )
        assert torch.all(r_node_mask == p_node_mask)

        p_example_data = {
            'X_t': p_ex_dense.X,
            'E_t': p_ex_dense.E,
            'y_t': example_batch['y'],
            'node_mask': p_node_mask
        }

        self.input_dims = {
            'X': example_batch['x'].size(1),
            'E': example_batch['edge_attr'].size(1),
            'y': example_batch['y'].size(1) + 1  # + 1 due to time conditioning
        }

        ex_extra_feat = extra_features(p_example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(p_example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        if use_context:
            self.input_dims['X'] += example_batch['x'].size(1)
            self.input_dims['E'] += example_batch['p_edge_attr'].size(1)

        self.output_dims = {
            'X': example_batch['x'].size(1),
            'E': example_batch['edge_attr'].size(1),
            'y': 0
        }

        print('Input dims:')
        for k, v in self.input_dims.items():
            print(f'\t{k} -> {v}')

        print('Output dims:')
        for k, v in self.output_dims.items():
            print(f'\t{k} -> {v}')

class RetroBridgeDataset(InMemoryDataset):
    types = {
        'N': 0, 'C': 1, 'O': 2, 'S': 3, 'Cl': 4, 'F': 5, 'B': 6, 'Br': 7, 'P': 8,
        'Si': 9, 'I': 10, 'Sn': 11, 'Mg': 12, 'Cu': 13, 'Zn': 14, 'Se': 15, '*': 16,
    }

    bonds = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }

    def __init__(self, stage, root, extra_nodes=False, swap=False):
        self.stage = stage
        self.extra_nodes = extra_nodes

        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        elif self.stage == 'test':
            self.file_idx = 2
        else:
            raise NotImplementedError

        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

        if swap:
            self.data = Data(
                x=self.data.p_x, edge_index=self.data.p_edge_index, edge_attr=self.data.p_edge_attr,
                p_x=self.data.x, p_edge_index=self.data.edge_index, p_edge_attr=self.data.edge_attr,
                y=self.data.y, idx=self.data.idx, r_smiles=self.data.p_smiles, p_smiles=self.data.r_smiles,
            )
            self.slices = {
                'x': self.slices['p_x'],
                'edge_index': self.slices['p_edge_index'],
                'edge_attr': self.slices['p_edge_attr'],
                'y': self.slices['y'],
                'idx': self.slices['idx'],
                'p_x': self.slices['x'],
                'p_edge_index': self.slices['edge_index'],
                'p_edge_attr': self.slices['edge_attr'],
                'r_smiles': self.slices['p_smiles'],
                'p_smiles': self.slices['r_smiles'],
            }

    @property
    def processed_dir(self) -> str:
        if self.extra_nodes:
            return os.path.join(self.root, f'processed_retrobridge_extra_nodes')
        else:
            return os.path.join(self.root, f'processed_retrobridge')

    @property
    def raw_file_names(self):
        return ['uspto50k_train.csv', 'uspto50k_val.csv', 'uspto50k_test.csv']

    @property
    def split_file_name(self):
        return ['uspto50k_train.csv', 'uspto50k_val.csv', 'uspto50k_test.csv']

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return [f'train.pt', f'val.pt', f'test.pt']

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        for fname in self.raw_file_names:
            print(f'Downloading {fname}')
            url = DOWNLOAD_URL_TEMPLATE.format(fname=fname)
            path = os.path.join(self.raw_dir, fname)
            subprocess.run(f'wget {url} -O {path}', shell=True)

    def process(self):
        table = pd.read_csv(self.split_paths[self.file_idx])
        data_list = []
        for i, reaction_smiles in enumerate(tqdm(table['reactants>reagents>production'].values)):
            reactants_smi, _, product_smi = reaction_smiles.split('>')
            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)
            r_num_nodes = rmol.GetNumAtoms()
            p_num_nodes = pmol.GetNumAtoms()
            assert p_num_nodes <= r_num_nodes

            if self.extra_nodes:
                new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes
                if r_num_nodes > new_r_num_nodes:
                    print(f'Molecule with |r|-|p| > max_n_dummy_nodes: r={r_num_nodes}, p={p_num_nodes}')
                    if self.stage in ['train', 'val']:
                        continue
                    else:
                        reactants_smi, product_smi = 'C', 'C'
                        rmol = Chem.MolFromSmiles(reactants_smi)
                        pmol = Chem.MolFromSmiles(product_smi)
                        p_num_nodes = pmol.GetNumAtoms()
                        new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes

                r_num_nodes = new_r_num_nodes

            try:
                mapping = self.compute_nodes_order_mapping(rmol)
                r_x, r_edge_index, r_edge_attr = self.compute_graph(
                    rmol, mapping, r_num_nodes, types=self.types, bonds=self.bonds
                )
                p_x, p_edge_index, p_edge_attr = self.compute_graph(
                    pmol, mapping, r_num_nodes, types=self.types, bonds=self.bonds
                )
            except Exception as e:
                print(f'Error processing molecule {i}: {e}')
                continue

            if self.stage in ['train', 'val']:
                assert len(p_x) == len(r_x)

            product_mask = ~(p_x[:, -1].bool()).squeeze()
            if len(r_x) == len(p_x) and not torch.allclose(r_x[product_mask], p_x[product_mask]):
                print(f'Incorrect atom mapping {i}')
                continue

            if self.stage == 'train' and len(p_edge_attr) == 0:
                continue

            # Shuffle nodes to avoid leaking
            if len(p_x) == len(r_x):
                new2old_idx = torch.randperm(r_num_nodes).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(r_num_nodes)

                r_x = r_x[new2old_idx]
                r_edge_index = torch.stack([old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]], dim=0)
                r_edge_index, r_edge_attr = self.sort_edges(r_edge_index, r_edge_attr, r_num_nodes)

                p_x = p_x[new2old_idx]
                p_edge_index = torch.stack([old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]], dim=0)
                p_edge_index, p_edge_attr = self.sort_edges(p_edge_index, p_edge_attr, r_num_nodes)

                product_mask = ~(p_x[:, -1].bool()).squeeze()
                assert torch.allclose(r_x[product_mask], p_x[product_mask])

            y = torch.zeros(size=(1, 0), dtype=torch.float)
            data = Data(
                x=r_x, edge_index=r_edge_index, edge_attr=r_edge_attr, y=y, idx=i,
                p_x=p_x, p_edge_index=p_edge_index, p_edge_attr=p_edge_attr,
                r_smiles=reactants_smi, p_smiles=product_smi,
            )

            data_list.append(data)

        print(f'Dataset contains {len(data_list)} reactions')
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    @staticmethod
    def compute_graph(molecule, mapping, max_num_nodes, types, bonds):
        max_num_nodes = max(molecule.GetNumAtoms(), max_num_nodes)  # in case |reactants|-|product| > max_n_dummy_nodes
        type_idx = [len(types) - 1] * max_num_nodes
        for i, atom in enumerate(molecule.GetAtoms()):
            type_idx[mapping[atom.GetAtomMapNum()]] = types[atom.GetSymbol()]

        num_classes = len(types)
        x = F.one_hot(torch.tensor(type_idx), num_classes=num_classes).float()

        row, col, edge_type = [], [], []
        for bond in molecule.GetBonds():
            start_atom_map_num = molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
            end_atom_map_num = molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
            start, end = mapping[start_atom_map_num], mapping[end_atom_map_num]
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()] + 1]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

        return x, edge_index, edge_attr

    @staticmethod
    def compute_nodes_order_mapping(molecule):
        # In case if atomic map numbers do not start from 1
        order = []
        for atom in molecule.GetAtoms():
            order.append(atom.GetAtomMapNum())
        order = {
            atom_map_num: idx
            for idx, atom_map_num in enumerate(sorted(order))
        }
        return order

    @staticmethod
    def sort_edges(edge_index, edge_attr, max_num_nodes):
        if len(edge_attr) != 0:
            perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        return edge_index, edge_attr


class AbstractDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, shuffle):
        super().__init__()
        self.dataloaders = None
        self.input_dims = None
        self.output_dims = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def __getitem__(self, idx):
        return self.dataloaders['train'][idx]

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.dataloaders['train']):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.dataloaders['train']):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def dummy_atoms_counts(self, max_n_dummy_nodes):
        dummy_atoms = np.zeros(max_n_dummy_nodes + 1)
        for data in self.dataloaders['train']:
            batch_counts = scatter(data.p_x[:, -1], data.batch, reduce='sum')
            for cnt in batch_counts.long().detach().cpu().numpy():
                if cnt > max_n_dummy_nodes:
                    continue
                dummy_atoms[cnt] += 1

        return torch.tensor(dummy_atoms) / dummy_atoms.sum()


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)  # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]

                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies

class RetroBridgeDataModule(MolecularDataModule):
    """
    
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        extra_nodes: bool = False,
        evaluation: bool = False,
        swap: bool = False,
    ):
        """Initialize a `RetroBridgeDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param shuffle: Whether to shuffle the train_dataloader. Defaults to `True`.
        :param extra_nodes: TODO: chech is this?. Defaults to `False`.
        :param evaluation: Whether to train on the validation set. TODO: why would you do this? Defaults to `False`.
        :param swap: Whether to swap data.x and data.p_x TODO: what is this? and why would you do this? Defaults to `False`.
        """
        super().__init__(batch_size, num_workers, shuffle)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.data_dir = data_dir
        self.batch_size_per_device = batch_size

        self.evaluation = evaluation
        self.swap = swap
        self.extra_nodes = extra_nodes

        self.train_smiles = []

    def prepare_data(self):
        """Nothing to download."""
        pass

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
        stage = 'val' if self.evaluation else 'train'
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = RetroBridgeDataset(stage=stage, root=self.data_dir, extra_nodes=self.extra_nodes, swap=self.swap)
            self.data_val = RetroBridgeDataset(stage='val', root=self.data_dir, extra_nodes=self.extra_nodes, swap=self.swap)
            self.data_test = RetroBridgeDataset(stage='test', root=self.data_dir, extra_nodes=self.extra_nodes, swap=self.swap)
        
        self.train_smiles = self.data_train.r_smiles

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
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

    def test_dataloader(self) -> DataLoader:
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
    retrobridge_data = RetroBridgeDataModule()
    retrobridge_data.setup()
    train_loader = retrobridge_data.train_dataloader()
    item = next(iter(train_loader))
    print(item)
    print(type(item))


