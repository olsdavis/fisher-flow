import os
from typing import Any, Sequence
import subprocess
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning import LightningDataModule

from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from util import NSphere
from .extra_features import ExtraFeatures, DummyExtraFeatures
from .extra_molecular_features import ExtraMolecularFeatures
from .sampling_metrics import compute_retrosynthesis_metrics, SamplingMolecularMetrics


DOWNLOAD_URL_TEMPLATE = 'https://zenodo.org/record/8114657/files/{fname}?download=1'
USPTO_MIT_DOWNLOAD_URL = 'https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip'


device = torch.device("cuda")
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
            info_dir = f'{datamodule.data_dir}/info_retrobridge_extra_nodes'
        else:
            info_dir = f'{datamodule.data_dir}/info_retrobridge'

        os.makedirs(info_dir, exist_ok=True)

        if datamodule.evaluation and os.path.exists(f'{info_dir}/dummy_nodes_dist.txt'):
            self.dummy_nodes_dist = torch.tensor(np.loadtxt(f'{info_dir}/dummy_nodes_dist.txt'))
            self.n_nodes = torch.tensor(np.loadtxt(f'{info_dir}/n_counts.txt'))
            self.max_n_nodes = len(self.n_nodes) - 1
            self.node_types = torch.tensor(np.loadtxt(f'{info_dir}/atom_types.txt'))
            self.edge_types = torch.tensor(np.loadtxt(f'{info_dir}/edge_types.txt'))
            self.valency_distribution = torch.tensor(np.loadtxt(f'{info_dir}/valencies.txt'))
            self.nodes_dist = DistributionNodes(self.n_nodes)
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
            self.nodes_dist = DistributionNodes(self.n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features, use_context):
        example_batch = next(iter(datamodule.train_dataloader()))
        r_ex_dense, r_node_mask = to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch
        )
        p_ex_dense, p_node_mask = to_dense(
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

        self.setup(stage="train")

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
            datasets = {
            'train' : RetroBridgeDataset(stage=stage, root=self.data_dir, extra_nodes=self.extra_nodes, swap=self.swap),
            'val' : RetroBridgeDataset(stage='val', root=self.data_dir, extra_nodes=self.extra_nodes, swap=self.swap),
            'test' : RetroBridgeDataset(stage='test', root=self.data_dir, extra_nodes=self.extra_nodes, swap=self.swap),
        }
        self.dataloaders = {}
        for split, dataset in datasets.items():
            self.dataloaders[split] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=self.shuffle,
            )

        self.train_smiles = datasets['train'].r_smiles

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        #assert self.data_train
        return self.dataloaders['train']

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        #assert self.data_val
        return self.dataloaders['val']

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        #assert self.data_test
        return self.dataloaders['test']

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


import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from .retrobridge_utils import *
from .diffusion_utils import assert_correctly_masked
from .layers import Xtoy, Etoy, masked_softmax


def str_to_activation(name: str) -> nn.Module:
    """
    Returns the activation function associated to the name `name`.
    """
    acts = {
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(0.01),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
        "swish": nn.SiLU(),
    }
    return acts[name]


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(
        self, 
        n_layers: int, 
        input_dims: dict, 
        hidden_mlp_dims: dict, 
        hidden_dims: dict,
        output_dims: dict, 
        act_fn_in: nn.ReLU(), 
        act_fn_out: nn.ReLU(), 
        addition=True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.addition = addition

        self.act_fn_in = str_to_activation(act_fn_in)
        self.act_fn_out = str_to_activation(act_fn_out)

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims['X'], hidden_mlp_dims['X']), self.act_fn_in,
            nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), self.act_fn_in)
        
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims['E'], hidden_mlp_dims['E']), self.act_fn_in,
            nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), self.act_fn_in)

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims['y'], hidden_mlp_dims['y']), self.act_fn_in,
            nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), self.act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), self.act_fn_out,
            nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), self.act_fn_out,
            nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), self.act_fn_out,
            nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        if self.addition:
            X = (X + X_to_out)
            E = (E + E_to_out)
            y = y + y_to_out

        E = E * diag_mask
        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def symmetrise_edges(E: torch.Tensor) -> torch.Tensor:
    """
    Symmetrises the edges tensor.
    """
    i, j = torch.triu_indices(E.size(1), E.size(2))
    E[:, i, j, :] = E[:, j, i, :]
    return E

def compute_extra_data(noisy_data, context=None, condition_on_t=True):
    """ At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input. """

    global extra_features_calc
    global domain_features

    extra_features = extra_features_calc(noisy_data)
    extra_molecular_features = domain_features(noisy_data)

    extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
    extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
    extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

    if context is not None:
        extra_X = torch.cat((extra_X, context.X), dim=-1)
        extra_E = torch.cat((extra_E, context.E), dim=-1)

    if condition_on_t:
        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)
    return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


def retrobridge_forward(net: nn.Module, noisy_data: dict[str, Any], extra_data: PlaceHolder, node_mask: torch.Tensor) -> torch.Tensor:
    X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
    E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
    y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
    return net(X, E, y, node_mask)


def mask_like_placeholder(node_mask: torch.Tensor, X: torch.Tensor, E: torch.Tensor, collapse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    x_mask = node_mask.unsqueeze(-1)
    e_mask1 = x_mask.unsqueeze(2)
    e_mask2 = x_mask.unsqueeze(1)

    if collapse:
        X = torch.argmax(X, dim=-1)
        E = torch.argmax(E, dim=-1)

        X[node_mask == 0] = - 1
        E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
    else:
        X[node_mask == 0, :] = 0
        E[~((e_mask1 * e_mask2).squeeze(-1)), :] = 0
        E = set_zero_diag(E)
        # assert torch.allclose(E, torch.transpose(E, 1, 2))
    return X, E


def set_zero_diag(E: torch.Tensor) -> torch.Tensor:
    """
    Sets the diagonal of `E` to all zeros. Taken from `retrobridge_utils`.
    """
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


@torch.no_grad()
def sample_molecule(
    manifold,
    model: nn.Module,
    data,
    inference_steps: int = 100,
) -> PlaceHolder:
    """
    Samples reactants given a product contained in `data`.
    """
    # generate molecules
    product, node_mask = to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
    product = product.mask(node_mask)
    X, E, y = (
        product.X,
        product.E,
        torch.empty((node_mask.shape[0], 0), device=device),
    )
    # do joint tangent Euler method
    dt = torch.tensor(1.0 / inference_steps, device=data.x.device)
    t = torch.zeros(data.batch_size, 1, device=data.x.device)
    context = product.clone()
    orig_edge_shape = E.shape
    # Masks for fixed and modifiable nodes  | from Retrobridge
    fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
    modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
    assert torch.all(fixed_nodes | modifiable_nodes)
    # start!
    for i in range(inference_steps):
        noisy_data = {
            "t": t,
            "E_t": E,
            "X_t": X,
            "y_t": y,
            "node_mask": node_mask,
        }
        extra_data = compute_extra_data(noisy_data, context=context)
        pred = retrobridge_forward(model, noisy_data, extra_data, node_mask)
        # prep
        target_edge_shape = (E.size(0), -1, E.size(-1))
        # make a step
        # print(t[0], pred.X.square().sum(dim=(-1, -2)).mean(), pred.E.square().sum(dim=(-1, -2, -3)).mean())
        X = manifold.exp_map(
            X, manifold.make_tangent(X, pred.X) * dt,
        )
        X = X * modifiable_nodes + product.X * fixed_nodes
        E = E.reshape(target_edge_shape)
        E = manifold.exp_map(
            E,
            manifold.make_tangent(
                E,
                pred.E.reshape((pred.E.size(0), -1, pred.E.size(-1))),
            ) * dt,
        )
        # E = self.manifold.masked_projection(E)
        y = pred.y
        t += dt
        E = E.reshape(orig_edge_shape)
        X, E = mask_like_placeholder(node_mask, X, E)
        E = symmetrise_edges(E)

    ret = PlaceHolder(
        X=X,
        E=E,
        y=y,
    ).mask(node_mask, collapse=True)

    # E = self.symmetrise_edges(E)
    return ret

"""
dataset = RetroBridgeDataModule("./data", batch_size=64, num_workers=32, shuffle=True, extra_nodes=True, evaluation=False, swap=False, pin_memory=False)
dataset_infos = RetroBridgeDatasetInfos(dataset)
extra_features_calc = (
    ExtraFeatures("all", dataset_info=dataset_infos)
)
domain_features = (
    DummyExtraFeatures()
)
dataset_infos.compute_input_output_dims(
    datamodule=dataset,
    extra_features=extra_features_calc,
    domain_features=domain_features,
    use_context=True,
)
"""


def retrobridge_eval(manifold, model, dataloader, samples_per_input=5):

    global dataset_infos, dataset
    samples_left_to_generate = 5
    samples_left_to_save = 0# self.samples_to_save
    chains_left_to_save =0# self.chains_to_save

    samples = []
    grouped_samples = []
    # grouped_scores = []
    ground_truth = []

    ident = 0

    for data in tqdm(dataloader, total=samples_left_to_generate // dataloader.batch_size):
        if samples_left_to_generate <= 0:
            break

        data = data.to(device)
        bs = len(data.batch.unique())
        to_generate = bs
        to_save = min(samples_left_to_save, bs)
        chains_save = min(chains_left_to_save, bs)
        batch_groups = []
        # batch_scores = []
        for sample_idx in range(samples_per_input):
            """molecule_list, true_molecule_list, products_list, scores, _, _ = self.sample_batch(
                data=data,
                batch_id=ident,
                batch_size=to_generate,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps_to_save=self.number_chain_steps_to_save,
                sample_idx=sample_idx,
            )"""
            mol_sample = sample_molecule(manifold, model, data, inference_steps=100)
            molecule_list = create_pred_reactant_molecules(mol_sample.X, mol_sample.E, data.batch, to_generate)
            samples.extend(molecule_list)
            batch_groups.append(molecule_list)
            # batch_scores.append(scores)
            if sample_idx == 0:
                ground_truth.extend(
                    create_true_reactant_molecules(data, to_generate)
                )

        ident += to_generate
        samples_left_to_save -= to_save
        samples_left_to_generate -= to_generate
        chains_left_to_save -= chains_save

        # Regrouping sampled reactants for computing top-N accuracy
        for mol_idx_in_batch in range(bs):
            mol_samples_group = []
            mol_scores_group = []
            # for batch_group, scores_group in zip(batch_groups, batch_scores):
            for batch_group in batch_groups:
                mol_samples_group.append(batch_group[mol_idx_in_batch])
                # mol_scores_group.append(scores_group[mol_idx_in_batch])

            assert len(mol_samples_group) == samples_per_input
            grouped_samples.append(mol_samples_group)
            # grouped_scores.append(mol_scores_group)

    to_log = compute_retrosynthesis_metrics(
        grouped_samples=grouped_samples,
        ground_truth=ground_truth,
        atom_decoder=dataset_infos.atom_decoder,
        grouped_scores=None,
    )

    for metric_name, metric in to_log.items():
        # print(f"-- {metric_name}: {metric}")
        wandb.log({metric_name: metric})

    to_log = SamplingMolecularMetrics(dataset_infos, dataset.train_smiles)(samples)
    # val_molecular_metrics(samples)
    for metric_name, metric in to_log.items():
        # print(f"-- {metric_name}: {metric}")
        wandb.log({metric_name: metric})


def run_retrobridge_experiment(args):
    torch.manual_seed(42)
    np.random.seed(42)
    model = GraphTransformer(
        n_layers=5,
        input_dims={'X': 40, 'E': 10, 'y': 12},
        hidden_mlp_dims={'X': 256, 'E': 128, 'y': 128},
        hidden_dims={'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128},
        output_dims={'X': 17, 'E': 5, 'y': 0},
        act_fn_in="relu",
        act_fn_out="relu",
        addition=True,
    )
    global dataset
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manifold = NSphere()
    for epoch in range(100):
        model.train()
        losses = []
        for data in dataset.train_dataloader():
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Getting graphs of reactants (target) and product (context)
            reactants, r_node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch,
            )
            reactants = reactants.mask(r_node_mask)

            product, p_node_mask = to_dense(
                data.p_x, data.p_edge_index, data.p_edge_attr, data.batch,
            )
            product = product.mask(p_node_mask)

            t = torch.rand(data.batch_size, 1, device=data.x.device)

            # product is prior, reactants is targret
            # now, need to train over product of these buggers

            target_edge_shape = (product.E.size(0), -1, product.E.size(-1))
            edges_t = manifold.geodesic_interpolant(
                product.E.reshape(target_edge_shape), reactants.E.reshape(target_edge_shape), t
            )
            edges_target = manifold.log_map(product.E, reactants.E)
            edges_target = manifold.parallel_transport(product.E.reshape(target_edge_shape), edges_t, edges_target.reshape(target_edge_shape))
            edges_t = edges_t.reshape_as(product.E)
            edges_target = edges_target.reshape_as(product.E)
            # symmetrise edges_t, edges_target
            edges_t = symmetrise_edges(edges_t)
            edges_target = symmetrise_edges(edges_target)
            feats_t = manifold.geodesic_interpolant(product.X, reactants.X, t)
            feats_target = manifold.log_map(product.X, reactants.X)
            feats_target = manifold.parallel_transport(product.X, feats_t, feats_target)
            # mask things not included
            feats_t, edges_t = mask_like_placeholder(p_node_mask, feats_t, edges_t)
            feats_target, edges_target = mask_like_placeholder(
                p_node_mask,
                feats_target,
                edges_target,
            )


            # checks
            # assert (torch.isclose(feats_t.square().sum(dim=-1), torch.tensor(1.0)) | torch.isclose(feats_t.square().sum(dim=-1), torch.tensor(0.0))).all()
            # assert (torch.isclose(edges_t.square().sum(dim=-1), torch.tensor(1.0)) | torch.isclose(edges_t.square().sum(dim=-1), torch.tensor(0.0))).all()
            # f_mask = torch.isclose(feats_t.square().sum(dim=-1), torch.tensor(1.0))
            # assert self.manifold.all_belong_tangent(feats_t[f_mask], feats_target[f_mask])
            # flat_e = edges_t.reshape(edges_t.size(0), -1, edges_t.size(-1))
            # e_mask = torch.isclose(flat_e.square().sum(dim=(-1)), torch.tensor(1.0))
            # assert self.manifold.all_belong_tangent(flat_e[e_mask], edges_target.reshape_as(flat_e)[e_mask])


            # define data
            noisy_data = {
                "t": t,
                "E_t": edges_t.reshape_as(product.E),
                "X_t": feats_t,
                "y": product.y,
                "y_t": product.y,
                "node_mask": r_node_mask,
            }

            # Computing extra features + context and making predictions
            context = product.clone()
            extra_data = compute_extra_data(noisy_data, context=context)

            pred = retrobridge_forward(model, noisy_data, extra_data, r_node_mask)
            # have two targets, need two projections
            modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
            fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
            assert torch.all(modifiable_nodes | fixed_nodes)
            loss_x_raw = (
                manifold.make_tangent(feats_t, pred.X) - feats_target
            ) * modifiable_nodes
            # reshape for B, K, D shape
            edges_t = edges_t.reshape(edges_t.size(0), -1, edges_t.size(-1))
            pred_reshaped = pred.E.reshape(pred.E.size(0), -1, pred.E.size(-1))
            loss_edges_raw = (
                manifold.make_tangent(edges_t, pred_reshaped) - edges_target.reshape_as(edges_t)
            )
            # loss_edges_raw = loss_edges_raw.reshape_as(product.E)

            # final_X, final_E = self.mask_like_placeholder(p_node_mask, loss_x_raw, loss_edges_raw)
            loss = (
                loss_x_raw.square().sum(dim=(-1, -2))
                + 5.0 * loss_edges_raw.square().sum(dim=(-1, -2, -3))  # 5.0* done in retrobridge
            ).mean()
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
        print(f"Epoch {epoch:03} -- Loss: {torch.tensor(losses).mean():.5f}")
        wandb.log({"train/loss": torch.tensor(losses).mean()})
        if (epoch + 1) % 5 == 0:
            with torch.inference_mode():
                model.eval()
                retrobridge_eval(manifold, model, dataset.val_dataloader(), samples_per_input=5)
