# From https://github.com/Dunni3/FlowMol/tree/main
import argparse
import atexit
import json
import pickle
import signal
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import tqdm
import yaml
from rdkit import Chem
from multiprocessing import Pool
import pandas as pd

from torch.nn.functional import one_hot
from multiprocessing import Pool

class MoleculeFeaturizer():

    def __init__(self, atom_map: str, n_cpus=1):
        self.n_cpus = n_cpus
        self.atom_map = atom_map
        self.atom_map_dict = {atom: i for i, atom in enumerate(atom_map)}

        if self.n_cpus == 1:
            self.pool = None
        else:
            self.pool = Pool(self.n_cpus)

        if 'H' in atom_map:
            self.explicit_hydrogens = True
        else:    
            self.explicit_hydrogens = False

    def featurize_molecules(self, molecules):


        all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs = [], [], [], [], []
        all_bond_order_counts = torch.zeros(5, dtype=torch.int64)

        if self.n_cpus == 1:
            for molecule in molecules:
                positions, atom_types, atom_charges, bond_types, bond_idxs, bond_order_counts = featurize_molecule(molecule, self.atom_map_dict)
                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

                if bond_order_counts is not None:
                    all_bond_order_counts += bond_order_counts

        else:
            args = [(molecule, self.atom_map_dict) for molecule in molecules]
            results = self.pool.starmap(featurize_molecule, args)
            for positions, atom_types, atom_charges, bond_types, bond_idxs, bond_order_counts in results:
                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

                if bond_order_counts is not None:
                    all_bond_order_counts += bond_order_counts

        # find molecules that failed to featurize and count them
        num_failed = 0
        failed_idxs = []
        for i in range(len(all_positions)):
            if all_positions[i] is None:
                num_failed += 1
                failed_idxs.append(i)

        # remove failed molecules
        all_positions = [pos for i, pos in enumerate(all_positions) if i not in failed_idxs]
        all_atom_types = [atom for i, atom in enumerate(all_atom_types) if i not in failed_idxs]
        all_atom_charges = [charge for i, charge in enumerate(all_atom_charges) if i not in failed_idxs]
        all_bond_types = [bond for i, bond in enumerate(all_bond_types) if i not in failed_idxs]
        all_bond_idxs = [idx for i, idx in enumerate(all_bond_idxs) if i not in failed_idxs]

        return all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs, num_failed, all_bond_order_counts



def featurize_molecule(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int], explicit_hydrogens=True):

    # if explicit_hydrogens is False, remove all hydrogens from the molecule
    if not explicit_hydrogens:
        molecule = Chem.RemoveHs(molecule)

    # get positions
    positions = molecule.GetConformer().GetPositions()
    positions = torch.from_numpy(positions)

    # get atom elements as a string
    # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_types_idx = torch.zeros(molecule.GetNumAtoms()).long()
    atom_charges = torch.zeros_like(atom_types_idx)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            atom_types_idx[i] = atom_map_dict[atom.GetSymbol()]
        except KeyError:
            print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return None, None, None, None, None, None
        
        atom_charges[i] = atom.GetFormalCharge()

    # get atom types as one-hot vectors
    atom_types = one_hot(atom_types_idx, num_classes=len(atom_map_dict)).bool()

    atom_charges = atom_charges.type(torch.int32)

    # get one-hot encoded of existing bonds only (no non-existing bonds)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True))
    edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

    # note that because we take the upper-triangular portion of the adjacency matrix, there is only one edge per bond
    # at training time for every edge (i,j) in edge_index, we will also add edges (j,i)
    # we also only retain existing bonds, but then at training time we will add in edges for non-existing bonds

    bond_types = adj[edge_index[:, 0], edge_index[:, 1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.type(torch.int32)
    # edge_attr = one_hot(bond_types, num_classes=5).bool() # five bond classes: no bond, single, double, triple, aromatic

    # count the number of pairs of atoms which are bonded
    n_bonded_pairs = edge_index.shape[0]

    # compute the number of upper-edge pairs
    n_atoms = atom_types.shape[0]
    n_pairs = n_atoms * (n_atoms - 1) // 2

    # compute the number of pairs of atoms which are not bonded
    n_unbonded = n_pairs - n_bonded_pairs

    # construct an array containing the counts of each bond type in the molecule
    bond_order_idxs, existing_bond_order_counts = torch.unique(edge_attr, return_counts=True)
    bond_order_counts = torch.zeros(5, dtype=torch.int64)
    for bond_order_idx, count in zip(bond_order_idxs, existing_bond_order_counts):
        bond_order_counts[bond_order_idx] = count

    bond_order_counts[0] = n_unbonded

    return positions, atom_types, atom_charges, edge_attr, edge_index, bond_order_counts


def compute_p_c_given_a(atom_charges: torch.Tensor, atom_types: torch.Tensor, atom_type_map: List[str]) -> torch.Tensor:
    """Computes the conditional distribution of charges given atom type, p(c|a)."""
    charge_idx_to_val = torch.arange(-2,4)
    charge_val_to_idx = {int(val): idx for idx, val in enumerate(charge_idx_to_val)}
    
    n_atom_types = len(atom_type_map)
    n_charges = len(charge_idx_to_val)

    # convert atom types from one-hots to indices
    atom_types = atom_types.float().argmax(dim=1)

    # create a tensor to store the conditional distribution of charges given atom type, p(c|a)
    p_c_given_a = torch.zeros(n_atom_types, n_charges, dtype=torch.float32)


    for atom_idx in range(n_atom_types):
        atom_type_mask = atom_types == atom_idx # mask for atoms with the current atom type
        unique_charges, charge_counts = torch.unique(atom_charges[atom_type_mask], return_counts=True)
        for unique_charge, charge_count in zip(unique_charges, charge_counts):
            charge_idx = charge_val_to_idx[int(unique_charge)]
            p_c_given_a[atom_idx, charge_idx] = charge_count

    row_sum = p_c_given_a.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0e-8
    p_c_given_a = p_c_given_a / row_sum
    return p_c_given_a

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(description='Process geometry')
    p.add_argument('--config', type=Path, help='config file path')
    p.add_argument('--chunk_size', type=int, default=1000, help='number of molecules to process at once')

    p.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use when computing partial charges for confomers')
    # p.add_argument('--dataset_size', type=int, default=None, help='number of molecules in dataset, only used to truncate dataset for debugging')

    args = p.parse_args()

    return args

def process_split(split_df, split_name, args, dataset_config):

    # get processed data directory and create it if it doesn't exist
    output_dir = Path(config['dataset']['processed_data_dir'])
    output_dir.mkdir(exist_ok=True) 

    raw_dir = Path(dataset_config['raw_data_dir']) 
    sdf_file = raw_dir / 'gdb9.sdf'
    bad_mols_file = raw_dir / 'uncharacterized.txt'

    # get the molecule ids to skip
    ids_to_skip = set()
    with open(bad_mols_file, 'r') as f:
        lines = f.read().split('\n')[9:-2]
        for x in lines:
            ids_to_skip.add(int(x.split()[0]) - 1)

    # get the molecule ids that are in our split
    mol_idxs_in_split = set(split_df.index.values.tolist())

    dataset_size = dataset_config['dataset_size']
    if dataset_size is None:
        dataset_size = np.inf

    # read all the molecules from the sdf file
    all_molecules = []
    all_smiles = []
    mol_reader = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
    for mol_idx, mol in enumerate(mol_reader):

        # skip molecules that are in the bad_mols_file or not in this split
        if mol_idx in ids_to_skip or mol_idx not in mol_idxs_in_split:
            continue

        all_molecules.append(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles is not None:
            all_smiles.append(smiles)  # Convert mol to smiles string and append to all_smiles

        if len(all_molecules) > dataset_size:
            break


    all_positions = []
    all_atom_types = []
    all_atom_charges = []
    all_bond_types = []
    all_bond_idxs = []
    all_bond_order_counts = torch.zeros(5, dtype=torch.int64)

    mol_featurizer = MoleculeFeaturizer(config['dataset']['atom_map'], n_cpus=args.n_cpus)

    # molecules is a list of rdkit molecules.  now we create an iterator that yields sub-lists of molecules. we do this using itertools:
    chunk_iterator = chunks(all_molecules, args.chunk_size)
    n_chunks = len(all_molecules) // args.chunk_size + 1

    tqdm_iterator = tqdm.tqdm(chunk_iterator, desc='Featurizing molecules', total=n_chunks)
    failed_molecules_bar = tqdm.tqdm(desc="Failed Molecules", unit="molecules")

    # create a tqdm bar to report the total number of molecules processed
    total_molecules_bar = tqdm.tqdm(desc="Total Molecules", unit="molecules", total=len(all_molecules))

    failed_molecules = 0
    for molecule_chunk in tqdm_iterator:

        # TODO: we should collect all the molecules from each individual list into a single list and then featurize them all at once - this would make the multiprocessing actually useful
        positions, atom_types, atom_charges, bond_types, bond_idxs, num_failed, bond_order_counts = mol_featurizer.featurize_molecules(molecule_chunk)

        failed_molecules += num_failed
        failed_molecules_bar.update(num_failed)
        total_molecules_bar.update(len(molecule_chunk))

        all_positions.extend(positions)
        all_atom_types.extend(atom_types)
        all_atom_charges.extend(atom_charges)
        all_bond_types.extend(bond_types)
        all_bond_idxs.extend(bond_idxs)
        all_bond_order_counts += bond_order_counts

    # get number of atoms in every data point
    n_atoms_list = [ x.shape[0] for x in all_positions ]
    n_bonds_list = [ x.shape[0] for x in all_bond_idxs ]

    # convert n_atoms_list and n_bonds_list to tensors
    n_atoms_list = torch.tensor(n_atoms_list)
    n_bonds_list = torch.tensor(n_bonds_list)

    # concatenate all_positions and all_features into single arrays
    all_positions = torch.concatenate(all_positions, dim=0)
    all_atom_types = torch.concatenate(all_atom_types, dim=0)
    all_atom_charges = torch.concatenate(all_atom_charges, dim=0)
    all_bond_types = torch.concatenate(all_bond_types, dim=0)
    all_bond_idxs = torch.concatenate(all_bond_idxs, dim=0)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
    node_idx_array = torch.zeros((len(n_atoms_list), 2), dtype=torch.int32)
    node_idx_array[:, 1] = torch.cumsum(n_atoms_list, dim=0)
    node_idx_array[1:, 0] = node_idx_array[:-1, 1]

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
    edge_idx_array = torch.zeros((len(n_bonds_list), 2), dtype=torch.int32)
    edge_idx_array[:, 1] = torch.cumsum(n_bonds_list, dim=0)
    edge_idx_array[1:, 0] = edge_idx_array[:-1, 1]

    all_positions = all_positions.type(torch.float32)
    all_atom_charges = all_atom_charges.type(torch.int32)
    all_bond_idxs = all_bond_idxs.type(torch.int32)

    # create a dictionary to store all the data
    data_dict = {
        'smiles': all_smiles,
        'positions': all_positions,
        'atom_types': all_atom_types,
        'atom_charges': all_atom_charges,
        'bond_types': all_bond_types,
        'bond_idxs': all_bond_idxs,
        'node_idx_array': node_idx_array,
        'edge_idx_array': edge_idx_array,
    }

    # determine output file name and save the data_dict there
    output_file = output_dir / f'{split_name}_processed.pt'
    torch.save(data_dict, output_file)

    # create histogram of number of atoms
    n_atoms, counts = torch.unique(n_atoms_list, return_counts=True)
    histogram_file = output_dir / f'{split_name}_n_atoms_histogram.pt'
    torch.save((n_atoms, counts), histogram_file)

    # compute the marginal distribution of atom types, p(a)
    p_a = all_atom_types.sum(dim=0)
    p_a = p_a / p_a.sum()

    # compute the marginal distribution of bond types, p(e)
    p_e = all_bond_order_counts / all_bond_order_counts.sum()

    # compute the marginal distirbution of charges, p(c)
    charge_vals, charge_counts = torch.unique(all_atom_charges, return_counts=True)
    p_c = torch.zeros(6, dtype=torch.float32)
    for c_val, c_count in zip(charge_vals, charge_counts):
        p_c[c_val+2] = c_count
    p_c = p_c / p_c.sum()

    # compute the conditional distribution of charges given atom type, p(c|a)
    p_c_given_a = compute_p_c_given_a(all_atom_charges, all_atom_types, dataset_config['atom_map'])

    # save p(a), p(e) and p(c|a) to a file
    marginal_dists_file = output_dir / f'{split_name}_marginal_dists.pt'
    torch.save((p_a, p_c, p_e, p_c_given_a), marginal_dists_file)

    # write all_smiles to its own file
    smiles_file = output_dir / f'{split_name}_smiles.pkl'
    with open(smiles_file, 'wb') as f:
        pickle.dump(all_smiles, f)


if __name__ == "__main__":

    # parse command-line args
    args = parse_args()

    # load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_config = config['dataset']
    if dataset_config['dataset_name'] != 'qm9':
        raise ValueError('This script only works with the qm9 dataset')

    ##########3
    # this must be changed for the qm9 dataset
    ############3

    # get qm9 csv file as a pandas dataframe
    qm9_csv_file = Path(dataset_config['raw_data_dir']) / 'gdb9.sdf.csv'
    df = pd.read_csv(qm9_csv_file)

    n_samples = df.shape[0]
    n_train = 100000
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    # print the number of samples in each split
    print(f"Number of samples in train split: {n_train}")
    print(f"Number of samples in test split: {n_test}")
    print(f"Number of samples in val split: {n_val}")

    # Shuffle dataset with df.sample, then split
    train, val, test = np.split(df.sample(frac=1, random_state=42), [n_train, n_val + n_train])
    

    split_names = ['train_data', 'val_data', 'test_data']
    for split_df, split_name in zip([train, val, test], split_names):
        process_split(split_df, split_name, args, dataset_config)