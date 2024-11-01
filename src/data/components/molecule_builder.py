# Taken from https://github.com/Dunni3/FlowMol
import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
import dgl
from typing import List, Dict
from torch.nn.functional import one_hot

bond_type_map = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]

bond_type_to_idx = { bond_type:idx for idx, bond_type in enumerate(bond_type_map)}


def rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    d = x_0.shape[1]
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = torch.zeros(1, d)
        x_1_mean = torch.zeros(1, d)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=0, keepdim=True)
        x_1_mean = x_1.mean(dim=0, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    H = x_0_c.T.mm(x_1_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    if pre_centered:
        t = torch.zeros(1, d)
    else:
        t = x_1_mean - R.mm(x_0_mean.T).T # has shape (1, D)

    # apply rotation to x_0_c
    x_0_aligned = x_0_c.mm(R.T)

    # move x_0_aligned to its original frame
    x_0_aligned = x_0_aligned + x_0_mean

    # apply the translation
    x_0_aligned = x_0_aligned + t

    return x_0_aligned


class SampledMolecule:

    def __init__(self, g: dgl.DGLGraph, atom_type_map: List[str], traj_frames: Dict[str, torch.Tensor] = None, exclude_charges: bool = False, align_traj: bool = True):
        """Represents a molecule sampled from a model. Converts the DGL graph to an rdkit molecule and keeps all associated information."""

        self.exclude_charges = exclude_charges
        self.align_traj = align_traj
        
        # save the graph
        self.g = g

        self.positions, self.atom_types, self.atom_charges, self.bond_types, self.bond_src_idxs, self.bond_dst_idxs = extract_moldata_from_graph(g, atom_type_map, exclude_charges=exclude_charges)

        self.atom_type_map = atom_type_map
        self.num_atoms = g.num_nodes()
        self.num_atom_types = len(atom_type_map)

        # build rdkit molecule
        self.rdkit_mol = self.build_molecule()

        # compute valencies on every atom
        self.valencies = self.compute_valencies()

        self.traj_frames = traj_frames

        if traj_frames is not None:
            self.traj_mols = self.process_traj_frames(traj_frames) # convert frames into a list of rdkit molecules

    @classmethod
    def from_rdkit_mol(cls, mol: Chem.Mol, atom_type_map: List[str] = None):
        """Creates a SampledMolecule from an rdkit molecule."""

        if atom_type_map is None:
            atom_types = [ atom.GetSymbol() for atom in mol.GetAtoms() ]
            atom_type_map = list(set(atom_types))

        atom_type_to_idx = {atom_type: i for i, atom_type in enumerate(atom_type_map)}

        # get positions
        positions = mol.GetConformer().GetPositions()
        positions = torch.from_numpy(positions)

        # get atom elements as a string
        # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        atom_types_idx = torch.zeros(mol.GetNumAtoms()).long()
        atom_charges = torch.zeros_like(atom_types_idx)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_types_idx[i] = atom_type_to_idx[atom.GetSymbol()]
            atom_charges[i] = atom.GetFormalCharge()

        # get one-hot encoded of existing bonds only (no non-existing bonds)
        adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
        edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

        bond_types = adj[edge_index[:, 0], edge_index[:, 1]]
        bond_src_idxs = edge_index[:, 0]
        bond_dst_idxs = edge_index[:, 1]
        bond_types[bond_types == 1.5] = 4
        edge_attr = bond_types.type(torch.int32)

        # create graph
        g = dgl.graph((bond_src_idxs, bond_dst_idxs), num_nodes=mol.GetNumAtoms())

        # add data to the graph
        g.ndata['x_1'] = positions
        g.ndata['a_1'] = one_hot(atom_types_idx.long(), num_classes=len(atom_type_map)).float()
        g.ndata['c_1'] = one_hot(atom_charges.long() + 2, num_classes=6).float()
        g.edata['e_1'] = one_hot(edge_attr.long(), num_classes=5).float()
        g.edata['ue_mask'] = torch.ones(g.num_edges()).bool()

        return cls(g, atom_type_map=atom_type_map)
    
    # this code is adapted from MiDi: https://github.com/cvignac/MiDi/blob/ba07fc5b1313855c047ba0b90e7aceae47e34e38/midi/analysis/rdkit_functions.py
    def build_molecule(self):
        mol = build_molecule(self.positions, self.atom_types, self.atom_charges, self.bond_src_idxs, self.bond_dst_idxs, self.bond_types)
        return mol
    
    def compute_valencies(self):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""
        adj = torch.zeros((self.num_atoms, self.num_atoms))
        adjusted_bond_types = self.bond_types.clone()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adjusted_bond_types = adjusted_bond_types.float()
        adj[self.bond_src_idxs, self.bond_dst_idxs] = adjusted_bond_types
        adj[self.bond_dst_idxs, self.bond_src_idxs] = adjusted_bond_types
        valencies = torch.sum(adj, dim=-1).long()
        return valencies
    
    def process_traj_frames(self, traj_frames: Dict[str, torch.Tensor]):
        """Converts the trajectory frames to a list of rdkit molecules."""
        # convert the frames to a list of rdkit molecules
        g_dummy = copy_graph(self.g)

        n_frames = traj_frames['x'].shape[0]

        x_final = traj_frames['x'][-1] # has shape (n_atoms, 3)

        traj_mols = []
        for frame_idx in range(n_frames):

            # put current frame data into graph
            for feat in traj_frames.keys():
                if feat == 'e':
                    g_dummy.edata['e_1'] = traj_frames[feat][frame_idx].clone()
                else:
                    g_dummy.ndata[f'{feat}_1'] = traj_frames[feat][frame_idx].clone()

            # extract mol data from graph
            positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs = extract_moldata_from_graph(g_dummy, self.atom_type_map)

            # align positions to final frame
            if self.align_traj:
                positions = rigid_alignment(positions, x_final)

            # build rdkit molecule
            mol = build_molecule(positions, atom_types, atom_charges, bond_src_idxs, bond_dst_idxs, bond_types)

            # add mol to list
            traj_mols.append(mol)

        traj_mols = [mol for mol in traj_mols if mol is not None]

        if len(traj_mols) < len(traj_frames):
            print(f'WARNING: {len(traj_frames) - len(traj_mols)} frames were not converted to rdkit molecules')

        return traj_mols

    def __repr__(self):
        return f"{self.bond_types}, {self.atom_types}, {self.atom_charges}, {self.positions}"
    

def extract_moldata_from_graph(g: dgl.DGLGraph, atom_type_map: List[str], exclude_charges: bool = False):

    # extract node-level features
    positions = g.ndata['x_1']

    # extract node-level features
    positions = g.ndata['x_1']
    atom_types = g.ndata['a_1'].argmax(dim=1)
    atom_types = [atom_type_map[int(atom)] for atom in atom_types]

    if exclude_charges:
        atom_charges = None
    else:
        atom_charges = g.ndata['c_1'].argmax(dim=1) - 2 # implicit assumption that index 0 charge is -2


    # get bond types and atom indicies for every edge, convert types from simplex to integer
    bond_types = g.edata['e_1'].argmax(dim=1)
    bond_src_idxs, bond_dst_idxs = g.edges()

    # get just the upper triangle of the adjacency matrix
    upper_edge_mask = g.edata['ue_mask']
    bond_types = bond_types[upper_edge_mask]
    bond_src_idxs = bond_src_idxs[upper_edge_mask]
    bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

    # get only non-zero bond types
    bond_mask = bond_types != 0
    bond_types = bond_types[bond_mask]
    bond_src_idxs = bond_src_idxs[bond_mask]
    bond_dst_idxs = bond_dst_idxs[bond_mask]


    return positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs


def build_molecule(positions, atom_types, atom_charges, bond_src_idxs, bond_dst_idxs, bond_types):
    """Builds a rdkit molecule from the given atom and bond information."""
    # create a rdkit molecule and add atoms to it
    mol = Chem.RWMol()
    for atom_type, charge in zip(atom_types, atom_charges):
        a = Chem.Atom(atom_type)
        if charge != 0:
            a.SetFormalCharge(int(charge))
        mol.AddAtom(a)

    # add bonds to rdkit molecule
    for bond_type, src_idx, dst_idx in zip(bond_types, bond_src_idxs, bond_dst_idxs):
        src_idx = int(src_idx)
        dst_idx = int(dst_idx)
        mol.AddBond(src_idx, dst_idx, bond_type_map[bond_type])

    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        return None

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = positions[i]
        x, y, z = float(x), float(y), float(z)
        conf.SetAtomPosition(i, Point3D(x,y,z))
    mol.AddConformer(conf)

    return mol


def copy_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    
    # get edges
    edges = g.edges(form='uv')

    # get number of nodes
    num_nodes = g.num_nodes()

    g_copy = dgl.graph(edges, num_nodes=num_nodes, device=g.device)

    # transfer over node features
    for nfeat in g.ndata.keys():
        g_copy.ndata[nfeat] = g.ndata[nfeat].detach().clone()

    # transfer over edge features
    for efeat in g.edata.keys():
        g_copy.edata[efeat] = g.edata[efeat].detach().clone()


    return g_copy