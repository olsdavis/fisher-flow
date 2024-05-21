from typing import Union
import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from .gvp import GVPConv, GVP, _rbf, _norm_no_nan

class EndpointVectorField(nn.Module):

    def __init__(self, n_atom_types: int,
                    canonical_feat_order: list = ['x', 'a', 'c', 'e'],
                    n_charges: int = 6,
                    n_bond_types: int = 5, 
                    n_vec_channels: int = 16,
                    n_cp_feats: int = 0, 
                    n_hidden_scalars: int = 64,
                    n_hidden_edge_feats: int = 64,
                    n_recycles: int = 1,
                    n_molecule_updates: int = 2, 
                    convs_per_update: int = 2,
                    n_message_gvps: int = 3, 
                    n_update_gvps: int = 3,
                    separate_mol_updaters: bool = False,
                    message_norm: Union[float, str] = 100,
                    update_edge_w_distance: bool = False,
                    rbf_dmax = 20,
                    rbf_dim = 16,
                    exclude_charges: bool = False
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.exclude_charges = exclude_charges
        self.canonical_feat_order = canonical_feat_order

        if self.exclude_charges:
            self.n_charges = 0

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        assert n_vec_channels >= 3, 'n_vec_channels must be >= 3'


        self.scalar_embedding = nn.Sequential(
            nn.Linear(n_atom_types + n_charges + 1, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats)
        )

        self.conv_layers = []
        for conv_idx in range(convs_per_update*n_molecule_updates):
            self.conv_layers.append(GVPConv(
                scalar_size=n_hidden_scalars,
                vector_size=n_vec_channels,
                n_cp_feats=n_cp_feats,
                edge_feat_size=n_hidden_edge_feats,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim
            )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for _ in range(n_updaters - 1):
            self.node_position_updaters.append(NodePositionUpdate(n_hidden_scalars, n_vec_channels, n_gvps=3, n_cp_feats=n_cp_feats))
            self.edge_updaters.append(EdgeUpdate(n_hidden_scalars, n_hidden_edge_feats, update_edge_w_distance=update_edge_w_distance, rbf_dim=rbf_dim))


        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges)
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types)
        )

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False):
        """Predict x_1 (trajectory destination) given x_t"""
        device = g.device

        with g.local_scope():
            # gather node and edge features for input to convolutions
            node_scalar_features = [
                g.ndata['a_t'],
                t[node_batch_idx].unsqueeze(-1)
            ]

            # if we are not excluding charges, include them in the node scalar features
            if not self.exclude_charges:
                node_scalar_features.append(g.ndata['c_t'])

            node_scalar_features = torch.cat(node_scalar_features, dim=-1)
            node_scalar_features = self.scalar_embedding(node_scalar_features)

            node_positions = g.ndata['x_t']

            num_nodes = g.num_nodes()

            # initialize the vector features for every node to be zeros
            node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)
            # i thought setting the first three channels to the identity matrix would be a good idea,
            # but this actually breaks rotational equivariance
            # node_vec_features[:, :3, :] = torch.eye(3, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)

            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)

            x_diff, d = self.precompute_distances(g)
            for recycle_idx in range(self.n_recycles):
                for conv_idx, conv in enumerate(self.conv_layers):

                    # perform a single convolution which updates node scalar and vector features (but not positions)
                    node_scalar_features, node_vec_features = conv(g, 
                            scalar_feats=node_scalar_features, 
                            coord_feats=node_positions,
                            vec_feats=node_vec_features,
                            edge_feats=edge_features,
                            x_diff=x_diff,
                            d=d
                    )

                    # every convs_per_update convolutions, update the node positions and edge features
                    if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:

                        if self.separate_mol_updaters:
                            updater_idx = conv_idx // self.convs_per_update - 1
                        else:
                            updater_idx = 0

                        node_positions = self.node_position_updaters[updater_idx](node_scalar_features, node_positions, node_vec_features)

                        x_diff, d = self.precompute_distances(g, node_positions)

                        edge_features = self.edge_updaters[updater_idx](g, node_scalar_features, edge_features, d=d)

            
            # predict final charges and atom type logits
            node_scalar_features = self.node_output_head(node_scalar_features)
            atom_type_logits = node_scalar_features[:, :self.n_atom_types]
            if not self.exclude_charges:
                atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

            # predict the final edge logits
            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            # project node positions back into zero-COM subspace
            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {
            'x': node_positions,
            'a': atom_type_logits,
            'e': edge_logits
        }
        if not self.exclude_charges:
            dst_dict['c'] = atom_charge_logits

        # apply softmax to categorical features, if requested
        # at training time, we don't want to apply softmax because we use cross-entropy loss which includes softmax
        # at inference time, we want to apply softmax to get a vector which lies on the simplex
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in ['a', 'c', 'e']: # if this is a categorical feature
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1) # apply softmax to this feature

        return dst_dict
    
    def precompute_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            if node_positions is None:
                g.ndata['x_d'] = g.ndata['x_t']
            else:
                g.ndata['x_d'] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            x1_weight = alpha_t_prime_i[feat_idx]*(s_i - t_i)/(1 - alpha_t_i[feat_idx])
            xt_weight = 1 - x1_weight

            if feat == "e":
                g_data_src = g.edata

                # set the edge features so that corresponding upper and lower triangle edges have the same value
                x1 = torch.zeros_like(g.edata['e_0'])
                x1[upper_edge_mask] = dst_dict[feat]
                x1[~upper_edge_mask] = dst_dict[feat]
            else:
                g_data_src = g.ndata
                x1 = dst_dict[feat]

            g_data_src[f'{feat}_t'] = x1_weight*x1 + xt_weight*g_data_src[f'{feat}_t']

        return g


class VectorField(EndpointVectorField):

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False):
        
        dst_dict = super().forward(g, t, node_batch_idx, upper_edge_mask, apply_softmax, remove_com)
        dst_dict['x'] = dst_dict['x'] - g.ndata['x_t']
        return dst_dict
    
    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        
        # predict the destination of the trajectory given the current timepoint
        vec_field = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=False,
            remove_com=False
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == "e":
                x_t = g.edata[f'e_t'][upper_edge_mask]
            else:
                x_t = g.ndata[f'{feat}_t']

            # x_s = x_t + vec_field*(s - t)
            x_s = x_t + vec_field[feat]*(s_i - t_i)

            # set x_t = x_s
            if feat == "e":
                x_t = torch.zeros_like(g.edata['e_0'])
                x_t[upper_edge_mask] = x_s
                x_t[~upper_edge_mask] = x_s
                g.edata[f'{feat}_t'] = x_t
            else:
                x_t = x_s
                g.ndata[f'{feat}_t'] = x_t

        # remove COM from x_t
        g.ndata['x_t'] = g.ndata['x_t'] - dgl.readout_nodes(g, feat='x_t', op='mean')[node_batch_idx]

        return g


class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)


class EdgeUpdate(nn.Module):

    def __init__(self, n_node_scalars, n_edge_feats, update_edge_w_distance=False, rbf_dim=16):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars*2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )

        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d):
        

        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges()

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]

        if self.update_edge_w_distance:
            mlp_inputs.append(d)

        edge_feats = self.edge_norm(edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1)))
        return edge_feats