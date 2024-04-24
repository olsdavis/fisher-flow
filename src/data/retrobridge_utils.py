import torch
import torch_geometric

from torch_geometric.utils import to_dense_adj, to_dense_batch, scatter


class PlaceHolder:
    def __init__(self, X, E, y=None):
        self.X = X
        self.E = E
        self.y = y if y is not None else torch.zeros(size=(self.X.shape[0], 0), dtype=torch.float, device=X.device)

    def type_as(self, x: torch.Tensor):
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

        return self

    def clone(self):
        if self.y is not None:
            return PlaceHolder(self.X.clone(), self.E.clone(), self.y.clone())
        else:
            return PlaceHolder(self.X.clone(), self.E.clone())

    def detach(self):
        if self.y is not None:
            return PlaceHolder(self.X.clone().detach(), self.E.clone().detach(), self.y.clone().detach())
        else:
            return PlaceHolder(self.X.clone().detach(), self.E.clone().detach())


class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p


def to_dense(x, edge_index, edge_attr, batch, explicitly_encode_no_edge=True):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    if explicitly_encode_no_edge:
        E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E

    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def create_true_reactant_molecules(data, batch_size):
    reactants, r_node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    reactants = reactants.mask(r_node_mask, collapse=True)
    n_nodes = scatter(torch.ones_like(data.batch), data.batch, reduce='sum')
    true_molecule_list = []
    for i in range(batch_size):
        n = n_nodes[i]
        atom_types = reactants.X[i, :n].cpu()
        edge_types = reactants.E[i, :n, :n].cpu()
        true_molecule_list.append([atom_types, edge_types])

    return true_molecule_list


def create_pred_reactant_molecules(X, E, batch_mask, batch_size):
    molecule_list = []
    n_nodes = scatter(torch.ones_like(batch_mask), batch_mask, reduce='sum')
    for i in range(batch_size):
        n = n_nodes[i]
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()
        molecule_list.append([atom_types, edge_types])

    return molecule_list


def create_input_product_molecules(data, batch_size):
    products, p_node_mask = to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
    products = products.mask(p_node_mask, collapse=True)
    n_nodes = scatter(torch.ones_like(data.batch), data.batch, reduce='sum')
    products_list = []
    for i in range(batch_size):
        n = n_nodes[i]
        atom_types = products.X[i, :n].cpu()
        edge_types = products.E[i, :n, :n].cpu()
        products_list.append([atom_types, edge_types])

    return products_list
