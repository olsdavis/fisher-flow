import torch
import torch.nn as nn

from rdkit import Chem
from src.experiments.retrosynthesis import BasicMolecularMetrics, build_molecule
from torch import Tensor
from torchmetrics import MeanSquaredError, MeanAbsoluteError, Metric, MetricCollection


from pdb import set_trace


class DummySamplingMolecularMetrics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, molecules: list):
        return {}

    def reset(self):
        pass


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, train_smiles):
        super().__init__()
        di = dataset_infos
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)
        self.generated_node_dist = GeneratedNodesDistribution(di.output_dims['X'])
        self.generated_edge_dist = GeneratedEdgesDistribution(di.output_dims['E'])
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)

        n_target_dist = di.n_nodes.type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer('n_target_dist', n_target_dist)

        node_target_dist = di.node_types.type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        self.register_buffer('node_target_dist', node_target_dist)

        edge_target_dist = di.edge_types.type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        self.register_buffer('edge_target_dist', edge_target_dist)

        valency_target_dist = di.valency_distribution.type_as(self.generated_valency_dist.edgepernode_dist)
        valency_target_dist = valency_target_dist / torch.sum(valency_target_dist)
        self.register_buffer('valency_target_dist', valency_target_dist)

        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = train_smiles
        self.dataset_info = di

    def forward(self, molecules: list):
        metrics = BasicMolecularMetrics(self.dataset_info, self.train_smiles)
        to_log = metrics.evaluate(molecules)

        self.generated_n_dist(molecules)
        generated_n_dist = self.generated_n_dist.compute()
        self.n_dist_mae(generated_n_dist)

        self.generated_node_dist(molecules)
        generated_node_dist = self.generated_node_dist.compute()
        self.node_dist_mae(generated_node_dist)

        self.generated_edge_dist(molecules)
        generated_edge_dist = self.generated_edge_dist.compute()
        self.edge_dist_mae(generated_edge_dist)

        self.generated_valency_dist(molecules)
        generated_valency_dist = self.generated_valency_dist.compute()
        self.valency_dist_mae(generated_valency_dist)

        for i, atom_type in enumerate(self.dataset_info.atom_decoder):
            generated_probability = generated_node_dist[i]
            target_probability = self.node_target_dist[i]
            to_log[f'sampled_atom_dist/{atom_type}_dist'] = (generated_probability - target_probability).item()

        for j, bond_type in enumerate(['No bond', 'Single', 'Double', 'Triple', 'Aromatic']):
            generated_probability = generated_edge_dist[j]
            target_probability = self.edge_target_dist[j]
            to_log[f'sampled_bond_dist/bond_{bond_type}_dist'] = (generated_probability - target_probability).item()

        for valency in range(6):
            generated_probability = generated_valency_dist[valency]
            target_probability = self.valency_target_dist[valency]
            to_log[f'sampled_valency_dist/valency_{valency}_dist'] = (generated_probability - target_probability).item()

        to_log['sampled_mae/n_mae'] = self.n_dist_mae.compute()
        to_log['sampled_mae/node_mae'] = self.node_dist_mae.compute()
        to_log['sampled_mae/edge_mae'] = self.edge_dist_mae.compute()
        to_log['sampled_mae/valency_mae'] = self.valency_dist_mae.compute()

        return to_log

    def reset(self):
        for metric in [self.n_dist_mae, self.node_dist_mae, self.edge_dist_mae, self.valency_dist_mae]:
            metric.reset()


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state('n_dist', default=torch.zeros(max_n + 1, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state('node_dist', default=torch.zeros(num_atom_types, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert int(atom_type) != -1, "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state('edge_dist', default=torch.zeros(num_edge_types, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class MeanNumberEdge(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state('total_edge', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state('edgepernode_dist', default=torch.zeros(3 * max_n - 2, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """ Compute the distance between histograms. """
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


class MSEPerClass(MeanSquaredError):
    full_state_update = False

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds = preds[..., self.class_id]
        target = target[..., self.class_id]
        super().update(preds, target)


class HydroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SnMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class MgMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CuMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ZnMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


# Bonds MSE

class NoBondMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetrics(MetricCollection):
    def __init__(self, dataset_infos):
        self.atom_decoder = dataset_infos.atom_decoder
        class_dict = {'H': HydroMSE, 'C': CarbonMSE, 'N': NitroMSE, 'O': OxyMSE, 'F': FluorMSE, 'B': BoronMSE,
                      'Br': BrMSE, 'Cl': ClMSE, 'I': IodineMSE, 'P': PhosphorusMSE, 'S': SulfurMSE, 'Se': SeMSE,
                      'Si': SiMSE, 'Sn': SnMSE, 'Mg': MgMSE, 'Cu': CuMSE, 'Zn': ZnMSE}

        metrics_list = []
        for i, atom_type in enumerate(self.atom_decoder):
            metrics_list.append(class_dict[atom_type](i))

        super().__init__(metrics_list)


class BondMetrics(MetricCollection):
    def __init__(self):
        mse_no_bond = NoBondMSE(0)
        mse_SI = SingleMSE(1)
        mse_DO = DoubleMSE(2)
        mse_TR = TripleMSE(3)
        mse_AR = AromaticMSE(4)
        super().__init__([mse_no_bond, mse_SI, mse_DO, mse_TR, mse_AR])


def compute_retrosynthesis_metrics(grouped_samples, ground_truth, atom_decoder, grouped_scores=None):
    """
    Compute top-N accuracy. Inputs are matrices of atom types and bonds.
    """

    total = 0
    top_1_success = 0
    top_3_success = 0
    top_5_success = 0

    top_1_success_scoring = 0
    top_3_success_scoring = 0
    top_5_success_scoring = 0

    for i, sampled_reactants in enumerate(grouped_samples):
        true_reactants = ground_truth[i]
        true_mol = build_molecule(true_reactants[0], true_reactants[1], atom_decoder)
        true_smi = Chem.MolToSmiles(true_mol)
        if true_smi is None:
            continue

        sampled_smis = []
        for sample in sampled_reactants:
            sampled_mol = build_molecule(sample[0], sample[1], atom_decoder)
            try:
                sampled_smi = Chem.MolToSmiles(sampled_mol)
            except RuntimeError as e:
                print("Error encountered when trying to produce")
                print(e)
                sampled_smi = None
            if sampled_smi is None:
                continue
            sampled_smis.append(sampled_smi)

        total += 1
        top_1_success += true_smi in set(sampled_smis[:1])
        top_3_success += true_smi in set(sampled_smis[:3])
        top_5_success += true_smi in set(sampled_smis[:5])

        if grouped_scores is not None:
            scores = grouped_scores[i]
            sorted_sampled_smis = [
                sampled_smis[j]
                for j, _ in sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
            ]

            top_1_success_scoring += true_smi in set(sorted_sampled_smis[:1])
            top_3_success_scoring += true_smi in set(sorted_sampled_smis[:3])
            top_5_success_scoring += true_smi in set(sorted_sampled_smis[:5])

    to_log = {
        'top_1_accuracy': top_1_success / total,
        'top_3_accuracy': top_3_success / total,
        'top_5_accuracy': top_5_success / total,
    }
    if grouped_scores is not None:
        to_log['top_1_accuracy_scoring'] = top_1_success_scoring / total
        to_log['top_3_accuracy_scoring'] = top_3_success_scoring / total
        to_log['top_5_accuracy_scoring'] = top_5_success_scoring / total

    print('======= Retrosynthesis Metrics =======')
    print(f'Total: {total}')
    print(f'Top-1 accuracy: {top_1_success / total}')
    print(f'Top-3 accuracy: {top_3_success / total}')
    print(f'Top-5 accuracy: {top_5_success / total}')
    print('======================================')

    return to_log
