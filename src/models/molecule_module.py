"""
A module for molecular generative tasks, namely: QM9, GeomDrugs, Retrosynthesis.
"""
from abc import ABC, abstractmethod
from typing import Any
from torch_geometric.data import Batch as TGBatch
import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import MeanMetric
import dgl
from dgl import DGLGraph
from lightning import LightningModule


from src.sfm import Manifold, manifold_from_name
from src.data.components.qm_utils import (
    get_batch_idxs, get_upper_edge_mask, build_edge_idxs,
)
from src.data.components.sample_analyzer import (
    SampleAnalyzer, SampledMolecule,
)
from src.data.components.molecule_prior import align_prior


GraphData = TGBatch | DGLGraph


class Prior(ABC):
    """
    A class for prior distributions.
    """

    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    @abstractmethod
    def sample(self, n: int, k: int, d: int) -> Tensor:
        """
        Samples `n` points from the prior.
        """


class UniformPrior(Prior):
    """
    A uniform prior distribution.
    """

    def sample(self, n: int, k: int, d: int) -> Tensor:
        return self.manifold.uniform_prior(n, k, d)


class GaussianPrior(Prior):
    """
    A Gaussian prior distribution.
    """

    def sample(self, n: int, k: int, d: int) -> Tensor:
        return torch.randn((n, k, d))


def _get_prior(name: str, manifold: Manifold) -> Prior:
    """
    Returns a prior distribution.
    """
    if name == "uniform":
        return UniformPrior(manifold)
    if name == "gaussian":
        return GaussianPrior(manifold)
    raise ValueError(f"unknown prior: {name}")


class MoleculeModule(LightningModule):
    """Module for molecular tasks."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        conditional: bool,
        features_manifolds: dict[str, str],
        atom_type_map: list[str] | None = None,
        n_atoms_hist_file: str = "data/qm9/train_data_n_atoms_histogram.pt",
        inference_steps: int = 100,
        features_priors: dict[str, str] | None = None,
        loss_weights: dict[str, float] | None = None,
        eval_mols_every: int = 1,
        n_eval_mols: int = 128,
    ):
        """
        Parameters:
            - `conditional`: `True` iff the molecular generation is conditonal, which
            assumes that the batches contain the `x_0` data.
        """
        super().__init__()
        torch.set_float32_matmul_precision("high")
        assert features_priors is None or not conditional,\
            "conditional models do not have priors"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        if not conditional:
            self.atom_type_map = atom_type_map
            self.build_n_atoms_dist(n_atoms_hist_file=n_atoms_hist_file)
            self.features_lengths = {
                "x": 3,
                "a": len(atom_type_map),
                "c": 6,
                "e": 5,
            }
        self.conditional = conditional
        self.features_manifolds = {
            key: manifold_from_name(value) for key, value in features_manifolds.items()
        }
        self.features_priors = {
            key: _get_prior(value, self.features_manifolds[key])
            for key, value in features_priors.items()
        }
        self.loss_weights = loss_weights
        self.inference_steps = inference_steps
        self.eval_mols_every = eval_mols_every
        self.n_eval_mols = n_eval_mols

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # same but for individual features
        for key in self.features_manifolds:
            setattr(self, f"train/{key}_loss", MeanMetric())
            setattr(self, f"val/{key}_loss", MeanMetric())
            setattr(self, f"test/{key}_loss", MeanMetric())

    def forward(self, x: GraphData, t: Tensor) -> Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    @torch.no_grad()
    def sample_unconditional_molecules(
        self,
        n_atoms: Tensor,
        n_timesteps: int,
    ):
        """
        Sample molecules with the given number of atoms.
        """
        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=self.device)
            g.append(g_i)

        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.initialize_random_noise(g, do_x=True)

        # integrate trajectories
        itg_result = self.net.integrate(
            g,
            node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            n_timesteps=n_timesteps,
            visualize=False,
        )

        g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        molecules = []
        for g_i in dgl.unbatch(g):
            args = [g_i, self.atom_type_map]

            molecules.append(SampledMolecule(*args, exclude_charges=False))

        return molecules

    def sample_conditional_molecules(self, x_0: TGBatch) -> TGBatch:
        raise NotImplementedError("")

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()

    def conditional_step(self, inp: TGBatch) -> dict[str, Tensor]:
        """
        Computes losses for conditional models.
        """
        raise NotImplementedError("conditional_step must be implemented for conditional models")

    def initialize_random_noise(self, inp: DGLGraph, do_x: bool = False) -> DGLGraph:
        """
        Initializes random noise for the graph `inp`, in the case of unconditional models.
        """
        assert not self.conditional, "conditional models do not require random init noise"
        # initialise priors
        for key, prior in self.features_priors.items():
            if key == "e":
                inp.edata["e_0"] = (
                    prior.sample(inp.num_edges(), 1, self.features_lengths["e"]).squeeze()
                ).to(self.device)
                upper = get_upper_edge_mask(inp)
                inp.edata["e_0"][~upper] = inp.edata["e_0"][upper]
            else:
                if key == "x" and not do_x:
                    continue
                # x is set in the dataset because it requires the alignment to be done on CPU
                p = prior.sample(inp.num_nodes(), 1, self.features_lengths[key]).squeeze()
                inp.ndata[f"{key}_0"] = p.to(self.device)

        for key, manifold in self.features_manifolds.items():
            # ensure is on the manifold
            if key == "e":
                inp.edata[f"{key}_0"] = manifold.project(inp.edata[f"{key}_0"])
            else:
                inp.ndata[f"{key}_0"] = manifold.project(inp.ndata[f"{key}_0"])
        return inp

    def unconditional_step(self, inp: DGLGraph) -> dict[str, Tensor]:
        """
        Computes losses for unconditional models.
        """
        with torch.no_grad():
            inp = self.initialize_random_noise(inp)
            node_batch_idx, edge_batch_idx = get_batch_idxs(inp)
            t = torch.rand(inp.batch_size, device=self.device)
        # now, forward through model
        inp = self.net.sample_conditional_path(inp, t, node_batch_idx, edge_batch_idx)
        output = self.net(inp, t, node_batch_idx, edge_batch_idx, apply_softmax=False, remove_com=False)
        losses = {}
        for key in self.features_manifolds.keys():
            if key == "x":
                losses[key] = F.mse_loss(
                    output[key], inp.ndata[f"{key}_1_true"], reduction="mean",
                )
            else:
                losses[key] = F.cross_entropy(
                    output[key], inp.ndata[f"{key}_1_true"] if key != "e" else inp.edata[f"{key}_1_true"], reduction="mean",
                )
        return losses

    def model_step(self, inp: GraphData) -> dict[str, Tensor]:
        """
        Given an input graph, returns named losses.
        """
        if self.conditional:
            assert isinstance(inp, TGBatch), "conditional models require TGBatch inputs"
            return self.conditional_step(inp)
        assert isinstance(inp, DGLGraph), "unconditional models require DGLGraph inputs"
        return self.unconditional_step(inp)

    def training_step(
        self, batch: GraphData, batch_idx: int
    ) -> Tensor:
        """
        Perform a single training step on a batch of data from the training set.
        """
        losses = self.model_step(batch)
        for key, loss in losses.items():
            getattr(self, f"train/{key}_loss")(loss)

        if self.loss_weights is not None:
            loss = sum(
                weight * losses[key] for key, weight in self.loss_weights.items()
            )
        else:
            loss = sum(losses.values())
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Lightning hook that is called when a training epoch ends."""

    def validation_step(self, batch: GraphData, batch_idx: int):
        """
        Perform a single validation step on a batch of data from the validation set.
        """
        losses = self.model_step(batch)
        for key, loss in losses.items():
            getattr(self, f"val/{key}_loss")(loss)

        if self.loss_weights is not None:
            loss = sum(
                weight * losses[key] for key, weight in self.loss_weights.items()
            )
        else:
            loss = sum(losses.values())
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Lightning hook that is called when a validation epoch ends."""
        if self.current_epoch % self.eval_mols_every == 0:
            molecules = self.sample_random_sizes(self.n_eval_mols, n_timesteps=self.inference_steps)
            analyzer = SampleAnalyzer()
            stats = analyzer.analyze(molecules)
            for key, value in stats.items():
                self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: GraphData, batch_idx: int):
        """
        Perform a single test step on a batch of data from the test set.
        """
        losses = self.model_step(batch)
        for key, loss in losses.items():
            getattr(self, f"test/{key}_loss")(loss)

        if self.loss_weights is not None:
            loss = sum(
                weight * losses[key] for key, weight in self.loss_weights.items()
            )
        else:
            loss = sum(losses.values())
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        """Lightning hook that is called when a test epoch ends."""

    def setup(self, stage: str):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        """Builds the distribution of the number of atoms in a ligand."""
        n_atoms, n_atom_counts = torch.load(n_atoms_hist_file)
        n_atoms_prob = n_atom_counts / n_atom_counts.sum()
        self.n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
        self.n_atoms_map = n_atoms

    def sample_n_atoms(self, n_molecules: int):
        """Draw samples from the distribution of the number of atoms in a ligand."""
        n_atoms = self.n_atoms_dist.sample((n_molecules,))
        return self.n_atoms_map[n_atoms]

    def sample_random_sizes(self, n_molecules: int, n_timesteps: int):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        # get the number of atoms that will be in each molecules
        atoms_per_molecule = self.sample_n_atoms(n_molecules).to(self.device)

        return self.sample_unconditional_molecules(atoms_per_molecule, n_timesteps=n_timesteps)
