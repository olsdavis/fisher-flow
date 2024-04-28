import os
from typing import Any, List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
from torch_geometric.data import Batch

from src.experiments.retrosynthesis import compute_retrosynthesis_metrics, TrainMolecularMetricsDiscrete
from src.experiments.retrosynthesis import SamplingMolecularMetrics, MolecularVisualization
from src.experiments.retrosynthesis import ExtraFeatures, DummyExtraFeatures, ExtraMolecularFeatures
from src.markov_bridge import PredefinedNoiseScheduleDiscrete, InterpolationTransition, sample_discrete_features, reverse_tensor
from src.experiments.retrosynthesis import TrainLossDiscrete, TrainLossVLB
from src.data import PlaceHolder, to_dense, create_pred_reactant_molecules, create_true_reactant_molecules, create_input_product_molecules

"""
TODO: class and methods
"""

class RetroBridgeModule(LightningModule):
    """
    Module for the Toy DFM dataset.
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        input_dims: dict = {'X': 40, 'E': 10, 'y': 12},
        output_dims: dict = {'X': 17, 'E': 5, 'y': 0},
        loss_type: str = 'vlb',
        experiment_name: str = 'test',
        logs: str = 'logs',
        checkpoints: str = 'checkpoints',
        diffusion_steps: int = 500,
        transition: str = None,
        lr: float = 0.0002,
        weight_decay: float = 0.000000000001,
        extra_features: str = "all",
        extra_molecular_features: bool = False,
        use_context: bool = True,
        diffusion_noise_schedule: str = "cosine",
        number_chain_steps_to_save: int = 50,
        log_every_steps: int = 50,
        sample_every_val: int = 20,
        samples_to_generate: int = 128,
        samples_to_save: int = 128,
        samples_per_input: int = 5,
        chains_to_save: int = 5,
        fix_product_nodes: bool = True,
        lambda_train: List[int] = [5, 0],
    ) -> None:
        """
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        if compile:
            self.net = torch.compile(net)
        else:
            self.net = net

        # TODO: check if these are needed        
        # for averaging loss across batches
        # self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        assert loss_type in ['cross_entropy', 'vlb']

        self.name = experiment_name
        self.chains_dir = os.path.join(logs, 'chains', experiment_name)
        self.graphs_dir = os.path.join(logs, 'graphs', experiment_name)
        self.checkpoints_dir = os.path.join(checkpoints, experiment_name)

        self.model_dtype = torch.float32
        self.T = diffusion_steps
        self.transition = transition

        self.lr = lr
        self.weight_decay = weight_decay

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        # self.trainer.train_dataloaders.dataset -- > datamodule
        self.dataset_infos = self.trainer.train_dataloaders.dataset_infos
        self.train_metrics = TrainMolecularMetricsDiscrete(self.dataset_infos)
        self.sampling_metrics = SamplingMolecularMetrics(
            self.dataset_infos, self.trainer.train_dataloaders.dataset.train_smiles)
        self.train_loss = TrainLossDiscrete(lambda_train) if loss_type != 'vlb' else TrainLossVLB(lambda_train)
        self.val_loss = TrainLossDiscrete(lambda_train) if loss_type != 'vlb' else TrainLossVLB(lambda_train)

        self.visualization_tools = MolecularVisualization(self.dataset_infos)
        self.extra_features = (
            ExtraFeatures(extra_features, dataset_info=self.dataset_infos)
            if extra_features is not None
            else DummyExtraFeatures()
        )
        
        self.domain_features = (
            ExtraMolecularFeatures(dataset_infos=self.dataset_infos)
            if extra_molecular_features
            else DummyExtraFeatures()
        )
        self.use_context = use_context
    
        self.nodes_dist = self.dataset_infos.nodes_dist
        
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=diffusion_noise_schedule,
            timesteps=diffusion_steps,
        )
        self.transition_model = InterpolationTransition(
            x_classes=self.Xdim_output,
            e_classes=self.Edim_output,
            y_classes=self.ydim_output
        )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            ignore=[
                self.train_metrics, 
                self.sampling_metrics, 
                self.visualization_tools, 
                self.dataset_infos,
            ]
        )

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None

        self.number_chain_steps_to_save = number_chain_steps_to_save
        self.log_every_steps = log_every_steps
        self.sample_every_val = sample_every_val
        self.samples_to_generate = samples_to_generate
        self.samples_to_save = samples_to_save
        self.samples_per_input = samples_per_input
        self.chains_to_save = chains_to_save
        self.val_counter = 0

        self.fix_product_nodes = fix_product_nodes
        self.loss_type = loss_type

    def forward(self, noisy_data: dict, extra_data: PlaceHolder, node_mask: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.net(X, E, y, node_mask)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def training_step(
        self, data: Batch, i: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param data: xxx.
        :param i: xxx.
        :return: xxx.
        """
        reactants, product, pred, node_mask, noisy_data, _ = self.process_and_forward(data)
        if self.loss_type == 'vlb':
            return self.compute_training_VLB(
                reactants=reactants,
                pred=pred,
                node_mask=node_mask,
                noisy_data=noisy_data,
                i=i,
            )
        else:
            return self.compute_training_CE_loss_and_metrics(reactants=reactants, pred=pred, i=i)

    def compute_training_CE_loss_and_metrics(self, reactants, pred, i):
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y,
        )
        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=reactants.X,
            true_E=reactants.E,
        )

        if i % self.log_every_steps == 0:
            self.log(f'train_loss/batch_CE', loss.detach())
            for metric_name, metric in self.train_loss.compute_metrics().items():
                self.log(f'train_loss/{metric_name}', metric)
            for metric_name, metric in self.train_metrics.compute_metrics().items():
                self.log(f'train_detailed/{metric_name}/train', metric)

            self.train_loss.reset()
            self.train_metrics.reset()

        return {'loss': loss}

    def compute_validation_CE_loss(self, reactants, pred, i):
        loss = self.val_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y,
        )

        if i % self.log_every_steps == 0:
            self.log(f'val_loss/batch_CE', loss.detach())
            for metric_name, metric in self.val_loss.compute_metrics().items():
                self.log(f'val_loss/{metric_name}', metric)

            self.train_loss.reset()
            self.train_metrics.reset()

        return {'loss': loss}

    def compute_training_VLB(self, reactants, pred, node_mask, noisy_data, i):
        z_t = PlaceHolder(X=noisy_data['X_t'], E=noisy_data['E_t'], y=noisy_data['y_t'])
        z_T_true = reactants
        z_T_pred = pred
        t = noisy_data['t']

        true_pX, true_pE = self.compute_q_zs_given_q_zt(z_t, z_T_true, node_mask, t=t)
        pred_pX, pred_pE = self.compute_p_zs_given_p_zt(z_t, z_T_pred, node_mask, t=t)

        loss = self.train_loss(
            masked_pred_X=pred_pX,
            masked_pred_E=pred_pE,
            true_X=true_pX,
            true_E=true_pE,
        )
        if i % self.log_every_steps == 0:
            self.log(f'train_loss/batch_CE', loss.detach())
            for metric_name, metric in self.train_loss.compute_metrics().items():
                self.log(f'train_loss/{metric_name}', metric)

            self.train_loss.reset()

        return {'loss': loss}

    def compute_validation_VLB(self, reactants, pred, node_mask, noisy_data, i):
        z_t = PlaceHolder(X=noisy_data['X_t'], E=noisy_data['E_t'], y=noisy_data['y_t'])
        z_T_true = reactants
        z_T_pred = pred
        t = noisy_data['t']

        true_pX, true_pE = self.compute_q_zs_given_q_zt(z_t, z_T_true, node_mask, t=t)
        pred_pX, pred_pE = self.compute_p_zs_given_p_zt(z_t, z_T_pred, node_mask, t=t)

        loss = self.val_loss(
            masked_pred_X=pred_pX,
            masked_pred_E=pred_pE,
            true_X=true_pX,
            true_E=true_pE,
        )
        if i % self.log_every_steps == 0:
            self.log(f'val_loss/batch_CE', loss.detach())
            for metric_name, metric in self.train_loss.compute_metrics().items():
                self.log(f'val_loss/{metric_name}', metric)

            self.train_loss.reset()

        return {'loss': loss}

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.sampling_metrics.reset()

    def on_train_epoch_start(self):
        self.train_loss.reset()
        self.train_metrics.reset()

    def process_and_forward(self, data):
        # Getting graphs of reactants (target) and product (context)
        reactants, r_node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        reactants = reactants.mask(r_node_mask)

        product, p_node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(p_node_mask)

        assert torch.allclose(r_node_mask, p_node_mask)
        node_mask = r_node_mask

        # Getting noisy data
        # Note that here products and reactants are swapped
        noisy_data = self.apply_noise(
            X=product.X, E=product.E, y=product.y,
            X_T=reactants.X, E_T=reactants.E, y_T=reactants.y,
            node_mask=node_mask,
        )

        # Computing extra features + context and making predictions
        context = product.clone() if self.use_context else None
        extra_data = self.compute_extra_data(noisy_data, context=context)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Masking unchanged part
        if self.fix_product_nodes:
            fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
            modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
            assert torch.all(fixed_nodes | modifiable_nodes)
            pred.X = pred.X * modifiable_nodes + product.X * fixed_nodes
            pred.X = pred.X * node_mask.unsqueeze(-1)

        return reactants, product, pred, node_mask, noisy_data, context

    def on_train_epoch_end(self) -> None:
        """Nothing to do."""
    
    # TODO: typing
    def validation_step(self, data: Batch, i: int,) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param : xxx.
        :param : xxx.
        :return: xxx.
        """
        reactants, product, pred, node_mask, noisy_data, context = self.process_and_forward(data)
        if self.loss_type == 'vlb':
            return self.compute_validation_VLB(
                reactants=reactants,
                pred=pred,
                node_mask=node_mask,
                noisy_data=noisy_data,
                i=i,
            )
        else:
            return self.compute_validation_CE_loss(reactants=reactants, pred=pred, i=i)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_counter += 1
        if self.val_counter % self.sample_every_val == 0:
            self.sample()
            self.trainer.save_checkpoint(os.path.join(self.checkpoints_dir, 'last.ckpt'))

    def sample(self):
        samples_left_to_generate = self.samples_to_generate
        samples_left_to_save = self.samples_to_save
        chains_left_to_save = self.chains_to_save

        samples = []
        grouped_samples = []
        grouped_scores = []
        ground_truth = []

        ident = 0
        print(f'Sampling epoch={self.current_epoch}')

        dataloader = self.trainer.datamodule.val_dataloader()
        for data in tqdm(dataloader, total=samples_left_to_generate // dataloader.batch_size):
            if samples_left_to_generate <= 0:
                break

            data = data.to(self.device)
            bs = len(data.batch.unique())
            to_generate = bs
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            batch_groups = []
            batch_scores = []
            for sample_idx in range(self.samples_per_input):
                molecule_list, true_molecule_list, products_list, scores, _, _ = self.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=to_generate,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps_to_save=self.number_chain_steps_to_save,
                    sample_idx=sample_idx,
                )
                samples.extend(molecule_list)
                batch_groups.append(molecule_list)
                batch_scores.append(scores)
                if sample_idx == 0:
                    ground_truth.extend(true_molecule_list)

            ident += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

            # Regrouping sampled reactants for computing top-N accuracy
            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                mol_scores_group = []
                for batch_group, scores_group in zip(batch_groups, batch_scores):
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                    mol_scores_group.append(scores_group[mol_idx_in_batch])

                assert len(mol_samples_group) == self.samples_per_input
                grouped_samples.append(mol_samples_group)
                grouped_scores.append(mol_scores_group)

        to_log = compute_retrosynthesis_metrics(
            grouped_samples=grouped_samples,
            ground_truth=ground_truth,
            atom_decoder=self.dataset_infos.atom_decoder,
            grouped_scores=grouped_scores,
        )
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        to_log = self.sampling_metrics(samples)
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        self.sampling_metrics.reset()

    def apply_noise(self, X, E, y, X_T, E_T, y_T, node_mask):
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(
            alpha_bar_t=alpha_t_bar,
            X_T=X_T,
            E_T=E_T,
            y_T=y_T,
            node_mask=node_mask,
            device=self.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        assert (len(Qtb.X.shape) == 4 and len(Qtb.E.shape) == 5)
        assert (abs(Qtb.X.sum(dim=3) - 1.) < 1e-4).all(), Qtb.X.sum(dim=3) - 1
        assert (abs(Qtb.E.sum(dim=4) - 1.) < 1e-4).all()

        probX = (X.unsqueeze(-2) @ Qtb.X).squeeze(-2)  # (bs, n, dx_out)
        probE = (E.unsqueeze(-2) @ Qtb.E).squeeze(-2)  # (bs, n, n, de_out)

        sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            't_int': t_int,
            't': t_float,
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            'X_t': z_t.X,
            'E_t': z_t.E,
            'y_t': z_t.y,
            'node_mask': node_mask
        }
        return noisy_data

    @torch.no_grad()
    def sample_batch(
            self,
            data,
            batch_id,
            batch_size,
            keep_chain,
            number_chain_steps_to_save,
            save_final,
            sample_idx,
            save_true_reactants=True,
            use_one_hot=False,
    ):
        """
        :param data
        :param batch_id: int
        :param batch_size: int
        :param number_chain_steps_to_save: int
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param sample_idx: int
        :param save_true_reactants: bool
        :param use_one_hot: convert predictions to one hot before computing transition matrices
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        chain_X, chain_E, true_molecule_list, products_list, molecule_list, _, nll, ell = self.sample_chain(
            data=data,
            batch_size=batch_size,
            keep_chain=keep_chain,
            number_chain_steps_to_save=number_chain_steps_to_save,
            save_true_reactants=save_true_reactants,
            use_one_hot=use_one_hot,
        )

        if self.visualization_tools is not None:
            self.visualize(
                chain_X=chain_X,
                chain_E=chain_E,
                true_molecule_list=true_molecule_list,
                products_list=products_list,
                molecule_list=molecule_list,
                sample_idx=sample_idx,
                batch_id=batch_id,
                save_final=save_final
            )

        return molecule_list, true_molecule_list, products_list, [0] * len(molecule_list), nll, ell

    def sample_chain_no_true_no_save(self, data, batch_size, use_one_hot=False):
        # Context product
        product, node_mask = to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(node_mask)

        # Creating context
        context = product.clone() if self.use_context else None

        # Masks for fixed and modifiable nodes
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # z_T – starting state (product)
        X, E, y = product.X, product.E, torch.empty((node_mask.shape[0], 0), device=self.device)

        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in tqdm(reversed(range(0, self.T)), total=self.T):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, node_log_likelihood, edge_log_likelihood = self.sample_p_zs_given_zt(
                s=s_norm,
                t=t_norm,
                X_t=X,
                E_t=E,
                y_t=y,
                X_T=product.X,
                E_T=product.E,
                y_T=product.y,
                node_mask=node_mask,
                context=context,
                use_one_hot=use_one_hot,
            )

            # Masking unchanged part
            if self.fix_product_nodes:
                sampled_s.X = sampled_s.X * modifiable_nodes + product.X * fixed_nodes
                sampled_s = sampled_s.mask(node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        molecule_list = create_pred_reactant_molecules(X, E, data.batch, batch_size)

        return molecule_list

    def sample_chain(
            self, data, batch_size, keep_chain, number_chain_steps_to_save, save_true_reactants, use_one_hot=False
    ):
        # Context product
        product, node_mask = to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(node_mask)

        # Discrete context product
        product_discrete, _ = to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product_discrete = product_discrete.mask(node_mask, collapse=True)

        # Creating context
        context = product.clone() if self.use_context else None

        # Masks for fixed and modifiable nodes
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # z_T – starting state (product)
        X, E, y = product.X, product.E, torch.empty((node_mask.shape[0], 0), device=self.device)

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps_to_save < self.T

        chain_X_size = torch.Size((number_chain_steps_to_save, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps_to_save, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        sampled_s = None
        nll = torch.zeros(batch_size, device=X.device, dtype=torch.float64)
        ell = torch.zeros(batch_size, device=X.device, dtype=torch.float64)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in tqdm(reversed(range(0, self.T)), total=self.T):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, node_log_likelihood, edge_log_likelihood = self.sample_p_zs_given_zt(
                s=s_norm,
                t=t_norm,
                X_t=X,
                E_t=E,
                y_t=y,
                X_T=product.X,
                E_T=product.E,
                y_T=product.y,
                node_mask=node_mask,
                context=context,
                use_one_hot=use_one_hot,
            )

            # Masking unchanged part
            if self.fix_product_nodes:
                sampled_s.X = sampled_s.X * modifiable_nodes + product.X * fixed_nodes
                sampled_s = sampled_s.mask(node_mask)
                discrete_sampled_s = sampled_s.clone()
                discrete_sampled_s = discrete_sampled_s.mask(node_mask, collapse=True)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps_to_save) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            nll += node_log_likelihood
            ell += edge_log_likelihood

        # Save raw predictions for further scoring
        pred = sampled_s.clone()

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = reverse_tensor(chain_X)
            chain_E = reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.shape[0] == (number_chain_steps_to_save + 10)

        # Saving true and predicted reactants
        true_molecule_list = create_true_reactant_molecules(data, batch_size) if save_true_reactants else []
        products_list = create_input_product_molecules(data, batch_size)
        molecule_list = create_pred_reactant_molecules(X, E, data.batch, batch_size)

        return (
            chain_X, chain_E, true_molecule_list, products_list, molecule_list, pred,
            nll.detach().cpu().numpy().tolist(),
            ell.detach().cpu().numpy().tolist(),
        )

    def visualize(
            self,
            chain_X,
            chain_E,
            true_molecule_list,
            products_list,
            molecule_list,
            sample_idx,
            batch_id,
            save_final
    ):
        current_samples_path = os.path.join(self.graphs_dir, f'epoch{self.current_epoch}_b{batch_id}')
        current_chains_dir = os.path.join(self.chains_dir, f'epoch_{self.current_epoch}')

        if sample_idx == 0:
            # 1. Visualize chains
            num_molecules = chain_X.shape[1]
            for i in range(num_molecules):
                results_path = os.path.join(current_chains_dir, f'molecule_{batch_id + i}')
                os.makedirs(results_path, exist_ok=True)
                self.visualization_tools.visualize_chain(
                    path=results_path,
                    nodes_list=chain_X[:, i, :].numpy(),
                    adjacency_matrix=chain_E[:, i, :].numpy(),
                )

            # 2. Visualize true reactants
            self.visualization_tools.visualize(
                path=current_samples_path,
                molecules=true_molecule_list,
                num_molecules_to_visualize=save_final,
                prefix='true_',
            )

            # 3. Visualize input products
            self.visualization_tools.visualize(
                path=current_samples_path,
                molecules=products_list,
                num_molecules_to_visualize=save_final,
                prefix='input_product_',
            )

        # Visualize predicted reactants
        self.visualization_tools.visualize(
            path=current_samples_path,
            molecules=molecule_list,
            num_molecules_to_visualize=save_final,
            prefix=f'pred_',
            suffix=f'_{sample_idx}'
        )

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, X_T, E_T, y_T, node_mask, context=None, use_one_hot=False):
        # Hack: in direct MB we consider flipped time flow
        bs, n = X_t.shape[:2]
        t = 1 - t
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data, context=context)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        if use_one_hot:
            x_mask = node_mask.unsqueeze(-1).float()
            e_mask1 = x_mask.unsqueeze(2).float()
            e_mask2 = x_mask.unsqueeze(1).float()
            pred_X = F.one_hot(pred.X.argmax(dim=-1), num_classes=self.Xdim_output).float() * x_mask
            pred_E = F.one_hot(pred.E.argmax(dim=-1), num_classes=self.Edim_output).float() * e_mask1 * e_mask2

        # Compute transition matrices given prediction
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,
            X_T=pred_X,
            E_T=pred_E,
            y_T=y_T,
            node_mask=node_mask,
            device=self.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        # Node transition probabilities
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt.X  # bs, n, 1, d_t
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)  # bs, n, d_t
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        # Edge transition probabilities
        E_T_flat = E_t.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t)
        Qt_E_flat = Qt.E.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t-1, d_t)
        unnormalized_prob_E = E_T_flat.unsqueeze(-2) @ Qt_E_flat  # bs, N, 1, d_t
        unnormalized_prob_E = unnormalized_prob_E.squeeze(-2)  # bs, N, d_t
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        # set_trace()
        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        # Likelihood
        node_log_likelihood = torch.log(prob_X) + torch.log(pred_X)
        node_log_likelihood = (node_log_likelihood * X_s).sum(-1).sum(-1)

        edge_log_likelihood = torch.log(prob_E) + torch.log(pred_E)
        edge_log_likelihood = (edge_log_likelihood * E_s).sum(-1).sum(-1).sum(-1)

        return (
            out_one_hot.mask(node_mask).type_as(y_t),
            out_discrete.mask(node_mask, collapse=True).type_as(y_t),
            node_log_likelihood,
            edge_log_likelihood,
        )

    def compute_q_zs_given_q_zt(self, z_t, z_T, node_mask, t):
        X_t = z_t.X.to(torch.float32)
        E_t = z_t.E.to(torch.float32)

        # Hack: in direct MB we consider flipped time flow
        bs, n = X_t.shape[:2]
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)

        # Normalize predictions
        X_T = z_T.X.to(torch.float32)  # bs, n, d0
        E_T = z_T.E.to(torch.float32)  # bs, n, n, d0
        y_T = z_T.y

        # Compute transition matrices given prediction
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,
            X_T=X_T,
            E_T=E_T,
            y_T=y_T,
            node_mask=node_mask,
            device=self.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        # Node transition probabilities
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt.X  # bs, n, 1, d_t
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)  # bs, n, d_t
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        # Edge transition probabilities
        E_T_flat = E_t.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t)
        Qt_E_flat = Qt.E.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t-1, d_t)
        unnormalized_prob_E = E_T_flat.unsqueeze(-2) @ Qt_E_flat  # bs, N, 1, d_t
        unnormalized_prob_E = unnormalized_prob_E.squeeze(-2)  # bs, N, d_t
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, E_T.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X, prob_E

    def compute_p_zs_given_p_zt(self, z_t, pred, node_mask, t):
        p_X_T = F.softmax(pred.X, dim=-1)  # bs, n, d
        p_E_T = F.softmax(pred.E, dim=-1)  # bs, n, n, d

        prob_X = torch.zeros_like(p_X_T)  # bs, n, d
        prob_E = torch.zeros_like(p_E_T)  # bs, n, n, d

        for i in range(self.Xdim_output):
            X_T_i = F.one_hot(torch.ones_like(p_X_T[..., 0]).long() * i, num_classes=self.Xdim_output).float()
            E_T_i = F.one_hot(torch.zeros_like(p_E_T[..., 0]).long(), num_classes=self.Edim_output).float()
            z_T = PlaceHolder(X_T_i, E_T_i)
            prob_X_i, _ = self.compute_q_zs_given_q_zt(z_t, z_T, node_mask, t)  # bs, n, d
            prob_X += prob_X_i * p_X_T[..., i].unsqueeze(-1)  # bs, n, d

        for i in range(self.Edim_output):
            X_T_i = F.one_hot(torch.zeros_like(p_X_T[..., 0]).long(), num_classes=self.Xdim_output).float()
            E_T_i = F.one_hot(torch.ones_like(p_E_T[..., 0]).long() * i, num_classes=self.Edim_output).float()
            z_T = PlaceHolder(X_T_i, E_T_i)
            _, prob_E_i = self.compute_q_zs_given_q_zt(z_t, z_T, node_mask, t)  # bs, n, n, d
            prob_E += prob_E_i * p_E_T[..., i].unsqueeze(-1)  # bs, n, n, d

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X, prob_E

    def compute_extra_data(self, noisy_data, context=None, condition_on_t=True):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

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


if __name__ == "__main__":
    RetroBridgeModule(None, None, None, False)
