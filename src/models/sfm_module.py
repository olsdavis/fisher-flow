import copy
import os
import time
from collections import defaultdict
from typing import Any, List
import pandas as pd
import numpy as np
import yaml

from functools import partial
from typing import Any
import torch
from torch.nn import functional as F
import lightning as pl
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch.optim.optimizer import Optimizer
from torchmetrics import MeanMetric, MinMetric, MaxMetric
from torch_ema import ExponentialMovingAverage

from src.sfm import (
    OTSampler,
    estimate_categorical_kl,
    manifold_from_name,
    ot_train_step,
)

from src.models.net import expand_simplex
from src.dfm import (
    DirichletConditionalFlow,
    sample_cond_prob_path,
    simplex_proj,
    load_flybrain_designed_seqs,
    get_wasserstein_dist,
    update_ema,
)

from src.models.net import TangentWrapper, CNNModel
from src.data.components import SeiEval, upgrade_state_dict

class GeneralModule(pl.LightningModule):
    def __init__(
        self,
        validate: bool = False,
        print_freq: int = 100,
        model_dir: str = 'tmp',
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        #self.args = args
        self.validate = validate
        self.print_freq = print_freq
        self.model_dir = model_dir

        self.iter_step = -1
        self._log = defaultdict(list)
        self.generator = np.random.default_rng()
        self.last_log_time = time.time()


    def try_print_log(self):

        step = self.iter_step if self.validate else self.trainer.global_step
        if (step + 1) % self.print_freq == 0:
            print(self.model_dir)
            log = self._log
            log = {key: log[key] for key in log if "iter_" in key}

            log = self.gather_log(log, self.trainer.world_size)
            mean_log = self.get_log_mean(log)
            mean_log.update(
                {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
            if self.trainer.is_global_zero:
                print(str(mean_log))
                self.log_dict(mean_log, batch_size=1)
                for metric_name, metric in mean_log.items():
                    self.log(f'{metric_name}', metric)
            for key in list(log.keys()):
                if "iter_" in key:
                    del self._log[key]

    # def lg(self, key, data):
    #     if isinstance(data, torch.Tensor):
    #         data = data.detach().cpu().item()
    #     log = self._log
    #     if self.args.validate or self.stage == 'train':
    #         log["iter_" + key].append(data)
    #     log[self.stage + "_" + key].append(data)

    def on_train_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update(
            {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        if self.trainer.is_global_zero:
            print(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            for metric_name, metric in mean_log.items():
                self.log(f'{metric_name}', metric)

        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def on_validation_epoch_end(self):
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update(
            {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        if self.trainer.is_global_zero:
            print(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            for metric_name, metric in mean_log.items():
                self.log(f'{metric_name}', metric)

            path = os.path.join(
                self.model_dir, f"val_{self.trainer.global_step}.csv"
            )
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]



    def gather_log(self, log, world_size):
        if world_size == 1:
            return log
        log_list = [None] * world_size
        torch.distributed.all_gather_object(log_list, log)
        log = {key: sum([l[key] for l in log_list], []) for key in log}
        return log

    def get_log_mean(self, log):
        out = {}
        for key in log:
            try:
                out[key] = np.nanmean(log[key])
            except:
                pass
        return out


class SFMModule(GeneralModule):
    """
    Module for the Toy DFM dataset and DNA Enhancer dataset.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        manifold: str = "sphere",
        ot_method: str = "exact",
        kl_eval: bool = False,
        kl_samples: int = 512_000,
        label_smoothing: float | None = None,
        ema: bool = False,
        ema_decay: float = 0.99,
        tangent_euler: bool = True,
        debug_grads: bool = False,
        inference_steps: int = 100,
        dataset_type: str = "toy", # "toy", "promoter", "enhancer"
        validate: bool = False,
        print_freq: int = 100,
        cls_ckpt_hparams: str | None = None,
        clean_cls_ckpt_hparams: str | None = None,
        clean_cls_model: str = 'cnn',
        cls_ckpt: str | None = None,
        clean_cls_ckpt: str | None = None,
        target_class: int = 0,
    ) -> None:
        """
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(validate=validate, print_freq=print_freq)  

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # if basically zero or zero
        self.smoothing = label_smoothing if label_smoothing and label_smoothing > 1e-6 else None
        self.tangent_euler = tangent_euler
        self.manifold = manifold_from_name(manifold)
        self.net = TangentWrapper(self.manifold, net).to(self.device)
        if ema:
            self.ema = ExponentialMovingAverage(self.net.parameters(), decay=ema_decay).to(self.device)
        else:
            self.ema = None
        self.sampler = OTSampler(self.manifold, ot_method) if ot_method != "None" else None
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.sp_mse = MeanMetric()
        self.min_grad = MinMetric()
        self.max_grad = MaxMetric()
        self.mean_grad = MeanMetric()
        self.kl_eval = kl_eval
        self.kl_samples = kl_samples
        self.debug_grads = debug_grads
        self.inference_steps = inference_steps
        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.dataset_type = dataset_type
        self.validate = validate
        self.print_freq = print_freq
        self.cls_ckpt_hparams = cls_ckpt_hparams
        self.clean_cls_ckpt_hparams = clean_cls_ckpt_hparams
        self.clean_cls_model = clean_cls_model
        self.cls_ckpt = cls_ckpt
        self.clean_cls_ckpt = clean_cls_ckpt
        self.target_class = target_class

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'cls_model' not in k and 'distill_model' not in k}

    # def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     """Perform a forward pass through the model `self.net`."""
    #     return self.net(x, t)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.sp_mse.reset()

    def model_step(
        self, x_1: torch.Tensor, signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a single model step on a batch of data.
        """
        # points are on the simplex
        x_1 = self.manifold.project(x_1)
        return ot_train_step(
            self.manifold.smooth_labels(x_1, mx=self.smoothing) if self.smoothing else x_1,
            self.manifold,
            self.net,
            self.sampler,
            signal=signal,
            closed_form_drv=False,
        )[0]

    def training_step(
        self, batch: torch.Tensor | list[torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        self.stage = 'train'
        if isinstance(batch, list):
            seq, signal = batch
            loss = self.model_step(seq, signal)
        else:
            seq = batch
            loss = self.model_step(seq)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/perplexity', torch.exp(loss.mean())[None].expand(seq.size(0)).mean())
        return loss

    def validation_step(self, batch: torch.Tensor | list[torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if isinstance(batch, list):
            seq, signal = batch
            loss = self.model_step(seq, signal)
        else:
            seq = batch
            loss = self.model_step(seq)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/perplexity', torch.exp(loss.mean())[None].expand(seq.size(0)).mean())
        if self.dataset_type == 'promoter':
            eval_model = partial(self.net, signal=signal)
            pred = self.manifold.tangent_euler(
                self.manifold.uniform_prior(*seq.shape[:-1], 4).to(seq.device),
                eval_model,
                steps=self.inference_steps,
                tangent=self.tangent_euler,
            )
            mx = torch.argmax(pred, dim=-1)
            one_hot = F.one_hot(mx, num_classes=4)
            mse = SeiEval().eval_sp_mse(one_hot, seq, batch_idx)
            self.sp_mse(mse)
            self.log("val/sp-mse", self.sp_mse, on_step=False, on_epoch=True, prog_bar=True)
        elif self.dataset_type == 'enhancer':
            self.stage = "val"
            self.general_step(batch, batch_idx)

    def test_step(self, batch: torch.Tensor | list[torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if isinstance(batch, list):
            seq, signal = batch
            loss = self.model_step(seq, signal)
        else:
            seq = batch
            loss = self.model_step(seq)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/perplexity', torch.exp(loss.mean())[None].expand(seq.size(0)).mean())
        if self.dataset_type == 'promoter':
            eval_model = partial(self.net, signal=signal)
            pred = self.manifold.tangent_euler(
                self.manifold.uniform_prior(*seq.shape[:-1], 4).to(seq.device),
                eval_model,
                steps=self.inference_steps,
                tangent=self.tangent_euler,
            )
            mx = torch.argmax(pred, dim=-1)
            one_hot = F.one_hot(mx, num_classes=4)
            mse = SeiEval().eval_sp_mse(one_hot, seq, batch_idx)
            self.sp_mse(mse)
            self.log("test/sp-mse", self.sp_mse, on_step=False, on_epoch=True, prog_bar=True)
        elif self.dataset_type == 'enhancer':
            self.stage = "val"
            self.general_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        self.train_out_initialized = True
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update(
            {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        if self.trainer.is_global_zero:
            print(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            for name, metric in mean_log.items():
                self.log(name, metric)

        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        if self.kl_eval:
            # evaluate KL
            real_probs = self.trainer.val_dataloaders.dataset.probs.to(self.device)
            kl = estimate_categorical_kl(
                self.net,
                self.manifold,
                real_probs,
                self.kl_samples // 10,
                batch=self.hparams.get("kl_batch", 2048),
                silent=True,
                tangent=self.tangent_euler,
                inference_steps=self.inference_steps,
            )
            self.log("val/kl", kl, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.dataset_type == 'enhancer':
            #import pdb; pdb.set_trace()
            if self.trainer.is_global_zero:
                print("on_validation_epoch_end")
            self.generator = np.random.default_rng()
            log = self._log
            log = {key: log[key] for key in log if "val_" in key}
            log = self.gather_log(log, self.trainer.world_size)
            mean_log = self.get_log_mean(log)
            mean_log.update(
                {'val_nan_inf_step_fraction': self.nan_inf_counter / self.inf_counter}
            )
            mean_log.update(
                {
                    'epoch': float(self.trainer.current_epoch),
                    'step': float(self.trainer.global_step), 
                    'iter_step': float(self.iter_step),
                }
            )
            if self.clean_cls_ckpt:
                if not self.target_class == self.net.num_cls:
                    probs = torch.softmax(torch.cat(self.val_outputs['logits_cleancls_generated']), dim=-1) # (10505, 81)
                    target_prob = probs[:, self.target_class]
                    mean_log.update({'cleancls_mean_target_prob': target_prob.detach().cpu().mean()})
                # calculate FID/FXD metrics:
                embeds_gen = torch.cat(self.val_outputs['embeddings_cleancls_generated']).detach().cpu().numpy() # (10505, 128)
                if not self.validate:
                    train_clss = torch.cat(self.train_outputs['clss_cleancls']).squeeze().detach().cpu().numpy() # (83726, )
                    train_embeds = torch.cat(self.train_outputs['embeddings_cleancls']).detach().cpu().numpy() # (83726, 128)
                    mean_log.update({'val_fxd_generated_to_allseqs_allTrainSet': get_wasserstein_dist(embeds_gen, train_embeds)})
                    if not self.target_class == self.net.num_cls:
                        embeds_cls_specific = train_embeds[train_clss == self.target_class] # (2562, 128)
                        mean_log.update({'val_fxd_generated_to_targetclsseqs_allTrainSet': get_wasserstein_dist(embeds_gen, embeds_cls_specific)})
                clss = torch.cat(self.val_outputs['clss_cleancls']).squeeze().detach().cpu().numpy() # (10505, )
                embeds = torch.cat(self.val_outputs['embeddings_cleancls']).detach().cpu().numpy() # (10505, 128)
                embeds_rand = torch.randint(0,4, size=embeds_gen.shape).numpy()
                mean_log.update({'val_fxd_randseq_to_allseqs': get_wasserstein_dist(embeds_rand, embeds)})
                mean_log.update({'val_fxd_generated_to_allseqs': get_wasserstein_dist(embeds_gen, embeds)})
                if not self.target_class == self.net.num_cls:
                    embeds_cls_specific = embeds[clss == self.target_class] # (326, 128)
                    mean_log.update({'val_fxd_generated_to_targetclsseqs': get_wasserstein_dist(embeds_gen, embeds_cls_specific)})
                if self.taskiran_seq_path is not None:
                    embeds_taskiran = torch.cat(self.val_outputs['embeddings_cleancls_taskiran']).detach().cpu().numpy()
                    mean_log.update({'val_fxd_taskiran_to_allseqs': get_wasserstein_dist(embeds_taskiran, embeds)})
                    if not self.target_class == self.net.num_cls:
                        mean_log.update({'val_fxd_taskiran_to_targetclsseqs': get_wasserstein_dist(embeds_taskiran, embeds_cls_specific)})
            self.mean_log_ema = update_ema(current_dict=mean_log, prev_ema=self.mean_log_ema, gamma=0.9)
            mean_log.update(self.mean_log_ema)
            if self.trainer.is_global_zero:
                print(str(mean_log))
                self.log_dict(mean_log, batch_size=1)

            for key in list(log.keys()):
                if "val_" in key:
                    del self._log[key]
            self.val_outputs = defaultdict(list)


    def general_step(self, batch: torch.Tensor | list[torch.Tensor], batch_idx: int):
        self.iter_step += 1
        if isinstance(batch, list):
            seq, signal = batch
        else:
            seq = batch
        B = seq.size(0)

        if self.stage == "val":
            eval_model = partial(self.net, signal=signal)
            pred = self.manifold.tangent_euler(
                self.manifold.uniform_prior(*seq.shape[:-1], self.net.dim).to(seq.device),
                eval_model,
                steps=self.inference_steps,
                tangent=self.tangent_euler,
            )
            seq_pred = torch.argmax(pred, dim=-1)
            self.val_outputs['seqs'].append(seq_pred.cpu())
            if self.cls_ckpt is not None:
                #self.run_cls_model(seq_pred, cls, log_dict=self.val_outputs, clean_data=False, postfix='_noisycls_generated', generated=True)
                self.run_cls_model(seq, signal, log_dict=self.val_outputs, clean_data=False, postfix='_noisycls', generated=False)
            if self.clean_cls_ckpt is not None:
                self.run_cls_model(seq_pred, signal, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls_generated', generated=True)
                self.run_cls_model(seq, signal, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls', generated=False)
                if self.taskiran_seq_path is not None:
                    indices = torch.randperm(len(self.taskiran_fly_seqs))[:B].to(self.device)
                    self.run_cls_model(self.taskiran_fly_seqs[indices].to(self.device), signal, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls_taskiran', generated=True)
        
        self.log('dur', time.time() - self.last_log_time)
        if not self.train_out_initialized and self.clean_cls_ckpt is not None:
            self.run_cls_model(seq, signal, log_dict=self.train_outputs, clean_data=True, postfix='_cleancls', generated=False, run_log=False)
        self.last_log_time = time.time()

    @torch.no_grad()
    def run_cls_model(self, seq, cls, log_dict, clean_data=False, postfix='', generated=False, run_log=True):
        cls = cls.squeeze()
        if generated:
            cls = (torch.ones_like(cls,device=self.device) * self.target_class).long()

        xt, alphas = xxx #sample_cond_prob_path(self.mode, self.fix_alpha, self.alpha_scale, seq, self.net.dim)
        if self.cls_expanded_simplex:
            xt, _ = xxx #expand_simplex(xt, alphas, self.prior_pseudocount)

        cls_model = self.clean_cls_model if clean_data else self.cls_model
        logits, embeddings = cls_model(xt if not clean_data else seq, t=alphas, return_embedding=True)
        cls_pred = torch.argmax(logits, dim=-1)

        if run_log:
            if not self.target_class == self.net.num_cls:
                losses = self.crossent_loss(logits, cls)
                self.log(f'cls_loss{postfix}', losses.mean())
                self.log(f'cls_accuracy{postfix}', cls_pred.eq(cls).float().mean())

        log_dict[f'embeddings{postfix}'].append(embeddings.detach().cpu())
        log_dict[f'clss{postfix}'].append(cls.detach().cpu())
        log_dict[f'logits{postfix}'].append(logits.detach().cpu())
        log_dict[f'alphas{postfix}'].append(alphas.detach().cpu())
        if not clean_data and not self.target_class == self.net.num_cls: # num_cls stands for the masked class
            scores = self.get_cls_score(xt, alphas)
            log_dict[f'scores{postfix}'].append(scores.detach().cpu())

    def on_validation_epoch_start(self):
        if not self.loaded_classifiers:
            self.load_classifiers(load_cls=self.cls_ckpt is not None, load_clean_cls=self.clean_cls_ckpt is not None)
            self.loaded_classifiers = True
        self.inf_counter = 1
        self.nan_inf_counter = 0

    @torch.no_grad()
    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_train_epoch_start(self):
        self.inf_counter = 1
        self.nan_inf_counter = 0
        if not self.loaded_classifiers:
            self.load_classifiers(load_cls=self.cls_ckpt is not None, load_clean_cls=self.clean_cls_ckpt is not None)
            self.loaded_classifiers = True
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.debug_grads:
            norms = grad_norm(self.net, norm_type=2).values()
            self.min_grad(min(norms))
            self.max_grad(max(norms))
            self.mean_grad(sum(norms) / len(norms))
            self.log("train/min_grad", self.min_grad, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/max_grad", self.max_grad, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/mean_grad", self.mean_grad, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        """Evaluates KL if required."""
        if self.kl_eval:
            # evaluate KL
            real_probs = self.trainer.test_dataloaders.dataset.probs.to(self.device)
            kl = estimate_categorical_kl(
                self.net,
                self.manifold,
                real_probs,
                self.kl_samples,
                batch=self.hparams.get("kl_batch", 2048),
                tangent=self.tangent_euler,
            )
            self.log("test/kl", kl, on_step=False, on_epoch=True, prog_bar=False)

    def load_classifiers(self, load_cls, load_clean_cls, requires_grad = False):
        if self.trainer.is_global_zero:
            print("load_classifiers")
        if load_cls:
            with open(self.cls_ckpt_hparams) as f:
                hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            if self.cls_model == 'cnn':
                self.cls_model = CNNModel(
                    dim=self.net.dim, # the same as alphabet_size
                    hidden=hparams.hidden,
                    mode=hparams.mode,
                    num_cls=self.net.num_cls, 
                    depth=hparams.depth,
                    dropout=hparams.dropout,
                    prior_pseudocount=hparams.prior_pseudocount,
                    cls_expanded_simplex=hparams.cls_expanded_simplex,
                    clean_data=hparams.clean_data,
                    classifier=True,
                    classifier_free_guidance=hparams.classifier_free_guidance,
                )
            elif self.cls_model == 'mlp':
                #self.cls_model = MLPModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
                raise NotImplementedError()
            elif self.cls_model == 'transformer':
                #self.cls_model = TransformerModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
                raise NotImplementedError()
            elif self.cls_model == 'deepflybrain':
                #self.cls_model = DeepFlyBrainModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            self.cls_model.load_state_dict(upgrade_state_dict(torch.load(self.cls_ckpt, map_location=self.device)['state_dict'],prefixes=['model.']))
            self.cls_model.eval()
            self.cls_model.to(self.device)
            for param in self.cls_model.parameters():
                param.requires_grad = requires_grad

        if load_clean_cls:
            with open(self.clean_cls_ckpt_hparams) as f:
                hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            if self.clean_cls_model == 'cnn':
                #import pdb; pdb.set_trace()
                self.clean_cls_model = CNNModel(
                    dim=self.net.dim, # the same as alphabet_size, 
                    hidden=hparams['args'].hidden_dim,
                    mode=hparams['args'].mode,
                    num_cls=self.net.num_cls,
                    depth=hparams['args'].num_cnn_stacks,
                    dropout=hparams['args'].dropout,
                    prior_pseudocount=hparams['args'].prior_pseudocount,
                    cls_expanded_simplex=hparams['args'].cls_expanded_simplex,
                    clean_data=hparams['args'].clean_data,
                    classifier=True,
                    classifier_free_guidance=hparams['args'].cls_free_guidance,
                )
            elif self.clean_cls_model == 'mlp':
                raise NotImplementedError()
                #self.clean_cls_model = MLPModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.clean_cls_model == 'transformer':
                raise NotImplementedError()
                #self.clean_cls_model = TransformerModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.clean_cls_model == 'deepflybrain':
                raise NotImplementedError()
                #self.clean_cls_model = DeepFlyBrainModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            else:
                raise NotImplementedError()
            self.clean_cls_model.load_state_dict(upgrade_state_dict(torch.load(self.clean_cls_ckpt, map_location=self.device)['state_dict'], prefixes=['model.']))
            self.clean_cls_model.eval()
            self.clean_cls_model.to(self.device)
            for param in self.clean_cls_model.parameters():
                param.requires_grad = requires_grad

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()

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
        optimizer = self.hparams.optimizer(params=self.net.parameters())
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
    SFMModule(None, None, None, False)
