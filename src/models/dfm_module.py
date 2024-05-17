# Adaptation of: https://github.com/HannesStark/dirichlet-flow-matching/blob/main/lightning_modules/dna_module.py
import copy
import os
import time
from collections import defaultdict
from typing import Any, List
import pandas as pd
import numpy as np
import yaml

import lightning as pl
import torch
from torch.distributions import Dirichlet
from torchmetrics import MeanMetric

from selene_sdk.utils import NonStrandSpecific

from src.models.net import expand_simplex
from src.dfm import (
    DirichletConditionalFlow,
    sample_cond_prob_path,
    simplex_proj,
    load_flybrain_designed_seqs,
    get_wasserstein_dist,
    update_ema,
)

from src.data.components import Sei, upgrade_state_dict
from src.models.net import PromoterModel, CNNModel

"""
test module import

> python -m src.models.dfm_module

Run DNA enhancer experiment with Dirichlet flow matching.

> python -m src.train experiment=enhancer_dfm logger=wandb
"""

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

class DNAModule(GeneralModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        mode: str,
        kl_samples: int = 512_000,
        vectorfield_addition: bool = False,
        guidance_scale: float = 0.5,
        cls_expanded_simplex: bool = False,
        taskiran_seq_path: str | None = None,
        alpha_max: float = 8.0,
        alpha_scale: float = 2.0,
        fix_alpha: float | None = None,
        prior_pseudocount: float = 2.0,
        cls_free_guidance: bool = False,
        binary_guidance: bool = False,
        num_integration_steps: int = 100,
        scale_cls_score: bool = False,
        analytic_cls_score: bool = False,
        target_class: int = 0,
        cls_free_noclass_ratio: float = 0.3,
        dataset_type: str = 'toy_sampled',
        cls_ckpt: str | None = None,
        clean_cls_ckpt: str | None = None,
        score_free_guidance: bool = False,
        all_class_inference: bool = False,
        probability_tilt: bool = False,
        probability_addition: bool = False,
        adaptive_prob_add: bool = False,
        flow_temp: float = 1.0,
        cls_guidance: bool = False,
        sep_x_y: bool = False,
        distill_ckpt: str | None = None,
        distill_ckpt_hparams: str | None = None,
        cls_ckpt_hparams: str | None = None,
        clean_cls_ckpt_hparams: str | None = None,
        clean_cls_model: str = 'cnn',
        cls_model: str = 'cnn',
        allow_nan_cfactor: str = False,
        toy_simplex_dim: int = None,
        print_freq: int = 100,
        validate: bool = False,
    ):
        super().__init__(validate=validate, print_freq=print_freq)  
        self.save_hyperparameters(logger=False)
        self.kl_samples = kl_samples
        self.cls_guidance = cls_guidance
        self.flow_temp = flow_temp
        self.alpha_max = alpha_max
        self.adaptive_prob_add = adaptive_prob_add
        self.probability_addition = probability_addition
        self.probability_tilt = probability_tilt
        self.all_class_inference = all_class_inference
        self.score_free_guidance = score_free_guidance
        self.taskiran_seq_path = taskiran_seq_path
        self.cls_ckpt = cls_ckpt
        self.clean_cls_ckpt = clean_cls_ckpt
        self.dataset_type = dataset_type
        self.cls_free_noclass_ratio = cls_free_noclass_ratio
        self.target_class = target_class
        self.analytic_cls_score = analytic_cls_score
        self.scale_cls_score = scale_cls_score
        self.num_integration_steps = num_integration_steps
        self.binary_guidance = binary_guidance
        self.cls_free_guidance = cls_free_guidance
        self.prior_pseudocount = prior_pseudocount
        self.fix_alpha = fix_alpha
        self.alpha_scale = alpha_scale
        self.cls_expanded_simplex = cls_expanded_simplex
        self.vectorfield_addition = vectorfield_addition
        self.guidance_scale = guidance_scale
        self.mode = mode
        self.net = torch.compile(net) if compile else net
        self.condflow = DirichletConditionalFlow(K=self.net.dim, alpha_spacing=0.001, alpha_max=alpha_max)
        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_outputs: dict[str, list] = defaultdict(list)
        self.train_outputs: dict[str, list] = defaultdict(list)
        self.train_out_initialized = False
        self.loaded_classifiers = False
        self.loaded_distill_model = False
        if taskiran_seq_path is not None:
            self.taskiran_fly_seqs = load_flybrain_designed_seqs(taskiran_seq_path).to(self.device)
        # added:
        self.stage = "train"
        self.sep_x_y = sep_x_y
        self.distill_ckpt = distill_ckpt
        self.distill_ckpt_hparams = distill_ckpt_hparams
        self.cls_ckpt_hparams = cls_ckpt_hparams
        self.clean_cls_ckpt_hparams = clean_cls_ckpt_hparams
        self.cls_model = cls_model
        self.clean_cls_model = clean_cls_model
        self.allow_nan_cfactor = allow_nan_cfactor

        self.toy_simplex_dim = toy_simplex_dim

        self.mean_log_ema = {}

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'cls_model' not in k and 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.sep_x_y:
            seq, _ = batch
        else:
            seq = batch
        self.log('train/perplexity', torch.exp(loss.mean())[None].expand(seq.size(0)).mean())
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.sep_x_y:
            seq, _ = batch
        else:
            seq = batch
        self.log('val/perplexity', torch.exp(loss.mean())[None].expand(seq.size(0)).mean())

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.sep_x_y:
            seq, _ = batch
        else:
            seq = batch
        self.log('test/perplexity', torch.exp(loss.mean())[None].expand(seq.size(0)).mean())

    @torch.inference_mode()
    def estimate_kl(self, real_probs, n_samples: int):
        with torch.inference_mode():
            to_draw = n_samples
            acc = torch.zeros((self.net.k, self.net.dim), device=self.device)
            concentration = torch.ones((self.net.k, self.net.dim), device=self.device)
            while to_draw > 0:
                n = min(to_draw, 2048)
                x_0 = Dirichlet(concentration).sample((n,))
                samples = self.dirichlet_flow_inference(x_0, None, self.net)[1]
                acc += torch.nn.functional.one_hot(samples.argmax(dim=-1), self.net.dim).sum(dim=0)
                to_draw -= n
        acc /= acc.sum(dim=-1, keepdim=True)
        real_probs = real_probs.to(self.device)
        kl = (acc * (acc.log() - real_probs.log())).sum(dim=-1).mean().item()
        return kl

    def on_test_epoch_end(self):
        self.log("test/kl", self.estimate_kl(self.trainer.test_dataloaders.dataset.probs.to(self.device), self.kl_samples), on_step=False, on_epoch=True, prog_bar=False)

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        if self.sep_x_y:
            seq, cls = batch
        else:
            seq = batch
            cls = None
        B = seq.size(0)
        xt, alphas = sample_cond_prob_path(self.mode, self.fix_alpha, self.alpha_scale, seq.long(), self.net.dim)
        if self.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
            else:
                logits_distill, xt = self.dirichlet_flow_inference(seq, cls, model=self.net)
                seq_distill = torch.argmax(logits_distill, dim=-1)
            alphas = alphas * 0
        xt_inp = xt
        if self.mode == 'dirichlet' or self.mode == 'riemannian':
            xt_inp, _ = expand_simplex(xt,alphas, self.prior_pseudocount)

        if self.cls_free_guidance:
            if self.binary_guidance:
                cls_inp = cls.clone()
                cls_inp[cls != self.target_class] = self.net.num_cls
            else:

                cls_inp = torch.where(torch.rand(B, device=self.device) >= self.cls_free_noclass_ratio, cls.squeeze(), self.net.num_cls) # set fraction of the classes to the unconditional class
        else:
            cls_inp = None

        logits = self.net(xt_inp, t=alphas, cls=cls_inp)

        if self.mode == 'sfm':
            seq = seq.argmax(dim=-1)

        losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),
            seq_distill if self.mode == 'distill' else seq,
            reduction='mean',
        )

        #self.log('perplexity', torch.exp(losses.mean())[None].expand(B))
        if self.stage == "val":
            if self.mode == 'dirichlet':
                logits_pred, _ = self.dirichlet_flow_inference(seq, cls, model=self.net)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.mode == 'riemannian':
                logits_pred = self.riemannian_flow_inference(seq)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.mode == 'ardm' or self.mode == 'lrar':
                seq_pred = self.ar_inference(seq)
            elif self.mode == 'distill':
                logits_pred = self.distill_inference(seq)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            else:
                raise NotImplementedError()

            if self.dataset_type == 'toy_fixed':
                self.log_data_similarities(seq_pred)

            self.val_outputs['seqs'].append(seq_pred.cpu())
            if self.cls_ckpt is not None:
                #self.run_cls_model(seq_pred, cls, log_dict=self.val_outputs, clean_data=False, postfix='_noisycls_generated', generated=True)
                self.run_cls_model(seq, cls, log_dict=self.val_outputs, clean_data=False, postfix='_noisycls', generated=False)
            if self.clean_cls_ckpt is not None:
                self.run_cls_model(seq_pred, cls, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls_generated', generated=True)
                self.run_cls_model(seq, cls, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls', generated=False)
                if self.taskiran_seq_path is not None:
                    indices = torch.randperm(len(self.taskiran_fly_seqs))[:B].to(self.device)
                    self.run_cls_model(self.taskiran_fly_seqs[indices].to(self.device), cls, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls_taskiran', generated=True)
        
        self.log('dur', time.time() - self.last_log_time)
        if not self.train_out_initialized and self.clean_cls_ckpt is not None:
            self.run_cls_model(seq, cls, log_dict=self.train_outputs, clean_data=True, postfix='_cleancls', generated=False, run_log=False)
        self.last_log_time = time.time()
        return losses

    @torch.no_grad()
    def distill_inference(self, seq):
        B, L = seq.shape
        K = self.net.dim
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K, device=seq.device)).sample()
        logits = self.net(x0, t=torch.zeros(B, device=self.device))
        return logits

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq, cls, model):
        if self.dataset_type == 'enhancer':
            B, L = seq.shape
            K = model.dim # the same as alphabet_size
        else:
            B, L, K = seq.shape
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K, device=seq.device)).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()

        t_span = torch.linspace(1, self.alpha_max, self.num_integration_steps, device=self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            xt_expanded, _ = expand_simplex(xt, s[None].expand(B), self.prior_pseudocount)
            if self.cls_free_guidance:
                logits = model(xt_expanded, t=s[None].expand(B), cls=cls if self.all_class_inference else (torch.ones(B, device=self.device) * self.target_class).long())
                probs_cond = torch.nn.functional.softmax(logits / self.flow_temp, -1)  # [B, L, K]
                if self.score_free_guidance:
                    flow_probs = probs_cond
                else:
                    logits_uncond = model(xt_expanded, t=s[None].expand(B), cls=(torch.ones(B, device=self.device) * model.num_cls).long())
                    probs_unccond = torch.nn.functional.softmax(logits_uncond / self.flow_temp, -1)  # [B, L, K]
                    if self.probability_tilt:
                        flow_probs = probs_cond ** (1 - self.guidance_scale) * probs_unccond ** (self.guidance_scale)
                        flow_probs = flow_probs / flow_probs.sum(-1)[...,None]
                    elif self.probability_addition:
                        if self.adaptive_prob_add:
                            #TODO this is wrong for some reason and we get negative probabilities ?!?!??!
                            potential_scales = probs_cond / (probs_cond - probs_unccond)
                            max_guide_scale = potential_scales.min(-1)[0]
                            flow_probs = probs_cond * (1 - max_guide_scale[...,None]) + probs_unccond * max_guide_scale[...,None]
                        else:
                            flow_probs = probs_cond * self.guidance_scale + probs_unccond *(1 - self.guidance_scale)
                    else:
                        flow_probs = self.get_cls_free_guided_flow(xt, s + 1e-4, logits_uncond, logits)

            else:
                logits = model(xt_expanded, t=s[None].expand(B))
                flow_probs = torch.nn.functional.softmax(logits / self.flow_temp, -1) # [B, L, K]

            if self.cls_guidance:
                probs_cond, _ = self.get_cls_guided_flow(xt, s + 1e-4, flow_probs)
                flow_probs = probs_cond * self.guidance_scale + flow_probs * (1 - self.guidance_scale)

            if not torch.allclose(flow_probs.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)

            self.inf_counter += 1
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
                if self.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                    self.nan_inf_counter += 1
                else:
                    raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            if self.vectorfield_addition:
                flow_cond = (probs_cond.unsqueeze(-2) * cond_flows).sum(-1)
                flow_uncond = (probs_unccond.unsqueeze(-2) * cond_flows).sum(-1)
                flow = flow_cond * self.guidance_scale + (1 - self.guidance_scale) * flow_uncond

            xt = xt + flow * (t - s)

            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt)
        return logits, xt  # NOTE: this used to be x0; but it seems that it was a mistake

    @torch.no_grad()
    def riemannian_flow_inference(self, seq):
        if self.dataset_type == 'enhancer':
            B, L = seq.shape
            K = self.net.dim # the same as alphabet_size
        else:
            B, L, K = seq.shape
        xt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        eye = torch.eye(K).to(self.device)

        t_span = torch.linspace(0, 1, self.num_integration_steps, device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            xt_expanded, _ = expand_simplex(xt, s[None].expand(B), self.prior_pseudocount)
            logits = self.net(xt_expanded, s[None].expand(B))
            probs = torch.nn.functional.softmax(logits, -1)
            cond_flows = (eye - xt.unsqueeze(-1)) / (1 - s)
            flow = (probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
        return xt
    
    @torch.no_grad()
    def ar_inference(self, seq, signal):
        B, L = seq.shape
        order = np.arange(L)
        if self.mode =='ardm': np.random.shuffle(order)
        curr = (torch.ones((B, L), device=self.device) * 4).long()
        for i, k in enumerate(order):
            t = torch.tensor( i/L ,device=self.device)
            logits = self.model(torch.nn.functional.one_hot(curr, num_classes=5), signal, t[None].expand(B))
            curr[:, k] = torch.distributions.Categorical(probs=torch.nn.functional.softmax(logits[:, k] / self.flow_temp, -1)).sample()
        return curr

    def get_cls_free_guided_flow(self, xt, alpha, logits, logits_cond,):
        B, L, K = xt.shape
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_cond = torch.nn.functional.softmax(logits_cond, dim=-1)

        cond_scores_mats = ((alpha - 1) * (torch.eye(self.net.dim).to(xt)[None, :] / xt[..., None]))  # [B, L, K, K]

        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(2)[:, :, None, :]  # [B, L, K, K] now the columns sum up to 0

        score = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, probs)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        score_cond = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, probs_cond)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        score_guided = (1 - self.guidance_scale) * score + self.guidance_scale * score_cond

        Q_mats = cond_scores_mats.clone()  # [B, L, K, K]
        Q_mats[:, :, -1, :] = torch.ones((B, L, K))  # [B, L, K, K]
        score_guided_ = score_guided.clone()  # [B, L, K]
        score_guided_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        flow_guided = torch.linalg.solve(Q_mats, score_guided_)  # [B, L, K]
        return flow_guided

    def get_cls_guided_flow(self, xt, alpha, p_x0_given_xt):
        B, L, K = xt.shape
        # get the matrix of scores of the conditional probability flows for each simplex corner
        cond_scores_mats = ((alpha - 1) * (torch.eye(self.net.dim).to(xt)[None, :] / xt[..., None]))  # [B, L, K, K]
        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(2)[:, :, None, :]  # [B, L, K, K] now the columns sum up to 0
        # assert torch.allclose(cond_scores_mats.sum(2), torch.zeros((B, L, K)),atol=1e-4), cond_scores_mats.sum(2)

        score = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, p_x0_given_xt)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        # assert torch.allclose(score.sum(2), torch.zeros((B, L)),atol=1e-4)

        cls_score = self.get_cls_score(xt, alpha[None].expand(B))
        if self.scale_cls_score:
            norm_score = torch.norm(score, dim=2, keepdim=True)
            norm_cls_score = torch.norm(cls_score, dim=2, keepdim=True)
            cls_score = torch.where(norm_cls_score != 0, cls_score * norm_score / norm_cls_score, cls_score)
        guided_score = cls_score + score

        Q_mats = cond_scores_mats.clone()  # [B, L, K, K]
        Q_mats[:, :, -1, :] = torch.ones((B, L, K))  # [B, L, K, K]
        guided_score_ = guided_score.clone()  # [B, L, K]
        guided_score_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_x0_given_xt_y = torch.linalg.solve(Q_mats, guided_score_) # [B, L, K]
        """
        # for debugging whether these probabilities also have negative entries and are off of the simplex in other ways
        cls_score_ = cls_score.clone()  # [B, L, K]
        cls_score_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_xt_given_y = torch.linalg.solve(Q_mats, cls_score_)

        score_guided_ = score.clone()  # [B, L, K]
        score_guided_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_x0_given_xt_back = torch.linalg.solve(Q_mats, score_guided_)
        """
        if torch.isnan(p_x0_given_xt_y).any():
            print("Warning: there were this many nans in the probs_cond of the classifier score: ", torch.isnan(p_x0_given_xt_y).sum(), "We are setting them to 0.")
            p_x0_given_xt_y = torch.nan_to_num(p_x0_given_xt_y)
        return p_x0_given_xt_y, cls_score

    def get_cls_score(self, xt, alpha):
        with torch.enable_grad():
            xt_ = xt.clone().detach().requires_grad_(True)
            xt_.requires_grad = True
            if self.cls_expanded_simplex:
                xt_, _ = expand_simplex(xt, alpha[None].expand(xt_.shape[0]), self.prior_pseudocount)
            if self.analytic_cls_score:
                assert self.dataset_type == 'toy_fixed', 'analytic_cls_score can only be calculated for fixed dataset'
                B, _, _ = xt.shape

                x0_given_y = self.toy_data.data_class1.to(self.device) # num_seq, L
                x0_given_y_expanded = x0_given_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1,-1) # B, num_seq, L, 1
                xt_expanded = xt_.unsqueeze(1).expand(-1,x0_given_y_expanded.shape[1], -1, -1) # B, num_seq, L, K
                selected_xt = torch.gather(xt_expanded, dim=3, index=x0_given_y_expanded).squeeze() # [B, num_seq, L] where the indices of cls1_data were used to select entries in the K dimension
                p_xt_given_x0_y = selected_xt ** (alpha[:,None,None] - 1) # [B, num_seq, L]
                p_xt_given_y = p_xt_given_x0_y.mean(1)  # [B, L] because the probs for each x0 are the same

                x0_all = torch.cat([self.toy_data.data_class1, self.toy_data.data_class2], dim= 0).to(self.device)  # num_seq * 2, L
                x0_expanded = x0_all.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)  # B, num_seq*2, L, 1
                xt_expanded = xt_.unsqueeze(1).expand(-1, x0_expanded.shape[1], -1, -1)  # B, num_seq*2, L, K
                selected_xt = torch.gather(xt_expanded, dim=3,index=x0_expanded).squeeze()  # [B, num_seq, L] where the indices of cls1_data were used to select entries in the K dimension
                p_xt_given_x0 = selected_xt ** (alpha[:, None, None] - 1)  # [B, num_seq, L]
                p_xt = p_xt_given_x0.mean(1)  # [B, L] because the probs for each x0 are the same

                p_y_given_xt = p_xt_given_y / 2 / p_xt # [B,L] divide by 2 becaue that is p(y) (but it does not really matter because it is just a constant)
                p_y_given_xt = p_y_given_xt.prod(-1) # per sequence probabilities. Works because positions are independent.
                log_prob = torch.log(p_y_given_xt).sum()
                cls_score = torch.autograd.grad(log_prob, [xt_])[0]
            else:
                cls_logits = self.cls_model(xt_, t=alpha)
                loss = torch.nn.functional.cross_entropy(cls_logits, torch.ones(len(xt), dtype=torch.long, device=xt.device) * self.target_class).mean()
                assert not torch.isnan(loss).any()
                cls_score = - torch.autograd.grad(loss,[xt_])[0]  # need the minus because cross entropy loss puts a minus in front of log probability.
                assert not torch.isnan(cls_score).any()
        cls_score = cls_score - cls_score.mean(-1)[:,:,None]
        return cls_score.detach()

    @torch.no_grad()
    def run_cls_model(self, seq, cls, log_dict, clean_data=False, postfix='', generated=False, run_log=True):
        cls = cls.squeeze()
        if generated:
            cls = (torch.ones_like(cls,device=self.device) * self.target_class).long()

        xt, alphas = sample_cond_prob_path(self.mode, self.fix_alpha, self.alpha_scale, seq, self.net.dim)
        if self.cls_expanded_simplex:
            xt, _ = expand_simplex(xt, alphas, self.prior_pseudocount)

        cls_model = self.clean_cls_model if clean_data else self.cls_model
        logits, embeddings = cls_model(xt if not clean_data else seq, t=alphas, return_embedding=True)
        cls_pred = torch.argmax(logits, dim=-1)

        if run_log:
            if not self.target_class == self.net.num_cls:
                losses = self.crossent_loss(logits, cls)
                # self.log(f'cls_loss{postfix}', losses)
            # self.log(f'cls_accuracy{postfix}', cls_pred.eq(cls).float())

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

    def on_validation_epoch_end(self):
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
        if self.dataset_type == 'toy_sampled':
            all_seqs = torch.cat(self.val_outputs['seqs'], dim=0).cpu()
            all_seqs_one_hot = torch.nn.functional.one_hot(all_seqs, num_classes=self.toy_simplex_dim)
            counts = all_seqs_one_hot.sum(0).float()
            empirical_dist = counts / counts.sum(dim=-1, keepdim=True)
            kl = (empirical_dist * (torch.log(empirical_dist) - torch.log(self.toy_data.probs[self.target_class]))).sum(-1).mean()
            rkl = (self.toy_data.probs[self.target_class] * (torch.log(self.toy_data.probs[self.target_class]) - torch.log(empirical_dist))).sum(-1).mean()
            sanity_self_kl = (empirical_dist * (torch.log(empirical_dist) - torch.log(empirical_dist))).sum(-1).mean()
            mean_log.update({'val_kl': kl.cpu().item(), 'val_rkl': rkl.cpu().item(), 'val_sanity_self_kl': sanity_self_kl.cpu().item()})

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
            #wandb.log(mean_log)
            if self.dataset_type == 'toy_sampled':
                #pil_dist_comp = self.plot_empirical_and_true(empirical_dist, self.toy_data.probs[self.args.target_class])
                #wandb.log({'fig': [wandb.Image(pil_dist_comp)], 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
                pass
            if self.cls_ckpt is not None:
                #pil_probs, pil_score_norms = self.plot_score_and_probs()
                #wandb.log({'fig': [wandb.Image(pil_probs), wandb.Image(pil_score_norms)], 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
                pass

            # path = os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv")
            # pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]
        self.val_outputs = defaultdict(list)

        if self.trainer.is_global_zero:
            # print("on_validation_epoch_end_end")
            pass

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            #logger.info("on_train_epoch_start")
            pass
        self.inf_counter = 1
        self.nan_inf_counter = 0
        if not self.loaded_distill_model and self.distill_ckpt is not None:
            self.load_distill_model()
            self.loaded_distill_model = True
        if not self.loaded_classifiers:
            self.load_classifiers(load_cls=self.cls_ckpt is not None, load_clean_cls=self.clean_cls_ckpt is not None)
            self.loaded_classifiers = True

    def on_train_epoch_end(self):
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

    def load_distill_model(self):
        with open(self.distill_ckpt_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            self.distill_args = copy.deepcopy(hparams['args'])
        if self.distill_args.model == 'cnn':
            self.distill_model = CNNModel(
                dim=self.distill_args.dim, # dim = alphabet_size
                hidden=self.distill_args.hidden,
                mode=self.distill_args.mode,
                num_cls=self.net.num_cls,
                depth=self.distill_args.depth,
                dropout=self.distill_args.dropout,
                prior_pseudocount=self.distill_args.prior_pseudocount,
                cls_expanded_simplex=self.distill_args.cls_expanded_simplex,
                clean_data=self.distill_args.clean_data,
                classifier=self.distill_args.classifier,
                classifier_free_guidance=self.distill_args.classifier_free_guidance,
            )
        elif self.distill_args.model == 'mlp':
            raise NotImplementedError()
            #self.distill_model = MLPModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'transformer':
            raise NotImplementedError()
            # self.distill_model = TransformerModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'deepflybrain':
            raise NotImplementedError()
            #self.distill_model = DeepFlyBrainModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        else:
            raise NotImplementedError()
        upgraded_dict = upgrade_state_dict(torch.load(self.distill_ckpt, map_location=self.device)['state_dict'], prefixes=['model.'])
        no_cls_dict = {k: v for k, v in upgraded_dict.items() if 'cls_model' not in k}
        self.distill_model.load_state_dict(no_cls_dict)
        self.distill_model.eval()
        self.distill_model.to(self.device)
        for param in self.distill_model.parameters():
            param.requires_grad = False

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


class PromoterModule(GeneralModule):
    def __init__(
        self,
        model: PromoterModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        mode: str = 'dirichlet',
        alpha_max: int = 8,
        ckpt_iterations: List[int] | Any = None,
        validate: bool = False,
        print_freq: int = 100,
        prior_pseudocount: int = 2,
        fix_alpha: float = None,
        alpha_scale: float = 2.0,
        flow_temp: float = 1.0,
        num_integration_steps: int = 20,
        distill_ckpt: str = None,
        distill_ckpt_hparams: str = None,
        net: Any = None, # TODO: why is this param being added to the yaml config?
    ) -> None:
        super().__init__(validate=validate, print_freq=print_freq)
        
        self.mode = mode
        self.ckpt_iterations = ckpt_iterations
        self.validate = validate
        self.prior_pseudocount = prior_pseudocount
        self.alpha_max = alpha_max
        self.fix_alpha = fix_alpha
        self.alpha_scale = alpha_scale
        self.flow_temp = flow_temp
        self.num_integration_steps = num_integration_steps
        self.distill_ckpt = distill_ckpt
        self.distill_ckpt_hparams = distill_ckpt_hparams

        if compile:
            self.model = torch.compile(model)
        else:
            self.model = model

        self.condflow = DirichletConditionalFlow(
            K=self.model.alphabet_size, alpha_spacing=0.01, alpha_max=alpha_max)

        self.seifeatures = pd.read_csv('data/promoter_design/target.sei.names', sep='|', header=None)
        self.sei_cache = {}
        self.loaded_distill_model = False

        self.model_dir = 'tmp'

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        if self.ckpt_iterations is not None and self.trainer.global_step in self.ckpt_iterations:
            self.trainer.save_checkpoint(os.path.join(self.model_dir, f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt"))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        if self.validate:
            self.try_print_log()
        self.log("val/loss", loss)
        B = batch.shape[0]
        self.log('val/perplexity', torch.exp(loss)[None].expand(B).mean())

        seq_one_hot = batch[:, :, :4]
        seq = torch.argmax(seq_one_hot, dim=-1)
        signal = batch[:, :, 4:5] # [128, 1024, 1]

        if self.mode == 'dirichlet':
            logits_pred, _ = self.dirichlet_flow_inference(seq, signal, self.model,
                alpha_max=self.alpha_max, prior_pseudocount=self.prior_pseudocount, flow_temp=self.flow_temp)
            seq_pred = torch.argmax(logits_pred, dim=-1)
        elif self.mode == 'riemannian':
            logits_pred = self.riemannian_flow_inference(seq, signal)
            seq_pred = torch.argmax(logits_pred, dim=-1)
        elif self.mode == 'ardm' or self.mode == 'lrar':
            seq_pred = self.ar_inference(seq, signal)
        elif self.mode == 'distill':
            logits_pred = self.distill_inference(seq, signal)
            seq_pred = torch.argmax(logits_pred, dim=-1)
        else:
            raise NotImplementedError()

        seq_pred_one_hot = torch.nn.functional.one_hot(seq_pred, num_classes=self.model.alphabet_size).float()

        if batch_idx not in self.sei_cache:
            sei_profile = self.get_sei_profile(seq_one_hot)
            self.sei_cache = sei_profile
        else:
            sei_profile = self.sei_cache[batch_idx]

        sei_profile_pred = self.get_sei_profile(seq_pred_one_hot)
        self.log("val/sp-mse", ((sei_profile - sei_profile_pred) ** 2).mean())

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/perplexity', torch.exp(loss.mean())[None].expand(batch.size(0)).mean())
        seq_one_hot = batch[:, :, :4]
        seq = torch.argmax(seq_one_hot, dim=-1)
        signal = batch[:, :, 4:5] # [128, 1024, 1]

        if self.mode == 'dirichlet':
            logits_pred, _ = self.dirichlet_flow_inference(seq, signal, self.model,
                alpha_max=self.alpha_max, prior_pseudocount=self.prior_pseudocount, flow_temp=self.flow_temp)
            seq_pred = torch.argmax(logits_pred, dim=-1)
        elif self.mode == 'riemannian':
            logits_pred = self.riemannian_flow_inference(seq, signal)
            seq_pred = torch.argmax(logits_pred, dim=-1)
        elif self.mode == 'ardm' or self.mode == 'lrar':
            seq_pred = self.ar_inference(seq, signal)
        elif self.mode == 'distill':
            logits_pred = self.distill_inference(seq, signal)
            seq_pred = torch.argmax(logits_pred, dim=-1)
        else:
            raise NotImplementedError()

        seq_pred_one_hot = torch.nn.functional.one_hot(seq_pred, num_classes=self.model.alphabet_size).float()

        if batch_idx not in self.sei_cache:
            sei_profile = self.get_sei_profile(seq_one_hot)
            self.sei_cache = sei_profile
        else:
            sei_profile = self.sei_cache[batch_idx]

        sei_profile_pred = self.get_sei_profile(seq_pred_one_hot)
        self.log("test/sp-mse", ((sei_profile - sei_profile_pred) ** 2).mean())

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        seq_one_hot = batch[:, :, :4] # [128, 1024, 4]
        seq = torch.argmax(seq_one_hot, dim=-1)
        signal = batch[:, :, 4:5] # [128, 1024, 1]
        
        B, L = seq.shape

        if self.mode == 'dirichlet' or self.mode == 'riemannian':
            #import pdb; pdb.set_trace()
            # xt [128, 1024, 4] real numbers in [0,1]
            xt, alphas = sample_cond_prob_path(
                mode=self.mode, fix_alpha=self.fix_alpha,
                alpha_scale=self.alpha_scale, seq=seq, alphabet_size=self.model.alphabet_size) # [128, 1024, 4], [128]
            xt, prior_weights = expand_simplex(
                xt, alphas, self.prior_pseudocount) # [128, 1024, 8]
            self.log('prior_weight', prior_weights.mean())
        elif self.mode == 'ardm' or self.mode == 'lrar':
            mask_prob = torch.rand(1, device=self.device)
            mask = torch.rand(seq.shape, device=self.device) < mask_prob
            if self.mode == 'lrar': mask = ~(torch.arange(L, device=self.device) < (1-mask_prob) * L)
            xt = torch.where(mask, 4, seq) # mask token has idx 4
            xt = torch.nn.functional.one_hot(xt, num_classes=5)
            alphas = mask_prob.expand(B)
        elif self.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
                xt = torch.ones((B,L, self.model.alphabet_size), device=self.device)
                xt = xt / xt.sum(-1)[..., None]
            else:
                logits_distill, xt = self.dirichlet_flow_inference(seq, signal, model=self.distill_model,
                    alpha_max=self.alpha_max, prior_pseudocount=self.prior_pseudocount, flow_temp=self.flow_temp)
                seq_distill = torch.argmax(logits_distill, dim=-1)
            alphas = torch.zeros(B, device=self.device)

        logits = self.model(xt, signal=signal, t=alphas) # [128, 1024, 4]

        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq_distill if self.mode == 'distill' else seq, reduction='none')
        losses = losses.mean(-1) # (batch_size,)

        # TODO: remove .mean() from log
        self.log('alpha', alphas.mean())
        #self.log('loss', losses.mean()) # logging done in the train and val step functions
        self.log('perplexity', torch.exp(losses.mean())[None].expand(B).mean())
        self.log('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B).mean())
        if self.stage == "val":

            if self.mode == 'dirichlet':
                logits_pred, _ = self.dirichlet_flow_inference(seq, signal, self.model,
                    alpha_max=self.alpha_max, prior_pseudocount=self.prior_pseudocount, flow_temp=self.flow_temp) # [128, 1024, 4]
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.mode == 'riemannian':
                logits_pred = self.riemannian_flow_inference(seq, signal)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.mode == 'ardm' or self.mode == 'lrar':
                seq_pred = self.ar_inference(seq, signal)
            elif self.mode == 'distill':
                logits_pred = self.distill_inference(seq, signal)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            else:
                raise NotImplementedError()
            # TODO: add to log
            #self.log('seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq_pred])
            seq_pred_one_hot = torch.nn.functional.one_hot(seq_pred, num_classes=self.model.alphabet_size).float() # [128, 1024, 4]

            if batch_idx not in self.sei_cache:
                sei_profile = self.get_sei_profile(seq_one_hot) # (128, )
                self.sei_cache = sei_profile
            else:
                sei_profile = self.sei_cache[batch_idx]

            sei_profile_pred = self.get_sei_profile(seq_pred_one_hot) # (128, )
            # TODO: remove .mean() from log
            self.log('sp-mse', ((sei_profile - sei_profile_pred) ** 2).mean())
            self.log('recovery', seq_pred.eq(seq).float().mean(-1).mean())

        self.last_log_time = time.time()
        return losses.mean()

    def get_sei_profile(self, seq_one_hot):
        B, L, K = seq_one_hot.shape
        sei_inp = torch.cat([torch.ones((B, 4, 1536), device=self.device) * 0.25,
                             seq_one_hot.transpose(1, 2),
                             torch.ones((B, 4, 1536), device=self.device) * 0.25], 2) # batchsize x 4 x 4,096
        sei_out = self.sei(sei_inp).cpu().detach().numpy() # batchsize x 21,907
        sei_out = sei_out[:, self.seifeatures[1].str.strip().values == 'H3K4me3'] # batchsize x 2,350
        predh3k4me3 = sei_out.mean(axis=1) # batchsize
        return predh3k4me3

    @torch.no_grad()
    def distill_inference(self, seq,signal):
        B, L = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K, device=seq.device)).sample()
        logits = self.model(x0,signal, t=torch.zeros(B, device=self.device))
        return logits

    @torch.no_grad()
    def riemannian_flow_inference(self, seq, signal, batch_idx=None):
        B, L = seq.shape
        K = self.model.alphabet_size
        xt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        eye = torch.eye(K).to(self.device)

        t_span = torch.linspace(0, 1, self.num_integration_steps, device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), self.prior_pseudocount)
            logits = self.model(xt_expanded, signal, s[None].expand(B))
            probs = torch.nn.functional.softmax(logits, -1)
            cond_flows = (eye - xt.unsqueeze(-1)) / (1 - s)
            flow = (probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
        return logits

    @torch.no_grad()
    def ar_inference(self, seq, signal):
        B, L = seq.shape
        order = np.arange(L)
        if self.mode =='ardm': np.random.shuffle(order)
        curr = (torch.ones((B, L), device=self.device) * 4).long()
        for i, k in enumerate(order):
            t = torch.tensor( i/L ,device=self.device)
            logits = self.model(torch.nn.functional.one_hot(curr, num_classes=5), signal, t[None].expand(B))
            curr[:, k] = torch.distributions.Categorical(probs=torch.nn.functional.softmax(logits[:, k] / self.flow_temp, -1)).sample()
        return curr

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq, signal, model, alpha_max, prior_pseudocount, flow_temp):
        B, L = seq.shape
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, model.alphabet_size, device=seq.device)).sample()
        eye = torch.eye(model.alphabet_size).to(x0)
        xt = x0

        t_span = torch.linspace(1, alpha_max, self.num_integration_steps, device=self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            prior_weight = prior_pseudocount / (s + prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)

            logits = model(seq_xt, signal, s[None].expand(B))
            out_probs = torch.nn.functional.softmax(logits / flow_temp, -1)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                c_factor = torch.nan_to_num(c_factor)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative.')
                xt = simplex_proj(xt)

        return logits, xt  # NOTE: this used to be x0; but it seems that it was a mistake

    @torch.no_grad()
    def on_validation_epoch_start(self) -> None:
        print('Loading sei model')
        self.sei = NonStrandSpecific(Sei(4096, 21907))
        self.sei.load_state_dict(upgrade_state_dict(
            torch.load('data/promoter_design/best.sei.model.pth.tar', map_location='cpu')['state_dict'],
            prefixes=['module.']))
        self.sei.to(self.device)
        self.generator = np.random.default_rng(seed=137)
    
    @torch.no_grad()
    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def load_distill_model(self):
        with open(self.distill_ckpt_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            self.distill_args = copy.deepcopy(hparams['args'])
        self.distill_model =  PromoterModel(
            mode=self.distill_args.mode, embed_dim=self.distill_args.embed_dim,
            time_dependent_weights=self.distill_args.time_dependent_weights,
            time_step=self.distill_args.time_step)
        upgraded_dict = upgrade_state_dict(torch.load(self.distill_ckpt, map_location=self.device)['state_dict'], prefixes=['model.'])
        self.distill_model.load_state_dict(upgraded_dict)
        self.distill_model.eval()
        self.distill_model.to(self.device)
        for param in self.distill_model.parameters():
            param.requires_grad = False


    def on_train_epoch_start(self) -> None:
        if not self.loaded_distill_model and self.distill_ckpt is not None:
            self.load_distill_model()
            self.loaded_distill_model = True

    def on_validation_epoch_end(self):
        del self.sei
        torch.cuda.empty_cache()
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'epoch': self.trainer.current_epoch, 'step': self.trainer.global_step, 'iter_step': self.iter_step})

        if self.trainer.is_global_zero:
            print(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            for metric_name, metric in mean_log.items():
                self.log(metric_name, metric)

            path = os.path.join(self.model_dir, f"val_{self.trainer.global_step}.csv")
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]

    # def lg(self, key, data):
    #     if isinstance(data, torch.Tensor):
    #         data = data.detach().cpu().numpy()
    #     log = self._log
    #     if self.args.validate or self.stage == 'train':
    #         log["iter_" + key].extend(data)
    #     log[self.stage + "_" + key].extend(data)

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
    # model = PromoterModel(mode='dirichlet')
    # PromoterModule(model, None, None)
    net = CNNModel(dim=4, hidden=128, mode='dirichlet', num_cls=4, depth=3, dropout=0.1, prior_pseudocount=2, cls_expanded_simplex=False, clean_data=False, classifier=True, classifier_free_guidance=False)
    DNAModule(net, None, None, False, False)