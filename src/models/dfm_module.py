# Adaptation of: https://github.com/HannesStark/dirichlet-flow-matching/blob/main/lightning_modules/dna_module.py
from collections import defaultdict
from typing import Any
import lightning as pl
import torch
from torch.distributions import Dirichlet
from torchmetrics import MeanMetric


from src.models.net import expand_simplex
from src.dfm import (
    DirichletConditionalFlow,
    sample_cond_prob_path,
    simplex_proj,
    load_flybrain_designed_seqs,
)


class DNAModule(pl.LightningModule):
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
        num_integration_steps: int = 20,
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
    ):
        super().__init__()
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
        self.condflow = DirichletConditionalFlow(K=self.net.k, alpha_spacing=0.001, alpha_max=alpha_max)
        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_outputs: dict[str, list] = defaultdict(list)
        # self.train_outputs: dict[str, list] = defaultdict(list)
        self.train_out_initialized = False
        self.loaded_classifiers = False
        self.loaded_distill_model = False
        if taskiran_seq_path is not None:
            self.taskiran_fly_seqs = load_flybrain_designed_seqs(taskiran_seq_path).to(self.device)
        # added:
        self.stage = "train"

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'cls_model' not in k and 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        # evaluate KL
        with torch.inference_mode():
            to_draw = self.kl_samples
            acc = torch.zeros((self.net.k, self.net.dim), device=self.device)
            concentration = torch.ones((self.net.k, self.net.dim), device=self.device)
            while to_draw > 0:
                n = min(to_draw, 512)
                x_0 = Dirichlet(concentration).sample((n,))
                samples = self.dirichlet_flow_inference(x_0, None, self.net)[1]
                acc += torch.nn.functional.one_hot(samples.argmax(dim=-1), self.net.dim).sum(dim=0)
                to_draw -= n
        acc /= acc.sum(dim=-1, keepdim=True)
        real_probs = self.trainer.test_dataloaders.dataset.probs.to(self.device)
        kl = (acc * (acc.log() - real_probs.log())).sum(dim=-1).mean().item()
        self.log("test/kl", kl, on_step=False, on_epoch=True, prog_bar=False)

    def general_step(self, batch, batch_idx=None):
        seq = batch
        B = seq.size(0)

        xt, alphas = sample_cond_prob_path(self.mode, self.fix_alpha, self.alpha_scale, seq, self.net.dim)
        if self.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
            else:
                logits_distill, xt = self.dirichlet_flow_inference(seq, None, model=self.net)
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
        losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),
            seq_distill if self.mode == 'distill' else seq.argmax(dim=-1),
            reduction='mean',
        )

        # self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        if self.stage == "val":
            if self.mode == 'dirichlet':
                logits_pred, _ = self.dirichlet_flow_inference(seq, None, model=self.net)
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
        # if not self.train_out_initialized and self.clean_cls_ckpt is not None:
        #     self.run_cls_model(seq, cls, log_dict=self.train_outputs, clean_data=True, postfix='_cleancls', generated=False, run_log=False)
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
                # self.lg(f'cls_loss{postfix}', losses)
            # self.lg(f'cls_accuracy{postfix}', cls_pred.eq(cls).float())

        log_dict[f'embeddings{postfix}'].append(embeddings.detach().cpu())
        log_dict[f'clss{postfix}'].append(cls.detach().cpu())
        log_dict[f'logits{postfix}'].append(logits.detach().cpu())
        log_dict[f'alphas{postfix}'].append(alphas.detach().cpu())
        if not clean_data and not self.target_class == self.net.num_cls: # num_cls stands for the masked class
            scores = self.get_cls_score(xt, alphas)
            log_dict[f'scores{postfix}'].append(scores.detach().cpu())

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        self.val_outputs = defaultdict(list)

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        self.train_out_initialized = True
        """log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = self.gather_log(log, self.trainer.world_size)

        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]"""

    """
    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        if self.args.validate or self.stage == 'train':
            log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)
    """

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
