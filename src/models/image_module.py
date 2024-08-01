"""
Module for flow models over images.
"""
from abc import ABC, abstractmethod
import os
import numpy as np
from lightning import LightningModule
import torch
from torch.func import jvp
from torch import Tensor, nn, vmap
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchmetrics.image import FrechetInceptionDistance as FID
from torchvision.utils import make_grid
from einops import rearrange
from src.sfm import NSimplex, manifold_from_name


def geodesic(manifold, start_point, end_point):
    # https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/manifolds/utils.py#L6
    shooting_tangent_vec = manifold.logmap(start_point, end_point)

    def path(t):
        """Generate parameterized function for geodesic curve.
        Parameters
        ----------
        t : array-like, shape=[n_points,]
            Times at which to compute points of the geodesics.
        """
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path


class FlowModule(ABC, LightningModule):
    """
    Class for all flow models.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        x1_pred: bool = False,
        fast_matmul: bool = True,
        manifold: str = "simplex",
        inference_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "fid"])
        self.fid = FID(feature=2048, normalize=True)
        self.fid = self.fid.to(self.device)
        self.net = net
        self.x1_pred = x1_pred
        self.total_train_loss = MeanMetric()
        self.total_val_loss = MeanMetric()
        self.total_test_loss = MeanMetric()
        self.manifold = manifold_from_name(manifold)
        self.inference_steps = inference_steps
        if fast_matmul:
            torch.set_float32_matmul_precision("high")

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model during training.
        """

    def training_step(self, batch: Tensor, batch_idx: int):
        loss = self.forward(batch)
        self.total_train_loss(loss)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        loss = self.forward(batch)
        self.total_val_loss(loss)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: Tensor, batch_idx: int):
        loss = self.forward(batch)
        self.total_test_loss(loss)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    # default misc

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
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


class ImageFlowModule(FlowModule):
    """
    Fisher Flow module for images.
    """

    def __init__(
        self,
        fid_freq: int = 10,
        x1_pred: bool = False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.fid_freq = fid_freq
        self.x1_pred = x1_pred

    def compute_target(self, x_0: Tensor, x_1: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        with torch.inference_mode(False):
            def cond_u(x0, x1, t):
                path = geodesic(self.manifold.sphere, x0, x1)
                x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
                return x_t, u_t
            x_t, target = vmap(cond_u)(x_0, x_1, time)
        x_t = x_t.squeeze()
        target = target.squeeze()
        return x_t, target

    def forward(self, x: Tensor | list[Tensor]) -> Tensor:
        if isinstance(x, list):
            # labelled dataset, e.g., CIFAR-10
            x = x[0]

        # B, C, H, W, bits is x_0
        b, c, h, w, bits = x.shape
        # sample noise
        x_0 = self.manifold.uniform_prior(b, h * w * c, bits).to(device=self.device)
        x_1 = x.reshape(b, h * w * c, bits)
        t = torch.rand(b, 1, device=self.device)

        # loss for x_1 pred mode
        if self.x1_pred:
            x_t = self.manifold.geodesic_interpolant(x_0, x_1, t)
            # x_t = x_t.reshape(b, c, h, w, bits).permute(0, 1, 4, 2, 3).reshape(b, c * bits, h, w)
            x_t = rearrange(x_t, "b (h w c) k -> b (c k) h w", h=h, w=w, c=c, b=b, k=bits)
            out = self.net(x_t, t)
            return F.cross_entropy(
                # out.reshape(b, c, bits, h, w).permute(0, 2, 1, 3, 4),
                rearrange(out, "b (c k) h w -> b k c h w", h=h, w=w, c=c, b=b, k=bits),
                x.permute(0, 1, 4, 2, 3).argmax(dim=2),
                reduction="mean",
            )

        # loss for vectors
        x_t, target = self.compute_target(x_0, x_1, t)
        # out = self.net(x_t.reshape(b, h, w, c * bits).permute(0, 3, 1, 2), t)
        x_t = rearrange(x_t, "b (h w c) k -> b (c k) h w", h=h, w=w, c=c, b=b, k=bits)
        out = self.net(x_t, t)
        out = rearrange(out, "b (c k) h w -> b (h w c) k", h=h, w=w, c=c, b=b, k=bits)
        # out = out.reshape(b, c, bits, h, w).permute(0, 1, 3, 4, 2).reshape(b, h * w * c, bits)
        loss = F.mse_loss(out, target, reduction="mean")
        return loss

    @torch.inference_mode()
    def generate_image_batch(self, batch_size: int = 256) -> Tensor:
        h = 32
        w = 32
        c = 3
        bits = 256

        manifold_shape = (batch_size, h * w * c, bits)
        x_0 = self.manifold.uniform_prior(
            *manifold_shape
        ).to(device=self.device)
        x = x_0
        times = torch.linspace(0, 1, self.inference_steps, device=self.device)

        for t, s in zip(times[:-1], times[1:]):
            # s is the next time step
            dt = s - t
            # prepare x's shape
            x_prev = x
            x = x.reshape(batch_size, h, w, c, bits).permute(0, 3, 4, 1, 2)
            x = x.reshape(batch_size, c * bits, h, w)
            # through model
            out = self.net(x, t)
            # now, prepare step
            if self.x1_pred:
                out = out.reshape(batch_size, c, bits, h, w).permute(0, 1, 3, 4, 2).softmax(dim=-1)
                out = out.reshape(batch_size, h * w * c, bits)
                vec = self.manifold.log_map(
                    x_prev,
                    NSimplex().send_to(
                        out,
                        type(self.manifold),
                    ),
                )
            else:
                # TODO: make tangent!!
                vec = out.reshape(batch_size, c, bits, h, w).permute(0, 1, 3, 4, 2)\
                    .reshape(batch_size, h * w * c, bits)
            x = self.manifold.exp_map(x_prev, vec * dt)

        # argmax the colours
        x = x.reshape(batch_size, c, h, w, bits)
        x = x.argmax(dim=-1).float()
        # make images between 0 and 1, and floats
        return x.float() / 255.0

    def generate_image_collection(self, n: int = 10000, batch_size: int = 256) -> Tensor:
        images = []
        while sum(t.shape[0] for t in images) < n:
            print("starting to generate...")
            images += [self.generate_image_batch(min(batch_size, n - len(images)))]
            print("done batch!")
        return torch.cat(images, dim=0)

    @torch.inference_mode()
    def compute_fid(self, real: Tensor, fake: Tensor, stride: int = 256) -> Tensor:
        """
        Computes the FID between the real images, `real`, and the generated ones,
        `fake`. Updates the FID metric `stride` images at a time.
        """
        self.fid.reset()
        for i in range(0, len(real), stride):
            self.fid.update(real[i:min(i+stride, real.shape[0])].to(self.fid.device), real=True)
        for i in range(0, len(fake), stride):
            self.fid.update(fake[i:min(i+stride, fake.shape[0])], real=False)
        return self.fid.compute().detach().item()

    def log_images(self, images: Tensor, logs_folder: str, rows: int = 8, cols: int = 8):
        with torch.inference_mode():
            final_images = np.random.choice(images.shape[0], rows * cols, replace=False)
            grid = make_grid(final_images, nrow=rows)
            self.log(f"{logs_folder}/example_images", grid, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self.current_epoch % self.fid_freq != 0 or self.current_epoch == 0:
            return
        with torch.inference_mode():
            generated = self.generate_image_collection(1000)
            fid = self.compute_fid(
                real=self.trainer.datamodule.get_all_test_set(),
                fake=generated,
            )
            self.log(
                "val/fid",
                fid,
                prog_bar=True, on_step=False, on_epoch=True,
            )
            self.log_images(generated, f"val_{self.current_epoch:03d}")

    def on_test_epoch_end(self):
        os.makedirs(name="./final_fid", exist_ok=True)
        with torch.inference_mode():
            generated = self.generate_image_collection()
            fid = self.compute_fid(
                real=self.trainer.datamodule.get_all_test_set(),
                fake=generated,
            )
            self.log(
                "test/fid",
                fid,
                prog_bar=True, on_step=False, on_epoch=True
            )
            self.log_images(generated, "test_final", rows=10, cols=10)
