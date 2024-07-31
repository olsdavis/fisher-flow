"""
Module for flow models over images.
"""
from abc import ABC, abstractmethod
import os
from lightning import LightningModule
import torch
from torch.func import jvp
from torch import Tensor, nn, vmap
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchmetrics.image import FrechetInceptionDistance as FID
from torchvision.utils import save_image
from src.sfm import manifold_from_name


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
        fast_matmul: bool = True,
        manifold: str = "simplex",
        inference_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "fid"])
        self.fid = FID(feature=2048)
        self.fid = self.fid.to(self.device)
        self.net = net
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
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.fid_freq = fid_freq

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

        # x contains the ground truth, start from noise
        # Batch, H x W, C x 3
        manifold_shape = (x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
        permuted_shape = (x.shape[0], x.shape[2], x.shape[3], x.shape[1])
        x_0 = self.manifold.uniform_prior(
            *manifold_shape
        ).to(device=x.device).reshape(permuted_shape)
        # sample random times
        times = torch.rand(x.shape[0], 1, device=x.device)
        # compute loss
        x_t, target = self.compute_target(x_0, x.permute(0, 2, 3, 1), times)
        out = self.net(x_t.permute(0, 3, 1, 2), times)
        return F.mse_loss(out, target.reshape_as(out), reduction="mean")

    @torch.inference_mode()
    def generate_image_batch(self, batch_size: int = 256) -> Tensor:
        manifold_shape = (batch_size, 32 ** 2, 256 * 3,)
        x_0 = self.manifold.uniform_prior(
            *manifold_shape
        ).to(device=self.device)
        x = x_0
        times = torch.linspace(0, 1, self.inference_steps, device=self.device)
        for t, s in zip(times[:-1], times[1:]):
            # s is the next time step
            dt = s - t
            # euler integration
            vec = self.net(x.reshape(batch_size, 32, 32, 256 * 3).permute(0, 3, 1, 2), t)
            vec = vec.permute(0, 2, 3, 1).reshape(batch_size, 32 ** 2, 256 * 3)
            x = self.manifold.exp_map(
                x,
                vec * dt,
            )
        x = x.reshape(batch_size, 32, 32, 3, 256).argmax(dim=-1).permute(0, 3, 1, 2)
        return x.to(torch.uint8)

    def generate_image_collection(self, n: int = 10000, batch_size: int = 256) -> Tensor:
        images = []
        while sum(t.shape[0] for t in images) < n:
            print("starting to generate...")
            images += [self.generate_image_batch(min(batch_size, n - len(images)))]
            print("done batch!")
        return torch.cat(images, dim=0)

    @torch.inference_mode()
    def compute_fid(self, real: Tensor, fake: Tensor) -> Tensor:
        self.fid.reset()
        for i in range(0, len(real), 256):
            self.fid.update(real[i:min(i+256, real.shape[0])].to(self.fid.device), real=True)
        for i in range(0, len(fake), 256):
            self.fid.update(fake[i:min(i+256, fake.shape[0])], real=False)
        return self.fid.compute().detach().item()

    def on_validation_epoch_end(self):
        if self.current_epoch % self.fid_freq != 0 and self.current_epoch != 0:
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
                prog_bar=True, on_step=False, on_epoch=True
            )
            # save the made images too
            for i, im in enumerate(generated):
                # log images
                folder = f"./val_{self.current_epoch:03d}"
                os.makedirs(name=folder, exist_ok=True)
                save_image(im.float() / 255.0, f"{folder}/generated_{i:04d}.png")

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
            # save the made images too
            for i, im in enumerate(generated):
                # log images
                save_image(im.float() / 255.0, f"./final_fid/generated_{i}.png")
