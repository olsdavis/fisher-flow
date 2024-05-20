"""Distributional utils."""
import random
import torch
from torch import Tensor, nn
import numpy as np
from scipy.linalg import sqrtm
import tqdm
from src.sfm import Manifold, NSimplex


def set_seeds(seed: int = 0):
    """
    Sets the seeds for torch, numpy and random to `seed`.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def estimate_categorical_kl(
    model: nn.Module,
    manifold: Manifold,
    real_dist: Tensor,
    n: int,
    batch: int = 512,
    inference_steps: int = 100,
    sampling_mode: str = "max",
    silent: bool = False,
    tangent: bool = True,
) -> float:
    """
    Estimates the categorical KL divergence between points produced by the
    model `model` and `real_dist`. Done by sampling `n` points and estimating
    thus the different probabilities.

    Parameters:
        - `model`: the model;
        - `manifold`: manifold over which the model was trained;
        - `real_dist`: the real distribution tensor of shape `(k, d)`;
        - `n`: the number of points over which the estimate should be done;
        - `batch`: the number of points to draw per batch;
        - `inference_steps`: the number of steps to take for inference;
        - `sampling_mode`: how to sample points; if "sample", then samples
            from the distribution produced by the model; if "max" then takes
            the argmax of the distribution.

    Returns:
        An estimate of the KL divergence of the model's distribution from
        the real distribution, i.e., "KL(model || real_dist)".
    """
    assert sampling_mode in ["sample", "max"], "not a valid sampling mode"
    # init acc
    acc = torch.zeros_like(real_dist, device=real_dist.device).int()

    model.eval()
    to_sample = [batch] * (n // batch)
    if n % batch != 0:
        to_sample += [n % batch]
    for draw in (tqdm.tqdm(to_sample) if not silent else to_sample):
        x_0 = manifold.uniform_prior(
            draw, real_dist.size(0), real_dist.size(1),
        ).to(real_dist.device)
        x_1 = manifold.tangent_euler(x_0, model, inference_steps, tangent=tangent)
        x_1 = manifold.send_to(x_1, NSimplex)
        if sampling_mode == "sample":
            # TODO: remove or fix for Categorical
            raise NotImplementedError("Sampling from Dirichlet not implemented")
            # dist = Dirichlet(x_1)
            # samples = dist.sample()
            # acc += samples.sum(dim=0)
        else:
            samples = nn.functional.one_hot(
                x_1.argmax(dim=-1),
                real_dist.size(-1),
            )
            acc += samples.sum(dim=0)
    acc = acc.float()
    acc /= n
    # acc.clamp_min_(1e-12)
    if not silent:
        print(acc)
    ret = (acc * (acc.log() - real_dist.log())).sum(dim=-1).mean().item()
    return ret


def get_wasserstein_dist(embeds1, embeds2):
    # Taken from: https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/flow_utils.py#L38
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist
