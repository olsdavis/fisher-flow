import re


import torch
from torch import Tensor
import numpy as np


from src.sfm import get_wasserstein_dist
from src.models.net import CNNModel


def _upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder."]):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


class FBD(torch.nn.Module):
    """
    Class to evaluate the Fréchet Biological Distance (FBD).
    """

    def __init__(
        self,
        dim: int,
        k: int,
        hidden: int,
        num_cls: int,
        depth: int,
        ckpt_path: str,
    ):
        super().__init__()
        self.cls_model = CNNModel(
            dim=dim,
            k=k,
            num_cls=num_cls,
            hidden=hidden,
            mode="",  # ignored
            depth=depth,
            dropout=0.0,
            prior_pseudocount=2.0,  # unused
            cls_expanded_simplex=False,
            clean_data=True,
            classifier=True,
            classifier_free_guidance=False,
        )
        self.cls_model.load_state_dict(
            _upgrade_state_dict(
                torch.load(ckpt_path, map_location="cpu")['state_dict'], prefixes=['model.'],
            )
        )
        self.cache = {}

    @torch.inference_mode()
    def forward(self, x: Tensor, gt: Tensor, batch_index: int | None) -> np.ndarray | float:
        self.cls_model.eval()
        if batch_index is not None and batch_index in self.cache:
            gt_embeddings = self.cache[batch_index]
        else:
            gt_embeddings = self.cls_model(gt, t=None, return_embedding=True)[1].cpu().numpy()
            if batch_index is not None:
                self.cache[batch_index] = gt_embeddings
        embeddings = self.cls_model(x, t=None, return_embedding=True)[1].cpu().numpy()
        return get_wasserstein_dist(embeddings, gt_embeddings)
