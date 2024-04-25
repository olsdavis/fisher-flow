# Adaptation from: https://github.com/HannesStark/dirichlet-flow-matching/blob/main/lightning_modules/promoter_module.py
import pandas as pd
import torch
from torch import Tensor
from selene_sdk.utils import NonStrandSpecific

from .sei import Sei

_sei_loaded = False
_sei = NonStrandSpecific(Sei(4096, 21907))
_sei_features = pd.read_csv('data/promoter_design/target.sei.names', sep='|', header=None)


def get_sei_profile(seq_one_hot: Tensor) -> Tensor:
    """
    Get the SEI profile from the one-hot encoded sequence.

    Parameters:
        - `seq_one_hot`: The one-hot encoded sequence tensor.

    Returns:
        The SEI profile tensor.
    """
    if not _sei_loaded:
        _sei.load_state_dict(torch.load('data/promoter_design/best.sei.model.pth.tar'))
        _sei.to(seq_one_hot.device)
        _sei.eval()
    B, _, _ = seq_one_hot.shape
    sei_inp = torch.cat([torch.ones((B, 4, 1536), device=seq_one_hot.device) * 0.25,
                            seq_one_hot.transpose(1, 2),
                            torch.ones((B, 4, 1536), device=seq_one_hot.device) * 0.25], 2) # batchsize x 4 x 4,096
    sei_out = _sei(sei_inp).cpu().detach().numpy() # batchsize x 21,907
    sei_out = sei_out[:, _sei_features[1].str.strip().values == 'H3K4me3'] # batchsize x 2,350
    predh3k4me3 = sei_out.mean(axis=1) # batchsize
    return predh3k4me3


def eval_sp_mse(seq_one_hot: Tensor, target: Tensor) -> Tensor:
    """
    Evaluate the mean squared error of the SEI profile prediction.

    Parameters:
        - `seq_one_hot`: The one-hot encoded sequence tensor.
        - `target`: The target tensor.

    Returns:
        The mean squared error tensor.
    """
    pred_prof = get_sei_profile(seq_one_hot)
    return (pred_prof - target) ** 2
