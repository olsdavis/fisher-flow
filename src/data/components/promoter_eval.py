# Adaptation from: https://github.com/HannesStark/dirichlet-flow-matching/blob/main/lightning_modules/promoter_module.py
import re
import pandas as pd
import torch
from torch import Tensor
from selene_sdk.utils import NonStrandSpecific
from .sei import Sei


def upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder."]):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


class SeiEval:
    """Singleton class for SEI evaluation."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sei = NonStrandSpecific(Sei(4096, 21907))
            cls._instance._sei_features = (
                pd.read_csv('data/promoter_design/target.sei.names', sep='|', header=None)
            )
            cls._instance._sei_loaded = False
            cls._instance._sei_cache = {}
        return cls._instance

    @torch.no_grad()
    def get_sei_profile(self, seq_one_hot: Tensor) -> Tensor:
        """
        Get the SEI profile from the one-hot encoded sequence.

        Parameters:
            - `seq_one_hot`: The one-hot encoded sequence tensor.

        Returns:
            The SEI profile tensor.
        """
        if not self._sei_loaded:
            self._sei.load_state_dict(
                upgrade_state_dict(
                    torch.load('data/promoter_design/best.sei.model.pth.tar', map_location="cpu")['state_dict'],
                    prefixes=["module."],
                )
            )
            self._sei.to(seq_one_hot.device)
            self._sei.eval()
            self._sei_loaded = True
        B, _, _ = seq_one_hot.shape
        sei_inp = torch.cat([torch.ones((B, 4, 1536), device=seq_one_hot.device) * 0.25,
                                seq_one_hot.transpose(1, 2),
                                torch.ones((B, 4, 1536), device=seq_one_hot.device) * 0.25], 2) # batchsize x 4 x 4,096
        sei_out = self._sei(sei_inp).cpu().detach().numpy() # batchsize x 21,907
        sei_out = sei_out[:, self._sei_features[1].str.strip().values == 'H3K4me3'] # batchsize x 2,350
        predh3k4me3 = sei_out.mean(axis=1) # batchsize
        return predh3k4me3

    def eval_sp_mse(self, seq_one_hot: Tensor, target: Tensor, b_index: int) -> Tensor:
        """
        Evaluate the mean squared error of the SEI profile prediction.

        Parameters:
            - `seq_one_hot`: The one-hot encoded sequence tensor.
            - `target`: The target tensor;
            - `b_index`: The batch index of the target Tensor; avoids recalculating
                the profile all the time.

        Returns:
            The mean squared error tensor.
        """
        if b_index in self._sei_cache:
            target_prof = self._sei_cache[b_index]
        else:
            target_prof = self.get_sei_profile(target)
            self._sei_cache[b_index] = target_prof
        pred_prof = self.get_sei_profile(seq_one_hot)
        return (pred_prof - target_prof) ** 2
