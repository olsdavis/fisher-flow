"""Miscellaneous utils."""
import gc
import torch


def reset_memory():
    """Resets memory (CUDA cache + GC)."""
    torch.cuda.empty_cache()
    gc.collect()
