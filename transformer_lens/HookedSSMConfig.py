"""Hooked SSM Config.

Module with a dataclass for storing the configuration of a
:class:`transformer_lens.SSM` model.
"""
from __future__ import annotations

import logging
import pprint
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from transformer_lens import utils

SUPPORTED_ACTIVATIONS = ["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"]


@dataclass
class HookedSSMConfig:
    """
    Configuration class to store the configuration of a HookedSSM model.

    See further_comments.md for more details on the more complex arguments.
    
    Args:
        d_model (int): The dimensionality of the embeddings
        n_layers (int): The number of mamba blocks
        d_vocab (int): The size of the vocabulary. Defaults to -1, which means not set. If not set, will be
            automatically set from the tokenizer's vocab size.
        pad_vocab_size_multiple (int): # TODO add documentation
        device(str): The device to use for the model. Defaults to 'cuda' if
            available, else 'cpu'. Must be 'cuda' if `n_devices` > 1.
        seed (int, *optional*): The seed to use for the model.
            Used to set sources of randomness (Python, PyTorch and
            NumPy) and to initialize weights. Defaults to None. We recommend setting a seed, so your experiments are reproducible.
        # TODO add docs for all the Mamba specifc params
    """
    d_model: int
    n_layers: int
    vocab_size: int
    d_vocab: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    pad_vocab_size_multiple: int = 1
    device: Optional[str]
    seed: Optional[int]
    
    def __post_init__(self):
        if self.seed is not None:
            self.set_seed_everywhere(self.seed)
        if self.device is None:
            self.device = utils.get_device()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedSSMConfig:
        """
        Instantiates a `HookedSSMConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedSSMConfig:\n" + pprint.pformat(self.to_dict())

    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
