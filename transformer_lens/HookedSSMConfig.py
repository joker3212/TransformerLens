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
    """
    

    def __post_init__(self):
        pass

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
