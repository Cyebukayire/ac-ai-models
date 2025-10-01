from __future__ import print_function, division

# Ignore warnings
import warnings

from mamba_ssm import Mamba2

from deepkin.modules.layer_norm import FusedRMSNorm

warnings.filterwarnings("ignore")

from dataclasses import dataclass

import torch.nn as nn



# From: https://raw.githubusercontent.com/alxndrTL/mamba.py/main/mambapy/mamba2.py
@dataclass
class Mamba2Config:
    d_model: int  # D
    n_layers: int
    d_head: int  # todo : plutot n_heads non ?
    d_state: int = 64  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4
    n_groups: int = 1  # todo : ??

    A_init_range: tuple = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple = (0.0, float("inf"))
    conv_init = None

    learnable_init_states: bool = False
    activation: str = "swish"  # "swish" or "silu"

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    chunk_size: int = 256
    use_mem_eff_path: bool = True
    dtype = None
    device = None

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments
        self.n_heads = self.d_inner // self.d_head
        assert self.d_inner % self.d_head == 0

        assert (self.d_inner / self.d_head) % 8 == 0, f"requierement ({self.d_inner}/{self.d_head}={(self.d_inner / self.d_head)}) [need align 8] of causal_conv1d"

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba2Mixer(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype):
        caches = [layer.allocate_inference_cache(batch_size, max_seqlen, dtype) for layer in self.layers]
        return caches

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()

        # self.mixer = Mamba2Block(config)
        self.mixer = Mamba2(# This module uses roughly 3 * expand * d_model^2 parameters
                            d_model = config.d_model,  # Model dimension d_model
                            headdim = config.d_head, # Head Dim
                            d_state = config.d_state,  # SSM state expansion factor, typically 64 or 128
                            d_conv = config.d_conv,  # Local convolution width
                            expand = config.expand_factor,  # Block expansion factor
                            )
        self.norm = FusedRMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        conv_state, ssm_state = cache

        output, cache = self.mixer.step(self.norm(x), conv_state, ssm_state)
        output = output + x
        return output, cache
