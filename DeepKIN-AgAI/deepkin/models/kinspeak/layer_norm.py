from typing import Mapping, Any, Union, List, Tuple

import torch
from torch import Tensor

from apex.normalization import FusedLayerNorm as _FusedLayerNorm
from apex.normalization import FusedRMSNorm as _FusedRMSNorm

from apex.normalization.fused_layer_norm import fused_layer_norm, fused_layer_norm_affine, \
    fused_rms_norm_affine, fused_rms_norm


class FusedLayerNorm(_FusedLayerNorm):

    def super_forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling() or not x.is_cuda:
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.elementwise_affine:
            return fused_layer_norm_affine(
                x, self.weight, self.bias, self.normalized_shape, self.eps, self.memory_efficient
            )
        else:
            return fused_layer_norm(x, self.normalized_shape, self.eps, self.memory_efficient)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            return self.super_forward(x)
        else:
            with torch.cuda.device(x.device):
                return self.super_forward(x)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # Modify state_dict before loading
        modified_state_dict = {}
        for k, v in state_dict.items():
            if "gamma" in k:
                modified_state_dict[k.replace("gamma", "weight")] = v
            elif "beta" in k:
                modified_state_dict[k.replace("beta", "bias")] = v
            else:
                modified_state_dict[k] = v
        super().load_state_dict(modified_state_dict, strict=strict)


class TransFusedLayerNorm(_FusedLayerNorm):

    def super_forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling() or not x.is_cuda:
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.elementwise_affine:
            return fused_layer_norm_affine(
                x, self.weight, self.bias, self.normalized_shape, self.eps, self.memory_efficient
            )
        else:
            return fused_layer_norm(x, self.normalized_shape, self.eps, self.memory_efficient)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            x = x.transpose(1, -1)
            x = self.super_forward(x)
            x = x.transpose(1, -1)
            return x
        else:
            with torch.cuda.device(x.device):
                x = x.transpose(1, -1)
                x = self.super_forward(x)
                x = x.transpose(1, -1)
                return x

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # Modify state_dict before loading
        modified_state_dict = {}
        for k, v in state_dict.items():
            if "gamma" in k:
                modified_state_dict[k.replace("gamma", "weight")] = v
            elif "beta" in k:
                modified_state_dict[k.replace("beta", "bias")] = v
            else:
                modified_state_dict[k] = v
        super().load_state_dict(modified_state_dict, strict=strict)


class FusedRMSNorm(_FusedRMSNorm):
    def super_forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling() or not x.is_cuda:
            # return manual_rms_norm(x, self.normalized_shape, self.weight, self.eps)
            return torch.nn.functional.rms_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)
        if self.elementwise_affine:
            return fused_rms_norm_affine(
                x, self.weight, self.normalized_shape, self.eps, self.memory_efficient
            )
        else:
            return fused_rms_norm(x, self.normalized_shape, self.eps, self.memory_efficient)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            return self.super_forward(x)
        else:
            with torch.cuda.device(x.device):
                return self.super_forward(x)
