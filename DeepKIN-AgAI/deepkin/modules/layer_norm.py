import torch

try:
    import apex
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
except ImportError:
    from torch.nn import LayerNorm as _FusedLayerNorm

try:
    import apex
    from apex.normalization import FusedRMSNorm as _FusedRMSNorm
except ImportError:
    from torch.nn import RMSNorm as _FusedRMSNorm

class FusedLayerNorm(_FusedLayerNorm):
    @torch.jit.unused
    def forward(self, x):
        if not x.is_cuda:
            return super().forward(x)
        else:
            with torch.cuda.device(x.device):
                return super().forward(x)

    def load_state_dict(self, state_dict, strict=True):
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
    @torch.jit.unused
    def forward(self, x):
        if not x.is_cuda:
            x = x.transpose(1, -1)
            x = super().forward(x)
            x = x.transpose(1, -1)
            return x
        else:
            with torch.cuda.device(x.device):
                x = x.transpose(1, -1)
                x = super().forward(x)
                x = x.transpose(1, -1)
                return x

    def load_state_dict(self, state_dict, strict=True):
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
    @torch.jit.unused
    def forward(self, x):
        if not x.is_cuda:
            return super().forward(x)
        else:
            with torch.cuda.device(x.device):
                return super().forward(x)
