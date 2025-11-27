import math
import re
from typing import Optional, Tuple, List

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import get_window
from torch import nn
from torch.amp import custom_fwd
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from deepkin.data.kinya_norm import tts_symbols, text_to_sequence
from deepkin.modules.tts_arguments import TTSArguments
from deepkin.modules.tts_attentions import Encoder, SingleLayerEncoder, TransLayerNorm
from deepkin.modules.tts_commons import init_weights, get_padding, sequence_mask, fused_add_tanh_sigmoid_multiply, \
    generate_path, rand_slice_segments, intersperse


def maximum_path(neg_cent, mask):
  from monotonic_align.core import maximum_path_c
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
  path = np.zeros(neg_cent.shape, dtype=np.int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class STFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction

class MS_ISTFT_Generator(torch.nn.Module):

    def __init__(self, tts_args: TTSArguments):
        super(MS_ISTFT_Generator, self).__init__()
        # Copied from settings
        initial_channel = tts_args.initial_channel
        resblock = tts_args.resblock
        resblock_kernel_sizes = tts_args.resblock_kernel_sizes
        resblock_dilation_sizes = tts_args.resblock_dilation_sizes
        upsample_rates = tts_args.upsample_rates
        upsample_initial_channel = tts_args.upsample_initial_channel
        upsample_kernel_sizes = tts_args.upsample_kernel_sizes
        gen_istft_n_fft = tts_args.gen_istft_n_fft
        gen_istft_hop_size = tts_args.gen_istft_hop_size
        subbands = tts_args.subbands

        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock == 1 else ResBlock2

        # self.ups: nn.ModuleList[ConvTranspose1dInterface] = nn.ModuleList()
        # for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        i, (u, k) = 0, (upsample_rates[0], upsample_kernel_sizes[0])
        self.ups0 = weight_norm(ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2))
        i, (u, k) = 1, (upsample_rates[1], upsample_kernel_sizes[1])
        self.ups1 = weight_norm(ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2))

        # self.resblocks: nn.ModuleList[ResBlockInterface] = nn.ModuleList()
        # ch = upsample_initial_channel // (2 ** len(self.ups))
        # for i in range(len(self.ups)):
        #     ch = upsample_initial_channel // (2 ** (i + 1))
        #     for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        #         self.resblocks.append(resblock(ch, k, d))

        i = 0
        ch = upsample_initial_channel // (2 ** (i + 1))
        j, (k, d) = 0, (resblock_kernel_sizes[0], resblock_dilation_sizes[0])
        self.resblocks_0_0 = resblock(ch, k, d)
        j, (k, d) = 1, (resblock_kernel_sizes[1], resblock_dilation_sizes[1])
        self.resblocks_0_1 = resblock(ch, k, d)
        j, (k, d) = 2, (resblock_kernel_sizes[2], resblock_dilation_sizes[2])
        self.resblocks_0_2 = resblock(ch, k, d)

        i = 1
        ch = upsample_initial_channel // (2 ** (i + 1))
        j, (k, d) = 0, (resblock_kernel_sizes[0], resblock_dilation_sizes[0])
        self.resblocks_1_0 = resblock(ch, k, d)
        j, (k, d) = 1, (resblock_kernel_sizes[1], resblock_dilation_sizes[1])
        self.resblocks_1_1 = resblock(ch, k, d)
        j, (k, d) = 2, (resblock_kernel_sizes[2], resblock_dilation_sizes[2])
        self.resblocks_1_2 = resblock(ch, k, d)

        self.post_n_fft = gen_istft_n_fft
        self.ups0.apply(init_weights)
        self.ups1.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []

        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands * (self.post_n_fft + 2), 7, 1, padding=3))

        self.subband_conv_post.apply(init_weights)

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        #self.multistream_conv_post = weight_norm(Conv1d(4, 1, kernel_size=63, bias=False, padding=get_padding(63, 1)))
        self.multistream_conv_post = weight_norm(Conv1d(self.subbands, 1, kernel_size=63, bias=False, padding=get_padding(63, 1))) # from MB-iSTFT-VITS-44100-Ja
        self.multistream_conv_post.apply(init_weights)

        self.stft = STFT(filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size, win_length=self.gen_istft_n_fft)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_pre(x)  # [B, ch, length]

        x = F.leaky_relu(x, 0.1)
        x = self.ups0(x, output_size=None)
        xs = self.resblocks_0_0(x, x_mask=None) + self.resblocks_0_1(x, x_mask=None) + self.resblocks_0_2(x, x_mask=None)
        x = xs / float(self.num_kernels)

        x = F.leaky_relu(x, 0.1)
        x = self.ups1(x, output_size=None)
        xs = self.resblocks_1_0(x, x_mask=None) + self.resblocks_1_1(x, x_mask=None) + self.resblocks_1_2(x, x_mask=None)
        x = xs / float(self.num_kernels)

        # for i in range(self.num_upsamples):
        #     x = F.leaky_relu(x, 0.1)
        #     ci: ConvTranspose1dInterface = self.ups[i]
        #     x = ci.forward(x, output_size=None)
        #
        #     xs: torch.Tensor = torch.zeros(1)
        #     for j in range(self.num_kernels):
        #         if j == 0:
        #             ri: ResBlockInterface = self.resblocks[i * self.num_kernels + j]
        #             xs = ri.forward(x, x_mask=None)
        #         else:
        #             ri: ResBlockInterface = self.resblocks[i * self.num_kernels + j]
        #             xs += ri.forward(x, x_mask=None)
        #     x = xs / float(self.num_kernels)

        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1] // self.subbands, x.shape[-1]))

        spec = torch.exp(x[:, :, :self.post_n_fft // 2 + 1, :])
        phase = math.pi * torch.sin(x[:, :, self.post_n_fft // 2 + 1:, :])

        y_mb_hat = self.stft.inverse(
            torch.reshape(spec, (spec.shape[0] * self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1])),
            torch.reshape(phase, (phase.shape[0] * self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1])))
        y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1]))
        y_mb_hat = y_mb_hat.squeeze(-2)

        #y_mb_hat = F.conv_transpose1d(y_mb_hat, self.updown_filter.cuda(x.device) * self.subbands, stride=self.subbands)
        y_mb_hat = F.conv_transpose1d(y_mb_hat, self.updown_filter.to(x.device) * self.subbands, stride=self.subbands)

        y_g_hat = self.multistream_conv_post(y_mb_hat)

        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        remove_weight_norm(self.ups0)
        remove_weight_norm(self.ups1)

        self.resblocks_0_0.remove_weight_norm()
        self.resblocks_0_1.remove_weight_norm()
        self.resblocks_0_2.remove_weight_norm()

        self.resblocks_1_0.remove_weight_norm()
        self.resblocks_1_1.remove_weight_norm()
        self.resblocks_1_2.remove_weight_norm()

        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.subband_conv_post)
        remove_weight_norm(self.multistream_conv_post)


class TextEncoder(nn.Module):
    def __init__(self, tts_args: TTSArguments):
        super().__init__()
        out_channels = tts_args.out_channels
        hidden_channels = tts_args.hidden_channels
        filter_channels = tts_args.filter_channels
        n_heads = tts_args.n_heads
        n_layers = tts_args.n_layers
        kernel_size = tts_args.kernel_size
        p_dropout = tts_args.p_dropout
        encoder_gin_channels = tts_args.encoder_gin_channels
        window_size = tts_args.window_size

        self.n_vocab = tts_args.n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.encoder_gin_channels = encoder_gin_channels
        self.window_size = window_size
        self.emb = nn.Embedding(tts_args.n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = Encoder(encoder_gin_channels, hidden_channels, filter_channels,
                               n_heads, n_layers, kernel_size, p_dropout, self.window_size)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1).contiguous()  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

class Flip(nn.Module):

    def forward(self, x, reverse: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x, None
class WNLayer(torch.nn.Module):
    def __init__(self, i_idx_layer, n_layers, hidden_channels, kernel_size, dilation_rate, p_dropout=0.0):
        super(WNLayer, self).__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.i_idx_layer = i_idx_layer
        dilation = dilation_rate ** i_idx_layer
        padding = int((kernel_size * dilation - dilation) / 2)
        self.in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                   dilation=dilation, padding=padding)
        self.in_layer = weight_norm(self.in_layer, name='weight')
        if i_idx_layer < n_layers - 1:
            res_skip_channels = 2 * hidden_channels
        else:
            res_skip_channels = hidden_channels

        self.res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        self.res_skip_layer = weight_norm(self.res_skip_layer, name='weight')

    def forward(self, n_channels_tensor: torch.Tensor, x: torch.Tensor, x_mask: torch.Tensor, output: torch.Tensor, g:Optional[torch.Tensor]=None):
        x_in = self.in_layer(x)
        if g is not None:
            cond_offset = self.i_idx_layer * 2 * self.hidden_channels
            g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
        else:
            g_l = torch.zeros_like(x_in)

        acts = fused_add_tanh_sigmoid_multiply(x_in,g_l, n_channels_tensor)
        acts = F.dropout(acts, self.p_dropout, self.training)# self.drop(acts)

        res_skip_acts = self.res_skip_layer(acts)
        if self.i_idx_layer < self.n_layers - 1:
            res_acts = res_skip_acts[:, :self.hidden_channels, :]
            x = (x + res_acts) * x_mask
            output = output + res_skip_acts[:, self.hidden_channels:, :]
        else:
            output = output + res_skip_acts
        return x, output

    def remove_weight_norm(self):
        remove_weight_norm(self.in_layer)
        remove_weight_norm(self.res_skip_layer)

class WN4(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, gin_channels=0, p_dropout=0.0):
        super(WN4, self).__init__()
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.n_layers = 4

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * self.n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name='weight')

        self.wn_layer_0 = WNLayer(0, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_1 = WNLayer(1, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_2 = WNLayer(2, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_3 = WNLayer(3, self.n_layers, hidden_channels, kernel_size, dilation_rate)

    def forward(self, x, x_mask, g:Optional[torch.Tensor]=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        x,output = self.wn_layer_0(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_1(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_2(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_3(n_channels_tensor, x, x_mask, output, g=g)

        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        self.wn_layer_0.remove_weight_norm()
        self.wn_layer_1.remove_weight_norm()
        self.wn_layer_2.remove_weight_norm()
        self.wn_layer_3.remove_weight_norm()

class WN16(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, gin_channels=0, p_dropout=0.0):
        super(WN16, self).__init__()
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.n_layers = 16

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * self.n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name='weight')

        self.wn_layer_0 = WNLayer(0, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_1 = WNLayer(1, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_2 = WNLayer(2, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_3 = WNLayer(3, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_4 = WNLayer(4, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_5 = WNLayer(5, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_6 = WNLayer(6, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_7 = WNLayer(7, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_8 = WNLayer(8, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_9 = WNLayer(9, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_10 = WNLayer(10, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_11 = WNLayer(11, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_12 = WNLayer(12, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_13 = WNLayer(13, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_14 = WNLayer(14, self.n_layers, hidden_channels, kernel_size, dilation_rate)
        self.wn_layer_15 = WNLayer(15, self.n_layers, hidden_channels, kernel_size, dilation_rate)

    def forward(self, x, x_mask, g:Optional[torch.Tensor]=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        x,output = self.wn_layer_0(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_1(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_2(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_3(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_4(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_5(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_6(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_7(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_8(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_9(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_10(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_11(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_12(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_13(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_14(n_channels_tensor, x, x_mask, output, g=g)
        x,output = self.wn_layer_15(n_channels_tensor, x, x_mask, output, g=g)

        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        self.wn_layer_0.remove_weight_norm()
        self.wn_layer_1.remove_weight_norm()
        self.wn_layer_2.remove_weight_norm()
        self.wn_layer_3.remove_weight_norm()
        self.wn_layer_4.remove_weight_norm()
        self.wn_layer_5.remove_weight_norm()
        self.wn_layer_6.remove_weight_norm()
        self.wn_layer_7.remove_weight_norm()
        self.wn_layer_8.remove_weight_norm()
        self.wn_layer_9.remove_weight_norm()
        self.wn_layer_10.remove_weight_norm()
        self.wn_layer_11.remove_weight_norm()
        self.wn_layer_12.remove_weight_norm()
        self.wn_layer_13.remove_weight_norm()
        self.wn_layer_14.remove_weight_norm()
        self.wn_layer_15.remove_weight_norm()

class ResidualCouplingTransformersLayer2(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0.0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.pre_transformer = SingleLayerEncoder(hidden_channels, hidden_channels, 2, 1, kernel_size, p_dropout, 4)
        self.enc = WN4(
            hidden_channels,
            kernel_size,
            dilation_rate,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g:Optional[torch.Tensor]=None, reverse:bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = h + self.pre_transformer(h * x_mask, x_mask)  # vits2 residual connection
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x, None

class ResidualCouplingTransformersBlock(nn.Module):  # vits2
    def __init__(self, tts_args: TTSArguments):
        super().__init__()
        inter_channels = tts_args.inter_channels
        hidden_channels = tts_args.hidden_channels
        kernel_size = 5
        dilation_rate = 1
        n_layers = 4
        n_flows = 4
        gin_channels = tts_args.gin_channels

        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # self.flows = nn.ModuleList()
        # for i in range(n_flows):
        #     self.flows.append(
        #         ResidualCouplingTransformersLayer2(
        #             inter_channels,
        #             hidden_channels,
        #             kernel_size,
        #             dilation_rate,
        #             n_layers,
        #             gin_channels=gin_channels,
        #             mean_only=True,
        #         )
        #     )
        #     self.flows.append(Flip())
        self.flow0_rcl = ResidualCouplingTransformersLayer2(inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True)
        self.flip0 = Flip()

        self.flow1_rcl = ResidualCouplingTransformersLayer2(inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True)
        self.flip1 = Flip()

        self.flow2_rcl = ResidualCouplingTransformersLayer2(inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True)
        self.flip2 = Flip()

        self.flow3_rcl = ResidualCouplingTransformersLayer2(inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True)
        self.flip3 = Flip()

    def forward(self, x, x_mask, g: Optional[torch.Tensor]=None, reverse:bool=False) -> torch.Tensor:
        if not reverse:
            x, _ = self.flow0_rcl(x, x_mask, g=g, reverse=reverse)
            x, _ = self.flip0(x, reverse=reverse)

            x, _ = self.flow1_rcl(x, x_mask, g=g, reverse=reverse)
            x, _ = self.flip1(x, reverse=reverse)

            x, _ = self.flow2_rcl(x, x_mask, g=g, reverse=reverse)
            x, _ = self.flip2(x, reverse=reverse)

            x, _ = self.flow3_rcl(x, x_mask, g=g, reverse=reverse)
            x, _ = self.flip3(x, reverse=reverse)
        else:
            x, _ = self.flip3(x, reverse=reverse)
            x, _ = self.flow3_rcl(x, x_mask, g=g, reverse=reverse)

            x, _ = self.flip2(x, reverse=reverse)
            x, _ = self.flow2_rcl(x, x_mask, g=g, reverse=reverse)

            x, _ = self.flip1(x, reverse=reverse)
            x, _ = self.flow1_rcl(x, x_mask, g=g, reverse=reverse)

            x, _ = self.flip0(x, reverse=reverse)
            x, _ = self.flow0_rcl(x, x_mask, g=g, reverse=reverse)

        return x

    def remove_weight_norm(self):
        self.flow0_rcl.enc.remove_weight_norm()
        self.flow1_rcl.enc.remove_weight_norm()
        self.flow2_rcl.enc.remove_weight_norm()
        self.flow3_rcl.enc.remove_weight_norm()


class DurationPredictor(nn.Module):
    def __init__(self, tts_args: TTSArguments):
        super().__init__()

        in_channels = tts_args.hidden_channels
        filter_channels = 256
        kernel_size = 3
        p_dropout = 0.5
        gin_channels = tts_args.gin_channels

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = TransLayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = TransLayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class DurationDiscriminator2(nn.Module):  # vits2 - DurationDiscriminator2
    # TODO : not using "spk conditioning" for now according to the paper.
    # Can be a better discriminator if we use it.
    def __init__(self, tts_args: TTSArguments):
        super().__init__()
        in_channels = tts_args.hidden_channels
        filter_channels = tts_args.hidden_channels
        kernel_size = 3
        p_dropout = 0.1
        gin_channels = tts_args.gin_channels

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = TransLayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = TransLayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = TransLayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = TransLayerNorm(filter_channels)

        # if gin_channels != 0:
        #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g:Optional[torch.Tensor]=None) -> torch.Tensor:
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = x * x_mask
        x = x.transpose(1, 2).contiguous()
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g:Optional[torch.Tensor]=None) -> List[List[torch.Tensor]]:
        x = torch.detach(x)
        # if g is not None:
        #   g = torch.detach(g)
        #   x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        output_probs: List[List[torch.Tensor]] = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append([output_prob])

        return output_probs

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.conv0 = weight_norm(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)))
        self.conv1 = weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)))
        self.conv2 = weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)))
        self.conv3 = weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)))
        self.conv4 = weight_norm(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0)))
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        x = F.leaky_relu(self.conv0(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv1(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv2(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv3(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv4(x), 0.1)
        fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def remove_weight_norm(self):
        remove_weight_norm(self.conv0)
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)
        remove_weight_norm(self.conv3)
        remove_weight_norm(self.conv4)
        remove_weight_norm(self.conv_post)

class DiscriminatorS(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        self.conv0 = weight_norm(Conv1d(1, 16, 15, 1, padding=7))
        self.conv1 = weight_norm(Conv1d(16, 64, 41, 4, groups=4, padding=20))
        self.conv2 = weight_norm(Conv1d(64, 256, 41, 4, groups=16, padding=20))
        self.conv3 = weight_norm(Conv1d(256, 1024, 41, 4, groups=64, padding=20))
        self.conv4 = weight_norm(Conv1d(1024, 1024, 41, 4, groups=256, padding=20))
        self.conv5 = weight_norm(Conv1d(1024, 1024, 5, 1, padding=2))
        self.conv_post = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []

        x = F.leaky_relu(self.conv0(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv1(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv2(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv3(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv4(x), 0.1)
        fmap.append(x)

        x = F.leaky_relu(self.conv5(x), 0.1)
        fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def remove_weight_norm(self):
        remove_weight_norm(self.conv0)
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)
        remove_weight_norm(self.conv3)
        remove_weight_norm(self.conv4)
        remove_weight_norm(self.conv5)
        remove_weight_norm(self.conv_post)


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        self.disc0 = DiscriminatorS()
        self.disc1 = DiscriminatorP(periods[0])
        self.disc2 = DiscriminatorP(periods[1])
        self.disc3 = DiscriminatorP(periods[2])
        self.disc4 = DiscriminatorP(periods[3])
        self.disc5 = DiscriminatorP(periods[4])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs: List[torch.Tensor] = []
        y_d_gs: List[torch.Tensor] = []
        fmap_rs: List[List[torch.Tensor]] = []
        fmap_gs: List[List[torch.Tensor]] = []

        y_d_r, fmap_r = self.disc0(y)
        y_d_g, fmap_g = self.disc0(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        y_d_r, fmap_r = self.disc1(y)
        y_d_g, fmap_g = self.disc1(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        y_d_r, fmap_r = self.disc2(y)
        y_d_g, fmap_g = self.disc2(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        y_d_r, fmap_r = self.disc3(y)
        y_d_g, fmap_g = self.disc3(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        y_d_r, fmap_r = self.disc4(y)
        y_d_g, fmap_g = self.disc4(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        y_d_r, fmap_r = self.disc5(y)
        y_d_g, fmap_g = self.disc5(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def remove_weight_norm(self):
        self.disc0.remove_weight_norm()
        self.disc1.remove_weight_norm()
        self.disc2.remove_weight_norm()
        self.disc3.remove_weight_norm()
        self.disc4.remove_weight_norm()
        self.disc5.remove_weight_norm()

class PosteriorEncoder(nn.Module):
    def __init__(self, tts_args: TTSArguments):
        super().__init__()
        in_channels = tts_args.posterior_channels
        out_channels = tts_args.inter_channels
        hidden_channels = tts_args.hidden_channels
        kernel_size = 5
        dilation_rate = 1
        n_layers = 16
        gin_channels = tts_args.gin_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN16(hidden_channels, kernel_size, dilation_rate, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

class FlexTTS(nn.Module):
    """
    TTS Synthesizer for Training
    """
    def __init__(self, tts_args: TTSArguments):
        super().__init__()
        self.segment_size = tts_args.train_segment_size // tts_args.data_hop_length
        self.use_noise_scaled_mas = True
        self.mas_noise_scale_initial = 0.01
        self.noise_scale_delta = 2e-6
        self.n_speakers = tts_args.n_speakers

        self.current_mas_noise_scale = self.mas_noise_scale_initial
        self.use_spk_conditioned_encoder = tts_args.use_spk_conditioned_encoder

        if self.use_spk_conditioned_encoder and tts_args.encoder_gin_channels > 0:
            self.enc_gin_channels = tts_args.encoder_gin_channels
        else:
            self.enc_gin_channels = 0

        # Sub-Modules
        self.enc_p = TextEncoder(tts_args)
        self.dec = MS_ISTFT_Generator(tts_args)
        self.enc_q = PosteriorEncoder(tts_args)
        self.flow = ResidualCouplingTransformersBlock(tts_args)
        self.dp = DurationPredictor(tts_args)
        self.emb_g = nn.Embedding(tts_args.n_speakers, tts_args.gin_channels)

    @custom_fwd(device_type='cuda')
    @torch.jit.ignore
    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)  # vits2?
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2).contiguous(),
                                     s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2).contiguous(), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            if self.use_noise_scaled_mas:
                epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)

        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o, o_mb = self.dec(z_slice, g=g)
        return o, o_mb, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (x, logw, logw_)

    @torch.jit.export
    def infer(self, x, x_lengths, sid, noise_scale: float=1.0, length_scale: float=1.0, noise_scale_w:float = 1.0, max_len: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        logw = self.dp(x, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2).contiguous()).transpose(1,
                                                                                 2).contiguous()  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        o, o_mb = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, o_mb, attn, y_mask, (z, z_p, m_p, logs_p)

    def kinya_tts(self, sid:int, text: str, output_sampling_rate: int, output_wav_file_name: str, device, speed:float = 1.0, model_sampling_rate: int = 24000, intersperse_with_blank=True, volume_gain=3.0):
        text = text_to_sequence(re.sub(r"[\[\](){}]", "", text))
        if intersperse_with_blank:
            text = intersperse(text, 0)
        text = torch.LongTensor(text)
        with torch.no_grad():
            x_tst = text.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([text.size(0)]).to(device)
            sid = torch.LongTensor([sid]).to(device)
            audio = self.infer(x_tst, x_tst_lengths, sid, noise_scale=.667, length_scale=(1.0/speed))[0][0, 0].data.cpu().float()
        louder_vol = torchaudio.transforms.Vol(gain=volume_gain, gain_type="amplitude")
        audio = louder_vol(audio.unsqueeze(0))

        if output_sampling_rate != model_sampling_rate:
            resampler = T.Resample(model_sampling_rate, output_sampling_rate, dtype=audio.dtype)
            audio = resampler(audio)
        torchaudio.save(output_wav_file_name, audio, output_sampling_rate)
        return True

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.enc_q.remove_weight_norm()
        self.flow.remove_weight_norm()

def export_mobile_module(module: nn.Module, filename:str):
    from torch.utils.mobile_optimizer import optimize_for_mobile
    wrapper = torch.jit.script(module)
    scripted_model = torch.jit.script(wrapper)
    optimized_model = optimize_for_mobile(scripted_model)
    optimized_model._save_for_lite_interpreter(filename)
    print(f"Done! Generated: {filename}")

if __name__ == '__main__':

    my_tts_args = TTSArguments()
    my_tts_args.n_vocab = len(tts_symbols)
    my_tts_args.n_speakers = 50

    dec = MS_ISTFT_Generator(my_tts_args)
    dec.remove_weight_norm()
    export_mobile_module(dec, "dec.ptl")

    enc = TextEncoder(my_tts_args)
    export_mobile_module(enc, "enc.ptl")

    flow = ResidualCouplingTransformersBlock(my_tts_args)
    flow.remove_weight_norm()
    export_mobile_module(flow, "flow.ptl")

    dp = DurationPredictor(my_tts_args)
    export_mobile_module(dp, "dp.ptl")

    dur_disc = DurationDiscriminator2(my_tts_args)
    export_mobile_module(dur_disc, "dur_disc.ptl")

    net_disc = MultiPeriodDiscriminator()
    net_disc.remove_weight_norm()
    export_mobile_module(net_disc, "net_disc.ptl")

    enc_post = PosteriorEncoder(my_tts_args)
    enc_post.remove_weight_norm()
    export_mobile_module(enc_post, "enc_post.ptl")

    flex_tts = FlexTTS(my_tts_args)
    flex_tts.remove_weight_norm()
    del flex_tts.enc_q
    export_mobile_module(flex_tts, "flex_tts.ptl")

