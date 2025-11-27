from __future__ import print_function, division, annotations

from typing import Tuple, Mapping, Any, List, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter

from deepkin.modules.tts_arguments import TTSArguments
from deepkin.modules.tts_commons import slice_segments, clip_grad_value_
from deepkin.modules.tts_losses import discriminator_loss, kl_loss, feature_loss, generator_loss
from deepkin.modules.tts_mel import mel_spectrogram_torch
from deepkin.modules.tts_modules import FlexTTS, MultiPeriodDiscriminator, DurationDiscriminator2, \
    export_mobile_module
from deepkin.utils.arguments import FlexArguments
from deepkin.utils.misc_functions import time_now


class FlexTTSTrainer(object):
    """
    TTS Synthesizer for Training
    """
    def __init__(self, tts_args: TTSArguments, rank, device, epoch_str = 1):
        super().__init__()
        self.tts_args = tts_args

        self.flex_tts = FlexTTS(tts_args).to(device)
        self.output_discriminator = MultiPeriodDiscriminator().to(device)
        self.duration_discriminator = DurationDiscriminator2(tts_args).to(device)

        self.uses_ddp = False
        try:
            if dist.is_initialized():
                self.uses_ddp = (dist.get_world_size() > 1)
        except ValueError:
            pass

        if self.uses_ddp:
            self.flex_tts = DDP(self.flex_tts, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            self.output_discriminator = DDP(self.output_discriminator, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            self.duration_discriminator = DDP(self.duration_discriminator, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        self.epoch_str = epoch_str

        self.optim_output_generator = torch.optim.AdamW(self.flex_tts.parameters(), self.tts_args.train_learning_rate, betas=(self.tts_args.train_betas[0], self.tts_args.train_betas[1]), eps=self.tts_args.train_eps)
        self.optim_output_discriminator = torch.optim.AdamW(self.output_discriminator.parameters(), self.tts_args.train_learning_rate, betas=(self.tts_args.train_betas[0], self.tts_args.train_betas[1]), eps=self.tts_args.train_eps)
        self.optim_duration_discriminator = torch.optim.AdamW(self.duration_discriminator.parameters(), self.tts_args.train_learning_rate, betas=(self.tts_args.train_betas[0], self.tts_args.train_betas[1]), eps=self.tts_args.train_eps)

        self.scheduler_output_generator = torch.optim.lr_scheduler.ExponentialLR(self.optim_output_generator, gamma=self.tts_args.train_lr_decay, last_epoch=self.epoch_str - 2)
        self.scheduler_output_discriminator = torch.optim.lr_scheduler.ExponentialLR(self.optim_output_discriminator, gamma=self.tts_args.train_lr_decay, last_epoch=self.epoch_str - 2)
        self.scheduler_duration_discriminator = torch.optim.lr_scheduler.ExponentialLR(self.optim_duration_discriminator, gamma=self.tts_args.train_lr_decay, last_epoch=self.epoch_str - 2)

        self.training = True
        self.train_epoch = 0

    def train(self):
        self.training = True
        self.flex_tts.train()
        self.output_discriminator.train()
        self.duration_discriminator.train()

    def eval(self):
        self.training = False
        self.flex_tts.eval()
        self.output_discriminator.eval()
        self.duration_discriminator.eval()

    def float(self):
        self.flex_tts.float()
        self.output_discriminator.float()
        self.duration_discriminator.float()

    def half(self):
        self.flex_tts.half()
        self.output_discriminator.half()
        self.duration_discriminator.half()

    def zero_grad(self, set_to_none=True):
        self.flex_tts.zero_grad(set_to_none=True)
        self.output_discriminator.zero_grad(set_to_none=True)
        self.duration_discriminator.zero_grad(set_to_none=True)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for p in self.flex_tts.parameters(recurse=recurse):
            yield p
        for p in self.output_discriminator.parameters(recurse=recurse):
            yield p
        for p in self.duration_discriminator.parameters(recurse=recurse):
            yield p

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {'tts_args': self.tts_args.as_dict(),

                'flex_tts': (self.flex_tts.module.state_dict() if self.uses_ddp else self.flex_tts.state_dict()),
                'output_discriminator': (self.output_discriminator.module.state_dict() if self.uses_ddp else self.output_discriminator.state_dict()),
                'duration_discriminator': (self.duration_discriminator.module.state_dict() if self.uses_ddp else self.duration_discriminator.state_dict()),

                'optim_output_generator': self.optim_output_generator.state_dict(),
                'optim_output_discriminator': self.optim_output_discriminator.state_dict(),
                'optim_duration_discriminator': self.optim_duration_discriminator.state_dict(),

                'scheduler_output_generator': self.scheduler_output_generator.state_dict(),
                'scheduler_output_discriminator': self.scheduler_output_discriminator.state_dict(),
                'scheduler_duration_discriminator': self.scheduler_duration_discriminator.state_dict(),

                'epoch_str': self.epoch_str}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.tts_args = TTSArguments().from_dict(state_dict['tts_args'])
        self.epoch_str = state_dict['epoch_str']

        if self.uses_ddp:
            self.flex_tts.module.load_state_dict(state_dict['flex_tts'])
            self.output_discriminator.module.load_state_dict(state_dict['output_discriminator'])
            self.duration_discriminator.module.load_state_dict(state_dict['duration_discriminator'])
        else:
            self.flex_tts.load_state_dict(state_dict['flex_tts'])
            self.output_discriminator.load_state_dict(state_dict['output_discriminator'])
            self.duration_discriminator.load_state_dict(state_dict['duration_discriminator'])

        self.optim_output_generator.load_state_dict(state_dict['optim_output_generator'])
        self.optim_output_discriminator.load_state_dict(state_dict['optim_output_discriminator'])
        self.optim_duration_discriminator.load_state_dict(state_dict['optim_duration_discriminator'])

        self.scheduler_output_generator.load_state_dict(state_dict['scheduler_output_generator'])
        self.scheduler_output_discriminator.load_state_dict(state_dict['scheduler_output_discriminator'])
        self.scheduler_duration_discriminator.load_state_dict(state_dict['scheduler_duration_discriminator'])

    @staticmethod
    def from_pretrained(filename):
        print(time_now(), 'Loading FlexTTS from', filename)
        state_dict = torch.load(filename, map_location='cpu')
        steps = state_dict["steps"]
        state_dict = state_dict['model_state_dict']
        epoch_str = state_dict['epoch_str']
        tts_args = TTSArguments().from_dict(state_dict['tts_args'])
        model = FlexTTSTrainer(tts_args, 0, torch.device('cpu'), epoch_str=epoch_str)
        model.load_state_dict(state_dict)
        print(time_now(), 'FlexTTS train steps:', f'{(steps/1000):,.0f}K')
        print(time_now(), 'Loading FlexTTS from', filename, 'done!')
        return model

    def export_mobile_FlexKinyaTTS(self, filename: str):
        self.flex_tts.eval()
        self.flex_tts.zero_grad()
        self.flex_tts.remove_weight_norm()
        del self.flex_tts.enc_q
        flex_kinya_tts = FlexKinyaTTS(self.flex_tts)
        export_mobile_module(flex_kinya_tts, filename)

    def __call__(self, batch_idx: int, args: FlexArguments, model_cache, training_item: Tuple):
        (x, x_lengths, spec, spec_lengths, y, y_lengths, speaker_ids) = training_item

        # 1. Synthesizer
        y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_) = self.flex_tts(x, x_lengths, spec, spec_lengths, speaker_ids)
        y_mel = slice_segments(spec, ids_slice, self.tts_args.train_segment_size // self.tts_args.data_hop_length)

        with torch.cuda.amp.autocast(enabled=False):
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).to(dtype=torch.float32), self.tts_args.data_filter_length, self.tts_args.data_n_mel_channels, self.tts_args.data_sampling_rate, self.tts_args.data_hop_length, self.tts_args.data_win_length, self.tts_args.data_mel_fmin, self.tts_args.data_mel_fmax)

        # 2. Output discriminator
        y = slice_segments(y, ids_slice * self.tts_args.data_hop_length, self.tts_args.train_segment_size)
        y_d_hat_r, y_d_hat_g, _, _ = self.output_discriminator(y.detach(), y_hat.detach())
        with torch.cuda.amp.autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

        # 3. Duration discriminator
        y_dur_hat_r, y_dur_hat_g = self.duration_discriminator(hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach())
        with torch.cuda.amp.autocast(enabled=False):
            loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
            loss_dur_disc_all = loss_dur_disc

        # 4. Duration discriminator optimization
        self.optim_duration_discriminator.zero_grad()
        loss_dur_disc_all.backward()
        clip_grad_value_(self.duration_discriminator.parameters(), None)
        self.optim_duration_discriminator.step()

        # 5. Output discriminator optimization
        self.optim_output_discriminator.zero_grad()
        loss_disc_all.backward()
        clip_grad_value_(self.output_discriminator.parameters(), None)
        self.optim_output_discriminator.step()

        # 6. Synthesizer optimization follows
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.output_discriminator(y, y_hat)
        y_dur_hat_r, y_dur_hat_g = self.duration_discriminator(hidden_x, x_mask, logw_, logw)
        with torch.cuda.amp.autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.tts_args.train_c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.tts_args.train_c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)

            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_dur_gen

        # 5. Output discriminator optimization
        self.optim_output_generator.zero_grad()
        loss_gen_all.backward()
        clip_grad_value_(self.flex_tts.parameters(), None)
        self.optim_output_generator.step()

        if (batch_idx == 0) and self.training:
            if self.train_epoch > 0:
                self.scheduler_output_generator.step()
                self.scheduler_output_discriminator.step()
                self.scheduler_duration_discriminator.step()
            self.train_epoch += 1

        return [loss_disc, loss_dur_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_dur_gen]

class FlexKinyaTTS(nn.Module):

    def __init__(self, flex_tts):
        super().__init__()
        self.flex_tts = flex_tts

    def forward(self, id_seq: List[int], sid: int, speed: float = 1.0) -> torch.Tensor:
        x = torch.LongTensor(id_seq).unsqueeze(0)
        x_lengths = torch.LongTensor([x.size(1)])
        sid = torch.LongTensor([sid])
        device = next(self.parameters()).device
        audio_stream = self.flex_tts.infer(x.to(device), x_lengths.to(device), sid.to(device), noise_scale=0.667, length_scale=(1.0/speed))[0][0].cpu().data.float()
        return audio_stream

    @staticmethod
    def from_pretrained(device: torch.device, pretrained_model_file: str) -> FlexKinyaTTS:
        trained_model = FlexTTSTrainer.from_pretrained(pretrained_model_file)
        trained_model.flex_tts.eval()
        tts = FlexKinyaTTS(trained_model.flex_tts).to(device)
        tts.flex_tts.eval()
        tts.flex_tts.zero_grad()
        tts.flex_tts.remove_weight_norm()
        del tts.flex_tts.enc_q
        return tts


def compare_params(name, a, b):
    sa,sb = sum(p.numel() for p in a.parameters()),  sum(p.numel() for p in b.parameters())
    if sa != sb:
        print('xxxxxxxxx=====>', name, sa, '~', sb)

def key_replacements(keys):
    reps = dict()

    for l in range(10):
        for k in keys:
            if k.startswith(f'enc_p.encoder.attn_layers.{l}.'):
                reps[k] = k.replace(f'enc_p.encoder.attn_layers.{l}.', f'enc_p.encoder.encoder_layer{l}.attn_layer.')
            if k.startswith(f'enc_p.encoder.norm_layers_1.{l}.'):
                reps[k] = k.replace(f'enc_p.encoder.norm_layers_1.{l}.', f'enc_p.encoder.encoder_layer{l}.norm_layer_1.')
            if k.startswith(f'enc_p.encoder.norm_layers_2.{l}.'):
                reps[k] = k.replace(f'enc_p.encoder.norm_layers_2.{l}.', f'enc_p.encoder.encoder_layer{l}.norm_layer_2.')
            if k.startswith(f'enc_p.encoder.ffn_layers.{l}.'):
                reps[k] = k.replace(f'enc_p.encoder.ffn_layers.{l}.', f'enc_p.encoder.encoder_layer{l}.ffn_layer.')

    for l in range(2):
        for k in keys:
            if k.startswith(f'dec.ups.{l}.'):
                reps[k] = k.replace(f'dec.ups.{l}.', f'dec.ups{l}.')

    for l in range(6):
        i = l//3
        j = l%3
        for k in keys:
            if k.startswith(f'dec.resblocks.{l}.'):
                reps[k] = k.replace(f'dec.resblocks.{l}.', f'dec.resblocks_{i}_{j}.')

    for l in range(16):
        for k in keys:
            if k.startswith(f'enc_q.enc.in_layers.{l}.'):
                reps[k] = k.replace(f'enc_q.enc.in_layers.{l}.', f'enc_q.enc.wn_layer_{l}.in_layer.')
            if k.startswith(f'enc_q.enc.res_skip_layers.{l}.'):
                reps[k] = k.replace(f'enc_q.enc.res_skip_layers.{l}.', f'enc_q.enc.wn_layer_{l}.res_skip_layer.')


    for l in range(0,8,2):
        h = l//2
        for k in keys:
            if k.startswith(f'flow.flows.{l}.post.'):
                reps[k] = k.replace(f'flow.flows.{l}.post.', f'flow.flow{h}_rcl.post.')
            if k.startswith(f'flow.flows.{l}.pre.'):
                reps[k] = k.replace(f'flow.flows.{l}.pre.', f'flow.flow{h}_rcl.pre.')
            if k.startswith(f'flow.flows.{l}.enc.cond_layer.'):
                reps[k] = k.replace(f'flow.flows.{l}.enc.cond_layer.', f'flow.flow{h}_rcl.enc.cond_layer.')
            if k.startswith(f'flow.flows.{l}.pre_transformer.attn_layers.0.'):
                reps[k] = k.replace(f'flow.flows.{l}.pre_transformer.attn_layers.0.',
                                    f'flow.flow{h}_rcl.pre_transformer.encoder_layer0.attn_layer.')
            if k.startswith(f'flow.flows.{l}.pre_transformer.norm_layers_1.0.'):
                reps[k] = k.replace(f'flow.flows.{l}.pre_transformer.norm_layers_1.0.',
                                    f'flow.flow{h}_rcl.pre_transformer.encoder_layer0.norm_layer_1.')
            if k.startswith(f'flow.flows.{l}.pre_transformer.norm_layers_2.0.'):
                reps[k] = k.replace(f'flow.flows.{l}.pre_transformer.norm_layers_2.0.',
                                    f'flow.flow{h}_rcl.pre_transformer.encoder_layer0.norm_layer_2.')
            if k.startswith(f'flow.flows.{l}.pre_transformer.ffn_layers.0.'):
                reps[k] = k.replace(f'flow.flows.{l}.pre_transformer.ffn_layers.0.',
                                    f'flow.flow{h}_rcl.pre_transformer.encoder_layer0.ffn_layer.')
            for j in range(4):
                if k.startswith(f'flow.flows.{l}.enc.in_layers.{j}.'):
                    reps[k] = k.replace(f'flow.flows.{l}.enc.in_layers.{j}.',
                                        f'flow.flow{h}_rcl.enc.wn_layer_{j}.in_layer.')
                if k.startswith(f'flow.flows.{l}.enc.res_skip_layers.{j}.'):
                    reps[k] = k.replace(f'flow.flows.{l}.enc.res_skip_layers.{j}.',
                                        f'flow.flow{h}_rcl.enc.wn_layer_{j}.res_skip_layer.')
    return reps

def compare_state_dict(a, b):
    reps = key_replacements(a)
    a_keys = set([reps[k] if (k in reps.keys()) else k for k in a.keys()])
    b_keys = set(b.keys())
    for k in sorted(list(b_keys)):
        if k not in a_keys:
            print(k)
