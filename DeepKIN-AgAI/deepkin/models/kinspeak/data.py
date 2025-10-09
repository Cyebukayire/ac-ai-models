from typing import List
import math
import os
import random
import numpy as np
from datetime import datetime
import torch
import torchaudio
import torchaudio.transforms as T
import progressbar
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from mutagen.mp3 import MP3

from deepkin.clib.libkinlp.kinlpy import BOS_ID, EOS_ID
from deepkin.utils.misc_functions import time_now

audio_min_duration = 2.0
audio_max_duration = 25.0
labels_min_length = 4
labels_max_length = 1024

cv_audio_clips_dir = "data/cv-corpus-20.0-2024-12-06/rw/clips/"
top_cv_file = 'data/top_cv_train.tsv'

class SampleConfig:
    def __init__(self):
        self.resamplers = {}
        self.mel_spectrograms = {}
        self.resample_rate = 16000
        self.lowpass_filter_width = 64
        self.rolloff = 0.9475937167399596
        self.resampling_method = "kaiser_window"
        self.beta = 14.769656459379492
        self.n_fft = 1024
        self.n_mels = 80

def get_sampling_tools(cfg: SampleConfig, sample_rate, waveform):
    sr_kwd = str(sample_rate)
    if sr_kwd in cfg.resamplers:
        resampler = cfg.resamplers[sr_kwd]
        mel_spectrogram = cfg.mel_spectrograms[sr_kwd]
    else:
        win_length = cfg.resample_rate * 25 // 1000  # 25ms
        hop_length = cfg.resample_rate * 10 // 1000  # 10ms
        resampler = T.Resample(sample_rate, cfg.resample_rate, lowpass_filter_width=cfg.lowpass_filter_width,
                                 rolloff=cfg.rolloff, resampling_method=cfg.resampling_method,
                                 dtype=waveform.dtype,
                                 beta=cfg.beta, )  # .cuda()
        mel_spectrogram = T.MelSpectrogram(sample_rate=cfg.resample_rate, n_fft=cfg.n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length, center=True, pad_mode="reflect", power=2.0,
                                             norm="slaney", onesided=True, n_mels=cfg.n_mels,
                                             mel_scale="htk", )  # .cuda()
        cfg.resamplers[sr_kwd] = resampler
        cfg.mel_spectrograms[sr_kwd] = mel_spectrogram
    return resampler, mel_spectrogram


def form_input_data(waveform, sample_rate, sample_config, log_eps=1e-36):
    resampler, mel_spectrogram = get_sampling_tools(sample_config, sample_rate, waveform)
    with torch.no_grad():
        if int(sample_rate) != int(sample_config.resample_rate):
            waveform = resampler(waveform)
        waveform = torch.mean(waveform, 0, keepdim=False)
        input_data = mel_spectrogram(waveform)  # (F,L)
        input_data = torch.log(input_data + log_eps)
    return input_data # (F,L)

def time_to_spec_samples(secs):
    return int(secs * 1000) // 10 # 10ms hop length

def adaptive_spec_augment(log_mel_spectrogram, # (N,F,L)
                          log_mel_spec_lengths: List[int],
                          frequency_mask_param=27,
                          num_frequency_masks=2,
                          time_mask_ratio_ps = 0.05,
                          num_time_masks=10):
    cloned_spec = log_mel_spectrogram.clone()
    mask_value = cloned_spec.min().item()
    for i,length in enumerate(log_mel_spec_lengths):
        F = frequency_mask_param
        T = int(math.floor(length * time_mask_ratio_ps))
        for _ in range(num_frequency_masks):
            cloned_spec[i:(i+1),:,:length] = torchaudio.functional.mask_along_axis(cloned_spec[i:(i+1),:,:length],
                                                                                     F, mask_value, 1)
        for _ in range(num_time_masks):
            cloned_spec[i:(i+1),:,:length] = torchaudio.functional.mask_along_axis(cloned_spec[i:(i+1),:,:length],
                                                                                     T, mask_value, 2)
    return cloned_spec

# TOP_CV_TRAIN_FILES = 0
#
# def set_top_cv_train_files(n):
#     global TOP_CV_TRAIN_FILES
#     TOP_CV_TRAIN_FILES = n

def data_read_common_voice(cv_list_file, text_to_id_sequence_fun, bos_id, eos_id):
    # fl = open(top_cv_file,'r', encoding="utf-8")
    # top_cv_train_clips = [line.rstrip('\n') for line in fl if len(line.rstrip('\n')) > 0]
    # top_cv_train_clips = set(top_cv_train_clips[:TOP_CV_TRAIN_FILES])
    # fl.close()

    f = open(cv_list_file, 'r', encoding="utf-8")
    cv_lines = [line.rstrip('\n') for line in f if len(line.rstrip('\n')) > 0]
    f.close()
    cv_lines = cv_lines[1:]
    cv_data = []
    print(time_now(), 'Reading {} common voice audio files listed in {} ...'.format(len(cv_lines), cv_list_file), flush=True)
    with progressbar.ProgressBar(initial_value=0,
                                 max_value=len(cv_lines),
                                 redirect_stdout=True) as bar:
        for itr,line in enumerate(cv_lines):
            if (itr % 100000) == 0:
                bar.update(itr)
            tokens_list = line.split('\t')
            if len(tokens_list) == 13:
                (client_id,path,sentence_id,sentence,sentence_domain,up_votes,down_votes,age,gender,accents,variant,locale,segment) = tuple(tokens_list)
                # if toks[1] in top_cv_train_clips or ('test' in cv_list_file) or ('dev' in cv_list_file):
                labels = [bos_id] + text_to_id_sequence_fun(sentence) + [eos_id]
                audio_file = os.path.join(cv_audio_clips_dir, path)
                tags = (age.strip(), gender.strip())
                if (len(labels) >= labels_min_length) and (len(labels) <= labels_max_length):
                    try:
                        secs = MP3(audio_file).info.length
                        if ((secs >= audio_min_duration) and (secs <= audio_max_duration)):
                            cv_data.append((audio_file, secs, labels, tags))
                    except:
                        print('Can\'t read file {}'.format(audio_file))
    tot = sum([s for f, s, l,ttgg in cv_data])
    hr = int(tot / 3600)
    mn = int(tot / 60) % 60
    sc = int(tot) % 60
    print(time_now(), '==> Read total {} CV files: {}h{}m{}s from {}'.format(len(cv_data),hr,mn,sc,cv_list_file))
    return cv_data

def data_read_common_voice_with_syllabe_vocab(file):
    from deepkin.data.syllabe_vocab import text_to_id_sequence
    return data_read_common_voice(file, text_to_id_sequence, BOS_ID, EOS_ID)

class KinSpeakDataset(Dataset):
    def __init__(self, data_read_list, include_labels=False,
                 num_shuffle_buckets=100,
                 max_batch_seconds=80,
                 debug=False,
                 batch_amplification_factor = 1.2):
        self.include_labels = include_labels
        self.num_shuffle_buckets = num_shuffle_buckets
        self.max_batch_seconds = max_batch_seconds
        self.debug=debug
        self.batch_amplification_factor = batch_amplification_factor
        self.sample_config = SampleConfig()
        self.data_items = []
        for (file,file_read_fn) in data_read_list:
            self.data_items.extend(file_read_fn(file)) # format: List[(audio_filename,audio_length,labels_seq)]
        self.index = [i for i in range(len(self.data_items))]
        self.index.sort(key=lambda x: self.data_items[x][1], reverse=True)
        if self.debug:
            tot = sum([s for f,s,l,ttgg in self.data_items])
            hr = int(tot/3600)
            mn = int(tot/60) % 60
            sc = int(tot) % 60
            print(time_now(), 'Read {} audio files:> Total: {}h{}m{}s ==> {:.0f} -> {:.0f} secs'.format(len(self.data_items), hr, mn, sc, self.data_items[self.index[0]][1], self.data_items[self.index[-1]][1]), flush=True)
        total_length = int(sum([x[1] for x in self.data_items]))+1
        bucket_size = (total_length // self.num_shuffle_buckets) + 1
        self.buckets = []
        start = 0
        num_seconds = 0.0
        for end in range(len(self.index)):
            length = self.data_items[self.index[end]][1]
            if (num_seconds + length) > bucket_size:
                if end > start:
                    self.buckets.append((start, end))
                num_seconds = 0.0
                start = end
            num_seconds += length
        self.shuffle_buckets_and_mark_batches()

    def shuffle_buckets_and_mark_batches(self):
        if self.debug:
            print(time_now(), 'Dataset shuffling ...', flush=True)
        seed_val = datetime.now().microsecond + (13468 * os.getpid())
        seed_val = int(seed_val) % ((2 ** 32) - 1)
        np.random.seed(seed_val)
        random.seed(seed_val)
        torch.random.manual_seed(seed_val)
        # Shuffle within buckets
        for (start, end) in self.buckets:
            copy = self.index[start:end]
            random.shuffle(copy)
            self.index[start:end] = copy
        # Form batches
        new_batches = []
        start = 0
        num_seconds = 0.0
        excl = 0
        for end, idx in enumerate(self.index):
            length = self.data_items[idx][1]
            if (num_seconds + length) > self.max_batch_seconds:
                if end > start:
                    new_batches.append((start, end))
                num_seconds = 0.0
                excl = 0
                start = end
            num_seconds += length
            excl += 1
        random.shuffle(new_batches)
        self.batches = new_batches
        if self.debug:
            print(time_now(), 'Batching DONE: got {} batches; discarded {} examples ({:.1f} seconds)'.format(len(self.batches), excl,
                                                                                         num_seconds), flush=True)
    def __len__(self):
        return int(len(self.batches) * self.batch_amplification_factor) # x1.5 to ensure we will always reach len(batches)-1 to re-shuffle buckets --> Num epochs adjusted to 200 to to reflect this!

    def __getitem__(self, batch_idx):
        batch_idx = batch_idx % len(self.batches)
        (start, end) = self.batches[batch_idx]
        items = []
        # total_samples = 0
        for i in range(start,end):
            idx = self.index[i]
            (audio_filename, audio_length, labels_seq, tags) = self.data_items[idx]
            waveform, sample_rate = torchaudio.load(audio_filename)
            input_data = form_input_data(waveform, sample_rate, self.sample_config)
            items.append((input_data,labels_seq))
        if batch_idx == (len(self.batches) - 1):
            self.shuffle_buckets_and_mark_batches()
        return items

def kinspeak_collate(batch_items):
    items = batch_items[0]
    input_data = []
    target_data = []
    input_lengths = []
    target_lengths = []
    for (x,y) in items:
        x_len = x.size(1)
        y_len = len(y)
        input_data.append(x.transpose(0, 1))  # padding dimension must be 0
        target_data.extend(y)
        input_lengths.append(x_len)
        target_lengths.append(y_len)
    with torch.no_grad():
        input = pad_sequence(input_data, batch_first=True).transpose(1, 2)  # (N,F,L)
        target = torch.tensor(target_data, dtype=torch.long)  # (N,S)
    return (input, input_lengths, target, target_lengths)

def kinspeak_collate_with_spec_augment(batch_items):
    items = batch_items[0]
    input_data = []
    target_data = []
    input_lengths = []
    target_lengths = []
    for (x,y) in items:
        x_len = x.size(1)
        y_len = len(y)
        input_data.append(x.transpose(0, 1))  # padding dimension must be 0
        target_data.extend(y)
        input_lengths.append(x_len)
        target_lengths.append(y_len)
    with torch.no_grad():
        input = pad_sequence(input_data, batch_first=True).transpose(1, 2)  # (N,F,L)
        input = adaptive_spec_augment(input,  # (N,F,L)
                              input_lengths)
        target = torch.tensor(target_data, dtype=torch.long)  # (N,S)
    return (input, input_lengths, target, target_lengths)

def kinspeak_collate_without_labels(batch_items):
    items = batch_items[0]
    input_data = []
    input_lengths = []
    for (x,_) in items:
        x_len = x.size(1)
        input_data.append(x.transpose(0, 1))  # padding dimension must be 0
        input_lengths.append(x_len)
    with torch.no_grad():
        input = pad_sequence(input_data, batch_first=True).transpose(1, 2)  # (N,F,L)
    return (input, input_lengths, None, None)
