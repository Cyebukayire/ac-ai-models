from typing import List, Union

from tap import Tap

class TTSArguments(Tap):
    # Training hyperparameters
    train_learning_rate: float = 2e-4
    train_betas: List[float] = [0.8, 0.99]
    train_eps: float = 1e-7
    train_lr_decay: float = 0.999875
    # Training configuration
    train_segment_size: int = 8192
    train_c_mel: int = 45
    train_c_kl: float = 1.0
    train_fft_sizes: List[int] = [384, 683, 171]
    train_hop_sizes: List[int] = [30, 60, 10]
    train_win_lengths: List[int] = [150, 300, 60]
    train_window: str = "hann_window"
    # Data configuration
    data_max_wav_value: float = 32768.0
    data_sampling_rate: int = 24000
    data_filter_length: int = 1024
    data_hop_length: int = 256
    data_win_length: int = 1024
    data_n_mel_channels: int = 80
    data_mel_fmin: float = 0.0
    data_mel_fmax: Union[int,None] = None
    data_add_blank: bool = True
    # Vocab
    n_vocab: int = 126
    # Speakers
    n_speakers: int = 50
    use_spk_conditioned_encoder: bool = False
    encoder_gin_channels: int = 0
    use_mel_posterior_encoder: bool = True
    #Posterior Encoder
    posterior_channels: int = 80
    # Decoder
    inter_channels: int = 192
    initial_channel: int = 192
    resblock: int = 1
    resblock_kernel_sizes: List[int] = [3, 7, 11]
    resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates: List[int] = [4, 4]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = [16, 16]
    gen_istft_n_fft: int = 16
    gen_istft_hop_size: int = 4
    subbands: int = 4
    # Text Encoder
    out_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 4
    n_layers: int = 8
    kernel_size: int = 3
    p_dropout: int = 0.1
    gin_channels: int = 512
    window_size: int = 4

    def __init__(self, *args, **kwargs):
        super().__init__(explicit_bool=True,*args, **kwargs)

def tts_base_args(list_args=None) -> TTSArguments:
    if list_args is not None:
        args = TTSArguments().parse_args(list_args)
    else:
        args = TTSArguments().parse_args()
    return args
