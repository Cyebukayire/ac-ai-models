from __future__ import print_function, division

from tap import Tap

class FlexArguments(Tap):
    use_multi_morph: bool = False
    use_rope: bool = False
    max_batch_num_sequences: int = 1024
    enable_amp: bool = True
    use_bfloat16: bool = True
    enable_grad_norm_clipping: bool = False
    grad_norm_clipping_max_norm: float = 100.0
    tts_data_dir: str = None
    tts_train_data_file: str = None
    tts_dev_data_file: str = None
    qa_train_qpn_triples: str = None
    qa_train_query_text: str = None
    qa_train_query_id: str = None
    qa_train_passage_text: str = None
    qa_train_passage_id: str = None
    qa_dev_qpn_triples: str = None
    qa_dev_query_text: str = None
    qa_dev_query_id: str = None
    qa_dev_passage_text: str = None
    qa_dev_passage_id: str = None
    dist_backend:str = 'nccl'
    min_gpu_memory: int = 8
    max_gpu_memory: int = 23
    mlm_batch_C1: float = 0.000006
    mlm_batch_C2: float = 0.0008
    mlm_batch_C3: float = 0.08
    min_mlm_batch_size: int = 4
    max_mlm_batch_size: int = 256
    colbert_embedding_dim: int = 1024
    colbert_similarity_metric: str = "cosine" # or 'l2'
    colbert_num_special_types: int = 3
    info_nce_temperature: float = 0.02
    info_nce_query_scale_weight_lambda: float = 0.5
    info_nce_multiple: int = 4,
    gpu_memory_scale_constant: float = 0.00001
    accumulation_steps: int = 1
    batch_size: int = 8
    char_max_seq_len: int = 1024
    checkpoint_steps: int = 100
    checkpoint_offset: int = 0
    cls_dev_input0: str = None
    cls_dev_input1: str = None
    cls_dev_label: str = None
    cls_labels: str = "0"
    cls_test_input0: str = None
    cls_test_input1: str = None
    cls_test_label: str = None
    cls_test_prediction_file: str = None
    cls_train_input0: str = None
    cls_train_input1: str = None
    cls_train_label: str = None
    corpus_id: int = 1
    dataloader_num_workers: int = 1
    dataloader_persistent_workers: bool = True
    dataloader_pin_memory: bool = True
    dataset_max_seq_len: int = 512
    decoder_mamba_num_layers: int = 1
    dev_parsed_corpus: str = None
    dev_unparsed_corpus: str = None
    devbest_cls_model_save_file_path: str = None
    devbest_cls_output_file: str = None
    embed_dim: int = 960
    empty_cache_at_gradient_step: bool = False
    empty_cache_at_log_step: bool = False
    encoder_fine_tune: bool = True
    final_cls_model_save_file_path: str = None
    final_cls_output_file: str = None
    ft_component_wise_gradient_clip: float = 0.0
    ft_reinit_layers: int = 0
    ft_swa_epochs_ratio: float = 0.25
    ft_swa_lr_ratio: float = 0.05
    ft_proj_init_stdev: float = 0.01
    gpus: int = 1
    head_trunk: bool = False
    home_path: str = "KINLP"
    inference_model_file: str = None
    input_format: str = None
    kin2kin_use_bert: bool = True
    kin2kin_use_gpt: bool = False
    kinlp_conf: str = None
    use_rms_norm: bool = False
    label_smoothing_eps: float = 0.1
    layernorm_epsilon: float = 1e-6
    load_saved_model: bool = True
    local_rank: int = 0
    train_log_steps: int = 1
    lr_decay_style: str = "linear"
    main_sequence_encoder_dim_ffn: int = 3072
    main_sequence_encoder_dropout: float = 0.1
    main_sequence_encoder_max_rel_pos: int = 256
    main_sequence_encoder_max_seq_len: int = 512
    main_sequence_encoder_num_heads: int = 12
    main_sequence_encoder_num_layers: int = 12
    main_sequence_encoder_rel_pos_bins: int = 256
    max_dataset_chunk_size: int = 500_000
    max_input_lines: int = 99999999
    max_mlm_documents: int = 1
    max_batch_tokens: int = 4096
    min_lr: float = 1e-6
    model_keyword: str = None
    model_save_path: str = None
    model_variant: str = None
    morpho_dim_ffn: int = 768
    morpho_dim_hidden: int = 160
    morpho_dropout: float = 0.1
    morpho_max_rel_pos: int = 12
    morpho_max_token_len: int = 24
    morpho_num_heads: int = 4
    morpho_num_layers: int = 4
    morpho_rel_pos_bins: int = 12
    multi_task_weighting: bool = False
    num_epochs: int = 30
    num_iters: int = 200000
    num_losses: int = 1
    number_of_load_batches: int = 16000
    peak_lr: float = 6e-4
    pooler_dropout: float = 0.3
    post_mlm_epochs: int = 0
    pretrained_bert_model_dir: str = None
    pretrained_bert_model_file: str = 'model.pt'
    pretrained_gpt_model_dir: str = None
    pretrained_gpt_model_file: str = None
    regression_scale_factor: float = 5.0
    regression_target: bool = False
    stem_dim_hidden: int = 160
    stop_grad_norm: float = 1.0
    stop_loss: float = 1e-6
    sentence_stem_mult: int = 3
    syllabe_max_seq_len: int = 1024
    task_keyword: str = None
    train_parsed_corpus: str = None
    train_unparsed_corpus: str = None
    use_cross_positional_attn_bias: bool = True
    use_ddp: bool = True
    use_mtl_optimizer: bool = False
    use_iterable_dataset: bool = True
    validation_steps: int = 10_000
    validation_offset: int = 0
    warmup_iter: int = 2000
    wd: float = 0.01
    world_size: int = 1
    xlmr: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(explicit_bool=True,*args, **kwargs)

def morpho_small_args(list_args=None) -> FlexArguments:
    if list_args is not None:
        args = FlexArguments().parse_args(list_args)
    else:
        args = FlexArguments().parse_args()
    args.world_size = args.gpus
    args.morpho_max_token_len = 24
    args.morpho_rel_pos_bins = 24
    args.morpho_max_rel_pos = 24
    args.main_sequence_encoder_max_seq_len = 512
    args.main_sequence_encoder_rel_pos_bins = 256
    args.main_sequence_encoder_max_rel_pos = 256
    # Architectural dimensions: --------
    args.morpho_dim_hidden = 192
    args.stem_dim_hidden = 192
    args.sentence_stem_mult = 1
    # main_hidden_dim = 192*3 + 192*1 = 768
    args.morpho_dim_ffn = 512
    args.main_sequence_encoder_dim_ffn = 2048
    args.morpho_num_heads = 4
    args.main_sequence_encoder_num_heads = 8
    # ----------------------------------
    args.morpho_num_layers = 4
    args.main_sequence_encoder_num_layers = 6
    # ----------------------------------
    return args

def morpho_base_args(list_args=None) -> FlexArguments:
    if list_args is not None:
        args = FlexArguments().parse_args(list_args)
    else:
        args = FlexArguments().parse_args()
    args.world_size = args.gpus
    args.morpho_max_token_len = 24
    args.morpho_rel_pos_bins = 24
    args.morpho_max_rel_pos = 24
    args.main_sequence_encoder_max_seq_len = 512
    args.main_sequence_encoder_rel_pos_bins = 256
    args.main_sequence_encoder_max_rel_pos = 256
    # Architectural dimensions: --------
    args.morpho_dim_hidden = 192
    args.stem_dim_hidden = 192
    args.sentence_stem_mult = 1
    # main_hidden_dim = 192*3 + 192*1 = 768
    args.morpho_dim_ffn = 512
    args.main_sequence_encoder_dim_ffn = 2048
    args.morpho_num_heads = 4
    args.main_sequence_encoder_num_heads = 8
    # ----------------------------------
    args.morpho_num_layers = 4
    args.main_sequence_encoder_num_layers = 12
    # ----------------------------------
    return args

def morpho_large_args(list_args=None) -> FlexArguments:
    if list_args is not None:
        args = FlexArguments().parse_args(list_args)
    else:
        args = FlexArguments().parse_args()
    args.world_size = args.gpus
    args.morpho_max_token_len = 24
    args.morpho_rel_pos_bins = 24
    args.morpho_max_rel_pos = 24
    args.main_sequence_encoder_max_seq_len = 512
    args.main_sequence_encoder_rel_pos_bins = 256
    args.main_sequence_encoder_max_rel_pos = 256
    # Architectural dimensions: --------
    args.morpho_dim_hidden = 384
    args.stem_dim_hidden = 384
    args.sentence_stem_mult = 1
    # main_hidden_dim = 384*3 + 384*1 = 1536
    args.morpho_dim_ffn = 1024
    args.main_sequence_encoder_dim_ffn = 4096
    args.morpho_num_heads = 4
    args.main_sequence_encoder_num_heads = 16
    # ----------------------------------
    args.morpho_num_layers = 6
    args.main_sequence_encoder_num_layers = 11
    # ----------------------------------
    return args
