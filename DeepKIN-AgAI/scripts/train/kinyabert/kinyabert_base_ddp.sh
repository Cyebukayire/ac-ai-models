# e.g. Train on 10 RTX 4090 GPUs
#  --gpus=10 \
#  --batch_size=24  \
#  --accumulation_steps=340  \

python3 deepkin/train/flex_trainer.py  \
    --model_variant="kinyabert:base" \
    --gpus=10 \
    --batch_size=24  \
    --accumulation_steps=340  \
    --dataloader_num_workers=2  \
    --dataloader_persistent_workers=True  \
    --dataloader_pin_memory=True  \
    --use_ddp=True \
    --use_mtl_optimizer=False \
    --warmup_iter=2400 \
    --peak_lr=6e-4  \
    --lr_decay_style="linear" \
    --num_iters=50000  \
    --train_parsed_corpus="morpho_parsed_corpus.txt"  \
    --number_of_load_batches=4096000  \
    --dataset_max_seq_len=512  \
    --use_iterable_dataset=True  \
    --max_mlm_documents=5  \
    --max_dataset_chunk_size=6000000 \
    --train_log_steps=1  \
    --checkpoint_steps=100 \
    --validation_steps=1000 \
    --load_saved_model=True  \
    --model_save_path="kinyabert_base_ddp.pt"
