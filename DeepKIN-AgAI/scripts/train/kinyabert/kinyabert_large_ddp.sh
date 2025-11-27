## e.g. Train on 8 RTX 4090 GPUs
#  --gpus=8 \
#  --batch_size=8  \
#  --accumulation_steps=1024  \

python3 deepkin/train/flex_trainer.py  \
    --model_variant="kinyabert:large" \
    --gpus=8 \
    --batch_size=8  \
    --accumulation_steps=1024  \
    --dataloader_num_workers=2  \
    --dataloader_persistent_workers=True  \
    --dataloader_pin_memory=True  \
    --use_ddp=True \
    --use_mtl_optimizer=False \
    --warmup_iter=3000 \
    --peak_lr=4e-4  \
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
    --model_save_path="kinyabert_large_ddp.pt"
