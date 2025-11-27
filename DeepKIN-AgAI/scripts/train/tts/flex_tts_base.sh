# e.g. Train on 2 RTX 4090 GPUs

python3 deepkin/train/flex_trainer.py  \
    --model_variant="flex_tts:base" \
    --gpus=2 \
    --enable_amp=False \
    --use_ddp=False \
    --use_mtl_optimizer=False \
    --num_losses=8 \
    --warmup_iter=32000 \
    --peak_lr=2e-4  \
    --num_iters=2000000  \
    --batch_size=16 \
    --accumulation_steps=2  \
    --dataloader_num_workers=8  \
    --dataloader_persistent_workers=True  \
    --dataloader_pin_memory=True  \
    --use_iterable_dataset=False \
    --train_log_steps=10  \
    --checkpoint_steps=1000 \
    --load_saved_model=True  \
    --tts_data_dir="kinya-ag-tts" \
    --tts_train_data_file="tts_train_data.psv" \
    --model_save_path="kinya_flex_tts_base.pt"
