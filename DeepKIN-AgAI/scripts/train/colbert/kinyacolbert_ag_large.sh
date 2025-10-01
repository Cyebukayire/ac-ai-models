# Deploy on RTX 4090
#    --batch_size=6  \
#    --accumulation_steps=20  \
#    --use_ddp=True \

# Deploy on DGXH200
#    --batch_size=32  \
#    --accumulation_steps=4  \
#    --use_ddp=False \

python3 deepkin/train/flex_trainer.py  \
    --model_variant="kinyabert_qaret:large" \
    --colbert_embedding_dim=512 \
    --gpus=1 \
    --batch_size=32  \
    --accumulation_steps=4  \
    --dataloader_num_workers=4  \
    --dataloader_persistent_workers=True  \
    --dataloader_pin_memory=True  \
    --use_ddp=False \
    --use_mtl_optimizer=False \
    --warmup_iter=2000 \
    --peak_lr=1e-5  \
    --lr_decay_style="cosine" \
    --num_iters=152630  \
    --dataset_max_seq_len=512  \
    --use_iterable_dataset=False  \
    --train_log_steps=1  \
    --checkpoint_steps=1000 \
    --pretrained_bert_model_file="KINLP/data/kinyabert_large_ddp_2024-11-27.pt_50K.pt" \
    --qa_train_query_id="KINLP/data/AgQA_2025-05-22/AgQA_query_id_2025-05-22.txt" \
    --qa_train_query_text="KINLP/data/AgQA_2025-05-22/parsed_AgQA_query_text_2025-05-22.txt" \
    --qa_train_passage_id="KINLP/data/AgQA_2025-05-22/AgQA_passage_id_2025-05-22.txt" \
    --qa_train_passage_text="KINLP/data/AgQA_2025-05-22/parsed_AgQA_passage_text_2025-05-22.txt" \
    --qa_train_qpn_triples="KINLP/data/AgQA_2025-05-22/AgQA_qpntriplets_all_2025-05-22.tsv" \
    --qa_dev_query_id="KINLP/data/AgQA_2025-05-22/AgQA_query_id_2025-05-22.txt" \
    --qa_dev_query_text="KINLP/data/AgQA_2025-05-22/parsed_AgQA_query_text_2025-05-22.txt" \
    --qa_dev_passage_id="KINLP/data/AgQA_2025-05-22/AgQA_passage_id_2025-05-22.txt" \
    --qa_dev_passage_text="KINLP/data/AgQA_2025-05-22/parsed_AgQA_passage_text_2025-05-22.txt" \
    --qa_dev_qpn_triples="KINLP/data/AgQA_2025-05-22/AgQA_qpntriplets_dev_2025-05-22.tsv" \
    --load_saved_model=True  \
    --model_save_path="KINLP/data/kinyabert_qaret_large.pt"
