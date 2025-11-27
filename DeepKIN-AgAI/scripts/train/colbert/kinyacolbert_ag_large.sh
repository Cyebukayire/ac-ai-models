# e.g. Train on 10 RTX 4090 GPUs
#    --gpus=10 \
#    --batch_size=6  \
#    --accumulation_steps=20  \
#    --use_ddp=True \

# e.g. Train on 4 H200 GPUs
#    --gpus=4 \
#    --batch_size=32  \
#    --accumulation_steps=4  \
#    --use_ddp=False \

python3 deepkin/train/flex_trainer.py  \
    --model_variant="kinya_colbert:large" \
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
    --pretrained_bert_model_file="kinyabert_large_pretrained.pt" \
    --qa_train_query_id="kinya-ag-retrieval/rw_ag_retrieval_query_id.txt" \
    --qa_train_query_text="kinya-ag-retrieval/parsed_rw_ag_retrieval_query_text.txt" \
    --qa_train_passage_id="kinya-ag-retrieval/rw_ag_retrieval_passage_id.txt" \
    --qa_train_passage_text="kinya-ag-retrieval/parsed_rw_ag_retrieval_passage_text.txt" \
    --qa_train_qpn_triples="kinya-ag-retrieval/rw_ag_retrieval_qpntriplets_all.tsv" \
    --qa_dev_query_id="kinya-ag-retrieval/rw_ag_retrieval_query_id.txt" \
    --qa_dev_query_text="kinya-ag-retrieval/parsed_rw_ag_retrieval_query_text.txt" \
    --qa_dev_passage_id="kinya-ag-retrieval/rw_ag_retrieval_passage_id.txt" \
    --qa_dev_passage_text="kinya-ag-retrieval/parsed_rw_ag_retrieval_passage_text.txt" \
    --qa_dev_qpn_triples="kinya-ag-retrieval/rw_ag_retrieval_qpntriplets_dev.tsv" \
    --load_saved_model=True  \
    --model_save_path="kinya_colbert_large_rw_ag_retrieval.pt"
