# DeepKIN-AgAI

Kinyarwanda Deep Learning Models and Tools for IVR/RAG-based Agricultural Chatbot

The IVR chatbot is powered by Retrieval-Augmented Generation (RAG), and is designed specifically to serve Kinyarwanda speakers with high accuracy and accessibility.
By combining the retrieval model with speech processing models, the system seamlessly understands spoken Kinyarwanda, retrieves the most relevant answers from a domain-specific knowledge base, and delivers natural, clear responses in real time.

## 1. Installing DeepKIN-AgAI

### System Requirements

- x86_64 CPU
- 64 GB of System RAM
- 160 GB of Disk Storage
- Nvidia GPU
- Nvidia Drivers
- Nvidia CUDA Toolkit
- Docker
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- Python 3.10+
- [PyTorch](https://pytorch.org/) 2.0+
- TorchAudio
- [MorphoKIN](MorphoKIN.md) (optional) for modeling Kinyarwanda morphology (required by KinyaBERT* models)

This tutorial was tested on [AWS EC2](https://aws.amazon.com/ec2/) *g6e.4xlarge* instance (160 GB disk storage) with "Amazon/Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04) 20251101" AMI.

### 1.1. Installation Steps

You can follow the steps below to get the package and its requirements installed:

1. Ensure system requirements are met, including having [PyTorch](https://pytorch.org/) and compatible TorchAudio installed. It's better to use a Python virtual environment.
```shell
# 1. Example of Python installation and virtual environment creation on Ubuntu:

sudo apt install python3.12-full python-is-python3 python3.12-venv
python -m venv flex
source ~/flex/bin/activate

# 2. Examples of PyTorch and torchaudio installation:

# You need to chose the right cuda version to use with PyTorch based on the version of your installed Cuda Toolkit
# e.g.:$ nvcc --version

# For Cuda 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# or

# For Cuda 12.9
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

```

2. Clone DeepKIN-AgAI from GitHub
```shell

git clone git@github.com:c4ir-rw/ac-ai-models.git

cd ac-ai-models/DeepKIN-AgAI/

```

3. Install dependencies
```shell

pip install -r requirements.txt

```

4. Install these additional packages to improve performance: *flash-attn*, *mamba-ssm*, *causal-conv1d* and *[Nvidia Apex](https://github.com/NVIDIA/apex)*

```shell

MAX_JOBS=8 pip install flash-attn --no-build-isolation

MAX_JOBS=8 pip install mamba-ssm[causal-conv1d] --no-build-isolation

git clone https://github.com/NVIDIA/apex
cd apex/

NVCC_APPEND_FLAGS="--threads 8" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

```

5. Install *deepkin* and *monotonic_align* packages inside DeepKIN-AgAI
```shell

cd DeepKIN-AgAI/

pip install -e ./


cd DeepKIN-AgAI/monotonic_align/

python setup.py build_ext --inplace

pip install ./ --no-build-isolation

```

## 2. Using DeepKIN-AgAI

### 2.1 Training

#### 2.1.1. Training a KinyaBERT model from scratch

1. Prepare a Kinyarwanda text corpus file containing multiple documents by placing one sentence per line and an empty line between documents. Use the provided [sample_corpus.txt](scripts/scratch/sample_corpus.txt) file as a reference.
2. Parse the text corpus with MorphoKIN. Adjust the number of parallel threads based on your CPU core count using *--num_threads* option.
```shell

# Ensure you have the free license file, e.g.:
# cp KINLP_LICENSE_FILE.dat /home/ubuntu/MORPHODATA/licenses/

# Have the corpus text file, e.g.:
mkdir -p /home/ubuntu/DATA
cp DeepKIN-AgAI/scripts/scratch/sample_corpus.txt /home/ubuntu/DATA/

# Run MorphoKIN with docker (interactive mode)
docker run --rm -v /home/ubuntu/MORPHODATA:/MORPHODATA -v /home/ubuntu/DATA:/DATA --gpus all -it morphokin:latest bash

morphokin --morphokin_working_dir /MORPHODATA \
 --morphokin_config_file /MORPHODATA/data/analysis_config_file.conf \
 --task PTF --num_threads 14 \
 --kinlp_license /MORPHODATA/licenses/KINLP_LICENSE_FILE.dat \
 --ca_roots_pem_file /MORPHODATA/data/roots.pem \
 --input_file /DATA/sample_corpus.txt \
 --output_file /DATA/preparsed_sample_corpus.txt

# Quit docker
exit

```
The sentences in the generated file ("pre-parsed" by MorphoKIN) is not in original order (this is done for faster parallel processing);
you need to re-arrange the sentences by running the provided "post-ptf" python script.
```shell

python DeepKIN-AgAI/scripts/scratch/post_ptf.py /home/ubuntu/DATA/preparsed_sample_corpus.txt  /home/ubuntu/DATA/parsed_sample_corpus.txt

```

3. Run the provided KinyaBERT (e.g. base) training script.
Adjust your batch size and accumulation steps based on your available GPU VRAM to give a global batch size of ~ 8K documents.
The example configuration below (*--batch_size=48 --accumulation_steps=170*) is for a GPU with 48 GB of VRAM. 
The script allow for multi-gpu training (*--gpu=N*) using Distributed Data Parellelism (DDP i.e. *--use_ddp=True*).
When training on multiple GPUs, ensure *--accumulation_steps* is divisible by the number of GPUs.
```shell

# Base architecture: 107M parameter KinyaBERT Model: DeepKIN-AgAI/scripts/train/kinyabert/kinyabert_base_ddp.sh
# Large architecture: 365M parameter KinyaBERT Model: DeepKIN-AgAI/scripts/train/kinyabert/kinyabert_large_ddp.sh

# Example:

python3 DeepKIN-AgAI/deepkin/train/flex_trainer.py  \
    --model_variant="kinyabert:base" \
    --gpus=1 \
    --batch_size=48  \
    --accumulation_steps=170  \
    --dataloader_num_workers=2  \
    --dataloader_persistent_workers=True  \
    --dataloader_pin_memory=True  \
    --use_ddp=True \
    --use_mtl_optimizer=False \
    --warmup_iter=2400 \
    --peak_lr=6e-4  \
    --lr_decay_style="linear" \
    --num_iters=50000  \
    --train_parsed_corpus="/home/ubuntu/DATA/parsed_sample_corpus.txt"  \
    --number_of_load_batches=40960  \
    --dataset_max_seq_len=512  \
    --use_iterable_dataset=True  \
    --max_mlm_documents=5  \
    --max_dataset_chunk_size=60000 \
    --train_log_steps=1  \
    --checkpoint_steps=100 \
    --validation_steps=1000 \
    --load_saved_model=True  \
    --model_save_path="/home/ubuntu/DATA/kinyabert_base_ddp_new.pt"

```

#### 2.1.2. Fine-tuning a pretrained KinyaBERT model into a KinyaColBERT retrieval model
```shell

# TBD

```


#### 2.1.3. Training a RAGatouille-based ColBERT retrieval model (No morphological model needed)
```shell

# TBD

```

#### 2.1.4. Training a Text-to-Speech model from scratch
```shell

# TBD

```

### 2.2 Inference

#### 2.2.1. Evaluating a trained KinyaColBERT retrieval model
```shell

# TBD

```


#### 2.2.2. Evaluating a trained RAGatouille-based ColBERT retrieval model
```shell

# TBD

```

#### 2.2.3. Evaluating a trained Text-to-Speech model
```shell

# TBD

```

#### 2.2.4. Running an API server for KinyaColBERT retrieval
```shell

# TBD

```

#### 2.2.5. Running an API server for Text-to-Speech
```shell

# TBD

```



## References

[1] Antoine Nzeyimana. 2020. Morphological disambiguation from stemming data. In Proceedings of the 28th International Conference on Computational Linguistics, pages 4649–4660, Barcelona, Spain (Online). International Committee on Computational Linguistics.

[2] Antoine Nzeyimana and Andre Niyongabo Rubungo. 2022. KinyaBERT: a Morphology-aware Kinyarwanda Language Model. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5347–5363, Dublin, Ireland. Association for Computational Linguistics.

[3] Antoine Nzeyimana. 2023. KINLP at SemEval-2023 Task 12: Kinyarwanda Tweet Sentiment Analysis. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 718–723, Toronto, Canada. Association for Computational Linguistics.

[4] Antoine Nzeyimana. 2024. Low-resource neural machine translation with morphological modeling. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 182–195, Mexico City, Mexico. Association for Computational Linguistics.

[5] Antoine Nzeyimana, and Andre Niyongabo Rubungo. 2025. KinyaColBERT: A Lexically Grounded Retrieval Model for Low-Resource Retrieval-Augmented Generation. arXiv preprint arXiv:2507.03241.
