# DeepKIN-AgAI

Kinyarwanda Deep Learning Models and Tools for IVR-based Agricultural Chatbot

## TBD: Full Documentation and Tutorial

## Getting started

Some of the models depend on a Morphological analyzer/generator for Kinyarwanda.
In order to use the toolkit, you need to go through the following steps:
1. Download and install the morphological analyzer/generator
2. Get a free license for the morphological analyzer/generator
2. Clone and install DeepKIN toolkit and its dependencies
3. Run experimental code

## Downloading and installing Kinyarwanda morphological analyzer/generator

The current release (version 0.1.0) of the morphological analyzer is only compatible with Linux x84_64 platform.
In order to run the morphological analyzer/generator, the system must meet the following minimum requirements:
- Operating system: Linux x86_64 (amd64), we have tested it with Ubuntu 64-bit, 18.04, 20.04 and 22.04 versions
- Drive space: 45 GB, (64 GB recommended)
- System memory (RAM): 40 GB (64 GB recommended)

The morphological analyzer is available for download from the following Google Drive link:
https://drive.google.com/file/d/1Kt9YXhLw_UVMCefcRGworyHUdh-tyQRj/view
With the link, you can download it directly to your machine.
In order to download it from a terminal (i.e. on a remote server), you need to use an OAuth token as in the following steps:
1. Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
2. In the Select the Scope box, paste https://www.googleapis.com/auth/drive.readonly
3. Click Authorize APIs and then Exchange authorization code for tokens
4. Copy the Access token
5. Run the following command in terminal, where ACCESS_TOKEN is the access token copied above:
```
curl -H "Authorization: Bearer ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1Kt9YXhLw_UVMCefcRGworyHUdh-tyQRj?alt=media -o KINLP.tar.gz
```

The morphological analyzer/generator package installation directory needs to be refered as `KINLP_HOME` environmental variable or be installed in `/opt/KINLP` path as follow:
```
gunzip -c KINLP.tar.gz | tar x
rm KINLP.tar.gz
sudo ln -s </path/to/downloaded/KINLP> /opt/KINLP
```

You can configure the following environmental variables at the shell startup for the morphological analyze to be available; e.g:
```
UBUNTU_VERSION=$(lsb_release -r --short)
export KINLP_HOME=/opt/KINLP
export PATH=$PATH:$KINLP_HOME:$KINLP_HOME/bin/$UBUNTU_VERSION
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION
```

Before using the morphological analyzer/generator, the following packages are needed on a Linux: `gcc g++ make cmake libomp-dev libgsl-dev gsl-bin libgsl-dbg python3-pybind11 pybind11-dev unicode libicu-dev`.
You can install them on Ubuntu Linux as follows:
```
sudo apt update
sudo apt install -y nano gcc g++ make cmake ninja-build libomp-dev libgsl-dev gsl-bin libgsl-dbg python3-pybind11 pybind11-dev unicode libicu-dev
```
This has only been tested on Ubuntu versions 18.04, 20.04 and 22.04.

Currently, the morphological analyzer/generator can be run in three different use-cases:
#### 1. To check the license validity:
```
bash $KINLP_HOME/morphokin.sh license </path/to/LICENSE_FILE.dat>
```
#### 2. Sentence analysis via interactive shell:
```
bash $KINLP_HOME/morphokin.sh snt </path/to/LICENSE_FILE.dat>
```
To exit the snt shell, enter either one of `exit, EXIT, e, E, quit, QUIT, q, Q` commands on the shell.

#### 3. To run morphological analysis and synthesis server on a unix socket; e.g. for Python API calls:
```
nohup bash $KINLP_HOME/morphokin.sh rms </path/to/LICENSE_FILE.dat> &>> rms.log &
```

## Getting the free license for the morphological analyzer

The free license is only allowed for academic and non-commercial use of the morphological analyzer/generator. 
Refer to the [Terms and Conditions](https://docs.google.com/document/d/17elFQbP4lR8uSufsU1NymObH_t2z0dy7sq78fbIMU7M/view) for the morphological analyzer/generator.

To request a free license, fill in the registration form available at::
https://morphokin.kinlp.com/license/request
The form requests basic information about the user and their organization.
Once submitted, you will be required to verify the email address by clicking on the confirmation link sent to your email address.

Once approved, a free license file will be sent to your email address.
Use the license with morphological analyzer as suggested in the previous section, where `</path/to/LICENSE_FILE.dat>` is the path to the license file on your system.

## Cloning and installing DeepKIN toolkit and its dependencies

**DeepKIN** toolkit implements deep learning models for Kinyarwanda NLP tasks such as text classification, language modeling, named entity recognition and others. The toolkit depends on the Kinyarwanda morphological analyzer/generator at its core.
It also depends on PyTorch and other python packages.
We recommend to use [Anaconda](https://www.anaconda.com/download) distribution with virtual environment as we have tested with it.
You will also need a CUDA-enabled GPU with at least 12 GB or GPU RAM. Nvidia GPUs with Tensor Cores are most recommended.

Go through the following commands for the installation of the toolkit and its dependencies:
```shell

# 1. Install Python

sudo apt install python3.12-venv

# 2. Create virtual environment

python -m venv mycustomenv

source ~/mycustomenv/bin/activate

# 3. Install PyTorch

pip3 install torch torchvision

pip install torchcodec
pip install torchaudio

# 4. Install various dependencies

pip install Cython
pip install distro
pip install progressbar2
pip install seqeval
pip install youtokentome
pip install sacremoses
pip install fastBPE
pip install packaging
pip install mutagen
pip install pandas
pip install acoustics
pip install typed-argument-parser
pip install demoji
pip install librosa
pip install pyinflect
pip install webcolors
pip install typo
pip install colorama
pip install minineedle
pip install ragatouille
pip install transformers tokenizers
pip install g2pk2
pip install cn2an
pip install inflect
pip install eng_to_ipa
pip install opencc
pip install unidecode
pip install phonemizer
pip install pyopenjtalk
pip install ko_pron
pip install pypinyin
pip install jieba
pip install indic_transliteration
pip install num_thai
pip install tensorboardX
pip install torchmetrics
pip install flash-attn --no-build-isolation
pip install causal-conv1d>=1.4.0
pip install mamba-ssm[causal-conv1d]
pip install Cython
pip install distro
pip install progressbar2
pip install seqeval
pip install youtokentome
pip install sacremoses
pip install fastBPE
pip install packaging
pip install mutagen
pip install pandas
pip install acoustics
pip install typed-argument-parser
pip install demoji
pip install librosa
pip install pyinflect
pip install webcolors
pip install typo
pip install colorama
pip install minineedle
pip install ragatouille
pip install transformers tokenizers
pip install g2pk2
pip install cn2an
pip install inflect
pip install eng_to_ipa
pip install opencc
pip install unidecode
pip install phonemizer
pip install pyopenjtalk
pip install ko_pron
pip install pypinyin
pip install jieba
pip install indic_transliteration
pip install num_thai
pip install tensorboardX
pip install torchmetrics
pip install Cython
pip install fastBPE
sudo apt-get install python3-dev

pip install packaging

# 5. Install flash-attention and mamba-ssm packages

pip install flash-attn --no-build-isolation
pip install psutil
pip install flash-attn --no-build-isolation
pip install causal-conv1d>=1.4.0
pip install mamba-ssm[causal-conv1d]

# 6. Build and install Nvidia apex

git clone https://github.com/NVIDIA/apex
cd apex/
NVCC_APPEND_FLAGS="--threads 8" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# 7. Install fairseq

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --no-deps -e ./

# 8. Install DeepKIN-AgAI package
cd DeepKIN-AgAI

pip install --no-deps -e ./

```

## Citations


```
@inproceedings{nzeyimana-2020-morphological,
    title = "Morphological disambiguation from stemming data",
    author = "Nzeyimana, Antoine",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.409",
    doi = "10.18653/v1/2020.coling-main.409",
    pages = "4649--4660",
}

@inproceedings{nzeyimana-niyongabo-rubungo-2022-kinyabert,
    title = "{K}inya{BERT}: a Morphology-aware {K}inyarwanda Language Model",
    author = "Nzeyimana, Antoine  and
      Niyongabo Rubungo, Andre",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.367",
    doi = "10.18653/v1/2022.acl-long.367",
    pages = "5347--5363",
}

@article{nzeyimana2025kinyacolbert,
  title={KinyaColBERT: A Lexically Grounded Retrieval Model for Low-Resource Retrieval-Augmented Generation},
  author={Nzeyimana, Antoine and Rubungo, Andre Niyongabo},
  journal={arXiv preprint arXiv:2507.03241},
  year={2025}
}

```

