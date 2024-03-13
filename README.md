
## Installation 
For installation and other package requirements, please follow the instructions as follows. 
This codebase is tested on Ubuntu 20.04 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment.
```bash
# Create a conda environment
conda create -y -n spg python=3.8

# Activate the environment
conda activate spg

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone SPG code repository and install requirements.
```bash
# Clone SPG code base
# cd SoftPromptGenerationWithCGAN 

# Install requirements
pip install -r requirements.txt
```


## How to Run

We provide the running scripts in `scripts`, which allow you to reproduce the results on the paper.

Make sure you change the path in `DATA` and run the commands under the main directory `SPG/`.

### SPG-CoOp

All you need is `SPG/scripts/spg_coop/spg_coop.sh`, which contains two input arguments.

`DATASET` takes as input a dataset name, including `PACS`, `VLCS`, `office_home`, `terra_incognita` and `domainnet`. The valid names are the files' names in `SPG/configs/datasets/`.

`BACKBONE` means basic backbone name, including `RN50` and `ViT-B/16`.

### SPG-CGAN

All you need is `SPG/scripts/spg_cgan/spg_cgan.sh`, which contains three input arguments.

`DATASET` takes as input a dataset name, including `PACS`, `VLCS`, `office_home`, `terra_incognita` and `domainnet`. The valid names are the files' names in `SPG/configs/datasets/`.

`CFG` means which config file to use (see `SPG/configs/trainers/SPG_CGAN/`).

`BACKBONE` means basic backbone name, including `RN50` and `ViT-B/16`.

Below we provide examples on how to run SPG on PACS with RN50 backbone.

**SPG_CoOp (Get the optimal prompt for each domain)**:
`bash scripts/spg_coop/spg_coop.sh pacs RN50`

**SPG_CGAN**:
`bash scripts/spg_cgan/spg_cgan.sh pacs spg RN50`


After the experiments are finished, you can obtain the average results looking into the log files. Say the structure of `outputs/` is

```
outputs
|–– SPG/
|   |–– SPG_CoOp/
|   |   |–– pacs/
|   |   |   |–– seed_1/b32_ep50/RN50/
|   |   |   |   |–– a/
|   |   |   |   |–– c/
|   |   |   |   |–– p/
|   |   |   |   |–– s/
|   |–– SPG_CGAN/
|   |   |–– pacs/
|   |   |   |–– spg/RN50/
|   |   |   |   |–– a/
|   |   |   |   |   |–– seed_1/
|   |   |   |   |   |–– seed_2/
|   |   |   |   |   |–– seed_3/
|   |   |   |   |–– c/
|   |   |   |   |   |–– seed_1/
|   |   |   |   |   |–– seed_2/
|   |   |   |   |   |–– seed_3/
|   |   |   |   |–– p/
|   |   |   |   |   |–– seed_1/
|   |   |   |   |   |–– seed_2/
|   |   |   |   |   |–– seed_3/
|   |   |   |   |–– s/
|   |   |   |   |   |–– seed_1/
|   |   |   |   |   |–– seed_2/
|   |   |   |   |   |–– seed_3/
```

To observe the resultant change curve, you can run
`tensorflow --logdir=outputs/SPG/SPG_CGAN/pacs/spg/RN50/a/seed_1`.

