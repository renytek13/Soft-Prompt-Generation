# Soft Prompt Generation with CGAN
**We propose a new prompt learning paradigm Soft Prompt Generation (SPG), which offers a straightforward yet effective solution for Domain Generalization. SPG is designed to exclusively harness a generative model for prompt generation, leveraging the model’s inherent capability to encode domain and content knowledge directly into the generated prompts.**

Please follow instrcutions below to reproduce the results. 

**We only give an example on PACS dataset in this code space.**

<hr />

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
cd SoftPromptGenerationWithCGAN

# Install requirements
pip install -r requirements.txt
```


<hr />


## Data preparation
**Please download the datasets `PACS`, `VLCS`, `office_home`, `terra_incognita` and `domainnet`.**

Follow [DATASETS.md](DATASETS.md) to install the datasets.

<hr />


## Run Directly

We provide the running scripts in `scripts`, which allow you to reproduce the results on the paper. 
Make sure you **modify the path in `$DATA`**!

If you wanna use our produced data splits and prompt labels. Please follow the instructions as follows:

<!-- 1. copy the data splits files of [PACS](datasets/PACS/) to the downloaded root directory of PACS datasets. -->
1. Ensure that you have downloaded the [PACS](./DATASETS.md#pacs) dataset file as well as our segmentation file by following the steps above and our pre-trained prompt label directory [prompt_labels](prompt_labels/).

2. Run the bash file as follows.

### Training 
```bash
# Example: trains on PACS dataset with ResNet50 as backbone.
bash scripts/spg_cgan/spg_cgan.sh pacs spg RN50
```
### Evaluation
```bash
# An expample of test.
bash scripts/test_all.sh pacs spg RN50
```


<hr />


## Run Two-stage Training Paradigm

If you wanna use the data splits and prompt labels produced by yourself. Please follow the instructions as follows:

**First, when downloading the dataset, ignore the last step of downloading the split folder when following the [Data preparation](#data-preparation) steps above and just build the images directory as required.**

### Stage I -- Produce the domain prompt labels

All you need is `SPG/scripts/spg_coop/spg_coop.sh`, which contains two input arguments.
Make sure you **modify the path in `$DATA`**!

### Runing
```bash
# Example: trains on PACS dataset with ResNet50 as backbone.
bash scripts/spg_coop/spg_coop.sh pacs RN50
```


### Stage II -- CGAN Pretraining

Please refer to the section of **[Run Directly](#run-directly)**


<hr />


## Results

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
|   |   |   |   |–– c/
|   |   |   |   |   |–– seed_1/
|   |   |   |   |–– p/
|   |   |   |   |   |–– seed_1/
|   |   |   |   |–– s/
|   |   |   |   |   |–– seed_1/
```

To observe the resultant change curve, you can run
`tensorflow --logdir=outputs/SPG/SPG_CGAN/pacs/spg/RN50/a/seed_1`.

<!--
Below we provide a set of results from a direct evaluation using our pre-trained model [test_models](#test_models) on PACS dataset with ResNet50 as backbone.

**How to Run**
Run `bash scripts/test_all.sh pacs spg RN50`

**View Results**
See the results in file [outputs_test](outputs_test/).
```bash
art_painting: accuracy: 93.5%. error: 6.5%.
cartoon: accuracy: 93.1%. error: 6.9%.
photo: accuracy: 99.1%. error: 0.9%.
sketch: accuracy: 85.1%. error: 14.9%.
```
-->

## Single-source Domain Generation

See `scripts/spg_cgan/single.sh`.

## Cross-dataset Domain Generation

See `scripts/spg_cgan/cross.sh`.
