# Soft Prompt Generation for Domain Generalization [ECCV 2024]

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.19286)

**We propose a new prompt learning paradigm Soft Prompt Generation (SPG), which offers a straightforward yet effective solution for Domain Generalization.**

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
cd Soft-Prompt-Generation-with-CGAN

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

After the experiments are finished, you can obtain the average results looking into the log files.

To observe the resultant change curve, you can run
`tensorflow --logdir=outputs/SPG/SPG_CGAN/pacs/spg/RN50/a/seed_1`.

## Single-source Domain Generation

See `scripts/spg_cgan/single.sh`.

## Cross-dataset Domain Generation

See `scripts/spg_cgan/cross.sh`.


## Acknowledgements

Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) repository. We thank the authors for releasing their code.

## Citation

If you use our code, please consider citing our work:

```bibtex
@inproceedings{bai2024softpromptgenerationdomain,
  title={Soft Prompt Generation for Domain Generalization},
  author={Shuanghao, Bai and Yuedi, Zhang and Wanqi, Zhou and Zhirong, Luan and Badong, Chen},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## Contact

If you have any questions, please contact at [zyd993@stu.xjtu.edu.cn](mailto:zyd993@stu.xjtu.edu.cn).
