#!/bin/bash

DATA=/opt/data/private/code/promptDG/SPG_Baseline/datasets # your directory of dataset
TRAINER=SPG_CGAN

DATASET=$1
CFG=$2  # config file
BACKBONE=$3 # backbone name
DOMAIN=a

# bash scripts/test.sh pacs spg RN50

DIR=test_models/pacs/RN50/a

python train.py \
    --backbone ${BACKBONE} \
    --target-domains ${DOMAIN} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only
