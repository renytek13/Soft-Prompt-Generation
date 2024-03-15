#!/bin/bash

# source activate spg

DATA= # your directory of dataset
TRAINER=SPG_CGAN

CFG=$1          # config file
BACKBONE=$2     # backbone name

# bash scripts/spg_cgan/cross.sh spg RN50
# bash scripts/spg_cgan/cross.sh spg ViT-B/16

for SEED in 1
do
  for DATASET in 'd'
  do
    DIR=outputs/SPG/cross/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/seed_${SEED}

    if [ -d "$DIR" ]; then
      echo "Results are available in ${DIR}, so skip this job"
    else
      echo "Run this job and save the output to ${DIR}"
      
      python train.py \
        --backbone ${BACKBONE} \
        --source-datasets ${DATASET} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/cross_dataset/cross.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --seed ${SEED}      
    fi
  done

  for DATASET in 'o' 'p' 't' 'v'
  do
    DIR=outputs/SPG/cross/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/seed_${SEED}

    if [ -d "$DIR" ]; then
      echo "Results are available in ${DIR}, so skip this job"
    else
      echo "Run this job and save the output to ${DIR}"

      MODEL_DIR=outputs/SPG/cross/${TRAINER}/d/${CFG}/${BACKBONE//\//}/seed_${SEED}
      
      python train.py \
        --backbone ${BACKBONE} \
        --source-datasets ${DATASET} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/cross_dataset/cross_test.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --seed ${SEED} \
        --eval-only
    fi
  done
done
