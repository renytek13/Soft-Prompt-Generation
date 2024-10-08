#!/bin/bash

# source activate spg

DATA=   # ******* your data path ******* 
TRAINER=SPG_CGAN

DATASET=$1
CFG=$2          # config file
BACKBONE=$3     # backbone name
GPU=$4


# bash scripts/spg_cgan/spg_cgan.sh pacs spg RN50 0
# bash scripts/spg_cgan/spg_cgan.sh vlcs spg RN50 0
# bash scripts/spg_cgan/spg_cgan.sh office_home spg RN50 1
# bash scripts/spg_cgan/spg_cgan.sh terra_incognita spg RN50 1
# bash scripts/spg_cgan/spg_cgan.sh domainnet spg RN50 2

# bash scripts/spg_cgan/spg_cgan.sh pacs spg ViT-B/16 0
# bash scripts/spg_cgan/spg_cgan.sh vlcs spg ViT-B/16 0
# bash scripts/spg_cgan/spg_cgan.sh office_home spg ViT-B/16 1
# bash scripts/spg_cgan/spg_cgan.sh terra_incognita spg ViT-B/16 1
# bash scripts/spg_cgan/spg_cgan.sh domainnet spg ViT-B/16 2


if [ "$DATASET" = "pacs" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 's')
elif [ "$DATASET" = "vlcs" ]; then
  ALL_DOMAIN=('c' 'l' 'p' 's')
elif [ "$DATASET" = "office_home" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 'r')
elif [ "$DATASET" = "terra_incognita" ]; then
  ALL_DOMAIN=('1' '2' '3' '4')
elif [ "$DATASET" = "domainnet" ]; then
  ALL_DOMAIN=('c' 'i' 'p' 'q' 'r' 's')
fi

for SEED in 1 2 3
do
  for DOMAIN in "${ALL_DOMAIN[@]}"
  do
    DIR=outputs/SPG/multi_DG/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}

    if [ -d "$DIR" ]; then
      echo "Results are available in ${DIR}, so skip this job"
    else
      echo "Run this job and save the output to ${DIR}"
      
      python train.py \
        --backbone ${BACKBONE} \
        --target-domains ${DOMAIN} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/multi_source/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --seed ${SEED} \
        --gpu ${GPU}      
    fi
  done
done
