#!/bin/bash

# source activate spg

DATA= # your directory of dataset
TRAINER=SPG_CGAN

DATASET=$1
CFG=$2  # config file
BACKBONE=$3 # backbone name

# bash scripts/test_all.sh pacs spg RN50

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

for DOMAIN in "${ALL_DOMAIN[@]}"
do
  DIR=outputs_test/${TRAINER}/${CFG}/${BACKBONE//\//}/${DATASET}/${DOMAIN}

  if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}, so skip this job"
  else
    echo "Run this job and save the output to ${DIR}"

    MODEL_DIR=test_models/${DATASET}/${BACKBONE//\//}/${DOMAIN}

    python train.py \
      --backbone ${BACKBONE} \
      --target-domains ${DOMAIN} \
      --root ${DATA} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --model-dir ${MODEL_DIR} \
      --eval-only
  fi
done
