#!/bin/bash

# source activate bsh_prompt

DATA= # your directory of dataset
TRAINER=SPG_CoOp
CFG=b32_ep50    # config file
SEED=1

DATASET=$1
BACKBONE=$2     # backbone name


# bash scripts/spg_coop/spg_coop.sh pacs RN50
# bash scripts/spg_coop/spg_coop.sh vlcs RN50
# bash scripts/spg_coop/spg_coop.sh office_home RN50
# bash scripts/spg_coop/spg_coop.sh terra_incognita RN50
# bash scripts/spg_coop/spg_coop.sh domainnet RN50

# bash scripts/spg_coop/spg_coop.sh pacs ViT-B/16
# bash scripts/spg_coop/spg_coop.sh vlcs ViT-B/16
# bash scripts/spg_coop/spg_coop.sh office_home ViT-B/16
# bash scripts/spg_coop/spg_coop.sh terra_incognita ViT-B/16
# bash scripts/spg_coop/spg_coop.sh domainnet ViT-B/16

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
  DIR=outputs/SPG/${TRAINER}/${DATASET}/seed_${SEED}/${CFG}/${BACKBONE//\//}/${DOMAIN}

  if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}, so skip this job"
  else
    echo "Run this job and save the output to ${DIR}"
    
    python train.py \
      --backbone ${BACKBONE} \
      --target-domains ${DOMAIN} \
      --root ${DATA} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}_coop.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --seed ${SEED}      
  fi
done