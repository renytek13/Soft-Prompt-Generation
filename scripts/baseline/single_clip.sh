#!/bin/bash

source activate spg

# custom config
DATA=   # ******* your data path *******/
CFG=b32_ep10

BACKBONE=$1 # backbone name
TRAINER=$2
GPU=$3

# bash scripts/baseline/single_clip.sh RN50 CLIP_ZS 0
# bash scripts/baseline/single_clip.sh RN50 CLIP_LR 0

# bash scripts/baseline/single_clip.sh ViT-B/16 CLIP_ZS 0
# bash scripts/baseline/single_clip.sh ViT-B/16 CLIP_LR 0

DATASET=single_pacs
for SEED in 1
do
  for WARMUP in 1
  do
    for DOMAIN in 'a' 'c' 'p' 's'
    do
      DIR=outputs_baseline/single-dg/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"
        
        python train_baseline.py \
          --gpu ${GPU} \
          --backbone ${BACKBONE} \
          --source-domains ${DOMAIN} \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/single_source/${DATASET}.yaml \
          --config-file configs/trainers/BASELINE/${CFG}.yaml \
          --output-dir ${DIR} \
          --seed ${SEED} \
          --warmup_epoch ${WARMUP} \
          --eval-only
      fi
    done
  done
done

DATASET=single_vlcs
for SEED in 1
do
  for WARMUP in 1
  do
    for DOMAIN in 'c' 'l' 'p' 's'
    do
      DIR=outputs_baseline/single-dg/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"
        
        python train_baseline.py \
          --gpu ${GPU} \
          --backbone ${BACKBONE} \
          --source-domains ${DOMAIN} \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/single_source/${DATASET}.yaml \
          --config-file configs/trainers/BASELINE/${CFG}.yaml \
          --output-dir ${DIR} \
          --seed ${SEED} \
          --warmup_epoch ${WARMUP} \
          --eval-only
      fi
    done
  done
done

DATASET=single_office_home
for SEED in 1
do
  for WARMUP in 1
  do
    for DOMAIN in 'a' 'c' 'p' 'r'
    do
      DIR=outputs_baseline/single-dg/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"
        
        python train_baseline.py \
          --gpu ${GPU} \
          --backbone ${BACKBONE} \
          --source-domains ${DOMAIN} \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/single_source/${DATASET}.yaml \
          --config-file configs/trainers/BASELINE/${CFG}.yaml \
          --output-dir ${DIR} \
          --seed ${SEED} \
          --warmup_epoch ${WARMUP} \
          --eval-only
      fi
    done
  done
done

DATASET=single_terra_incognita
for SEED in 1
do
  for WARMUP in 1
  do
    for DOMAIN in '1' '2' '3' '4'
    do
      DIR=outputs_baseline/single-dg/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"
        
        python train_baseline.py \
          --gpu ${GPU} \
          --backbone ${BACKBONE} \
          --source-domains ${DOMAIN} \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/single_source/${DATASET}.yaml \
          --config-file configs/trainers/BASELINE/${CFG}.yaml \
          --output-dir ${DIR} \
          --seed ${SEED} \
          --warmup_epoch ${WARMUP} \
          --eval-only
      fi
    done
  done
done

DATASET=single_domainnet
for SEED in 1
do
  for WARMUP in 1
  do
    for DOMAIN in 'c' 'i' 'p' 'q' 'r' 's'
    do
      DIR=outputs_baseline/single-dg/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"
        
        python train_baseline.py \
          --gpu ${GPU} \
          --backbone ${BACKBONE} \
          --source-domains ${DOMAIN} \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/single_source/${DATASET}.yaml \
          --config-file configs/trainers/BASELINE/${CFG}.yaml \
          --output-dir ${DIR} \
          --seed ${SEED} \
          --warmup_epoch ${WARMUP} \
          --eval-only
      fi
    done
  done
done
