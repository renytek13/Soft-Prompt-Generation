#!/bin/bash

source activate spg

# custom config
DATA=   # ******* your data path *******/
CFG=b32_ep10_cross

BACKBONE=$1 # backbone name
TRAINER=$2
GPU=$3


# bash scripts/baseline/cross_clip.sh RN50 CLIP_ZS 0

# bash scripts/baseline/cross_clip.sh ViT-B/16 CLIP_ZS 0


DATASET=cross
for SEED in 1
do
  for WARMUP in 1
  do
    for DATASETS in 'd' #'o' 'p' 't' 'v'
    do
      DIR=outputs_baseline/cross-dg/${TRAINER}/${CFG}/${BACKBONE//\//}/${DATASETS}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"
        
        python train_baseline.py \
          --gpu ${GPU} \
          --backbone ${BACKBONE} \
          --source-datasets ${DATASETS} \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/cross_dataset/${DATASET}.yaml \
          --config-file configs/trainers/BASELINE/${CFG}.yaml \
          --output-dir ${DIR} \
          --seed ${SEED} \
          --warmup_epoch ${WARMUP} \
          --eval-only
      fi
    done
  done
done

DATASET=cross_test
for SEED in 1
do
  for WARMUP in 1
  do
    for DATASETS in 'o' 'p' 't' 'v'
    do
      DIR=outputs_baseline/cross-dg_test/${TRAINER}/${CFG}/${BACKBONE//\//}/${DATASETS}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"
      
        MODEL_DIR=outputs_baseline/cross-dg/${TRAINER}/${CFG}/${BACKBONE//\//}/d/seed_${SEED}/warmup_${WARMUP}
        
        python train_baseline.py \
          --gpu ${GPU} \
          --backbone ${BACKBONE} \
          --source-datasets ${DATASETS} \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/cross_dataset/${DATASET}.yaml \
          --config-file configs/trainers/BASELINE/${CFG}.yaml \
          --output-dir ${DIR} \
          --seed ${SEED} \
          --warmup_epoch ${WARMUP} \
          --eval-only
      fi
    done
  done
done

# for SEED in 1
# do
#   for WARMUP in 1
#   do
#     for DATASETS in 'd' 'p' 't' 'v'
#     do
#       DIR=outputs_baseline/cross-dg_test/${TRAINER}/${CFG}/${BACKBONE//\//}/${DATASETS}/seed_${SEED}/warmup_${WARMUP}

#       if [ -d "$DIR" ]; then
#         echo "Results are available in ${DIR}, so skip this job"
#       else
#         echo "Run this job and save the output to ${DIR}"
      
#         MODEL_DIR=outputs_baseline/cross-dg/${TRAINER}/${CFG}/${BACKBONE//\//}/o/seed_${SEED}/warmup_${WARMUP}
        
#         python train_baseline.py \
#           --gpu ${GPU} \
#           --backbone ${BACKBONE} \
#           --source-datasets ${DATASETS} \
#           --root ${DATA} \
#           --trainer ${TRAINER} \
#           --dataset-config-file configs/datasets/cross_dataset/${DATASET}.yaml \
#           --config-file configs/trainers/BASELINE/${CFG}.yaml \
#           --output-dir ${DIR} \
#           --seed ${SEED} \
#           --warmup_epoch ${WARMUP} \
#           --eval-only
#       fi
#     done
#   done
# done

# for SEED in 1
# do
#   for WARMUP in 1
#   do
#     for DATASETS in 'd' 'o' 't' 'v'
#     do
#       DIR=outputs_baseline/cross-dg_test/${TRAINER}/${CFG}/${BACKBONE//\//}/${DATASETS}/seed_${SEED}/warmup_${WARMUP}

#       if [ -d "$DIR" ]; then
#         echo "Results are available in ${DIR}, so skip this job"
#       else
#         echo "Run this job and save the output to ${DIR}"
      
#         MODEL_DIR=outputs_baseline/cross-dg/${TRAINER}/${CFG}/${BACKBONE//\//}/p/seed_${SEED}/warmup_${WARMUP}
        
#         python train_baseline.py \
#           --gpu ${GPU} \
#           --backbone ${BACKBONE} \
#           --source-datasets ${DATASETS} \
#           --root ${DATA} \
#           --trainer ${TRAINER} \
#           --dataset-config-file configs/datasets/cross_dataset/${DATASET}.yaml \
#           --config-file configs/trainers/BASELINE/${CFG}.yaml \
#           --output-dir ${DIR} \
#           --seed ${SEED} \
#           --warmup_epoch ${WARMUP} \
#           --eval-only
#       fi
#     done
#   done
# done

# for SEED in 1
# do
#   for WARMUP in 1
#   do
#     for DATASETS in 'd' 'o' 'p' 'v'
#     do
#       DIR=outputs_baseline/cross-dg_test/${TRAINER}/${CFG}/${BACKBONE//\//}/${DATASETS}/seed_${SEED}/warmup_${WARMUP}

#       if [ -d "$DIR" ]; then
#         echo "Results are available in ${DIR}, so skip this job"
#       else
#         echo "Run this job and save the output to ${DIR}"
      
#         MODEL_DIR=outputs_baseline/cross-dg/${TRAINER}/${CFG}/${BACKBONE//\//}/t/seed_${SEED}/warmup_${WARMUP}
        
#         python train_baseline.py \
#           --gpu ${GPU} \
#           --backbone ${BACKBONE} \
#           --source-datasets ${DATASETS} \
#           --root ${DATA} \
#           --trainer ${TRAINER} \
#           --dataset-config-file configs/datasets/cross_dataset/${DATASET}.yaml \
#           --config-file configs/trainers/BASELINE/${CFG}.yaml \
#           --output-dir ${DIR} \
#           --seed ${SEED} \
#           --warmup_epoch ${WARMUP} \
#           --eval-only
#       fi
#     done
#   done
# done

# for SEED in 1
# do
#   for WARMUP in 1
#   do
#     for DATASETS in 'd' 'o' 'p' 't'
#     do
#       DIR=outputs_baseline/cross-dg_test/${TRAINER}/${CFG}/${BACKBONE//\//}/${DATASETS}/seed_${SEED}/warmup_${WARMUP}

#       if [ -d "$DIR" ]; then
#         echo "Results are available in ${DIR}, so skip this job"
#       else
#         echo "Run this job and save the output to ${DIR}"
      
#         MODEL_DIR=outputs_baseline/cross-dg/${TRAINER}/${CFG}/${BACKBONE//\//}/v/seed_${SEED}/warmup_${WARMUP}
        
#         python train_baseline.py \
#           --gpu ${GPU} \
#           --backbone ${BACKBONE} \
#           --source-datasets ${DATASETS} \
#           --root ${DATA} \
#           --trainer ${TRAINER} \
#           --dataset-config-file configs/datasets/cross_dataset/${DATASET}.yaml \
#           --config-file configs/trainers/BASELINE/${CFG}.yaml \
#           --output-dir ${DIR} \
#           --seed ${SEED} \
#           --warmup_epoch ${WARMUP} \
#           --eval-only
#       fi
#     done
#   done
# done