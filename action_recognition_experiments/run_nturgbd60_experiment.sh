#!/usr/bin/env bash

PROJ_ROOT=D:/repos/REMIND ##

export PYTHONPATH=${PROJ_ROOT}
export KMP_DUPLICATE_LIB_OK=TRUE
cd ${PROJ_ROOT}/action_recognition_experiments

NTURGBD60_ROOT=D:/repos/GradientEpisodicMemory/data/raw ##
EXPT_NAME=remind_nturgbd60
GPU=0 ##


CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
REPLAY_SAMPLES=50
MAX_BUFFER_SIZE=959665

BASE_INIT_CLASSES=6
CLASS_INCREMENT=6
NUM_CLASSES=60
BASE_INIT_CKPT=./files/best_AGCN_ClassifyAfterLevel_6.pth # base init ckpt file
LABEL_ORDER_DIR=./files/indices # location of numpy label files


set +o posix ##
exec > >(tee run_nturgbd60_experiment.log) 2>&1 ##
export CUDA_VISIBLE_DEVICES=${GPU} 
python -u nturgbd60_experiment.py \
--expt_name ${EXPT_NAME} \
--label_dir ${LABEL_ORDER_DIR} \
--train_data_path=${NTURGBD60_ROOT}/train_data_joint.npy \
--train_label_path=${NTURGBD60_ROOT}/train_label.pkl \
--val_data_path=${NTURGBD60_ROOT}/val_data_joint.npy \
--val_label_path=${NTURGBD60_ROOT}/val_label.pkl \
--base_model_args "{level: 9, num_class: 60, num_point: 25, num_person: 2, graph: graph.ntu_rgb_d.Graph, graph_args: {labeling_mode: 'spatial'}}" \
--classifier_model_args "{level: 10, num_class: 60, num_point: 25, num_person: 2, graph: graph.ntu_rgb_d.Graph, graph_args: {labeling_mode: 'spatial'}}" \
--classifier_ckpt ${BASE_INIT_CKPT} \
--extract_features_from levels.8 \
--num_channels 256 \
--spatial_feat_dim 75,25 \
--weight_decay 1e-5 \
--batch_size 1 \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--lr_mode step_lr_per_class \
--lr_step_size 100 \
--start_lr 0.1 \
--end_lr 0.001 \
--use_random_resized_crops \
--use_mixup \
--mixup_alpha .1 \
--num_classes ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--streaming_min_class ${BASE_INIT_CLASSES} \
--streaming_max_class ${NUM_CLASSES}
