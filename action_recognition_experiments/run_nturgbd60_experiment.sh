#!/usr/bin/env bash

PROJ_ROOT=/home/ruien/REMIND ##

export PYTHONPATH=${PROJ_ROOT}
export KMP_DUPLICATE_LIB_OK=TRUE
cd ${PROJ_ROOT}/action_recognition_experiments

NTURGBD60_ROOT=/home/ltj/codes/MS-G3D/data/ntu_60/$1 ##
EXPT_NAME=ntu60$1$2_shuffled
GPU=$3 ##


CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
REPLAY_SAMPLES=15 # batch number -1
MAX_BUFFER_SIZE=12000 # divisible by batch size

BASE_INIT_CLASSES=50 # 6
CLASS_INCREMENT=10 # 6
NUM_CLASSES=60
BASE_INIT_CKPT=./files/50c210c_pretrained_without_blocks.pt # base init ckpt file
LABEL_ORDER_DIR=./files/indices # location of numpy label files


#set +o posix ##
#exec > >(tee run_nturgbd60_experiment.log) 2>&1 ##
export CUDA_VISIBLE_DEVICES=${GPU}
python -u nturgbd60_experiment.py \
--expt_name ${EXPT_NAME} \
--label_dir ${LABEL_ORDER_DIR} \
--dataset_name nturgbd60$1 \
--train_data_path=${NTURGBD60_ROOT}/train_data_$2.npy \
--train_label_path=${NTURGBD60_ROOT}/train_label.pkl \
--val_data_path=${NTURGBD60_ROOT}/val_data_$2.npy \
--val_label_path=${NTURGBD60_ROOT}/val_label.pkl \
--base_arch "MSG3D_EndAtSCGN3" \
--base_model_args "{num_class: 60, num_point: 25, num_person: 2, num_gcn_scales: 13, num_g3d_scales: 6, graph: graph.ntu_rgb_d.AdjMatrixGraph}" \
--classifier "MSG3D_StartAfterSCGN3" \
--classifier_model_args "{num_class: 60, num_point: 25, num_person: 2, num_gcn_scales: 13, num_g3d_scales: 6, graph: graph.ntu_rgb_d.AdjMatrixGraph}" \
--classifier_ckpt ${BASE_INIT_CKPT} \
--extract_features_from sgcn3.2 \
--num_channels 384 \
--num_instances 2 \
--spatial_feat_dim 75,25 \
--weight_decay 1e-5 \
--batch_size 16 \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--lr_mode step_lr_per_class \
--lr_step_size 100 \
--start_lr 0.05 \
--end_lr 0.05 \
--use_mixup \
--mixup_alpha .1 \
--num_classes ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--streaming_min_class ${BASE_INIT_CLASSES} \
--streaming_max_class ${NUM_CLASSES}

# --base_arch "AGCN_ClassifyAfterLevel" \
# --base_model_args "{level: 9, num_class: 60, num_point: 25, num_person: 2, graph: graph.ntu_rgb_d.Graph, graph_args: {labeling_mode: 'spatial'}}" \
# --classifier "AGCN_StartAtLevel" \
# --classifier_model_args "{level: 10, num_class: 60, num_point: 25, num_person: 2, graph: graph.ntu_rgb_d.Graph, graph_args: {labeling_mode: 'spatial'}}" \ 
# --extract_features_from levels.8 \
# --num_channels 256 \
