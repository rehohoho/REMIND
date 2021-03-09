#!/usr/bin/env bash

PROJ_ROOT=/home/ruien/REMIND ##

export PYTHONPATH=${PROJ_ROOT}
#source activate remind_proj
cd ${PROJ_ROOT}/action_recognition_experiments

NTURGBD60_ROOT=/home/ltj/codes/MS-G3D/data/ntu_60/$1 ##
BASE_MAX_CLASS=50
MODEL="MSG3D"
LABEL_ORDER_DIR=./files/indices ##
GPU=$3 ##

# set +o posix ##
# exec > >(tee train_base_init_network.log) 2>&1 ##
export CUDA_VISIBLE_DEVICES=${GPU} 
python train_base_init_network_from_scratch.py \
--arch ${MODEL} \
--model_args "{num_class: 60, num_point: 25, num_person: 2, num_gcn_scales: 13, num_g3d_scales: 6, graph: graph.ntu_rgb_d.AdjMatrixGraph}" \
--base_max_class ${BASE_MAX_CLASS} \
--labels_dir ${LABEL_ORDER_DIR} \
--train_data_path=${NTURGBD60_ROOT}/train_data_$2.npy \
--train_label_path=${NTURGBD60_ROOT}/train_label.pkl \
--val_data_path=${NTURGBD60_ROOT}/val_data_$2.npy \
--val_label_path=${NTURGBD60_ROOT}/val_label.pkl \
--ckpt_file ${MODEL}_${BASE_MAX_CLASS}.pth \
--batch_size=16 \
--workers=16 \
--epochs=1 \
--learning-rate=0.05 \
--dataset_name nturgbd60$1

# --model_args "{level: 10, num_class: 60, num_point: 25, num_person: 2, graph: graph.ntu_rgb_d.Graph, graph_args: {labeling_mode: 'spatial'}}" \
