#!/bin/bash

PARAM_DIR=$1
ITERATION=$2
IMG_ZIP_FN=$3
LABEL_ZIP_FN=$4
OUT_DIR=$5
N_GPUS=$6

predict() {
    CUDA_VISIBLE_DEVICES=$3 PYTHONWARNINGS=ignore PYTHONOPTIMIZE=1 \
    python labels_from_segnet.py \
    --param_dir ${PARAM_DIR} \
    --iteration ${ITERATION} \
    --gpu 0 \
    --img_zip_fn ${IMG_ZIP_FN} \
    --label_zip_fn ${LABEL_ZIP_FN} \
    --out_dir ${OUT_DIR} \
    --start_index $1 \
    --end_index $2 \
    --eval_shape 1024 2048 &
}

n_data=500
step=$(expr ${n_data} / ${N_GPUS} + 1)
gpu_i=0
i=0
while [ $i -lt $n_data ]
do
    start_i=$i
    i=$(expr $i + $step)
    if [ $i -ge $n_data ]
    then
        i=$n_data
    fi
    predict $start_i $i $gpu_i
    gpu_i=$(expr $gpu_i + 1)
done
