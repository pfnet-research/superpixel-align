#!/bin/bash

N_GPUS=$1
N_CLUSTERS=4
GRANULARITY=300
BATCHSIZE=30
EXPERIMENT_ID="estimated_train_labels"

echo "n_clusters: ${N_CLUSTERS}"
echo "granularity: ${GRANULARITY}"
echo "batchsize: ${BATCHSIZE}"
echo "experiment name: ${EXPERIMENT_ID}"

predict() {
    CUDA_VISIBLE_DEVICES=$3 PYTHONWARNINGS=ignore PYTHONOPTIMIZE=1 \
    python batch_spalign_kmeans.py \
    --superpixel_method felzenszwalb \
    --n_clusters ${N_CLUSTERS} \
    --y_rel_pos 0.75 \
    --x_rel_pos 0.5 \
    --y_rel_sigma 0.1 \
    --x_rel_sigma 0.1 \
    --n_anchors 10 \
    --n_neighbors 4 \
    --batchsize ${BATCHSIZE} \
    --felzenszwalb_scale ${GRANULARITY} \
    --felzenszwalb_sigma 0.8 \
    --felzenszwalb_min_size 20 \
    --use_feature_maps 7 \
    --out_dir results/${EXPERIMENT_ID} \
    --start_index $1 \
    --end_index $2 \
    --cityscapes_img_zip data/cityscapes_train_imgs.0.zip \
    --cityscapes_label_zip data/cityscapes_train_labels.0.zip \
    --gpu 0 &
}

n_data=2975
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
