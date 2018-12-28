#!/bin/bash

N_CLUSTERS=4     # Use 4
BATCHSIZE=30     # Use 30
GRANULARITY=300  # Use 300
EXPERIMENT_ID="estimated_train_random300"

echo "n_clusters: ${N_CLUSTERS}"
echo "granularity: ${GRANULARITY}"
echo "experiment name: ${EXPERIMENT_ID}"

cp data/weights/drn_c_26.npz models/

predict() {
    MPLBACKEND=Agg PYTHONWARNINGS=ignore PYTHONOPTIMIZE=1 \
    python3 batch_spalign_kmeans.py \
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
    --img_file_list data/train_images.txt \
    --label_file_list data/train_labels.txt \
    --gpu 0
}

predict $1 $2
