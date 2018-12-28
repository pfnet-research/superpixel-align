#!/bin/bash

EXPERIMENT_ID="estimate_labels_using_direct_feature_clustering"
echo "experiment name: ${EXPERIMENT_ID}"

cp data/weights/drn_c_26.npz models/

predict() {
    MPLBACKEND=Agg PYTHONWARNINGS=ignore PYTHONOPTIMIZE=1 \
    python3 superpixel_overlaps.py \
    --superpixel_method felzenszwalb \
    --n_clusters 4 \
    --y_rel_pos 0.75 \
    --x_rel_pos 0.5 \
    --y_rel_sigma 0.1 \
    --x_rel_sigma 0.1 \
    --n_anchors 10 \
    --n_neighbors 4 \
    --batchsize 30 \
    --felzenszwalb_scale 500 \
    --felzenszwalb_sigma 0.9 \
    --felzenszwalb_min_size 20 \
    --overlap_threshold 0.01 \
    --use_feature_maps 7 \
    --out_dir results/${EXPERIMENT_ID} \
    --start_index $1 \
    --end_index $2 \
    --img_file_list data/train_images.txt \
    --label_file_list data/train_labels.txt \
    --gpu 0 &
}

predict $1 $2
