#!/bin/bash

# Check train
python utils/run_train_rounds.py \
--img_zip_fn data/cityscapes_train_imgs.0.zip \
--label_zip_fn data/cityscapes_train_labels.0.zip \
--estimated_label_zip_fn results/estimated_train_labels.0.zip \
--test_mode \
--save_each

# Check train with soft label
python utils/run_train_rounds.py \
--img_zip_fn data/cityscapes_train_imgs.0.zip \
--label_zip_fn data/cityscapes_train_labels.0.zip \
--estimated_label_zip_fn results/estimated_train_labels.0.zip \
--use_soft_label \
--test_mode

# Check train with mean squared error
python utils/run_train_rounds.py \
--img_zip_fn data/cityscapes_train_imgs.0.zip \
--label_zip_fn data/cityscapes_train_labels.0.zip \
--estimated_label_zip_fn results/estimated_train_labels.0.zip \
--use_mse \
--test_mode

# Check train_extra
python utils/run_train_rounds.py \
--img_zip_fn data/cityscapes_train_extra_imgs.0.zip \
--label_zip_fn data/cityscapes_train_extra_labels.0.zip \
--estimated_label_zip_fn results/estimated_train_extra_labels.0.zip \
--test_mode

# Check train_extra with soft label
python utils/run_train_rounds.py \
--img_zip_fn data/cityscapes_train_extra_imgs.0.zip \
--label_zip_fn data/cityscapes_train_extra_labels.0.zip \
--estimated_label_zip_fn results/estimated_train_extra_labels.0.zip \
--use_soft_label \
--test_mode

# Check train_extra with mean squared error
python utils/run_train_rounds.py \
--img_zip_fn data/cityscapes_train_extra_imgs.0.zip \
--label_zip_fn data/cityscapes_train_extra_labels.0.zip \
--estimated_label_zip_fn results/estimated_train_extra_labels.0.zip \
--use_mse \
--test_mode