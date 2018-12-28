#!/bin/bash

# Create a non-compressed zip file of random 300 training images and corresponding labels

cat data/random300_images.txt | zip -0r data/cityscapes_random_300_train_imgs.0.zip -@
echo 'created: data/cityscapes_random_300_train_imgs.0.zip'
cat data/random300_labels.txt | zip -0r data/cityscapes_random_300_train_labels.0.zip -@
echo 'created: data/cityscapes_random_300_train_labels.0.zip'

# Create a non-compressed zip file of all training images (2975 files)

find data/cityscapes/leftImg8bit/train -type f -name "*leftImg8bit.png" -print | zip -0r data/cityscapes_train_imgs.0.zip -@
echo 'created: data/cityscapes_train_imgs.0.zip'

# Create a non-compressed zip file of all training labels (2975 files)

find data/cityscapes/gtFine/train -type f -name "*labelIds.png" -print | zip -0r data/cityscapes_train_labels.0.zip -@
echo 'created: data/cityscapes_train_labels.0.zip'

# Ceate a non-compressed zip file of all train + train_extra images (2975 + 19998 = 22973 files)

find data/cityscapes/leftImg8bit/train data/cityscapes/leftImg8bit/train_extra -type f -name "*leftImg8bit.png" -print | zip -0r data/cityscapes_train_extra_imgs.0.zip -@
echo 'created: data/cityscapes_train_extra_imgs.0.zip'

# Ceate a non-compressed zip file of all train + train_extra labels (2975 + 19998 = 22973 files)

find data/cityscapes/gtFine/train data/cityscapes/gtCoarse/train_extra -type f -name "*labelIds.png" -print | zip -0r data/cityscapes_train_extra_labels.0.zip -@
echo 'created: data/cityscapes_train_extra_labels.0.zip'

# Create a non-compressed zip file of all validation images (500 files)

find data/cityscapes/leftImg8bit/val -type f -name "*leftImg8bit.png" -print | zip -0r data/cityscapes_val_imgs.0.zip -@
echo 'created: data/cityscapes_val_imgs.0.zip'

# Create a non-compressed zip file of all validation images (500 files)

find data/cityscapes/gtFine/val -type f -name "*labelIds.png" -print | zip -0r data/cityscapes_val_labels.0.zip -@
echo 'created: data/cityscapes_val_labels.0.zip'

# Create a non-comparessed zip file of fine resolution validation labels

find data/cityscapes/gtFine/val -name "*labelIds.png" | zip -0r data/cityscapes_gtFine_val_labels.0.zip -@
echo 'created: data/cityscapes_gtFine_val_labels.0.zip'
