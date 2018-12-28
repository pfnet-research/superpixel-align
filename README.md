Superpixel Align
================

This is the official implementation of ["Minimizing Supervision for Free-space Segmentation
"](https://arxiv.org/abs/1711.05998). BibTeX is [here](#reference).

Tested environment
------------------

- Ubuntu 16.04
- CUDA 9.0

Setup environment
-----------------

```bash
bash setup.sh
```

The above shell script creates miniconda environment under this directory and install requirements below. The `setup.sh` installs all of them **so you don't need to install the dependencies below by yourself.**

- Python: >=3.6.2
- External libraries: openmpi(--with-cuda)==2.1.1, cuDNN==7.0.3, nccl2==2.1.2
- Python packages: cython==0.27.3, tqdm==4.19.4, Pillow==4.3.0, scipy==1.0.0, scikit-image==0.13.1, opencv-python==3.3.0.10, pytorch==0.2.0, mpi4py==3.0.0, cupy==4.0.0b1, chainer==4.0.0b1, chainermn==git+https://github.com/mitmul/chainermn@support-cupy-v4.0.0b1, chainercv==0.7.0

Prepare
-------

Before starting all procedures below, please run this first to activate the environment:

```bash
source miniconda/bin/activate road-segm
```

### 1. Download Cityscapes dataset

Please make an account at the [Cityscapes dataset](https://www.cityscapes-dataset.com) site and download the files below from [the donwload page](https://www.cityscapes-dataset.com/downloads/). You need these files:

- [gtFine_trainvaltest.zip (241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
- [gtCoarse.zip (1.3GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=2)
- [leftImg8bit_trainvaltest.zip (11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
- [leftImg8bit_trainextra.zip (44GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4)

After you downloaded them, please extract all files in a same directory. That directory is referred as `[CITYSCAPES_DIR]` in the descriptions below.

### 2. Create symlink to the cityscapes dir

```bash
ln -s [CITYSCAPES_DIR] data/cityscapes
```

[CITYSCAPES_DIR] should be replaced with the path to the directory of the Cityscapes dataset. The directory should contains `gtCoarse`, `gtFine`, and `leftImg8bit` dirs. Each subdir should have the dirs below:

- data/cityscapes
  - gtCoarse
    - test
    - train
    - train_extra
    - val
  - gtFine
    - test
    - train
    - val
  - leftImg8bit
    - test
    - train
    - train_extra
    - val

### 3. Convert PyTorch model to Chainer model:

```bash
cd models
python convert_pth2ch.py
cd ..
```

### 4. Create zip files (dataset creation)

```bash
bash utils/create_zip_files.sh
```

Clustering superpixel align features
------------------------------------

Throughout all the commands below,

- Please replace `[NUMBER OF GPUS]` with the number of GPUs you want to use for this script.
- The GPU ID starts counting from 0, so if you want to specify the GPU IDs, please set `CUDA_VISIBLE_DEVICES` environment variable.
- Please run this first to activate the environment:

  ```bash
  source miniconda/bin/activate road-segm
  ```

### Generate labels of randomly selected 300 train images and evaluate them

```bash
MPLBACKEND=Agg bash utils/create_random300_labels.sh [NUMBER OF GPUS]

# Wait until it finishes (Check the processes has terminated)

python utils/mean_result.py \
results/estimated_train_random300_labels/result.json \
--n_imgs 300 --count_duplicated
```

The evaluation result is found [here](#random_300).

### Generate labels of validation images and evaluate them

```bash
MPLBACKEND=Agg bash utils/create_val_labels.sh [NUMBER OF GPUS]

# Wait until it finishes (Check the processes has terminated)

# Evaluate the estimation
python utils/mean_result.py results/estimated_val_labels/result.json
```

The evaluation result is found [here](#val).

### Generate labels of training images and evaluate them

```bash
MPLBACKEND=Agg bash utils/create_train_labels.sh [NUMBER OF GPUS]

# Wait until it finishes (Check the processes has terminated)

# Evaluate the estimation
python utils/mean_result.py results/estimated_train_labels/result.json

# Zip the generated labels
find results/estimated_train_labels -name "*leftImg8bit.npy" | zip -0r results/estimated_train_labels.0.zip -@
```

Then, please make sure that all the files below exist under `results` dir.

- estimated_train_labels.0.zip

This file is used as labels to train SegNet.

Train SegNet on estimated labels
--------------------------------

- Please change the value given to `--n_gpus` option to the number of GPUs you want to use for this experiment.
- Please change the value given to `--batchsize` option to the number you preferred when the default value 8 is too big to fit in your GPUs.
- The GPU ID starts counting from 0, so if you want to specify the GPU IDs, please set `CUDA_VISIBLE_DEVICES` environment variable.
- Please run this first to activate the environment:

  ```bash
  source miniconda/bin/activate road-segm
  ```


### Train SegNet on generated labels

The command below starts training of SegNetBasic on the generated labels created by the above script. The training goes for 2000 iterations with Adam optimizer.

```bash
LD_LIBRARY_PATH=miniconda/envs/road-segm/lib:$LD_LIBRARY_PATH \
MPLBACKEND=Agg \
python utils/run_train_rounds.py \
--img_zip_fn data/cityscapes_train_imgs.0.zip \
--label_zip_fn data/cityscapes_train_labels.0.zip \
--estimated_label_zip results/estimated_train_labels.0.zip \
--n_gpus 8 \
--batchsize 8
```

**See the `log` file in the `results/train_round1_[DATETIME]` dir for the evaluation scores of the generated labels by the trained SegNet on the estimated labels by superpixel align feature clustering.**


Evaluation results
------------------

### Superpixel align feature clustering

#### Randomly selected 300 train images<a name="random_300"></a>

| Metric    | Value              |
|:----------|:-------------------|
| Road IoU  | 0.8129520227337709 |
| Precision | 0.8835840497695169 |
| Recall    | 0.9166856000528959 |

#### Validation images<a name="val"></a>

| Metric    | Value              |
|:----------|:-------------------|
| Road IoU  | 0.7619056844993818 |
| Precision | 0.8799825987212356 |
| Recall    | 0.8919905105061199 |

### After SegNet training<a name="segnet"></a>

#### Evaluated on validation images

| Metric    | Value              |
|:----------|:-------------------|
| Road IoU  | 0.8345039286452565 |
| Precision | 0.897570349944977  |
| Recall    | 0.9232502418464443 |

Reference<a name='reference'></a>
---------------------------------

```
@InProceedings{Tsutsui_2018_CVPR_Workshops,
  author = {Tsutsui, Satoshi and Kerola, Tommi and Saito, Shunta and Crandall, David J.},
  title = {Minimizing Supervision for Free-Space Segmentation},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```
