#!/bin/bash

if [ ! -d lib ]; then
    mkdir lib
fi

install_miniconda () {
    if [ ! -d $PWD/miniconda ]; then
        curl -L -O https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh
        bash Miniconda3-4.3.21-Linux-x86_64.sh -b -p $PWD/miniconda
        rm -rf Miniconda3-4.3.21-Linux-x86_64.sh
    fi
}

install_cudnn () {
    install_prefix=$PWD/miniconda/envs/road-segm
    if [ ! -d lib/cudnn ]; then
        mkdir -p lib/cudnn && cd lib/cudnn
        if [[ $(nvcc --version | grep -o -E "V([0-9].[0-9])") == "V8.0" ]]; then
            curl -L -O http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.4/cudnn-8.0-linux-x64-v7.tgz
            tar -xzf cudnn-8.0-linux-x64-v7.tgz
            rm -rf cudnn-8.0-linux-x64-v7.tgz
        elif [[ $(nvcc --version | grep -o -E "V([0-9].[0-9])") == "V9.0" ]]; then
            curl -L -O http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.4/cudnn-9.0-linux-x64-v7.tgz
            tar -xzf cudnn-9.0-linux-x64-v7.tgz
            rm -rf cudnn-9.0-linux-x64-v7.tgz
        fi
        mv cuda/lib64/* $install_prefix/lib/
        mv cuda/include/* $install_prefix/include/
        cd ../..
    fi
}

install_nccl2 () {
    install_prefix=$PWD/miniconda/envs/road-segm
    if [ ! -d lib/nccl ]; then
        mkdir -p lib/nccl && cd lib/nccl
        if [[ $(nvcc --version | grep -o -E "V([0-9].[0-9])") == "V8.0" ]]; then
            curl -sL -o libnccl2_2.1.2-1+cuda8.0_amd64.deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.2-1+cuda8.0_amd64.deb
            curl -sL -o libnccl-dev_2.1.2-1+cuda8.0_amd64.deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.2-1+cuda8.0_amd64.deb
            ar vx libnccl2_2.1.2-1+cuda8.0_amd64.deb
            tar xvf data.tar.xz
            ar vx libnccl-dev_2.1.2-1+cuda8.0_amd64.deb
            tar xvf data.tar.xz
        elif [[ $(nvcc --version | grep -o -E "V([0-9].[0-9])") == "V9.0" ]]; then \
            curl -sL -o libnccl2_2.1.2-1+cuda9.0_amd64.deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.2-1+cuda9.0_amd64.deb
            curl -sL -o libnccl-dev_2.1.2-1+cuda9.0_amd64.deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.2-1+cuda9.0_amd64.deb
            ar vx libnccl2_2.1.2-1+cuda9.0_amd64.deb
            tar xvf data.tar.xz
            ar vx libnccl-dev_2.1.2-1+cuda9.0_amd64.deb
            tar xvf data.tar.xz
        fi
        mv ./usr/* ./
        rm -rf *.gz && rm -rf *.xz && rm -rf *.deb && rm -rf share && rm -rf usr && rm -rf debian-binary
        mv lib/x86_64-linux-gnu/* $install_prefix/lib/
        mv include/* $install_prefix/include/
        cd ../..
    fi
}

install_openmpi () {
    install_prefix=$PWD/miniconda/envs/road-segm
    if [ ! -d lib/openmpi ]; then
        cd lib
        curl -L -O https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.gz
        tar xvf openmpi-2.1.1.tar.gz
        rm -rf openmpi-2.1.1.tar.gz
        mv openmpi-2.1.1 openmpi
        cd openmpi
        ./configure --prefix $install_prefix --with-cuda
        make -j32 && make install
        cd ../..
    fi
    ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
}

install_opencv() {
    install_prefix=$PWD/miniconda/envs/road-segm
    if [ ! -d lib/opencv ]; then
        cd lib
        curl -L -O https://github.com/opencv/opencv/archive/3.3.1.zip
        unzip 3.3.1.zip
        rm -rf 3.3.1.zip
        mv opencv-3.3.1 opencv
        cd opencv
        mkdir build
        cd build
        cmake \
        -DBUILD_TESTS=OFF \
        -DENABLE_AVX=ON \
        -DENABLE_AVX2=ON \
        -DENABLE_FAST_MATH=ON \
        -DENABLE_NOISY_WARNINGS=ON \
        -DENABLE_SSE41=ON \
        -DENABLE_SSE42=ON \
        -DENABLE_SSSE3=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DBUILD_opencv_python3=ON \
        -DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.prefix)')/bin/python \
        -DPYTHON3_EXECUTABLE=$(python -c 'import sys; print(sys.prefix)')/bin/python \
        -DPYTHON_LIBRARY=$(python -c 'import sys; print(sys.prefix)')/lib \
        -DPYTHON3_LIBRARY=$(python -c 'import sys; print(sys.prefix)')/lib \
        -DPYTHON_LIBRARY_DEBUG=$(python -c 'import sys; print(sys.prefix)')/lib \
        -DPYTHON3_LIBRARY_DEBUG=$(python -c 'import sys; print(sys.prefix)')/lib \
        -DPYTHON3_PACKAGES_PATH=$(python -c 'import sys; print(sys.prefix)')/lib/python$(python -c "import sys; v = sys.version_info; print('{}.{}'.format(v.major, v.minor))")/site-packages \
        -DPYTHON3_INCLUDE_DIR=$(python -c 'import sys; print(sys.prefix)')/include/python$(python -c "import sys; v = sys.version_info; print('{}.{}'.format(v.major, v.minor))")m \
        -DPYTHON3_INCLUDE_DIR2=$(python -c 'import sys; print(sys.prefix)')/include \
        -DPYTHON_INCLUDE_DIR2=$(python -c 'import sys; print(sys.prefix)')/include \
        -DWITH_OPENMP=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENCLAMDBLAS=OFF \
        -DWITH_OPENCLAMDFFT=OFF \
        -DWITH_OPENCL_SVM=OFF \
        -DWITH_1394=OFF \
        -DWITH_TBB=ON \
        -DHAVE_MKL=ON \
        -DMKL_WITH_OPENMP=ON \
        -DMKL_WITH_TBB=ON \
        -DBUILD_TIFF=ON \
        -DBUILD_CUDA_STUBS=OFF \
        -DBUILD_opencv_cudaarithm=OFF \
        -DBUILD_opencv_cudabgsegm=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DBUILD_opencv_cudafeatures2d=OFF \
        -DBUILD_opencv_cudafilters=OFF \
        -DBUILD_opencv_cudaimgproc=OFF \
        -DBUILD_opencv_cudalegacy=OFF \
        -DBUILD_opencv_cudaobjdetect=OFF \
        -DBUILD_opencv_cudaoptflow=OFF \
        -DBUILD_opencv_cudastereo=OFF \
        -DBUILD_opencv_cudawarping=OFF \
        -DBUILD_opencv_cudev=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_CUBLAS=OFF \
        -DWITH_CUFFT=OFF \
        -DWITH_FFMPEG=ON \
        -DINSTALL_PYTHON_EXAMPLES=ON \
        -DINSTALL_C_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=$install_prefix \
        ../
        CPATH=$(python -c 'import sys; print(sys.prefix)')/include/python$(python -c "import sys; v = sys.version_info; print('{}.{}'.format(v.major, v.minor))")m make -j32 && \
        CPATH=$(python -c 'import sys; print(sys.prefix)')/include/python$(python -c "import sys; v = sys.version_info; print('{}.{}'.format(v.major, v.minor))")m make install
        cd ../../..
    fi
}

# Setup Python environment
install_miniconda
export PATH="$PWD/miniconda/bin:$PATH"
hash -r
conda create -n road-segm -y
source miniconda/bin/activate road-segm

# Install OpenMPI with CUDA option
install_openmpi

# Install NCCL2
install_nccl2

# Install cuDNN
install_cudnn

# Install dependencies
install_opencv
conda install pytorch==0.2.0 -c soumith
pip install mpi4py==3.0.0
pip install tqdm
conda install Cython
conda install matplotlib
conda install scipy
conda install scikit-image

export CFLAGS="-I$PWD/miniconda/envs/road-segm/include"
export CPATH="$PWD/miniconda/envs/road-segm/include:$CPATH"
export LDFLAGS="-L$PWD/miniconda/envs/road-segm/lib"
export LIBRARY_PATH="$PWD/miniconda/envs/road-segm/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/miniconda/envs/road-segm/lib:$LD_LIBRARY_PATH"

# Install ChainerMN
pip install git+https://github.com/mitmul/chainermn@support-cupy-v4.0.0b1

# Install ChainerCV
pip install chainercv==0.7.0

# Install CuPy
pip install cupy==4.0.0b1 --no-cache-dir -vvv

# Install Chainer
pip uninstall chainer -y
pip install chainer==4.0.0b1

# Check installation
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
python -c 'import chainer; print(chainer.cuda.available); print(chainer.cuda.cudnn_enabled)'
LD_LIBRARY_PATH=miniconda/envs/road-segm/lib:$LD_LIBRARY_PATH \
mpiexec -np 8 -x PATH -x LD_LIBRARY_PATH \
python -c "import chainermn; chainermn.create_communicator('single_node')"
which python