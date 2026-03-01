# Source image from pytorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Some important environment variables in Dockerfile
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# Custom Configuration
RUN sed -i  "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list && \
    sed -i  "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/*

# Install pip dependencies
COPY pip_requirements.txt /pip_requirements.txt
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --requirement /pip_requirements.txt
# Install 3DGS rasterizer
RUN git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization /diff-gaussian-rasterization
RUN TORCH_CUDA_ARCH_LIST="8.9+PTX" pip install /diff-gaussian-rasterization
# Install mmcv, mmdet, mmdet3d
RUN pip install -U openmim
RUN pip install mmengine
RUN pip install ninja psutil
RUN git clone https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
RUN git checkout v2.1.0
ARG HTTPS_PROXY=""
ENV MAX_JOBS=8
ENV https_proxy=${HTTPS_PROXY}
ENV MMCV_WITH_OPS="1"
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"
RUN pip install -e . -v
RUN mim install 'mmdet>=3.0.0'
RUN mim install 'mmdet3d>=1.1.0'
WORKDIR /