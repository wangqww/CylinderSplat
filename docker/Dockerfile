# Build context: the repository root, e.g.
#   docker build -f docker/Dockerfile -t cylindersplat .
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# Optional apt mirror, e.g. --build-arg APT_MIRROR=mirrors.ustc.edu.cn
ARG APT_MIRROR=""
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s/archive.ubuntu.com/$APT_MIRROR/g; s/security.ubuntu.com/$APT_MIRROR/g" /etc/apt/sources.list && \
      rm -f /etc/apt/sources.list.d/*; \
    fi && apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# GPUs to build the CUDA extensions for (the released models were trained on RTX 4090, sm_89)
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FORCE_CUDA=1 MAX_JOBS=8
# A proxy for the fetches: pass --build-arg https_proxy=... (a predefined build arg: visible to the
# RUN steps, never stored in the image environment or its history).

# Build tools. The source builds below import the installed torch, so they run with
# --no-build-isolation; torch 2.1's cpp_extension needs setuptools < 70 (pkg_resources.packaging).
RUN pip install "setuptools<70" wheel ninja psutil mmengine==0.10.7

# mmcv 2.1.0 with CUDA ops (must come before mmdet / mmdet3d)
RUN git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git /mmcv && \
    cd /mmcv && MMCV_WITH_OPS=1 pip install --no-build-isolation -e . -v

# pytorch3d 0.7.8
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

# Python dependencies (pinned; the file's --find-links line gets the torch-scatter wheel)
COPY docker/pip_requirements.txt /pip_requirements.txt
RUN pip install --requirement /pip_requirements.txt

# Local CUDA extensions: panorama Gaussian rasteriser (needs the glm submodule) and simple-knn
COPY pano_gaussian /cylindersplat/pano_gaussian
COPY simple-knn /cylindersplat/simple-knn
RUN pip install --no-build-isolation /cylindersplat/pano_gaussian /cylindersplat/simple-knn

WORKDIR /workspace
