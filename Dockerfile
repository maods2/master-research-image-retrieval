FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG CUDA="cu126"
ARG PYTORCH="2.6.0+cu126"
ARG FLASH_ATTN="2.5.8"
ARG TRANSFORMERS="4.42.3"
ARG DIFFUSERS="0.28.0"
ARG PEFT="0.11.1"
ARG TRL="0.9.4"
ARG BITSANDBYTES="0.43.1"
ARG DATASETS="2.19.1"
ARG ACCELERATE="0.30.1"
ARG EVALUATE="0.4.2"
ARG SENTENCE_TRANSFORMERS="2.7.0"
ARG DEEPSPEED="0.14.2"
ARG MAX_JOBS=4

RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y upgrade --only-upgrade systemd openssl cryptsetup && \
    apt-get install -y \
    build-essential \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    cmake \
    libprotobuf-dev \
    libaio-dev \
    protobuf-compiler \
    python3-dev \
    python3-pip \
    python3.10 \
    libsndfile1-dev \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}


RUN apt-get install -y unzip htop
# Update pip
RUN pip install --upgrade pip

# Install latest release PyTorch (PyTorch must be installed before any DeepSpeed c++/cuda ops.)
# RUN pip install --no-cache-dir -U torch==${PYTORCH} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA}
# RUN pip install --no-cache-dir -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA}


RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Install Hugging Face Libraries
# RUN pip install --upgrade --no-cache-dir \
#     transformers[sklearn,sentencepiece,vision]==${TRANSFORMERS} \
#     diffusers==${DIFFUSERS} \
#     datasets==${DATASETS} \
#     accelerate==${ACCELERATE} \
#     evaluate==${EVALUATE} \
#     peft==${PEFT} \
#     trl==${TRL} \
#     sentence-transformers==${SENTENCE_TRANSFORMERS} \
#     deepspeed==${DEEPSPEED} \
#     bitsandbytes==${BITSANDBYTES} \
#     tensorboard \
#     jupyter notebook

COPY requirements.txt requirements.txt
# Install Google Cloud Dependencies
RUN pip install -r requirements.txt