# first clone mujoco-py then replace its obsolete dockerfile with this one
# git clone https://github.com/openai/mujoco-py.git
# Install nvidia docker for GPU support
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install all needed tools
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openssh-server \
    curl \
    git \
    vim \
    wget \
    net-tools \
    software-properties-common \
    xserver-xorg-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libglew1.5 \
    libegl1-mesa \
    libxrandr2 \ 
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mujoco
RUN mkdir -p /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
RUN echo "LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}" >> ~/.bashrc
RUN echo "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}" >> ~/.bashrc

# Install anaconda
RUN wget \
    https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Anaconda3-2021.11-Linux-x86_64.sh -b \
    && rm -f Anaconda3-2021.11-Linux-x86_64.sh
ENV PATH="/root/anaconda3/bin:${PATH}"
RUN conda init bash

EXPOSE 22
EXPOSE 6006

# After create the image, users can run a container.
# The remaining steps are about creating the python env for mujoco-py. 
# Here is our example.
# conda create -n mujoco python=3.7
# conda activate mujoco
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install -U 'mujoco-py<2.2,>=2.1'