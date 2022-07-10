FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

RUN apt update
RUN apt install -y python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

RUN python3 -m pip install transformers yacs scipy
