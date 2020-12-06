#!/usr/bin/env bash

# install and setup
pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
pip install torch==1.4.0
pip install torchvision==0.5.0

git clone https://github.com/RZachLamberty/Yet-Another-EfficientDet-Pytorch.git

# download the benchmark speed test videos
# make sure you've attached a role with s3 read permissions, obviously
cd Yet-Another-EfficientDet-Pytorch
mkdir -p data/inputs
aws s3 sync s3://videoblocks-ml/data/object-detection-research/videoblocks/dev/boe-sub-sample-frame-paths data/inputs