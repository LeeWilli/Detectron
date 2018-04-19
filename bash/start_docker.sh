#!/bin/bash

DETECTRON=/detectron
sudo docker run --rm -it \
--name detectron \
--mount type=bind,source=/data/wangli/code/Detectron,target=$DETECTRON \
--mount type=bind,source=/data/wangli/fashionai_keypoint,target=$DETECTRON/lib/datasets/data/fashionai_keypoint \
--mount type=bind,source=/data/wangli/tmp,target=/tmp \
detectron:c2-cuda9-cudnn7 bash

python2 tools/train_net.py \
    --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x_fashion.yaml \
    OUTPUT_DIR /tmp/detectron-output USE_NCCL True
