#!/bin/bash
save_dir="./weight_model/flir_voc/"
dataset="flir_voc"
pretrained_path="/root/autodl-tmp/pretrained_model/resnet101_caffe.pth"
net="res101"

CUDA_VISIBLE_DEVICES=0 python -u da_train_net.py \
--max_epochs 7 --cuda --dataset ${dataset} \
--net ${net} --save_dir ${save_dir} \
--pretrained_path ${pretrained_path} \
--gc --lc --da_use_contex 

CUDA_VISIBLE_DEVICES=0 python test_scripts/test_voc_flir.py