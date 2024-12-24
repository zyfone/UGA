# Domain Adaptive Thermal Object Detection with Unbiased Granularity Alignment


![LAST COMMIT](https://img.shields.io/github/last-commit/zyfone/UEA) 
![ISSUES](https://img.shields.io/github/issues/zyfone/UEA)
![STARS](https://img.shields.io/github/stars/zyfone/UEA)


<div align=center>
  <img src="github_pic/framework.jpg" alt="Framework" width="450"/>
</div>


---
### RGB-to-RGB Benchmark：

[Cityscape-to-Foggycityscape branch](https://github.com/zyfone/UGA/tree/visible-to-visible) 

Performance is slightly higher than reported results.



## Requirements
* Ubuntu 18.04.5 LTS
* Python 3.6
* [CUDA 10.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.0.0](https://pytorch.org)
* [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)


## dataset download

We also provide the download URL of the dataset in the future.

[Dataste Download](https://github.com/AmineMarnissi/UDAT)




## Compile the code

Compile the cuda dependencies using following simple commands following [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0):
```bash
cd lib
python setup.py build develop
```

## Pre-trained Models


* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

## Training and Test

Train the model

```bash
sh ./train_scripts/train_flir.sh
```

Test the well-trained model:
```bash
python test_scripts/test_flir.py
```



## Reproducibility
Due to the instability of adversarial training, we do not report the best results, with our code you can get better results.

| ![Image 1](github_pic/report.png) | ![Image 2](github_pic/experiment.png) |![Image 3](github_pic/mAP-filr.jpg)|
|:-----------------------------------------:|:-----------------------------------------:|:-----------------------------------------:|



## :pencil:Related repos
Our project references the codes in the following repos:

* Megvii-Nanjing, [CR-DA-DET](https://github.com/Megvii-Nanjing/CR-DA-DET)


other code :
* https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch
* https://github.com/MCG-NJU/TIA/
* https://github.com/harsh-99/SCL
* https://github.com/AmineMarnissi/UDAT



if you have any questions , please contact me at 478756030@qq.com




```BibTeX
@article{10.1145/3665892,
author = {Shi, Caijuan and Zheng, Yuanfan and Chen, Zhen},
title = {Domain Adaptive Thermal Object Detection with Unbiased Granularity Alignment},
year = {2024},
doi = {10.1145/3665892},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
}
```
