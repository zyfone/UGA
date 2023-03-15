# Unsupervised Domain Adaptive Object Detection for CCDA(Title to be updated)
Code implementation for CCDA


We will update the code after the paper is officially accepted

## Requirements
* Ubuntu 18.04.5 LTS
* Python 3.6
* [CUDA 10.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.0.0](https://pytorch.org)
* [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)


## dataset download
 [FLIR](https://github.com/AmineMarnissi/UDAT)
 
 [Cityscape and Foggy Cityscape](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)

Compile the cuda dependencies using following simple commands following [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0):
```bash
cd lib
python setup.py build develop
```



## Pre-trained Models


* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)




## :pencil:Related repos
Our project references the codes in the following repos:

* Megvii-Nanjing,[CR-DA-DET](https://github.com/Megvii-Nanjing/CR-DA-DET)


other code :
* https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch
* https://github.com/ChenJinBIT/SIR
* https://github.com/MCG-NJU/TIA/
* https://github.com/harsh-99/SCL

