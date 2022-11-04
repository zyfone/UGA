# Unsupervised Domain Adaptive Object Detection for CAHR
Code implementation for CAHR


We will update the code after the paper is officially accepted

## Requirements
* Ubuntu 18.04.5 LTS
* Python 3.6
* [CUDA 9.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.0.0](https://pytorch.org)
* [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)





Remember to compile the cuda dependencies using following simple commands following [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0):
```bash
cd lib
python setup.py build develop
```



## Pre-trained Models

Follow the convention, two pre-trained models on ImageNet, i.e., VGG16 and ResNet101 are employed. 
Please download and place these two models in `pretrained_models/` from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)




## :pencil:Related repos
Our project references the codes in the following repos:


* Zhao _et al_.,[TIA](https://github.com/MCG-NJU/TIA/)
