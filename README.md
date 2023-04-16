# Unsupervised Domain Adaptive Object Detection for CCDA(Title to be updated)
Code implementation for CCDA



---
submitting to the chinaMM conference，we will update the code after the paper is officially accepted


We have been able to obtain improved performance over the reported results in the paper by adjusting some parameters
For fairness, we will also publish the code details of our comparison method via URL




## Requirements
* Ubuntu 18.04.5 LTS
* Python 3.6
* [CUDA 10.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.0.0](https://pytorch.org)
* [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)


## dataset download

We also provide the download URL of the dataset in the future

 [FLIR](https://github.com/AmineMarnissi/UDAT)
 
 [Cityscape and Foggy Cityscape](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)

Compile the cuda dependencies using following simple commands following [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0):
```bash
cd lib
python setup.py build develop
```


**Note that we find that our code is not stable due to adversarial training,require multiple testing attempts to achieve desired results**



## Pre-trained Models


* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)




## :pencil:Related repos
Our project references the codes in the following repos:

* Megvii-Nanjing,[CR-DA-DET](https://github.com/Megvii-Nanjing/CR-DA-DET)


other code :
* https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch
* https://github.com/MCG-NJU/TIA/
* https://github.com/harsh-99/SCL

