import os

net = "vgg16"
start_epoch = 6
max_epochs = 7
dataset = "cityscape"

step=[1000,2000,3000,4000,5000,6000,7000,8000,9000]


for i in range(start_epoch, max_epochs + 1):
    model_dir = "models/cityscape/vgg16/UGA/da_faster_rcnn_cityscape_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda  \
        --net {} --dataset {} --model_dir {} ".format(net, dataset,model_dir)
    os.system(command)
    for st in step:
        model_dir = "models/cityscape/vgg16/UGA/da_faster_rcnn_cityscape_{}_{}.pth".format(i,st)
        command = "CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda  \
        --net {} --dataset {} --model_dir {} ".format(net, dataset,model_dir)
        os.system(command)
