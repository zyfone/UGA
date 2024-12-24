import os

net = "res101"
part = "test"
start_epoch = 6
max_epochs = 7
output_dir = "./flir_city/result"
dataset = "flir_city"

step=[1000,2000,3000,4000,5000,6000,7000,8000,9000]


for i in range(start_epoch, max_epochs + 1):
    model_dir = "./weight_model/flir_city/flir_city_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=1 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
    for st in step:
        model_dir = "./weight_model/flir_city/flir_city_{}_{}.pth".format(i,st)
        command = "CUDA_VISIBLE_DEVICES=1 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i )
        os.system(command)