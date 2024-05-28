CUDA_VISIBLE_DEVICES=0  python -u trainval_net.py --dataset cityscape --net vgg16 --cuda \
--epochs 7 --gamma 3.0 --warmup --context --init --db \
--alpha1 0.1 --alpha2 1.0 --alpha3 1.0 \
--lamda1 1.0 --lamda2 1.0 --lamda3 0.01 \
--desp 'UGA'

CUDA_VISIBLE_DEVICES=0 python  test_city.py