#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/home/elliot/anaconda3/envs/pytorch041/bin/python" # python environment path
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch041/bin/tensorboard' # tensorboard environment path
    data_path='/home/elliot/data/pytorch/cifar10'
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=vgg11_bn_quan
# model=resnet20_quan
dataset=cifar10
epochs=200
train_batch_size=128
test_batch_size=128
optimizer=SGD

label_info=idx_34

attack_sample_size=128 # number of data used for BFA

tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info}/tb_log  #tensorboard log path

############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 100 150  --gammas 0.1 0.1 \
    --test_batch_size ${test_batch_size} \
    --workers 4 --ngpu 1 --gpu_id 1 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --clustering --lambda_coeff 1e-3    
    # --quan_bitwidth 2
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait