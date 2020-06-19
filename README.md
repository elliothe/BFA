> This repository is modified from prior [repository](https://github.com/elliothe/Neural_Network_Weight_Attack ) of ICCV-2019, which includes defense codes and other codes for profiling purpose. 
  
#  Bit-Flips Attack and Defense
  
  
![BFA](assets/BFA.jpg?raw=true "Bit Flip Attack")
  
This repository constains a Pytorch implementation of BFA and its defense as discussed in the papers:
  
* "[Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rakin_Bit-Flip_Attack_Crushing_Neural_Network_With_Progressive_Bit_Search_ICCV_2019_paper.pdf )", which is published in [ICCV-2019](http://iccv2019.thecvf.com/ ).
*  "[Defending and Harnessing the Bit-Flip based Adversarial Weight Attack]( )", which is published in [CVPR-2020](http://cvpr2020.thecvf.com/ ).
  
If you find this project useful to you, please cite [our work](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rakin_Bit-Flip_Attack_Crushing_Neural_Network_With_Progressive_Bit_Search_ICCV_2019_paper.pdf ):
  
```bibtex
@inproceedings{he2019bfa,
 title={Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search},
 author={Adnan Siraj Rakin and He, Zhezhi and Fan, Deliang},
 booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
 pages={1211-1220},
 year={2019}
}
  
@inproceedings{he2020defend,
 title={Defending and Harnessing the Bit-Flip based Adversarial Weight Attack},
 author={He, Zhezhi and Rakin, Adnan Siraj and Li, Jingtao and Chakrabarti, Chaitali and Fan, Deliang},
 booktitle={Proceedings of the IEEE International Conference on Computer Vision (CVPR)},
 year={2019}
}
```
  
##  Table of Contents
  
  
- [Bit-Flips Attack and Defense](#bit-flips-attack-and-defense )
  - [Table of Contents](#table-of-contents )
  - [Introduction](#introduction )
  - [Dependencies](#dependencies )
  - [Usage](#usage )
    - [1. Configurations](#1-configurations )
    - [2. Perform the BFA](#2-perform-the-bfa )
      - [2.1 Attack on the model trained in floating-point.](#21-attack-on-the-model-trained-in-floating-point )
        - [Example of ResNet-18 on ImageNet](#example-of-resnet-18-on-imagenet )
        - [What if I want to attack another Network architecture?](#what-if-i-want-to-attack-another-network-architecture )
        - [How to perform random bit-flips on a given model?](#how-to-perform-random-bit-flips-on-a-given-model )
      - [2.2 Training-based BFA defense](#22-training-based-bfa-defense )
        - [Binarization-aware training](#binarization-aware-training )
        - [Piecewise Weight Clustering](#piecewise-weight-clustering )
  - [Misc](#misc )
    - [Model quantization](#model-quantization )
    - [Bit Flipping](#bit-flipping )
  - [License](#license )
  
##  Introduction
  
  
This repository includes a Bit-Flip Attack (BFA) algorithm which search and identify the vulernable bits within a quantized deep neural network.
  
##  Dependencies
  
  
* Python 3.6 (Anaconda)
* [Pytorch](https://pytorch.org/ ) >=1.01
* [TensorboardX](https://github.com/lanpa/tensorboardX ) 
  
For more specific dependency, please refer [environment.yml](./environment.yml ) and [environment_setup.md](./docs/environment_setup.md )
  
##  Usage
  
  
  
  
###  1. Configurations
  
  
Please modify `"alhpha"`, `PYTHON=`, `TENSORBOARD=` and `data_path=` in the example bash code (`BFA_imagenet.sh`) before running the code. This configuration is extremely useful to run the same code on different nodes.
  
```bash
HOST=$(hostname)
echo "Current host is: $HOST"
  
# Automatic check the host and configuration
case $HOST in
"alpha") # alpha is the hostname (check your current host in terminal by 'hostname')
    PYTHON="/home/elliot/anaconda3/envs/pytorch041/bin/python" # python environment path
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch041/bin/tensorboard' # tensorboard environment path
    data_path='/home/elliot/data/imagenet' # imagenet/cifar10 dataset path
    ;;
esac
```
  
###  2. Perform the BFA
  
  
####  2.1 Attack on the model trained in floating-point.
  
  
#####  Example of ResNet-18 on ImageNet
  
  
> __Note__: BFA evalution can only be performed on signle GPU (i.e., data_parallel lead to bug). 
> __Note__: Keep the bit-width of weight quantization as 8-bit.
  
Here I show the BFA on the ResNet-18, where the ResNet-18 is from [pytorch pretrained model Zoo](https://pytorch.org/docs/stable/torchvision/models.html ). BFA can be performed by just running the following command in the terminal. 
```bash
$ bash BFA_imagenet.sh
# CUDA_VISIBLE_DEVICES=2 bash BFA_imagenet.sh  # to specify GPU id to ex. 2
```
  
The example output log file of BFA on ResNet18:
```txt
  **Test** Prec@1 69.498 Prec@5 88.976 Error@1 30.502
k_top is set to 10
Attack sample size is 128
**********************************
attacked module: conv1
attacked weight index: [42  2  4  5]
weight before attack: 21.0
weight after attack: -107.0
Iteration: [001/020]   Attack Time 1.824 (1.824)  [2020-05-06 21:14:43]
loss before attack: 0.6131
loss after attack: 0.8230
bit flips: 1
hamming_dist: 1
  **Test** Prec@1 67.538 Prec@5 87.756 Error@1 32.462
iteration Time 61.966 (61.966)
**********************************
attacked module: layer2.0.downsample.0
attacked weight index: [33 50  0  0]
weight before attack: -1.0
weight after attack: 127.0
Iteration: [002/020]   Attack Time 1.315 (1.569)  [2020-05-06 21:15:47]
loss before attack: 0.8230
loss after attack: 1.4941
bit flips: 2
hamming_dist: 2
  **Test** Prec@1 59.754 Prec@5 82.390 Error@1 40.246
iteration Time 62.318 (62.142)
**********************************
```
It shows to identify one bit througout the entire model only takes ~2 Second (i.e., Attack Time) using 128 sample images for BFA. 
  
#####   What if I want to attack another Network architecture?
  
  
Taken the MobileNet v2 as example, the step-by-step tutorial is listed as follow:
1.  the first step is find a [pretrained pytorch model online](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py ).
2.  create the model definition as ```./models/vanilla_models/vanilla_mobilenet_imagenet.py```, and copy the [model](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py ) into it. Then add the following line to the ```.models/__init__.py```:
  
```python
############# Mobilenet for ImageNet #######
from .vanilla_models.vanilla_mobilenet_imagenet import mobilenet_v2
```
And make sure you are using the pretrained model option is enabled by setting `pretrained=True` in ```./models/vanilla_models/vanilla_mobilenet_imagenet.py```:
```python
def mobilenet_v2(pretrained=True, progress=True, **kwargs):
  ...
```
  
3. Run the `bash eval_imagenet.sh` can see that accuracy on validation dataset is 71.878\%.
```txt
**Test** Prec@1 71.878 Prec@5 90.286 Error@1 28.122
```
  
4. To check the accuracy with 8-bit weight quantization. create a copy of quantized mobilenetv2 in `models/quan_mobilenet_imagenet.py`. The following modifications are made sequentially:
  - import the quantized convolution and fully-connected layer.
  ```python
  from .quantization import *
  ```
  - Change all `nn.Conv2d` and `nn.Linear` to `quan_Conv2d` and `quan_Linear`.
  - Add codes for proper model loading:
  ```python
        # Modification for proper model loading
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
  ```
  - Initialize the model in ```.models/__init__.py```:
  ```python
  from .quan_mobilenet_imagenet import mobilenet_v2_quan
  ```
  - To evaluate the accuracy of quantized version, run ```bash eval_imagenet_quan.sh```, then you get:
```txt
**Test** Prec@1 71.138 Prec@5 90.012 Error@1 28.862
```
5. To perform the BFA on mobilenet-v2, simply change the configuration in `BFA_imagenet.sh`:
```bash
model=mobilenet_v2_quan 
attack_sample_size=10 # reduce the data sampes to 10, otherwise GPU out-of-memory
```
The BFA result is:
```txt
  **Test** Prec@1 71.138 Prec@5 90.012 Error@1 28.862
k_top is set to 10
Attack sample size is 10
**********************************
attacked module: features.1.conv.0.0
attacked weight index: [6 0 1 2]
weight before attack: -41.0
weight after attack: 87.0
Iteration: [001/020]   Attack Time 1.004 (1.004)  [2020-05-07 04:26:19]
loss before attack: 1.1194
loss after attack: 13.1416
bit flips: 1
hamming_dist: 1
  **Test** Prec@1 0.238 Prec@5 0.866 Error@1 99.762
iteration Time 64.102 (64.102)
**********************************
```
Single bit-flip on 8-bit Mobilenet-V2 degrade the top-1 accuracy from 71.138% to 0.206%.
  
  
#####  How to perform random bit-flips on a given model?
  
  
The random attack is performed on all the possible weight bit (regardless MSB to LSB). Take the above MobileNet-v2 as example, you just need to add another line to enable the random bit flip `--random_bfa` in `BFA_imagent.sh`:
```bash
    ...
    --attack_sample_size ${attack_sample_size} \
    --random_bfa
    ...
```
  
####  2.2 Training-based BFA defense
  
  
#####  Binarization-aware training
  
  
Taken the ResNet-20 on CIFAR-10 as example:
  
1. Define a binarized ResNet20 in `models/quan_resnet_cifar.py`.
2. To use the weight binariztaion function. Comment out [multi-bit quantization](https://github.com/elliothe/BFA/blob/8a540ac0900f2599778394cfd1df56c0965c7cdf/models/quantization.py#L8-L142 ) and uncomment the [binarization modules](https://github.com/elliothe/BFA/blob/8a540ac0900f2599778394cfd1df56c0965c7cdf/models/quantization.py#L147-L290 ).
  
3. Perform the model training, where the binarized model is initialized in `models/__init__.py` as `resnet20_quan`. Then run `bash train_CIFAR.sh`  in terminal (Don't forget the path configuration!).
  
4. With binarized model trained and stored at `<path-to-model>/model_best.pth.tar`, make sure the following changes in the `BFA_CIFAR.sh`:
```bash
pretrained_model='<path-to-model>/model_best.pth.tar'
```
  
#####  Piecewise Weight Clustering
  
  
> The piecewise weight clutering should not be applied on the binarized NN. 
  
1. Make sure ```models/quantization.py``` use the multi-bit quantization, in constrast to the binarized counterpart. To change the bit-width, please access the code in ```models/quantization.py```. Under the definition of ```quan_Conv2d``` and ```quan_Linear```, change the arg ```self.N_bits = 8``` if you want 8-bit quantization.
  
2. In `train_CIFAR.sh`, enable (i.e., uncomment) the following command:
```bash
--clustering --lambda_coeff 1e-3
```
Then train the model by `bash train_CIFAR.sh`.
  
3. For the BFA evaluation, please refer the binarization case.
  
  
  
##  Misc
  
###  Model quantization
  
  
We direct adopt the post-training quantization on the DNN pretrained model provided by the [model-zoo](https://pytorch.org/docs/stable/torchvision/models.html ) of pytorch. 
  
> __Note__: for save the model in INT-8, additional data conversion is expected.
  
  
  
  
###  Bit Flipping
  
  
Considering the quantized weight <img src="https://latex.codecogs.com/gif.latex?w"/> is a integer ranging from <img src="https://latex.codecogs.com/gif.latex?-(2^{N-1})"/> to <img src="https://latex.codecogs.com/gif.latex?(2^{N-1}-1)"/>, if using <img src="https://latex.codecogs.com/gif.latex?N"/> bits quantization. For example, the value range is -128 to 127 with 8-bit representation. In this work, we use the two's complement as its binary format (<img src="https://latex.codecogs.com/gif.latex?b_{N-1},b_{N-2},...,b_0"/>), where the back and forth conversion can be described as:
  
<img src="https://latex.codecogs.com/gif.latex?W_b%20=%20-127%20+%202^7&#x5C;cdot%20B_7%20+%202^6%20&#x5C;cdot%20B_6%20+%20&#x5C;cdots&#x5C;cdots&#x5C;cdots%202^1&#x5C;cdot%20B_1%20+%202^0&#x5C;cdot%20B_0"/>
  
  
> __Warning__: The correctness of the code is also depends on the ```dtype``` setup for the quantized weight, when convert it back and forth between signed integer and two's complement (unsigned integer). By default, we use ```.short()``` for 16-bit signed integers to prevent overflowing.
  
  
##  License
  
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
  
The software is for educaitonal and academic research purpose only.
  
