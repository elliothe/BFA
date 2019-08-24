# Bit-Flips Attack:

This repository constains a Pytorch implementation of the paper "[Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search](https://arxiv.org/pdf/1903.12269.pdf)"

If you find this project useful to you, please cite [our work](https://arxiv.org/pdf/1903.12269.pdf):

```bibtex
@inproceedings{he2019bfa,
 title={Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search},
 author={Adnan Siraj Rakin and He, Zhezhi and Fan, Deliang},
 booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
 pages={},
 year={2019}
}
```

## Table of Contents

- [Introduction](#Introduction)
- [Env setup](#Env_setup)
<!-- - [Usage](#Usage)
    - [Train](#Train) --> -->


## Introduction

scale_distance
## Env setup
We leverage the docker to ensure the user can use our code.

## Usage

Following steps to perform Bit-Flip Attack (BFA):
1. Get a quantized model



```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

###Model quantization
For the goal that directly quantize the deep neural network without retraining it, we add the function ```--optimize_step``` to optimize the step-size of quantizer to minimize the loss (e.g., mean-square-error loss) between quantized weight and its full precision base. It is intriguing to find out that:

- directly apply the uniform quantizer can achieve higher accuracy (close to the full precision baseline) without optimize the quantizer, for high-bit quantization (e.g., 8-bit). 

- On the contrary, for the low-bit quantization (e.g., 4-bit), directly quantize the weight causes significant accuracy loss. With the ```--optimize_step``` enabled, accuracy can partially recover without retraining. 

Since for the ImageNet simulation, we want to use directly perform the weight quantization on the pretrained weight

## Bit Flipping

Considering the quantized weight $w$ is a integer ranging from $-(2^{N-1})$ to $(2^{N-1}-1)$, if using $N$ bits quantization. For example, the value range is -128 to 127 with 8-bit representation. In this work, we use the two's complement as its binary format ($b_{N-1}b_{N-2}b_0$), where the back and forth conversion:

Hereby, we choose the two's complement as the encoding method for 

$W_b = -127 + 2^7\cdot B_7 + 2^6 \cdot B_6 + \cdots\cdots\cdots 2^1\cdot B_1 + 2^0\cdot B_0$

In order to perform the BFA:
1. Identify the 

> __Note__: The correctness of the code is also depends on the ```dtype``` setup for the quantized weight, when convert it back and forth between signed integer and two's complement (unsigned integer). By default, we use ```.short()``` for 16-bit signed integers to prevent overflowing.


