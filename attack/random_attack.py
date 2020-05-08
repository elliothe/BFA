import random
import torch
from models.quantization import quan_Conv2d, quan_Linear, quantize
import operator
from attack.data_conversion import *


class random_flip(object):
    def __init__(self, model):
        self.module_list = []
        for name, m in model.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                self.module_list.append(name)

    def random_flip_one_bit(self, model):
        """
        Note that, the random bit-flip may not support on binary weight quantization.
        """
        chosen_module = random.choice(self.module_list)
        for name, m in model.named_modules():
            if name == chosen_module:
                flatten_weight = m.weight.detach().view(-1)
                chosen_idx = random.choice(range(flatten_weight.__len__()))
                # convert the chosen weight to 2's complement
                bin_w = int2bin(flatten_weight[chosen_idx], m.N_bits).short()
                # randomly select one bit
                bit_idx = random.choice(range(m.N_bits))
                mask = (bin_w.clone().zero_() + 1) * (2**bit_idx)
                bin_w = bin_w ^ mask
                int_w = bin2int(bin_w, m.N_bits).float()
                
                ##############################################
                ###   attack profiling
                ###############################################
                
                weight_mismatch = flatten_weight[chosen_idx] - int_w
                attack_weight_idx = chosen_idx
                
                print('attacked module:', chosen_module)
                
                attack_log = [] # init an empty list for profile
                
                
                weight_idx = chosen_idx
                weight_prior = flatten_weight[chosen_idx]
                weight_post = int_w

                print('attacked weight index:', weight_idx)
                print('weight before attack:', weight_prior)
                print('weight after attack:', weight_post)  
                
                tmp_list = ["module_idx", # module index in the net
                            self.bit_counter + 1, # current bit-flip index
                            "loss", # current bit-flip module
                            weight_idx, # attacked weight index in weight tensor
                            weight_prior, # weight magnitude before attack
                            weight_post # weight magnitude after attack
                            ] 
                attack_log.append(tmp_list)                            
                
                #################################
                
                flatten_weight[chosen_idx] = int_w
                m.weight.data = flatten_weight.view(m.weight.data.size())
            
                
        return attack_log
