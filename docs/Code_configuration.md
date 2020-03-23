# Detail configurations of the code


### Quantization bit-width

To change the bit-width, please access the code in ```models/quantization.py```. Under the definition of ```quan_Conv2d``` and ```quan_Linear```, change the arg ```self.N_bits = 8``` if you want 8-bit quantization.

### K-top

The ```k_top``` determines in current layer, the number of weights of top-gradient that used for the calculated the bit-gradient.

In the bash code, e.g., ```BFA_CIFAR.sh```, if dont assign ```k_top``` with a integer value, it will defaultly compute all the weight bits in that layer.


### n_iter

The argument ```n_iter``` set the maximum number of BFA iterations. However, the attack will automatically stop, if the accuracy is below the configured accuracy ```break_acc```.

