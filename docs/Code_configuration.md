# Detail configurations of the code

## Quantization

### Bit-width

To change the bit-width, please access the code in ```models/quantization.py```. Under the definition of ```quan_Conv2d``` and ```quan_Linear```, change the arg ```self.N_bits = 8``` if you want 8-bit quantization.

