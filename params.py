from thop import profile
import torch
from model import mrdn
import option

if __name__ == '__main__':
    model = mrdn.MRDN(option.args)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input,))
    print(flops / (1024 ** 3))
    print(params / 1024)
