import torch
from torchsummaryX import summary
import option

from model import mrdn

model = mrdn.MRDN(option.args)
# input = torch.randn(1, 3, 256, 256)
# summary(model, input)

# input LR x2, HR size is 720p
# summary(model, torch.zeros((1, 3, 640, 360)))

# input LR x3, HR size is 720p
# summary(model, torch.zeros((1, 3, 426, 240)))

# input LR x4, HR size is 720p
summary(model, torch.zeros((1, 3, 320, 180)))
