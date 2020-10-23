import pdb
import torch
import torch.nn as nn
import torchvision
from .vgg import Encoder_vgg
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class Encoder(nn.Module):
    def __init__(self, pretrained_path, device):
        super(Encoder, self).__init__()
        self.encoder_list = list(Encoder_vgg(in_channels=1, pretrained_path=pretrained_path).features.to(device))
        self.conv1x1 = conv_unit(in_ch=512, out_ch=512, kernel_size=1, activation='relu').to(device)

    def forward(self, x):
        ft_list = []
        for model_i, model in enumerate(self.encoder_list):
            x = model(x)
            if model_i % 2 == 0:
                ft_list.append(x)
        x = self.conv1x1(x)
        return x, ft_list[:]