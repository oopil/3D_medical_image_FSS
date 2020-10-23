import pdb
import torch
import torch.nn as nn
import torchvision
from .vgg import Encoder_vgg
from .attention import PAM_Module, CAM_Module
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class SupportEncoder(nn.Module):
    def __init__(self, pretrained_path, device):
        super(SupportEncoder, self).__init__()
        self.encoder_list = list(Encoder_vgg(in_channels=2, pretrained_path=pretrained_path).features.to(device))
        self.conv1x1 = conv_unit(in_ch=512, out_ch=512, kernel_size=1, activation='relu').to(device)

        self.attention=False #True
        if self.attention:
            self.pam = PAM_Module(in_dim=512).to(device)
            self.cam = CAM_Module(in_dim=512).to(device)

    def forward(self, x,y):
        out = torch.cat((x,y),dim=1)
        ft_list = []
        for model_i, model in enumerate(self.encoder_list):
            out = model(out)
            if model_i % 2 == 0:
                ft_list.append(out)

        if self.attention:
            out = self.pam(out)+self.cam(out)

        out = self.conv1x1(out)
        return out, ft_list[:]

class QueryEncoder(nn.Module):
    def __init__(self, pretrained_path, device):
        super(QueryEncoder, self).__init__()
        self.encoder_list = list(Encoder_vgg(in_channels=1, pretrained_path=pretrained_path).features.to(device))
        self.conv1x1 = conv_unit(in_ch=512, out_ch=512, kernel_size=1, activation='relu').to(device)

        self.attention=False #True
        if self.attention:
            self.pam = PAM_Module(in_dim=512).to(device)
            self.cam = CAM_Module(in_dim=512).to(device)

    def forward(self, x):
        ## data set preprocessing ( youtube vos, decathlon )
        ## get the encoded list -> skip connection
        ## change the vgg model from Few shot segmentation model

        # pdb.set_trace()
        ft_list = []
        for model_i, model in enumerate(self.encoder_list):
            x = model(x)
            if model_i % 2 == 0:
                ft_list.append(x)

        if self.attention:
            x = self.pam(x)+self.cam(x)

        x = self.conv1x1(x)
        return x, ft_list[:]
