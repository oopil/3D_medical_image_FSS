import pdb
import torch
import torch.nn as nn
import torchvision
import numpy as np
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class SupportConvLSTMCell(nn.Module):
    def __init__(self, channels = 512, height = 8, width = 8, device = 'cuda:0'):
        super(SupportConvLSTMCell, self).__init__()
        ## batch normalization in LSTM cell
        is_bn = False #False
        # Convolutions for gate computations
        self.Wxi = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Whi = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Wxf = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Whf = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Wxc = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Whc = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Wxo = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Who = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)

        # Matrices used for Hadamard product used in gate computations
        # maybe they are in CPU device
        self.Wci = torch.randn((channels, height, width), requires_grad = True).to(device)
        self.Wcf = torch.randn((channels, height, width), requires_grad = True).to(device)
        self.Wco = torch.randn((channels, height, width), requires_grad = True).to(device)
        nn.init.kaiming_uniform_(self.Wci)
        nn.init.kaiming_uniform_(self.Wcf)
        nn.init.kaiming_uniform_(self.Wco)

        # Since paper uses ReLU instead of the standard TanH function
        # self.gate_activation = nn.ReLU()
        self.gate_activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c_prev, h_prev):
        i = self.Wxi(x) + self.Whi(h_prev) + (self.Wci * c_prev)
        i = self.sigmoid(i)
        f = self.Wxf(x) + self.Whf(h_prev) + (self.Wcf * c_prev)
        f = self.sigmoid(f)
        c = self.gate_activation(self.Wxc(x) + self.Whc(h_prev))
        # c = (f * c_prev) + (i * c)
        c = ((f * c_prev) + (i * c))/2 ## scale c_state not to overflow
        o = self.Wxo(x) + self.Who(h_prev) + (self.Wco * c)
        o = self.sigmoid(o)
        h = o * self.gate_activation(c)
        return c, h

class QueryConvLSTMCell(nn.Module):
    def __init__(self, channels = 512, height = 8, width = 8, device = 'cuda:0'):
        super(QueryConvLSTMCell, self).__init__()
        ## batch normalization in LSTM cell
        is_bn = False #False
        # Convolutions for gate computations
        x_ch = channels
        h_ch = 2*channels
        self.Wxi = conv_unit(in_ch = x_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Whi = conv_unit(in_ch = h_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Wxf = conv_unit(in_ch = x_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Whf = conv_unit(in_ch = h_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Wxc = conv_unit(in_ch = x_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Wxo = conv_unit(in_ch = x_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Whc = conv_unit(in_ch = h_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)
        self.Who = conv_unit(in_ch = h_ch, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = is_bn).to(device)

        # Matrices used for Hadamard product used in gate computations
        # maybe they are in CPU device
        self.Wci = torch.randn((channels, height, width), requires_grad = True).to(device)
        self.Wcf = torch.randn((channels, height, width), requires_grad = True).to(device)
        self.Wco = torch.randn((channels, height, width), requires_grad = True).to(device)
        nn.init.kaiming_uniform_(self.Wci)
        nn.init.kaiming_uniform_(self.Wcf)
        nn.init.kaiming_uniform_(self.Wco)
        # Since paper uses ReLU instead of the standard TanH function
        # self.gate_activation = nn.ReLU()
        self.gate_activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c_prev, h_prev, h_supp):
        h = torch.cat((h_prev, h_supp),dim=1)
        i = self.Wxi(x) + self.Whi(h) + (self.Wci * c_prev)
        i = self.sigmoid(i)
        f = self.Wxf(x) + self.Whf(h) + (self.Wcf * c_prev)
        f = self.sigmoid(f)
        c = self.gate_activation(self.Wxc(x) + self.Whc(h))
        c = (f * c_prev) + (i * c)
        # c = ((f * c_prev) + (i * c))/2 ## scale c_state not to overflow
        o = self.Wxo(x) + self.Who(h) + (self.Wco * c)
        o = self.sigmoid(o)
        h = o * self.gate_activation(c)
        return c, h
