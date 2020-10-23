import pdb
import numpy as np
import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class MedicalFSS(nn.Module):
    def __init__(self, config, device):
        super(MedicalFSS, self).__init__()
        self.config=config
        resize_dim = self.config['input_size']
        self.encoded_h = int(resize_dim[0] / 2 ** self.config['n_pool'])
        self.encoded_w = int(resize_dim[1] / 2 ** self.config['n_pool'])

        self.encoder = Encoder(self.config['path']['init_path'], device)  # .to(device)
        self.decoder = Decoder(input_res=(self.encoded_h, self.encoded_w), output_res=resize_dim).to(device)
        self.q_slice_n = self.config['q_slice']
        self.ch = 256  # number of channels of embedding vector

    def forward(self, x):
        x = x.squeeze(3)
        x = x.squeeze(1) #[B,5,256,256]
        x_enc, ft_list  = self.encoder(x)
        yhat = self.decoder(x_enc, ft_list)  # [B, 1, 256, 256]
        out = [yhat[:, k, ...] for k in range(5)]
        # pdb.set_trace()
        return out