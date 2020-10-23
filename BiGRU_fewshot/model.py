import pdb
import numpy as np
import torch
import torch.nn as nn
from models.encoder import SupportEncoder, QueryEncoder
from models.convgru import ConvBGRU
from models.decoder import Decoder

class MedicalFSS(nn.Module):
    def __init__(self, config, device):
        super(MedicalFSS, self).__init__()
        self.config=config
        resize_dim = self.config['input_size']
        self.encoded_h = int(resize_dim[0] / 2 ** self.config['n_pool'])
        self.encoded_w = int(resize_dim[1] / 2 ** self.config['n_pool'])

        self.s_encoder = SupportEncoder(self.config['path']['init_path'], device)  # .to(device)
        self.q_encoder = QueryEncoder(self.config['path']['init_path'], device)  # .to(device)
        self.ConvBiGRU = ConvBGRU(in_channels=512,
                             hidden_channels=256,
                             kernel_size=(3, 3),
                             num_layers=self.config['n_layer'],
                             device=device).to(device)
        self.decoder = Decoder(input_res=(self.encoded_h, self.encoded_w), output_res=resize_dim).to(device)
        self.q_slice_n = self.config['q_slice']
        self.ch = 256  # number of channels of embedding vector
        self.n_shot = self.config['n_shot']
        self.reversed_idx = list(reversed(range(self.q_slice_n)))

        self.is_attention=self.config['is_attention']
        if self.is_attention:
            self.avgpool3d = nn.AvgPool3d((self.ch*2, self.encoded_w, self.encoded_h))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, s_x, s_y, q_x):
        s_x_encode, q_x_encode, q_ft_lists = [], [], []
        for frame_id in range(self.q_slice_n):
            s_xi = s_x[:, :, frame_id, :, :, :]  # [B, Support, 1, 256, 256]
            s_yi = s_y[:, :, frame_id, :, :, :]
            q_xi = q_x[:, frame_id, :, :, :]
            s_x_merge = s_xi.view(s_xi.size(0) * s_xi.size(1), 1, 256, 256)
            s_y_merge = s_yi.view(s_yi.size(0) * s_yi.size(1), 1, 256, 256)
            s_xi_encode_merge, s_ft_list = self.s_encoder(s_x_merge, s_y_merge)  # [B*S, ch, w, h]
            s_xi_encode = s_xi_encode_merge.view(s_xi.size(0), s_xi.size(1), self.ch, self.encoded_w, self.encoded_h)  # [B, S, ch, w, h]
            q_xi_encode, q_ft_list = self.q_encoder(q_xi)

            s_x_encode.append(s_xi_encode)  # [B,256(c),256,256]
            q_x_encode.append(q_xi_encode)  # [B,256(c),256,256]
            q_ft_lists.append(q_ft_list)

        s_xi_encode_frames = torch.stack(s_x_encode, dim=2) #[B, shot, slice, ch, w, h]

        gru_outputs = []
        for shot_id in range(self.n_shot):
            s_x_encode_batch = s_xi_encode_frames[:,shot_id, ...]
            # s_x_encode_batch = torch.stack(s_x_encode[:,shot_id,...], dim=1)
            q_x_encode_batch = torch.stack(q_x_encode, dim=1) #[B, slice, ch, w, h]
            x_encode_batch = torch.cat((s_x_encode_batch, q_x_encode_batch), dim=2)
            x_fwd = x_encode_batch
            x_rev = x_encode_batch[:, self.reversed_idx, ...]
            h_encode_gru = self.ConvBiGRU(x_fwd, x_rev)
            gru_outputs.append(h_encode_gru) #[B, slice, ch, w, h]

        gru_output = torch.stack(gru_outputs,dim=1)  #[B, shot, slice, ch, w, h]
        # gru_out = torch.sum(gru_output,dim=1) #[B, slice, ch, w, h]
        gru_out = torch.mean(gru_output,dim=1) #[B, slice, ch, w, h]

        out = []
        for frame_id in range(self.q_slice_n):
            hi = gru_out[:, frame_id, :, :, :]
            q_ft_list = q_ft_lists[frame_id]
            yhati = self.decoder(hi, q_ft_list)  # [B, 1, 256, 256]
            out.append(yhati)

        return out

    def get_attention_score(self):
        return self.attention_score