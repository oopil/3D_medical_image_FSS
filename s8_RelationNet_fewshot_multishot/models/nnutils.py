import torch
import torch.nn as nn

def conv_unit(in_ch, out_ch, kernel_size, stride = 1, padding = 0, activation = 'relu', batch_norm = True):
    seq_list = []
    seq_list.append(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))

    if batch_norm:
        seq_list.append(nn.BatchNorm2d(num_features = out_ch))
    
    if activation == 'relu':
        seq_list.append(nn.ReLU())
    elif activation == 'sigmoid':
        seq_list.append(nn.Sigmoid())
    
    return nn.Sequential(*seq_list)



# class VOSBaseArch(nn.Module):
#     def __init__(self, initializer, encoder, convlstmcell, decoder, cost_fn, optimizer):
#         super(VOSBaseArch, self).__init__()
#         self.initializer = initializer
#         self.encoder = encoder
#         self.convlstmcell = convlstmcell
#         self.decoder = decoder
#         self.cost_fn = cost_fn
#         self.optimizer = optimizer

#     def forward(self, x, y, t):
#         yhat_list = []
#         loss_list = []
#         loss_per_video = 0.0

#         print(x[:, 0, :, :, :].size(), y[:, 0, :, :, :].size())
#         ci, hi = initializer(x[:, 0, :, :, :] + y[:, 0, :, :, :])

#         for frame_id in range(1, x.size(1)):
#             xi = x[:, frame_id, :, :, :]
#             yi = y[:, frame_id, :, :, :]

#             xi = encoder(xi)
#             ci, hi = convlstmcell(xi, ci, hi)
#             yhati = decoder(hi)
#             yhat_list.append(yhati)

#             loss = cost_fn(yhati, yi)
#             loss_per_video += loss
#             loss_list.append(loss.item())

#         return yhat_list, loss_list, loss_per_video