import pdb
import torch
import torch.nn as nn
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class ConvGRUCell(nn.Module):
    """
    Basic CGRU cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias, device='cuda:0'):

        super(ConvGRUCell, self).__init__()
        self.device = device
        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.update_gate = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim, out_channels=self.hidden_dim,
                                     kernel_size=self.kernel_size, padding=self.padding,
                                     bias=self.bias)
        self.reset_gate = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim, out_channels=self.hidden_dim,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    bias=self.bias)

        self.out_gate = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim, out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur = cur_state
        # data size is [batch, channel, height, width]
        x_in = torch.cat([input_tensor, h_cur], dim=1)
        update = torch.sigmoid(self.update_gate(x_in))
        reset = torch.sigmoid(self.reset_gate(x_in))
        # print(h_cur.device)
        # pdb.set_trace()
        x_out = torch.tanh(self.out_gate(torch.cat([input_tensor, h_cur * reset], dim=1)))
        h_new = h_cur * (1 - update) + x_out * update

        return h_new

    def init_hidden(self, b, h, w):
        return torch.zeros(b, self.hidden_dim, h, w, device='cpu')


class ConvGRU(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, device='cuda:0'):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.device=device
        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvGRUCell(in_channels=cur_input_dim,
                                          hidden_channels=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        # if not self.batch_first:
        #     # (t, b, c, h, w) -> (b, t, c, h, w)
        #     input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvGRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx].to(self.device)
            output_inner = []
            for t in range(seq_len):

                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output # use encoded features as input again

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBGRU(nn.Module):
    # Constructor
    def __init__(self, in_channels, hidden_channels,
                 kernel_size, num_layers, bias=True, batch_first=False, device='cuda:0'):

        super(ConvBGRU, self).__init__()
        in_channels = 512
        hidden_channels = 512
        self.device=device
        self.forward_net = ConvGRU(in_channels, hidden_channels//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias, device=device).to(device)
        self.reverse_net = ConvGRU(in_channels, hidden_channels//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias, device=device).to(device)

    def forward(self, xforward, xreverse):
        """
        xforward, xreverse = B T C H W tensors.
        """

        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)

        y_out_fwd = y_out_fwd[-1] # outputs of last CGRU layer = B, T, C, H, W
        y_out_rev = y_out_rev[-1] # outputs of last CGRU layer = B, T, C, H, W

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, reversed_idx, ...] # reverse temporal outputs.
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)

        return ycat


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
