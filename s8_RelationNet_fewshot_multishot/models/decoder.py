import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class Decoder(nn.Module):
    def __init__(self, input_channels=512, input_res=(8, 14), init_channels=512, shrink_per_block=2, output_channels=1,
                 output_res=(256, 448)):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(512 + 512*1, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )  # 14 x 14
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(512 + 512*1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU()
        )  # 28 x 28
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(256 + 256*1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU()
        )  # 56 x 56
        self.double_conv4 = nn.Sequential(
            nn.Conv2d(128 + 128*1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )  # 112 x 112
        self.double_conv5 = nn.Sequential(
            nn.Conv2d(64 + 64 * 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            # nn.Conv2d(64, 2, kernel_size=1, padding=0),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),  # 1 for bce and 2 for cross entropy loss
            nn.Sigmoid()
        )  # 256 x 256
        # x = F.interpolate(x, orig_size, mode="bilinear")
        self._init_weights()

    def mask_process(self, mask):
        # x = F.interpolate(x, orig_size, mode="bilinear")
        mask = F.interpolate(mask, [16,16], mode="bilinear")

    def forward(self, hidden, ft_list):
        out = self.layer1(hidden)
        out = self.layer2(out)
        # out = self.upsample(out)  # block 1
        out = torch.cat((out, ft_list[-1]), dim=1)
        out = self.double_conv1(out)
        # out = self.upsample(out)  # block 2
        out = torch.cat((out, ft_list[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out)  # block 3
        out = torch.cat((out, ft_list[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out)  # block 4
        out = torch.cat((out, ft_list[-4]), dim=1)
        out = self.double_conv4(out)
        out = self.upsample(out)  # block 5
        out = torch.cat((out, ft_list[-5]), dim=1)
        out = self.double_conv5(out)
        # out = F.sigmoid(out)
        # out = torch.squeeze(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class Decoder_save(nn.Module):
    def __init__(self, input_channels=512, input_res=(8, 14), init_channels=512, shrink_per_block=2, output_channels=1,
                 output_res=(256, 448)):
        super(Decoder, self).__init__()
        self.input_channels = input_channels
        self.input_res = np.array(input_res)
        self.init_channels = init_channels
        self.shrink_per_block = shrink_per_block
        self.output_channels = output_channels
        self.output_res = np.array(output_res)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.net = self.network(self.input_channels, self.input_res, self.init_channels, self.shrink_per_block, self.output_channels, self.output_res)

    def network(self, input_channels, input_res, init_channels, shrink_per_block, output_channels, output_res):
        modules = []
        prev_channels = input_channels
        # print('0', prev_channels, input_channels, input_res, output_res)
        while True:
            # print('1', prev_channels, init_channels)

            modules.append(conv_unit(in_ch=prev_channels, out_ch=init_channels, kernel_size=5, stride=1, padding=2))

            # print('2', prev_channels, init_channels)

            if np.array_equal(input_res, output_res):
                modules.append(
                    conv_unit(in_ch=init_channels, out_ch=output_channels, kernel_size=5, stride=1, padding=2, activation='sigmoid'))
                # print('3', prev_channels, init_channels)

                break
            else:
                modules.append(conv_unit(in_ch=init_channels, out_ch=init_channels, kernel_size=5, stride=1, padding=2))
                # print('4', prev_channels, init_channels)

                modules.append(self.upsample)
                input_res *= 2

            prev_channels = init_channels
            if init_channels > 64:
                init_channels = int(init_channels / shrink_per_block)
            # print('5', prev_channels, init_channels)

        return nn.Sequential(*modules)

    def forward(self, h):
        return self.net(h)
