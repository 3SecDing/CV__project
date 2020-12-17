import torch
import torch.nn as nn

def conv_blocks(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    blocks = []

    blocks.append(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0))
    blocks.append(LReLU())
    blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=padding, groups=groups))
    blocks.append(LReLU())
    # blocks.append(nn.Conv2d(out_channels, out_channels, 1, 1, padding=0))
    return nn.Sequential(*blocks)

class LReLU(nn.Module):
    def forward(self, x):
        return torch.max(0.2 * x, x)

class LightResiualReduce(nn.Module):
    def __init__(self, upsample_type='deconv'):
        super(LightResiualReduce, self).__init__()
        in_channels = 4
        channels = [32, 64, 128, 256, 512]

        for i in range(len(channels)):
            self.add_module(f'layer{i + 1}', conv_blocks(in_channels, channels[i], 3, 1, 1, channels[i]))
            in_channels = channels[i]

        for i in range(len(channels))[::-1]:
            if i == 0:
                continue
            self.add_module(f'up_layer{i}', conv_blocks(channels[i], channels[i - 1], 3, 1, 1, channels[i - 1]))



        if upsample_type == 'deconv':
            for i in range(len(channels))[::-1]:
                if i == 0:
                    continue
                self.add_module(f'up{i + 1}', nn.ConvTranspose2d(channels[i], channels[i - 1], 2, 2))
        elif upsample_type == 'bilinear':
            for i in range(len(channels))[::-1]:
                if i == 0:
                    continue
                up = []
                up.append(nn.UpsamplingBilinear2d(scale_factor=2))
                up.append(nn.Conv2d(channels[i], channels[i - 1], 1, 1, 0))
                self.add_module(f'up{i + 1}', nn.Sequential(*up))
        else:
            print("Not support upsample type!!!")

        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(channels[0], 12, kernel_size=1, stride=1)
        self.init_weights()

    def forward(self, x):
        # print("x shape:", x.shape)
        x1 = self.layer1(x)
        x1_downsampled = self.downsample(x1)

        x2 = self.layer2(x1_downsampled)
        x2_downsampled = self.downsample(x2)

        x3 = self.layer3(x2_downsampled)
        x3_downsampled = self.downsample(x3)

        x4 = self.layer4(x3_downsampled)
        x4_downsampled = self.downsample(x4)

        x5 = self.layer5(x4_downsampled)

        x5_upsampled = self.up5(x5)
        x5_concat = torch.cat([x5_upsampled, x4], 1)
        # print("shape:", x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # print("shape:", x1_downsampled.shape, x2_downsampled.shape, x3_downsampled.shape, x4_downsampled.shape, x5_upsampled.shape, x5_concat.shape)
        x4_upsampled = self.up_layer4(x5_concat)
        x4_upsampled = self.up4(x4_upsampled)
        x4_concat = torch.cat((x4_upsampled, x3), 1)

        x3_upsampled = self.up_layer3(x4_concat)
        x3_upsampled = self.up3(x3_upsampled)
        x3_concat = torch.cat((x3_upsampled, x2), 1)

        x2_upsampled = self.up_layer2(x3_concat)
        x2_upsampled = self.up2(x2_upsampled)
        x2_concat = torch.cat((x2_upsampled, x1), 1)

        x1_upsampled = self.up_layer1(x2_concat)
        output = self.out_conv(x1_upsampled)
        output = nn.functional.pixel_shuffle(output, upscale_factor=2)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

        print("finished init model weights!!!")
