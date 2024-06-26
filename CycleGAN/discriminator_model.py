"""
Discriminator model for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn

class Block(nn.Module):
    '''Ck:
        4 x 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
        not use InstanceNorm for the first C64 layer
    '''
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels,
                4, 
                stride, 
                1, 
                bias=True,
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    '''Ck:
        4 x 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
        not use InstanceNorm for the first C64 layer
        After the last layer, we apply a convolution to produce a 1-dimensional output
        leaky ReLUs with a slope of 0.2
    '''
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # first C64 layer
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers      = []
        in_channels = features[0]
        
        # C128-C256-C512
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        
        # last layer1-dimensional output
        layers.append(nn.Conv2d(
                        in_channels,
                        1,
                        kernel_size=4,
                        stride=1,
                        padding=1,
                        padding_mode="reflect"))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

def test():
    x     = torch.randn((5, 3, 256, 256))
    model = Discriminator()
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
