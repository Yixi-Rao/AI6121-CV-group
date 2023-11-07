"""
Generator model for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    ''' c7s1-k: 7 x 7 Convolution-InstanceNormReLU layer with k filters and stride 1
        dk: 3 x 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
        uk: 3 x 3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2
    '''
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        
        # Downsampling layers or upsampling layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()  # ?
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    '''
        Rk: two 3 x 3 convolutional layers with the same number of filters on both layer.
    '''
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        
        # c7s1-64
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        
        # d128, d256
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, 
                    num_features * 2, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        
        # R256,R256,R256,R256,R256,R256,R256,R256,R256
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        
        # u128, u64
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        # c7s1-3
        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        # c7s1-64
        x = self.initial(x)
        # d128,d256
        for layer in self.down_blocks:
            x = layer(x)
        # R256,R256,R256,R256,R256,R256,R256,R256,R256
        x = self.res_blocks(x)
        # u128 u64
        for layer in self.up_blocks:
            x = layer(x)
        # c7s1-3
        return torch.tanh(self.last(x))

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)

if __name__ == "__main__":
    test()
