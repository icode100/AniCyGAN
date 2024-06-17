import torch
import torch.nn as nn

class DBlock(nn.Module):
    def __init__(self,in_channels,out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode='reflect'
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x): return self.conv(x)
class Discriminator(nn.Module):
    def __init__(self,in_channels, features = [64,128,256,512]):
        super().__init__()
        layers = list()
        init_channels = in_channels
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                DBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    stride = 1 if feature==features[-1] else 2
                )
            )
            in_channels = feature
        self.discriminator = nn.Sequential(
            #initial
            nn.Sequential(
                nn.Conv2d(
                    in_channels=init_channels,
                    out_channels=features[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode='reflect'
                ),
                nn.LeakyReLU(0.2)
            ),
            #intermediate
            nn.Sequential(*layers),
            #final
            nn.Conv2d(
                in_channels=in_channels,
                out_channels = 1,
                kernel_size=4,
                stride = 1,
                padding=1,
                padding_mode='reflect'
            )
        )
    def forward(self,x):
        return self.discriminator(x)
