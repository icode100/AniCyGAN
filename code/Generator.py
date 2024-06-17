import torch
import torch.nn as nn

class GBlock(nn.Module):
    def __init__(self,in_channels, out_channels, down = True, use_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    def forward(self,x): return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.block = nn.Sequential(
            GBlock(in_channels=channels,out_channels=channels,use_act=True,kernel_size = 3,padding=1),
            GBlock(in_channels=channels,out_channels=channels,use_act=False,kernel_size = 3, padding=1),
        )
    def forward(self,x): return x+self.block(x)

class Generator(nn.Module):
    def __init__(self,in_channels,num_residuals=9):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            GBlock(64,128,down=True, use_act = True,kernel_size = 3,stride = 2,padding = 1),
            GBlock(128,256,down=True, use_act = True,kernel_size = 3,stride = 2,padding = 1),
            *([ResidualBlock(256)]*num_residuals),
            GBlock(256,128,down=False, kernel_size = 3, stride = 2, padding=1,output_padding=1),
            GBlock(128,64,down=False, kernel_size = 3, stride = 2, padding=1,output_padding=1),
            nn.Conv2d(64,3,7,1,3,padding_mode="reflect")
        )
    def forward(self,x): return self.generator(x)
