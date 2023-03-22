# PyTorch Imports
import torch
import torchvision
import torch.nn as nn
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch import Tensor
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights
)



class ConvTranspose(nn.Module):
    def __init__(self, in_ch, out_ch, k, nonlinearity='hardswish'):
        super(ConvTranspose, self).__init__()
        if nonlinearity == 'hardswish':
            self.block = nn.Sequential(
                ConvTranspose2d(in_ch, out_ch, k, k),
                BatchNorm2d(out_ch, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.Hardswish(inplace=True)
            )
        elif nonlinearity == 'relu':
            self.block = nn.Sequential(
                ConvTranspose2d(in_ch, out_ch, k, k),
                BatchNorm2d(out_ch, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError('Invalid nonlinearity chosen for ConvTranspose Block')
    

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x
        


class Encoder(nn.Module):
    def __init__(self, pretrained: bool = False, hpos: int = 13):
        super(Encoder, self).__init__()
        model = mobilenet_v3_large()
        if pretrained:
            state_dict = MobileNet_V3_Large_Weights.IMAGENET1K_V2.get_state_dict()
            model.load_state_dict(state_dict)
        
        backbone = model.features
        backbone = nn.Sequential(
            *backbone[:hpos], backbone[hpos].block[0]
        )
        self.backbone = backbone
    

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return x



class Decoder(nn.Module):
    def __init__(self, depth: str = 'light'):
        super(Decoder, self).__init__()
        if depth == 'light':
            layers = nn.Sequential(
                ConvTranspose(672, 16, 4),
                ConvTranspose(16, 3, 4, 'relu'),
            )
        elif depth == 'deep':
            layers = nn.Sequential(
                ConvTranspose(672, 112, 1),
                ConvTranspose(112, 112, 2),
                ConvTranspose(112, 480, 1),
                ConvTranspose(480, 480, 2),
                ConvTranspose(480, 80, 1),
                ConvTranspose(80, 80, 2),
                ConvTranspose(80, 16, 1),
                ConvTranspose(16, 16, 2),
                ConvTranspose(16, 3, 1, 'relu')
            )
        else:
            raise ValueError('Invalid Decoder depth configuration')
        
        self.layers = layers
    

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, pretrained: bool = False, depth: str = 'light'):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder(depth)        
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x