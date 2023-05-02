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
from utils import unnormalize



class ConvTranspose(nn.Module):
    # Transposed Convolution Block
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



class ConvBlock(nn.Module):
    # Standard Convolution Block (with optional non-linearity)
    def __init__(self, in_ch, out_ch, k, nonlinearity='hardswish'):
        super(ConvBlock, self).__init__()

        if nonlinearity == 'hardswish':
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, k),
                BatchNorm2d(out_ch, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.Hardswish(inplace=True)
            )
        elif nonlinearity == 'relu':
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, k),
                BatchNorm2d(out_ch, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f'Nonlinearity {nonlinearity} is invalid for ConvBlock. Choose ReLU or Hardswish instead')
    

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x
        


class Encoder(nn.Module):
    # MobileNetV3 Large model that cuts off at C4 by default
    def __init__(self, pretrained: bool = False, hpos: int = 13):
        super(Encoder, self).__init__()
        model = mobilenet_v3_large()
        if pretrained:
            state_dict = MobileNet_V3_Large_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
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
    def __init__(self, depth: str = 'light', interpolation: str = 'transpose'):
        super(Decoder, self).__init__()

        if interpolation == 'transpose':
            if depth == 'light':
                layers = nn.Sequential(
                    ConvTranspose(672, 16, 4),
                    ConvTranspose(16, 3, 4, 'relu'),
                )
            elif depth == 'medium':
                layers = nn.Sequential(
                    ConvTranspose(672, 112, 1),
                    ConvTranspose(112, 80, 2),
                    ConvTranspose(80, 80, 2),
                    ConvTranspose(80, 16, 1),
                    ConvTranspose(16, 16, 4),
                    ConvTranspose(16, 3, 1, 'relu')
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
        
        elif interpolation == 'upsample':
            layers = nn.Sequential(
                ConvBlock(672, 112, 1),
                nn.Upsample(scale_factor=2, mode='bicubic'),
                ConvBlock(112, 480, 1),
                nn.Upsample(scale_factor=2, mode='bicubic'),
                ConvBlock(480, 80, 1),
                nn.Upsample(scale_factor=2, mode='bicubic'),
                ConvBlock(80, 16, 1),
                nn.Upsample(scale_factor=2, mode='bicubic'),
                ConvBlock(16, 3, 1, 'relu')
            )
        else:
            raise ValueError(f'interpolation type: {interpolation} is not supported\nUse transpose or upsample instead')
        
        self.layers = layers
    

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, pretrained: bool = False, depth: str = 'light', interpolation: str = 'transpose'):
        super(Autoencoder, self).__init__()
        self.interpolation = interpolation
        self.encoder = Encoder(pretrained)
        if pretrained:
            # Freeze the encoder by default
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.decoder = Decoder(depth, interpolation)        
        
    
    def inspect_result(self, x: Tensor):
        print("Max value of input is: {}".format( torch.max(x) ))
        print("Min value of input is: {}".format( torch.min(x) ))
        print("Std deviation of input is: {}".format( torch.std(x) ))
        print("Average of input is: {}".format( torch.mean(x) ))

        x = self.forward(x)
        x = unnormalize(x)

        print("Max value of output is: {}".format( torch.max(x) ))
        print("Min value of output is: {}".format( torch.min(x) ))
        print("Std deviation of output is: {}".format( torch.std(x) ))
        print("Average of output is: {}".format( torch.mean(x) ))

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x