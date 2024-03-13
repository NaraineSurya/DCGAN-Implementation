'''
Implementartion of Generator and Discriminator from DCGAN Paper 
'''

import torch 
import torch.nn as nn 

class Discriminator (nn.Module):
    def __init__(self, channels_img, feautres_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input : N x channeles_img x 64 x 64
            nn.Conv2d(channels_img, feautres_d, kernel_size=4, stride=2, padding=1),# 32x32
            nn.LeakyReLU(0.2),
            self._block(feautres_d, feautres_d*2,4,2,1), # 16x16
            self._block(feautres_d*2, feautres_d*4,4,2,1), # 8x8 
            self._block(feautres_d*4, feautres_d*8,4,2,1), # 4x4
            nn.Conv2d(feautres_d*8, 1, kernel_size=4, stride=2, padding=0), # 1x1
            nn.Sigmoid()
        )

    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d( 
                in_channels,out_channels,
                kernel_size,
                stride,
                padding,bias= False 
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)