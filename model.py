'''
Implementartion of Generator and Discriminator from DCGAN Paper 
'''

import torch 
import torch.nn as nn 

class Discriminator (nn.Module):
    def __init__(self, channels_img, feautres_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input : N x channels_img x 64 x 64
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
    

class Generator (nn.Module):
    def __init__(self, z_dim, channels_img, n_features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input = N x z_dim x 1 x 1
            self._block(z_dim, n_features*16, 4, 1, 0), # N x f*16 x 4 x 4
            self._block(n_features*16, n_features*8, 4, 2, 1), # 8 x 8
            self._block(n_features*8, n_features*4, 4, 2, 1), # 16 x 16
            self._block(n_features*4, n_features*2, 4, 2, 1), # 32 x 32

            nn.ConvTranspose2d(n_features*2, channels_img, 4, 2, 1), # 64 x 64 
            nn.Tanh(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.gen(x)
        
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    
def test():
    N, in_channels, H ,W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    Z = torch.randn((N, z_dim, 1, 1))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    assert gen(Z).shape == (N, in_channels, H, W)
    print("Success")

test()