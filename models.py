import os,sys
import torch.nn as nn
import torch.nn.functional as F
import torch

from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange

from helperFn import GaussianNoise, SpectralNorm, MiniBatchStdDev


class selfAttention(nn.Module):
    """ Self attention Layer"""
    ## May use not sure
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        proj_query = rearrange(self.query_conv(x), 'b c h w -> b (h w) c')
        proj_key = rearrange(self.key_conv(x), 'b c h w -> b c (h w)')
        proj_value = rearrange(self.value_conv(x), 'b c h w -> b (h w) c')
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=2)
        out = torch.bmm(attention, proj_value)
        out = x + self.gamma * rearrange(out, 'b (h w) c -> b c h w',
                                         **parse_shape(x, 'b c h w'))
        return out


# Define Transpose Conv Module with Spectralnorm


def conv2DTransposeBlock(in_channels, out_channels, kernel=(5, 4), stride=2, padding=1,
                         normalize=True, activation=nn.SELU(inplace=True), attention=False):
    # look into whether to add bias or not
    conv2DUpsample = nn.ConvTranspose2d(
        in_channels, out_channels, kernel, stride, padding, bias=False)
    layers = [SpectralNorm(conv2DUpsample)]

    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))

    if activation:
        layers.append(activation)
    else:
        layers.append(nn.ReLU(inplace=True))

        if attention:
            layers.append(selfAttention(out_channels))
    return layers

# Define Conv Module with Spectralnorm


def conv2DBlock(in_channels, out_channels, kernel=4, stride=2, padding=1,
                specNorm=True, activation=False, dropout=False, miniBatchNorm=False, attention=False):
    conv2DDownsample = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding,
                                 bias=False)
    if specNorm:
        layers = [SpectralNorm(conv2DDownsample)]
    else:
        layers = [conv2DDownsample]

    if miniBatchNorm:
        layers.append(MiniBatchStdDev())

    if activation:
        layers.append(activation)
    else:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    if dropout:
        layers.append(nn.Dropout(0.3))

    if attention:
        layers.append(selfAttention(out_channels))
    return layers



class carsCNNGenerator(nn.Module):
    def __init__(self, channels=512, latent_dim=100, kernel=4,class_dim=50, attention=False, activation=nn.Mish(inplace=True),conditional=False):
        super(carsCNNGenerator, self).__init__()
        self.attention = attention
        self.channels = channels
        self.conditional = conditional
        
        if self.conditional:
            linLayer = nn.Linear(latent_dim+class_dim,channels*(4*7))
            self.embed = nn.Embedding(class_dim,class_dim)
        else:
            linLayer = nn.Linear(latent_dim,channels*(4*7))
        self.model = nn.Sequential(
            linLayer,
            Rearrange('b (c h w)->b c h w',c=channels,h=4),
            *conv2DTransposeBlock(channels,channels//2,activation=activation,kernel=kernel,attention=False,
                                  padding=1,stride=(2)),
            *conv2DTransposeBlock(channels//2,channels//4,activation=activation,kernel=kernel,attention = False,
                                  padding=1,stride=(2)),
            *conv2DTransposeBlock(channels//4,channels//8,attention = False,activation=activation,kernel=(4),
                                  padding=1,stride=(2)),
            *conv2DTransposeBlock(channels//8,channels//8,attention = False,activation=activation,kernel=(4),
                                  padding=1,stride=(2)),
            *conv2DTransposeBlock(channels//8,3,normalize=False,attention = False,activation=nn.Tanh(),
                                  kernel=(4),padding=1,stride=(2)))
    def forward(self, x,c=None):
        ## synthesize image of size 128 x 224
        if self.conditional:
            y = self.embed(c)
            x = torch.cat([x,y],dim=1)
        return self.model(x)


class carsCNNDiscriminator(nn.Module):
    def __init__(self, channels, attention=False, activation=False,conditional=False,class_dim=50):
        super(carsCNNDiscriminator, self).__init__()
        self.conditional = conditional
        self.embed = nn.Embedding(class_dim,class_dim)
        self.fc = nn.Sequential(nn.Linear(class_dim+2,1),nn.LeakyReLU(0.2,inplace=True))
        if self.conditional:
            self.lastkernel = (4)
        else:
            self.lastkernel = (4,6)
        self.convBlock = nn.Sequential(
        GaussianNoise(),
        *conv2DBlock(3,channels,kernel=4,padding=1,activation=activation,stride=(2)),
        *conv2DBlock(channels,channels,kernel=4,padding=1,activation=activation,stride=(2)),
        *conv2DBlock(channels,channels*2,padding=1,activation=activation,stride=(2),kernel=(4)),
        *conv2DBlock(channels*2,channels*4,padding=1,activation=activation,attention=False,stride=(2),kernel=(4)),
        *conv2DBlock(channels*4,channels*8,padding=1,attention=False,activation=activation,stride=(2),kernel=(4)),
        *conv2DBlock(channels*8,1,kernel=self.lastkernel,padding=0,specNorm=False,dropout=False,miniBatchNorm=False,
                     activation=activation,stride=(2)),Rearrange('b c h w->b (c h w)'),)


    def forward(self, x,c=None):
        if self.conditional:
            y = self.embed(c)
            x= torch.cat([self.convBlock(x),y],dim=1)
            x= self.fc(x)
            return x
        else:
            return self.convBlock(x)


# Defin the GAN Network
class CNNNetwork():
    def __init__(self, channels, latentDim=100, attention=False, activationG=nn.Mish(inplace=True),
                 activationD=False,conditional=False):
        self.generator = carsCNNGenerator(channels=channels, latent_dim=latentDim,
                                           activation=activationG, attention=attention,conditional=conditional)
        self.discriminator = carsCNNDiscriminator(
            channels=channels//8, attention=attention, activation=activationD,conditional=conditional)
