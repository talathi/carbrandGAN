import torch.nn as nn
import torch.nn.functional as F
import torch

from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange

from helperFn import GaussianNoise, SpectralNorm, MiniBatchStdDev


class selfAttention(nn.Module):
    """ Self attention Layer"""

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


# Define a Scaled Conv Block Module-- Scaling factor is input to the block


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

# Define a progressively growing Generator Module


class celebCNNGenerator(nn.Module):
    def __init__(self, channels=512, latent_dim=100, kernel=(5, 4), attention=False, activation=nn.Mish(inplace=True),
                 layerCount=4):
        super(celebCNNGenerator, self).__init__()
        self.attention = attention
        self.channels = channels

        kernel = 4
        pad = 1
        stride = 2

        if layerCount < 0:
            kernel = 1
            stride = 1
            pad = 0

        # Common First Layer
        sequential = nn.Sequential(nn.Linear(latent_dim, channels*(4*4)),
                                   Rearrange('b (c h w)->b c h w', c=channels, h=4))

        # Intermediate Layers, depending on the size of image to synthesize
        moduleList = []
        if layerCount > 1:
            for i in range(layerCount):
                if i == layerCount-1:
                    outchannel = channels//2**i
                else:
                    outchannel = channels//2**(i+1)
                moduleList.append(conv2DTransposeBlock(
                    channels//2**i, outchannel, activation=activation, kernel=kernel, attention=False))

        elif layerCount == 1:
            i = 1
            moduleList.append(conv2DTransposeBlock(
                channels, channels//2**i, activation=activation, kernel=kernel, attention=False))

        else:
            i = 0

        if len(moduleList) > 0:
            for modules in moduleList:
                for module in modules:
                    sequential.add_module(str(len(sequential)), module)

        # Final Layer
        finalModule = conv2DTransposeBlock(
            channels//2**i, 3, padding=pad, normalize=False, activation=nn.Tanh(), kernel=kernel, stride=stride)

        for module in finalModule:
            sequential.add_module(str(len(sequential)+1), module)
        self.model = sequential

    def forward(self, x):
        return self.model(x)


class celebCNNDiscriminator(nn.Module):
    def __init__(self, channels, attention=False, activation=False, layerCount=3):
        super(celebCNNDiscriminator, self).__init__()
        self.attention = attention

        # First Layer
        sequential = nn.Sequential(GaussianNoise(),
                                   *conv2DBlock(3, channels, 4, 2,
                                                1, activation=activation))

        # Intermediate Layers
        moduleList = []
        firstConv = conv2DBlock(channels, channels, 4,
                                2, 1, activation=activation)
        if layerCount >= 0:
            for module in firstConv:
                sequential.add_module(str(len(sequential)), module)

        if layerCount > 0:
            for i in range(layerCount):
                moduleList.append(conv2DBlock(
                    channels*2**i, channels*2**(i+1), 4, 2, 1, activation=activation))
        else:
            i = -1

        if layerCount < -1:
            pad = 1
        else:
            pad = 0

        if len(moduleList) > 0:
            for modules in moduleList:
                for module in modules:
                    sequential.add_module(str(len(sequential)), module)

        # Common Final Layer
        lastConv = conv2DBlock(channels*2**(i+1), 1, 4, 2, pad, specNorm=False,
                               dropout=False, miniBatchNorm=False, activation=activation)

        for module in lastConv:
            sequential.add_module(str(len(sequential)), module)

        sequential.add_module(str(len(sequential)),
                              Rearrange('b c h w->b (c h w)'))

        self.convBlock = sequential


    def forward(self, x):
        return self.convBlock(x)


# Defin the GAN Network
class CNNNetwork():
    def __init__(self, channels, latentDim=100, layerCount=4, attention=False, activationG=nn.Mish(inplace=True),
                 activationD=False):
        self.generator = celebCNNGenerator(channels=channels, latent_dim=latentDim,
                                           activation=activationG, attention=attention,  layerCount=layerCount)
        self.discriminator = celebCNNDiscriminator(
            channels=channels//8, attention=attention, activation=activationD, layerCount=layerCount-1)
