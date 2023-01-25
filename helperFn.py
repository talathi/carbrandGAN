import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch

from einops import rearrange

from torchvision.models.inception import inception_v3


import numpy as np
from scipy.stats import entropy


HOME = os.environ['HOME']


def transferWeights(refModel, newModel,freeze=True):
    # refModel is a subset of newModel
    newModelStateDict = {k: v for k, v in newModel.state_dict().items()}
    refModelStateDict = {k: v for k, v in refModel.state_dict().items()}
    refKeys = list(refModelStateDict.keys())
    partialStatedict = {k: v for k, v in refModelStateDict.items(
    ) if k in newModelStateDict.keys() and v.shape == newModelStateDict[k].shape}
    missing=newModel.load_state_dict(partialStatedict,strict=False)
    ## freeze pre-trained weigths in new model
    if freeze:
        for name, params in newModel.named_parameters():
            if name in refKeys:
                params.requires_grad=False
        
    return newModel


def computeSimilarity(x):
    simVal = []
    if len(x.shape) == 4:
        x = rearrange(x, 'b c h w-> b (c h w)')
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            s = torch.cosine_similarity(
                torch.tanh(x[i]), torch.tanh(x[j]), dim=0)
            simVal.append(s)
    return torch.tensor(simVal)


def pdmpLoss(z, layer):
    zS = computeSimilarity(z)
    lS = computeSimilarity(layer)
    return torch.mean(lS/zS)


def sphericalInterpolation(t, p0, p1):
    # t is the interpolation parameter in range [0,1]
    # p0 and p1 are the two latent vectors
    if t <= 0:
        return p0
    elif t >= 1:
        return p1
    elif np.allclose(p0, p1):
        return p0

    # Convert p0 and p1 to unit vectors and find the angle between them (omega)
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    return np.sin((1.0 - t) * omega) / np.sin(omega) * p0 + np.sin(t * omega) / np.sin(omega) * p1


class EMA(object):
    # Exponential Moving average implementation for parameter update smoothing
    # theta_t = alpha x theta_t + (1-alpha) x theta_{t-1}
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def getFIDscore(fid, real, fake, device):
    assert real.shape[0] == fake.shape[0]
    fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
    fake = fake.to(torch.uint8).to(device)
    real = (255 * (real.clamp(-1, 1) * 0.5 + 0.5))
    real = real.to(torch.uint8).to(device)
    fid.update(real, real=True)
    fid.update(fake, real=False)
    return fid.compute().item()


def reparameterization(mu, logvar, latent_dim, device):
    std = torch.exp(logvar/2).to(device=device)
    sampled_z = torch.randn((mu.size(0), latent_dim),
                            requires_grad=True, device=device)
    z = sampled_z*std + mu
    return z

def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(10,10)) 
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(num_samples//cols + 1, cols, i + 1)
        #img = img[0].mul(255).add_(0.5).clamp(0,255).to("cpu",torch.uint8).numpy()
        img = img[0].clamp(-1,1).mul(0.5).add_(0.5).mul(255).clamp(0,255).to("cpu",torch.uint8).numpy()
        plt.imshow(rearrange(img,'c h w->h w c'))
        plt.xticks([])
        plt.yticks([])


def sampleImage(model, saveDir, batch, nRow, latentDim, device, epochs, batchesDone):
    imgDir = '%s/images' % saveDir
    if not os.path.isdir(imgDir):
        os.mkdir(imgDir)
    z = gaussianLatent(batch, latentDim, device)
    imgFake = model(z)

    save_image(imgFake, '%s/image_%d-%d.png' %
               (imgDir, epochs, batchesDone), nrow=nRow, normalize=True)


def gaussianLatent(batchSize, latentDim, device):
    return torch.randn((batchSize, latentDim), device=device)


def initWeigthts(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(model.weight.data, 0, 0.02)
    if classname.find("ConvTranspose2d") != -1:
        torch.nn.init.normal_(model.weight.data, 0, 0.02)
    if classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)


def inceptionScore(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(
        pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear',
                     align_corners=True).type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = batch.to(device="cuda")
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1, 1, 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class GaussianNoise(nn.Module):
    # sigma: sigma*pixel value = stdev of added noise from normal distribution
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        # noise is not considered model parameter
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            # scale of noise = stdev of gaussian noise = sigma * pixel value
            # detach so that scale is a constant and not differentiated over
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.expand(
                *x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x

class SpectralNorm(nn.Module):
    def __init__(self, module):
        super(SpectralNorm, self).__init__()
        self.module = nn.utils.spectral_norm(module)

    def forward(self, x):
        return self.module(x)

# goes after batchnorm in generator;


class PixelNorm(nn.Module):
    def __init__(self, alpha=1e-8):
        super(PixelNorm, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        y = (x**2).mean(dim=1, keepdim=True).add(self.alpha).sqrt()
        y = x/y
        return y

# std dev of each feature is calculated and averaged over minibatch
# goes on final layer of discriminator just before activation


class MiniBatchStdDev(nn.Module):
    def __init__(self, alpha=1e-8):
        super(MiniBatchStdDev, self).__init__()
        self.alpha = 1e-8

    def forward(self, x):
        b, _, h, w = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = (y**2).mean(dim=0, keepdim=True).add(self.alpha).sqrt()
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(b, 0, h, w)
        y = torch.cat([x, y], 1)
        return y
