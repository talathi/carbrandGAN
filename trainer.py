import sys
import torch
import numpy as np
import torch.nn as nn
import time
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from helperFn import gaussianLatent, initWeigthts, sampleImage, pdmpLoss, EMA, compute_gradient_penalty, transferWeights
from diffaug import DiffAugment
from models import CNNNetwork
policy = 'color,translation,cutout'


def allowEMA(generator, discriminator, smoothingFactor=0.9):
    movingAverageG = EMA(smoothingFactor)
    movingAverageD = EMA(smoothingFactor)
    movingAverageD.register(discriminator)
    movingAverageG.register(generator)
    return movingAverageD, movingAverageG

# Initial set up for GAN Training


def initialSetup(network, device, trainConfig, Gcheckpoint, Dcheckpoint, ema=False):
    discriminator = network.discriminator.train().to(device)
    generator = network.generator.train().to(device)
    movingAverageD = None
    movingAverageG = None

    if ema:
        movingAverageD, movingAverageG = allowEMA(generator, discriminator)

    discriminator.apply(initWeigthts)
    generator.apply(initWeigthts)

    optimizerG = torch.optim.Adam(
        generator.parameters(), lr=trainConfig['lr'], betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(
        discriminator.parameters(), lr=4*trainConfig['lr'], betas=(0.5, 0.999))

    if Gcheckpoint:
        gModel = torch.load(Gcheckpoint)
        generator.load_state_dict(gModel['model_state_dict'])
        optimizerG.load_state_dict(gModel['optimizer_state_dict'])

    if Dcheckpoint:
        dModel = torch.load(Dcheckpoint)
        discriminator.load_state_dict(dModel['model_state_dict'])
        optimizerD.load_state_dict(dModel['optimizer_state_dict'])

    return generator, discriminator, optimizerG, optimizerD, movingAverageG, movingAverageD


def updateNetwork(generator, discriminator, channels, latentDim, trainConfig, device, ema=False,
                  activation=nn.Mish(inplace=True),conditional=False):
    movingAverageD = None
    movingAverageG = None
    network = CNNNetwork(channels, latentDim,  attention=False,
                         activationG=activation, activationD=False,conditional=conditional)
    generator = transferWeights(generator, network.generator, freeze=False)
    discriminator = transferWeights(
        discriminator, network.discriminator, freeze=False)

    generator = generator.train().to(device)
    discriminator = discriminator.train().to(device)

    optimizerG = torch.optim.Adam(
        generator.parameters(), lr=trainConfig['lr'], betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(
        discriminator.parameters(), lr=4*trainConfig['lr'], betas=(0.5, 0.999))

    if ema:
        movingAverageD, movingAverageG = allowEMA(generator, discriminator)

    return generator, discriminator, optimizerG, optimizerD, movingAverageG, movingAverageD


def train(trainConfig, channels, latentDim, network, dataloader, device,
          Gcheckpoint=None, Dcheckpoint=None, startEpoch=0, hinge=False,
          ema=False, sim=False, diffAug=False, activation=nn.Mish(inplace=True), 
          proTrain=False,conditional=False,class_dim=50):

    generator, discriminator, optimizerG, optimizerD, movingAverageG, movingAverageD = initialSetup(
        network, device, trainConfig, Gcheckpoint, Dcheckpoint, ema=ema)

    # define midLayer getter
    layerofInterest = 10
    return_layers = {'convBlock.%d' % layerofInterest: 'LeakyReLU'}
    mid_getter = MidGetter(
        discriminator, return_layers=return_layers, keep_output=False)

    if trainConfig['lossType'] == 'BCE':
        GanLoss = nn.BCEWithLogitsLoss()
    if trainConfig['lossType'] == 'L2':
        GanLoss = nn.MSELoss()

    realLabels = 0.9*torch.ones(trainConfig['batchSize'], 1).to(device)
    fakeLabels = torch.zeros(trainConfig['batchSize'], 1).to(device)

    startTime = time.time()
    DLoss = []
    GLoss = []

    for epoch in range(startEpoch, trainConfig['numEpochs']):
        if proTrain:
            if epoch % 100 == 0 and epoch != 0:
                latentDim = latentDim+100
                if latentDim < 100:
                    return DLoss, GLoss

                generator, discriminator, optimizerG, optimizerD, movingAverageG, movingAverageD = updateNetwork(
                    generator, discriminator, channels, latentDim, trainConfig, device, ema=ema, 
                    activation=activation,conditional=conditional)

        DBLoss = []
        GBLoss = []
        for batchID, (imgs, realclass) in enumerate(dataloader):
            # Train Discrinimator:

            imgs = imgs.to(device)
            if diffAug:
                imgs = DiffAugment(imgs, policy)
            realclass = realclass.to(device)

            for _ in range(2):
                imgLabels = discriminator(imgs,realclass)

                z = gaussianLatent(trainConfig['batchSize'], latentDim, device)
                fakeclass = torch.randint(0,class_dim,(trainConfig['batchSize'],)).to(device)
                fakeimgs = generator(z,fakeclass)
                if diffAug:
                    fakeimgs = DiffAugment(fakeimgs, policy)
                fakeimgLabels = discriminator(fakeimgs.detach(),fakeclass)

                if hinge:
                    dLoss = nn.ReLU()(1+fakeimgLabels).mean() + nn.ReLU()(1-imgLabels).mean()
                else:
                    dLoss = GanLoss(imgLabels, realLabels) + \
                        GanLoss(fakeimgLabels, fakeLabels)

                # Add gradient penalty
                gpLoss = compute_gradient_penalty(
                    discriminator, imgs,realclass, fakeimgs, device)
                dLoss = dLoss + gpLoss

                optimizerD.zero_grad()
                dLoss.backward()
                optimizerD.step()

                if ema:
                    movingAverageD.update(discriminator)

            # Train Generator
            z = gaussianLatent(trainConfig['batchSize'], latentDim, device)
            fakeclass = torch.randint(0,class_dim,(trainConfig['batchSize'],)).to(device)
            fakeimgs = generator(z,fakeclass)
            if diffAug:
                fakeimgs = DiffAugment(fakeimgs, policy)

            fakeimgLabels = discriminator(fakeimgs,fakeclass)
            mid_outputs, _ = mid_getter(fakeimgs,fakeclass)
            layers = list(mid_outputs.items())[0][1]

            if sim:
                pdLoss = pdmpLoss(z, layers)
            else:
                pdLoss = torch.zeros(1).to(device).item()

            if hinge:
                adversarialLoss = -fakeimgLabels.mean()
            else:
                adversarialLoss = GanLoss(
                    fakeimgLabels, realLabels)

            gLoss = adversarialLoss + .1*pdLoss
            optimizerG.zero_grad()
            gLoss.backward()
            optimizerG.step()

            if ema:
                movingAverageG.update(generator)

            DBLoss.append(dLoss.item())
            GBLoss.append(gLoss.item())

            if trainConfig['consoleLogFreq'] is not None and batchID % trainConfig['consoleLogFreq'] == 0 and batchID != 0:
                batch = batchID+1/len(dataloader)
                ep = epoch+1

                sampleImage(
                    generator, trainConfig['saveDir'], 36, 6, latentDim, device, ep, batch,class_dim=class_dim)
                print('Gan Training: Time Elapses %.2f s |epoch=%d, batch=%d' %
                      ((time.time()-startTime), ep, batch))
                print('GLoss: %.3f DLoss: %.3f AdversarialLoss: %.3f PDLoss: %.3f' % (
                    gLoss, dLoss, adversarialLoss, pdLoss))
                print('\n')

        DLoss.append(np.mean(DBLoss))
        GLoss.append(np.mean(GBLoss))

        if epoch % 48 == 0:
            GcheckpointName = 'gModel-%d.pt' % epoch
            DcheckpointName = 'dModel-%d.pt' % epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                'loss': GLoss[-1],
            }, trainConfig['saveDir']+'/'+GcheckpointName)

            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                'loss': DLoss[-1],
            }, trainConfig['saveDir']+'/'+DcheckpointName)

    return DLoss, GLoss
