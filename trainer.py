import torch
import numpy as np
import torch.nn as nn
import time
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from helperFn import gaussianLatent, initWeigthts, sampleImage, pdmpLoss, EMA, compute_gradient_penalty
from helperFn import transferWeights
from models import CNNNetwork
from dataLoaders import celebaHQDataloader
from diffaug import DiffAugment
policy ='color,translation'

def allowEMA(generator, discriminator,smoothingFactor=0.9):
    movingAverageG = EMA(smoothingFactor)
    movingAverageD = EMA(smoothingFactor)
    movingAverageD.register(discriminator)
    movingAverageG.register(generator)
    return movingAverageD, movingAverageG

### Initial set up for GAN Training
def initialSetup(network, device, trainConfig, Gcheckpoint, Dcheckpoint, init=True, ema=False):
    discriminator = network.discriminator.train().to(device)
    generator = network.generator.train().to(device)
    movingAverageD = None
    movingAverageG = None

    if ema:
        movingAverageD, movingAverageG = allowEMA(generator,discriminator)   

    discriminator.apply(initWeigthts)
    if init:
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

### Update the setup to modify GAN architecture-- For progressive training

def updateNetwork(generator, discriminator, layerCount, latentDim, trainConfig, activation, device,ema=False,freeze=False):
    movingAverageD = None
    movingAverageG = None  
    network = CNNNetwork(512, latentDim, layerCount=layerCount, attention=False,
                        activationG=activation, activationD=False)
    generator = transferWeights(generator, network.generator,freeze=freeze)
    discriminator = transferWeights(
        discriminator, network.discriminator,freeze=freeze)

    generator = generator.train().to(device)
    discriminator = discriminator.train().to(device)

    optimizerG = torch.optim.Adam(
        generator.parameters(), lr=trainConfig['lr'], betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(
        discriminator.parameters(), lr=4*trainConfig['lr'], betas=(0.5, 0.999))
    
    if ema:
         movingAverageD, movingAverageG = allowEMA(generator,discriminator)

    return generator, discriminator, optimizerG, optimizerD, movingAverageG, movingAverageD


def train(trainConfig, latentDim, network, dataloader, activation, device,
          Gcheckpoint=None, Dcheckpoint=None, startEpoch=0, init=True, relativistic=False, hinge=False,
          attention=False, ema=False, relativisticHinge=False, layerCount=5, diffAug=False,proTrain=False,sim=False):

    generator, discriminator, optimizerG, optimizerD, movingAverageG, movingAverageD = initialSetup(
        network, device, trainConfig, Gcheckpoint, Dcheckpoint, init=init, ema=ema)

    # define midLayer getter
    layerofInterest = 2*layerCount if layerCount >0 else 2
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
    imgSize = trainConfig['imgSize']

    for epoch in range(startEpoch, trainConfig['numEpochs']):
        if proTrain:
            if epoch % 50 == 0 and epoch != 0:
                if latentDim<100:
                    break
            
                latentDim = latentDim-100 
                generator, discriminator, optimizerG, optimizerD,movingAverageG, movingAverageD = updateNetwork(
                    generator, discriminator, layerCount, latentDim, trainConfig, activation, device, ema=ema,freeze=False)

                # celebTrain = celebaHQDataloader(imgSize=imgSize)
                # celebTrain.getDataloader()
                # dataloader = celebTrain.dataloader

        DBLoss = []
        GBLoss = []
        for batchID, (imgs, realclass) in enumerate(dataloader):
            # Train Discrinimator:
            imgs = imgs.to(device)
            if diffAug:
                imgs = DiffAugment(imgs,policy)

            realclass = realclass.to(device)

            for runs in range(1):
                imgLabels = discriminator(imgs)

                z = gaussianLatent(trainConfig['batchSize'], latentDim, device)
                fakeimgs = generator(z)
                if diffAug:
                    fakeimgs = DiffAugment(fakeimgs,policy)

                fakeimgLabels = discriminator(fakeimgs.detach())
    
                if not relativistic and not hinge:
                    dLoss = GanLoss(imgLabels, realLabels) + \
                        GanLoss(fakeimgLabels, fakeLabels)

                if relativistic:
                    dLoss = GanLoss(imgLabels-fakeimgLabels.mean(dim=0, keepdim=True), realLabels) + \
                        GanLoss(fakeimgLabels-imgLabels.mean(dim=0,
                                keepdim=True), fakeLabels)

                if hinge:
                    # min(0,-y) = - max(0,y)
                    dLoss = nn.ReLU()(1+fakeimgLabels).mean() + nn.ReLU()(1-imgLabels).mean()

                if relativisticHinge:
                    dLoss = nn.ReLU()(1+(fakeimgLabels-imgLabels.mean(dim=0, keepdim=True))).mean()
                    + nn.ReLU()(1-(imgLabels-fakeimgLabels.mean(dim=0, keepdim=True))).mean()

                # Add gradient penalty
                gpLoss = compute_gradient_penalty(
                    discriminator, imgs, fakeimgs, device)
                dLoss = dLoss + gpLoss

                optimizerD.zero_grad()
                dLoss.backward()
                optimizerD.step()

                if ema:
                    movingAverageD.update(discriminator)

            # Train Generator
            z = gaussianLatent(trainConfig['batchSize'], latentDim, device)
            fakeclass = torch.randint_like(realclass, 0, 2).to(device)
            fakeimgs = generator(z)
            if diffAug:
                fakeimgs = DiffAugment(fakeimgs,policy)

            fakeimgLabels = discriminator(fakeimgs)

          
            mid_outputs, _ = mid_getter(fakeimgs)
            layers = list(mid_outputs.items())[0][1]
            
            if sim:
                pdLoss = pdmpLoss(z, layers)
            else:
                pdLoss = torch.zeros(1).to(device).item()

            if not relativistic and not hinge:
                adversarialLoss = GanLoss(
                    fakeimgLabels, realLabels)

            if relativistic:
                imgLabels = discriminator(imgs).detach()
                adversarialLoss = GanLoss(
                    fakeimgLabels-imgLabels.mean(dim=0, keepdim=True), realLabels)
            if hinge:
                adversarialLoss = -fakeimgLabels.mean()

            if relativisticHinge:
                adversarialLoss = - \
                    (fakeimgLabels-realLabels.mean(dim=0, keepdim=True)).mean()

            gLoss = adversarialLoss + .1*pdLoss

            optimizerG.zero_grad()
            gLoss.backward()
            optimizerG.step()

            if ema:
                movingAverageG.update(generator)

            DBLoss.append(dLoss.item())
            GBLoss.append(gLoss.item())

            if trainConfig['consoleLogFreq'] is not None and batchID % trainConfig['consoleLogFreq'] == 0 and batchID!=0:
                batch = batchID+1/len(dataloader)
                ep = epoch+1

                sampleImage(
                    generator, trainConfig['saveDir'], 36, 6, latentDim, device, ep, batch)
                print('Gan Training: Time Elapses %.2f s |epoch=%d, batch=%d' %
                      ((time.time()-startTime), ep, batch))
                print('GLoss: %.3f DLoss: %.3f AdversarialLoss: %.3f PDLoss: %.3f' % (
                    gLoss, dLoss, adversarialLoss, pdLoss))
                print('\n')

        DLoss.append(np.mean(DBLoss))
        GLoss.append(np.mean(GBLoss))

        if epoch % 24 == 0:
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
