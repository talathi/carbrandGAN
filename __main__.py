from dataLoaders import celebaHQDataloader
from models import CNNNetwork
from trainer import train
import pickle
from torchsummary import summary
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from helperFn import sampleImage

import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")


HOME = os.environ['HOME']

## Default

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DCN Gan on CelebA dataset"
    )
    parser.add_argument(
        "name", type=str, help="Name of the model for storing and loading purposes."
    )

    parser.add_argument(
        "--latentDim", type=int, help="Dimension of latent vector", default=100
    )

    parser.add_argument(
        "--numEpochs", type=int, help="num Training Epochs", default=10
    )

    parser.add_argument(
        "--startEpoch", type=int, help="start Epoch", default=0
    )

    parser.add_argument(
        "--batchSize", type=int, help="batchSize", default=32
    )

    parser.add_argument(
        "--logFreq", type=int, help="logFreq", default=500
    )

    parser.add_argument(
        "--channels", type=int, help="channels", default=512
    )

    parser.add_argument(
        "--imgSize", type=int, help="image size", default=128
    )

    parser.add_argument(
        "--lr", type=float, help="generator learning rate", default=0.0002
    )

    parser.add_argument(
        "--printarch", action="store_true", help="print architecture details"
    )

    parser.add_argument(
        "--root", help="Root path where checkpoint models are saved", type=str, default='%s/Models' % HOME
    )

    parser.add_argument(
        "--Gcheckpoint", help="Trained model generator checkpoint", type=str, default=None
    )

    parser.add_argument(
        "--Dcheckpoint", help="Trained model discriminator checkpoint", type=str, default=None
    )

    parser.add_argument("--activation", action="store_true",
                        help="default to relu")
    
    parser.add_argument("--train", action="store_true", help="run training")

    parser.add_argument("--eval", action="store_true", help="run evaluation")

    parser.add_argument("--l2loss", action="store_true",
                        help="use l2loss; default is BCELogit")

    parser.add_argument("--relativistic", action="store_true",
                        help="use relativistic loss")

    parser.add_argument("--attention", action="store_true",
                        help="add self attention module to generator and discriminator")

    parser.add_argument("--hinge", action="store_true",
                        help="adversarial hinge loss")

    parser.add_argument("--ema", action="store_true",
                        help="apply exponential moving average smoothing")

    parser.add_argument("--relativisticHinge", action="store_true",
                        help="add relativistic hinge loss")

    parser.add_argument("--diffAug", action="store_true",
                        help="add relativistic hinge loss")

    parser.add_argument("--proTrain", action="store_true",
                        help="progressive Training using variable Latent Vector Size")
    
    parser.add_argument("--sim", action="store_true",
                        help="invoke similarity loss to treat mode collapse")

    args = parser.parse_args()
    print(args)

    # set up folders
    if not os.path.isdir(args.root):
        os.mkdir(args.root)

    saveDir = '%s/%s' % (args.root, args.name)
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)

    if args.Gcheckpoint is not None:
        args.Gcheckpoint = saveDir+'/'+args.Gcheckpoint

    if args.Dcheckpoint is not None:
        args.Dcheckpoint = saveDir+'/'+args.Dcheckpoint

    # Get Data and Define Dataloader
    celebTrain = celebaHQDataloader(
        batchSize=args.batchSize, imgSize=args.imgSize)
    celebTrain.getDataloader()

    # print data dimension
    imgs, labels = next(iter(celebTrain.dataloader))
    if len(labels.shape) == 1:
        labels = rearrange(labels, '(c h)->c h', h=1)
    print(imgs.shape, labels.shape, labels.dtype)

    # Get Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    if args.activation:
        activation = nn.Mish(inplace=True)
    else:
        activation = nn.LeakyReLU(0.2,inplace=True)

    layerCount = int(np.log(args.imgSize)/np.log(2))-3


    # define Models
    network = CNNNetwork(args.channels, args.latentDim, layerCount=layerCount, attention=args.attention,
                        activationG=activation, activationD=False)
    
    
    if args.printarch:
        imgSize = (2**(layerCount+3), 2**(layerCount+3))
        print('Discriminator Network...')
        summary(network.discriminator.to(device),
                input_size=(3, imgSize[0], imgSize[1]))
        print('\n\n Generator Network')
        summary(network.generator.to(device), input_size=(args.latentDim,))

    # Define TrainConfig
    losstype = 'BCE'
    if args.l2loss:
        losstype = 'L2'

    trainingConfig = {
        'numEpochs': args.numEpochs,
        'batchSize': args.batchSize,
        'enableTensorboard': False,
        'consoleLogFreq': args.logFreq,
        'saveDir': saveDir,
        'lossType': losstype,
        'lr': args.lr,
        'imgSize': args.imgSize
    }

    if args.train:
        DLoss, GLoss = train(trainingConfig, args.latentDim, network, celebTrain.dataloader, activation, device,
                             Gcheckpoint=args.Gcheckpoint, Dcheckpoint=args.Dcheckpoint,
                             startEpoch=args.startEpoch, init=False,
                             relativistic=args.relativistic, hinge=args.hinge,
                             attention=args.attention, ema=args.ema,
                             relativisticHinge=args.relativisticHinge,
                             layerCount=layerCount, diffAug=args.diffAug, proTrain=args.proTrain,sim=args.sim)

        o = open('%s/trainData.pkl' % saveDir, 'wb')
        pickle.dump([DLoss, GLoss], o)
        o.close()

    if args.eval:
        assert args.Gcheckpoint is not None
        gModel = torch.load(args.Gcheckpoint)
        network.generator.load_state_dict(gModel['model_state_dict'])
        sampleImage(network.generator,
                    trainingConfig['saveDir'], 36, 6, args.latentDim, device, -1, -1)
