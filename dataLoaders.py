import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets


HOME = os.environ['HOME']


## celeb data loader (from torchvision,datasets)
class celebADataloader():
    def __init__(self, root='%s/Work/DataSets' % HOME, split='train',batchSize=32, workers=8, shuffle=True,imgSize=None):
        if imgSize is None:
            imgSize = (157,128)
        else:
            imgSize = (imgSize,imgSize)
        self.transform = transforms.Compose([transforms.Resize(size=imgSize), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5063, 0.4258, 0.3831],
                                                                 std=[0.3107, 0.2904, 0.2897], inplace=True)])
        self.root = root
        self.batchSize = batchSize
        self.workers = workers
        self.shuffle = shuffle
        self.split = split

    def getDataloader(self):
        celebA = datasets.CelebA(
            root=self.root, split=self.split, download=False, transform=self.transform)
        self.dataloader = DataLoader(
            celebA, batch_size=self.batchSize, num_workers=self.workers, shuffle=self.shuffle, 
            drop_last=True,pin_memory=True)

## celebHQ data loader
class celebaHQDataloader():
    def __init__(self,root='%s/Work/DataSets/celeba_hq256' % HOME, batchSize=32,workers=8,shuffle=True,imgSize=128):
        self.imgSize = (imgSize,imgSize)
        self.transform = transforms.Compose([transforms.Resize((imgSize, imgSize)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)])
        self.root = root
        self.batchSize = batchSize
        self.workers = workers
        self.shuffle = shuffle

    
    def getDataloader(self):
        dataset = datasets.ImageFolder(self.root, self.transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batchSize, shuffle=self.shuffle, num_workers=self.workers,drop_last=True)



class IgnoreLabelDataset(torch.utils.data.Dataset):
    # simple dataset class to read data without labels
    # wrapper to torchvision.dataset objects

    def __init__(self, orig):
        super(IgnoreLabelDataset, self).__init__()
        self.orig = orig
        # print (self.orig.shape)

    def __getitem__(self, index):
        if self.orig[index][0].shape[0] == 1:
            return self.orig[index][0].repeat(3, 1, 1)
        else:
            return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def getStats(dataloader):
    # get mean and stddev of data
    # useful for new data and then to be used for normalizing the dataset through transforms
    mean = torch.empty(3)
    stdDev = torch.empty(3)
    for batch, (imgs, _) in enumerate(dataloader):
        mean += rearrange(imgs, 'b c h w->c (b h w)').mean(axis=1)
        stdDev += (rearrange(imgs, 'b c h w->c (b h w)')**2).mean(axis=1)
    return mean/(batch+1), stdDev/(batch+1)

