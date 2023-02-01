import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets


HOME = os.environ['HOME']


class carsDataloader():
    def __init__(self,root='%s/Work/DataSets/car_brands/imgs' % HOME, batchSize=32,workers=8,shuffle=True,imgSize=(128,224)):
        ## Mean Image Size is: (175,182)
        ## Channel Mean and Channel stdDev is computed apriori using getStats
        ## Manual inspection of data to remove outlier images and car logo images
        self.transform = transforms.Compose([transforms.Resize(imgSize), transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),transforms.Normalize([0.4847, 0.4756, 0.4723], [0.3085, 0.3054, 0.3102], inplace=True)])
        self.root = root
        self.batchSize = batchSize
        self.workers = workers
        self.shuffle = shuffle

    
    def getDataloader(self):
        dataset = datasets.ImageFolder(self.root, self.transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batchSize, shuffle=self.shuffle, num_workers=self.workers,drop_last=True)


def getStats(dataloader):
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0
    
    for data in dataloader:
        images = data[0]

        for channel in range(3):
            channel_data = images[:, channel, :, :]

            pixel_sum[channel] += channel_data.view(channel_data.size(0), -1).sum()
            pixel_squared_sum[channel] += (channel_data.view(channel_data.size(0), -1) ** 2).sum()
        num_pixels += images.size(0)*images.size(2)*images.size(3)

    mean = pixel_sum / num_pixels
    stdDev = torch.sqrt(pixel_squared_sum / num_pixels - mean ** 2)

    return mean, stdDev