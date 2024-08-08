from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os 

def get_dataloader(rootDir,transforms,batchSize,shuffle=True):
    # create a dataset and use it to create a data loader
    ds=datasets.ImageFolder(root=rootDir,
                            transform=transforms)
    loader=DataLoader(ds,batch_size=batchSize,
                      shuffle=shuffle,
                      
                      pin_memory=True if config.DEVICE=="cuda" else False)
    return (ds,loader)