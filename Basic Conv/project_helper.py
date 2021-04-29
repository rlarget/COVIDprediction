import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torchxrayvision as xrv

class CovidDataset(Dataset):
    def __init__(self,
                 dataset_type,
                 path = ["..","..","Data","Dataset 1"],
                 split = {"train":list(range(7)),"valid":[7],"test":[8,9]},
                 transform = None):
        homedir = os.path.join(*path)
        filelist = []
        statuslist = []
        match_length = sum([len(matches) for matches in split.values()])
        i = 0
        for j,folder in enumerate(os.listdir(homedir)):
            for filename in os.listdir(os.path.join(homedir,folder)):
                if i % match_length in split[dataset_type]:
                    filelist.append(os.path.join(homedir,folder,filename))
                    statuslist.append(j)
                i += 1

        self.filenames = pd.Series(filelist)
        self.status = pd.Series(statuslist)
        self.transform = transform
        
    def __getitem__(self,index):
        img = Image.open(self.filenames[index])
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.status[index]
        
        return img, label

    def __len__(self):
        return self.status.shape[0]

class EnforceGrayscale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,img):
        img = img[:3,:,:]
        if (img.shape[-3] > 1):
            return transforms.functional.rgb_to_grayscale(img,num_output_channels=1)
        return img

## Current transform
xrv_transform = transforms.Compose([transforms.ToTensor(),
                                          EnforceGrayscale(),
                                          xrv.datasets.XRayCenterCrop(),
                                          xrv.datasets.XRayResizer(224)])


## old unused transform ##
#xray_transform = transforms.Compose([transforms.ToTensor(),
#                                     EnforceGrayscale(),
#                                     transforms.Resize((1000,1000)),
#                                     transforms.RandomCrop(700)])

def get_loader(dataset_type,batch_size=32,transform = None):
    
    if transform == None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        EnforceGrayscale()])
    
    ds = CovidDataset(dataset_type=dataset_type,
                         transform = transform)
    
    if dataset_type == "train":
        return DataLoader(dataset = ds,
                          batch_size = batch_size,
                          drop_last = True,
                          shuffle = True,
                          num_workers = 0)
    else:
        return DataLoader(dataset = ds,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 0)
