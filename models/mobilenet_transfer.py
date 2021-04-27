import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

import torchvision.models as models

torch.backends.cudnn.deterministic = True
#General structure of code thanks to Sebastain Rascha (@rasbt)

torch.manual_seed(0)

#Dummy data paths SDH
TRAIN_POS_PATH_CSV ='../data/COVID-19_Radiography_Dataset/COVIDtrain.xlsx'
TRAIN_NEG_PATH_CSV = '../data/COVID-19_Radiography_Dataset/Normaltrain.xlsx'
VALID_POS_PATH_CSV = '../data/COVID-19_Radiography_Dataset/COVIDvalid.xlsx'
VALID_NEG_PATH_CSV = '../data/COVID-19_Radiography_Dataset/Normalvalid.xlsx'
TRAIN_POS_PATH ='../data/COVID-19_Radiography_Dataset/COVID'
TRAIN_NEG_PATH ='../data/COVID-19_Radiography_Dataset/Normal'
BATCH_SIZE = 128

class COVIDdata(Dataset):
    """Custom Dataset for loading MORPH face images"""

    def __init__(self,
                 pos_path, pos_dir,  neg_path, neg_dir, transform=None):

        xl = pd.ExcelFile(pos_path)
        self.df = xl.parse("Sheet1")
        self.pos_num = self.df.shape[0]
        xl = pd.ExcelFile(neg_path)
        self.df_neg = xl.parse("Sheet1")
        self.df = pd.concat([self.df, self.df_neg]).reset_index(drop=True)
        if pos_path == TRAIN_POS_PATH_CSV:
            self.df = self.df.drop(self.df.index[[5000,9000]])
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir

        self.img_names = self.df['FILE NAME'].values
        self.transform = transform

    def __getitem__(self, index):
        #pos = 1, neg = 0
        label = 1 if index < self.pos_num else 0
        
        if label:
            img = Image.open(os.path.join(self.pos_dir,
                                      self.img_names[index]+'.png'))
        else:
#capataize NORMAL to Normal
            img = Image.open(os.path.join(self.neg_dir,
                                      self.img_names[index].capitalize()+'.png'))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.df.shape[0]

custom_transform = transforms.Compose([ transforms.Resize((128, 128)),
                                       transforms.ToTensor()])

train_dataset = COVIDdata(pos_path=TRAIN_POS_PATH_CSV,
                              pos_dir=TRAIN_POS_PATH,
                              neg_path = TRAIN_NEG_PATH_CSV,
                              neg_dir = TRAIN_NEG_PATH,
                              transform=custom_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size = 128, shuffle=1)

valid_dataset = COVIDdata(pos_path=VALID_POS_PATH_CSV,
                              pos_dir=TRAIN_POS_PATH,
                              neg_path = VALID_NEG_PATH_CSV,
                              neg_dir = TRAIN_NEG_PATH,
                              transform=custom_transform)

valid_loader = DataLoader(dataset=valid_dataset, batch_size = BATCH_SIZE, shuffle=1)

#were going to stick to mobilenetv2, similar acc as v3
model = models.mobilenet_v2(pretrained=False)
#add softmax and change output to 2
model.classifier[1] = torch.nn.Linear(1280,2, bias=1)
model.classifier = nn.Sequential(*list(model.classifier) + [nn.Softmax(1)])
#change input sizes
model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

#now lets load in our pre trained weights:
model.load_state_dict(torch.load('./models/pretrained.pt'))

#freese params:
#for param in model.features.parameters():
#    param.requires_grad=False

#only re train last few layers(modify range of loop)
#model.classifier[1].requires_grad=True
#for i in range(16,19):
#    model.features[i].requires_grad=True


optimizer = torch.optim.Adam(model.parameters(), lr=.005)

def compute_mae_and_mse(model, data_loader):
    mae, mse, num_examples = 0., 0., 0
    correct, alpha, beta = 0., 0., 0.
    for i, (features, targets) in enumerate(data_loader):

        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
        #accurcy
        for i in range(targets.size(0)):
            if targets[i] == predicted_labels[i]:
                correct += 1
            elif predicted_labels[i] == 1:
                alpha += 1
            elif predicted_labels[i] == 0:
                beta += 1


    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    false_pos = alpha / num_examples
    false_neg = beta / num_examples
    acc = correct / num_examples
    return mae, mse, false_pos, false_neg, acc*100

best_mae, best_rmse, best_epoch, num_epochs = 999, 999, -2, 10

for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        probas = model(features)
#same as CE loss, since softmax is output layer
        cost = F.nll_loss(torch.log(probas), targets)
        optimizer.zero_grad()
        
        cost.backward()

        optimizer.step()

        if not batch_idx % 50:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                 % (epoch+1, num_epochs, batch_idx,
                     len(train_dataset)//BATCH_SIZE, cost))
            print(s)
            with open("outputtransfer.txt", "a") as f:
                f.write('%s\n' % s)
    
    with torch.set_grad_enabled(False):
        valid_mae, valid_mse, v_false_pos, v_false_neg, v_acc = compute_mae_and_mse(model, valid_loader)
    if valid_mae < best_mae:
        best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
        ########## SAVE MODEL #############
        #torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


    s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d | acc: %2.2f %% |alpha/beta %.2f/%.2f ,' % (
        valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch, v_acc, v_false_pos, v_false_neg)
    print(s)
    with open("outputtransfer.txt", "a") as f:
        f.write('%s\n' % s)

