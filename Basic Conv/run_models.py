import torch

from torchvision import transforms

import torchxrayvision as xrv

from class_helpers.helper_train import train_model
from project_model import XRayConv2
from project_helper import get_loader, EnforceGrayscale

NUM_EPOCHS = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

initial_size = 224

models = [XRayConv2(initial_size,"conv1"),
          XRayConv2(initial_size,"conv2",settings=[(5,4,0,32),(3,3,0,64),(3,3,0,64)]),
          XRayConv2(initial_size,"conv3",settings=[(5,4,0.5,32),(3,3,0.5,64),(3,3,0.5,64)]),
          XRayConv2(initial_size,"conv4",settings=[(5,4,0.3,32),(3,3,0.1,64),(3,3,0,64)]),
          XRayConv2(initial_size,"conv5",settings=[(5,4,0.1,32),(3,3,0.1,64),(3,3,0.1,64)])]


xrv_transform = transforms.Compose([transforms.ToTensor(),
                                          EnforceGrayscale(),
                                          xrv.datasets.XRayCenterCrop(),
                                          xrv.datasets.XRayResizer(initial_size)])

train_xrv = get_loader("train",transform=xrv_transform)

valid_xrv = get_loader("valid",transform=xrv_transform)

test_xrv = get_loader("test",transform=xrv_transform)

for model in models:
    _, _, _ = train_model(model=model,
                                                                  num_epochs=NUM_EPOCHS,
                                                                  train_loader=train_xrv,
                                                                  valid_loader=valid_xrv,
                                                                  test_loader=test_xrv,
                                                                  optimizer=model.optimizer,
                                                                  device=DEVICE)
    model.save()
