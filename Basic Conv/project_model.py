import torch

def next_size(n,kernel,stride,padding=0):
    return int((n-kernel+2*padding)/stride)+1

class XRayConv2(torch.nn.Module):
    def __init__(self,initial_size,name,lr=0.001,settings = [(3,2,0,32),(3,2,0,64),(3,2,0,64)]):
        super().__init__()
        
        self.name = name
        
        size = [initial_size]
        layers = []
        for i, (kernel, stride, dropout, out_channels) in enumerate(settings):
            size.append(next_size(size[-1],kernel,stride))
            if i==0:
                in_channels = 1
            else:
                in_channels = settings[i-1][-1]
            layers.append(torch.nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=(kernel,kernel),
                                          stride=(stride,stride)))
            if dropout>0:
                layers.append(torch.nn.Dropout2d(dropout,inplace=True))
            layers.append(torch.nn.PReLU())

        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(settings[-1][-1]*(size[-1]**2),2))
        layers.append(torch.nn.LogSigmoid())
        
        self.network = torch.nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        
    def forward(self,x):
        x = self.network(x)
        return x
    
    def save(self):
        torch.save(self.state_dict(),f"./{self.name}_model.pt")
        torch.save(self.optimizer.state_dict(),f"./{self.name}_optim.pt")
        
    def load(self,name):
        self.load_state_dict(torch.load(name+"_model.pt"))
        self.optimizer.load_state_dict(torch.load(name+"_optim.pt"))
        return self