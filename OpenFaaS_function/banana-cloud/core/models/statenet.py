import torch.nn as nn
import torch.nn.functional as F

class StateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 6, kernel_size = 5, padding = 0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(6,16, kernel_size = 5, padding = 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(16,120, kernel_size = 5, padding = 0),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.AvgPool2d(2,2),

            
            nn.Flatten(),
            nn.Linear(69120,84), #da modificare
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(84, 6),
            #nn.Softmax(dim=1),
        )        
    
    def forward(self, xb):
        out = self.network(xb)
        #print('size tensore out layer: ', out.shape)        
        return out