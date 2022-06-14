import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 6),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)
            out = self.layers(x)
            return out