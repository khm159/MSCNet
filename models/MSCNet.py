import torch.nn as nn 

class MSCNet(nn.Module):
    def __init__(self):
        super.__init__()
    
    def forward(self, input):
        # Spatial Clues Exploitation 