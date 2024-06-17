import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as env

class LearnedEmbeddingModel(nn.Module):
    
    def __init__(self):
        super(LearnedEmbeddingModel, self).__init__()
        self.linear1 = nn.Linear(1536, 512)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(512, 768)
        self.linear3 = nn.Linear(768, 512)
        self.linear4 = nn.Linear(512, 768)


    def forward(self,x):
        combined_input = x
        x = self.linear1(combined_input.to(torch.float32))
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        x = self.gelu(x)
        x = self.linear4(x)       
        return x