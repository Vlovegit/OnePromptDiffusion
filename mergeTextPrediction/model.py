import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedEmbeddingModel(nn.Module):
    
    def __init__(self, cudaenv):
        super(LearnedEmbeddingModel, self).__init__()
        self.device = cudaenv
        self.linear1 = nn.Linear(1536, 512)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(512, 768)
        self.linear3 = nn.Linear(768, 512)
        self.linear4 = nn.Linear(512, 768)

    def forward(self, cond_embedding, uncond_embedding):
        combined_input = torch.cat([cond_embedding, uncond_embedding], dim=-1).to(self.device)
        # Pass through linear layers
        x = self.linear1(combined_input.to(torch.float32))
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        x = self.gelu(x)
        x = self.linear4(x)       
        return x