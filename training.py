import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AdamW,DistilBertModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
form data_process import *

class StanceDetect(nn.Module):
    def __init__(self, distilBert: DistilBertModel, num_pos: int, hidden_dims: list(int)):
        """Initialize the stance-detection model with DistilBert and Linear layers

        Args:
            distilBert (DistilBertModel): distilBert model block
            num_pos (int): number of logits/classes
            hidden_dim list(int): number of hidden dimensions in linear layers
        """
        super().__init__()
        
        self.distilbert = distilBert
        self.fc = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.classifier = nn.Linear(hidden_dims[1],num_pos)
    #     __init_weights__(hidden_dims)
        
    # def __init_weights__(self, hidden_dims: list(int)):
    #     # torch.seed(42)
    #     torch.random.normal(self.fc._weights,)
    def forward():
        pass
        

if __name__ == "main":
    
    train_lab, train_unlab, test = process()
    