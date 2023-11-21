import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel
from torch.nn import CrossEntropyLoss


class StanceDetect(nn.Module):
    def __init__(self, distilBert: DistilBertModel, num_pos: int):
        """Initialize the stance-detection model with DistilBert and Linear layers

        Args:
            distilBert (DistilBertModel): distilBert model block
            num_pos (int): number of logits/classes
            hidden_dim list(int): number of hidden dimensions in linear layers
        """
        super().__init__()
        
        self.distilbert = distilBert
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.3), # For regularization
            nn.Linear(768,num_pos) # classification)
        )
        
    def forward(self,input_ids,attention_mask):
        bert = self.distilbert(input_ids=input_ids,attention_mask=attention_mask)
        output = self.classifier(bert.last_hidden_state)
        return output