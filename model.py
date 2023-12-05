#model.py
import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel
from torch.utils.data import Dataset


class StanceDetect(nn.Module):
    def __init__(self, distilBert: DistilBertModel, num_pos: int, drop_rate: float):
        """Initialize the stance-detection model with DistilBert and Linear layers

        Args:
            distilBert (DistilBertModel): distilBert model block
            num_pos (int): number of logits/classes
            drop_rate float: percent of nodes dropped out between fully connected layers
        """
        super().__init__()
        
        self.distilbert = distilBert
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(drop_rate), # For regularization
            nn.Linear(768,num_pos), # classification)
            # nn.Softmax()
            # nn.Sigmoid()
        )
        self.soft = nn.Softmax(dim=1)
        
    def forward(self,input_ids,attention_mask):
        bert = self.distilbert(input_ids=input_ids,attention_mask=attention_mask)
        output = self.classifier(bert.last_hidden_state)
        output = output.mean(dim=1)
        vals = self.soft(output)
        return vals
    
class BinaryDataset(Dataset):
    """Dataset for POS tagging"""

    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data['labels'])
    

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.data.items()}