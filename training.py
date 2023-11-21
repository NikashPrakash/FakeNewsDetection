import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AdamW,DistilBertModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from clustering import *

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
        

def train(model, train, val, epoch):
    model.train()
    for i in range(epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        for _, (x,y) in train:
            pred = model(x)
    pass

if __name__ == "main":
    x_train, y_train, x_test, y_test = cluster_then_label()
    # concatenate x_train and y_train to pass into DataLoader
    x_train, y_train, x_test, y_test = x_train.tensor(), y_train.tensor(), x_test.tensor(), y_test.tensor()
    train = DataLoader(list(zip(x_train,y_train)), batch_size=64)
    test = DataLoader(list(zip(x_test,y_test)), batch_size=64)
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = StanceDetect(bert_model, 2)
    train(model,train)