import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AdamW,DistilBertModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from clustering import *
from model import StanceDetect
# import os

# def save_checkpoint(model, epoch, checkpoint_dir, stats):
#     """Save a checkpoint file to `checkpoint_dir`."""
#     state = {
#         "epoch": epoch,
#         "state_dict": model.state_dict(),
#         "stats": stats,
#     }

#     filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
#     torch.save(state, filename)    

def train(model, train, epoch, optimizer,collect_cycle=30, verbose=True): 
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0
    loss_fn = nn.CrossEntropyLoss()

    
    for i in range(epoch):
        model.train()
        for _, (x,y) in train:
            num_itr += 1
            pred = model(x)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                ))
    pass


# def early_stopping(stats, curr_count_to_patience, global_min_loss):
#     """Calculate new patience and validation loss.

#     Increment curr_patience by one if new loss is not less than global_min_loss
#     Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

#     Returns: new values of curr_patience and global_min_loss
#     """
#     if stats[-1][1] >= global_min_loss:
#       curr_count_to_patience += 1
#     else:
#       global_min_loss = stats[-1][1]
#       curr_count_to_patience = 0
#     return curr_count_to_patience, global_min_loss


def main():
    x_train, y_train, x_test, y_test = cluster_then_label()
    x_train, y_train, x_test, y_test = x_train.tensor(), y_train.tensor(), x_test.tensor(), y_test.tensor()
    # test = DataLoader(list(zip(x_test,y_test)), batch_size=64)
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = StanceDetect(bert_model, 2)
    # patience = 5
    # curr_count_to_patience = 0
    # global_min_loss = stats[0][1]
    
    optimizer = torch.optim.Adam(model.parameters())

    
    
    # while curr_count_to_patience < patience:
    train(model,torch.tensor(list(zip(x_train,y_train))))
    
    

if __name__ == "main":
    main()