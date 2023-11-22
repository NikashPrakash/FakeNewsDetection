#training.py

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AdamW,DistilBertModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import clustering
from model import StanceDetect
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from skorch import NeuralNetClassifier
import gensim.downloader
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

# def predictions(logits):
#     """Determine predicted class index given a tensor of logits.

#     Example: Given tensor([[0.2, -0.8], [-0.9, -3.1], [0.5, 2.3]]), return tensor([0, 0, 1])

#     Returns:
#         the predicted class output as a PyTorch Tensor
#     """
#     pred = logits.max(1)[1]
#     return pred

def cross_val(x, y): 
    # Use a logarithmic scale to sample weight decays
    params = {
        'lr': [5e-6,1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'max_epochs': [5, 10, 20, 30],
        'module__drop_rate': [0.1,0.3,0.5,0.6,0.75,0.9],
        'optimizer__weight_decay':[0, 1e-6, 1e-4, 1e-2]
    }
    model =  NeuralNetClassifier(
        StanceDetect,
        criterion=nn.CrossEntropyLoss(),
        lr=0.01,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
    )
    gs = GridSearchCV(model, params, cv=5, scoring='accuracy',refit=False)
    gs.fit(x,y)
    # best_lr = gs.best_params_['lr']
    # best_wd = gs.best_params_['wd']
    # best_params = gs.best_params_
    # best_model = gs.best_estimator_

    return gs.best_params_

def train(model, train, epoch, optimizer): 

    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0
    loss_fn = nn.CrossEntropyLoss()    

    for i in range(epoch):
        model.train()
        for x,y in train:
            num_itr += 1
            pred = model(x)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
                
            # if num_itr % collect_cycle == 0:  # Data collection cycle
            #     train_loss.append(loss.item())
            #     train_loss_ind.append(num_itr)
            # if verbose:
            #     print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
            #         epoch + 1,
            #         num_itr,
            #         loss.item()
            #         ))
            
            # model.eval()
            # val_epoch_loss = 0.0
            # val_correct = 0
            # val_total = 0
            # with torch.nograd():
            #     for val_x, val_y in val_loader:
                

def test(model, test_loader):
    model.eval()
    corect = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            #X.cuda()
            #y.cuda()
            output = model(X)#.cuda()
            # pred = predictions(output.data)
            correct += torch.sum(torch.eq(output,y).type(torch.IntTensor))
            total += y.size(dim=0)

    return correct / total



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
    # print(torch.cuda.is_available())
    print("aa")
    x_train, y_train, x_test, y_test = clustering.cluster_then_label()
    print("bb")
    # test = DataLoader(list(zip(x_test,y_test)), batch_size=64)
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    print("cc")
    # patience = 5
    # curr_count_to_patience = 0
    # global_min_loss = stats[0][1]
        
    # while curr_count_to_patience < patience:

    param = cross_val(x_train, y_train)
    print("dd")


    model = StanceDetect(bert_model, 2, param["module__drop_rate"])
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=param["lr"] , weight_decay=param["optimizer__weight_decay"])  


    train(model,torch.tensor(list(zip(x_train,y_train))), param['max_epochs'], optimizer)

    accuracy = test(model, torch.tensor(list(zip(x_test,y_test))))

    print(accuracy)

if __name__ == "__main__":
    print('hey')
    # glove = gensim.downloader.load('glove-wiki-gigaword-200')
    main()

