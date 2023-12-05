#training.py

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AdamW,DistilBertModel,DistilBertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import clustering
from model import StanceDetect, BinaryDataset
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,GridSearchCV
# from skorch import NeuralNetClassifier
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

def predictions(logits):
    """Determine predicted class index given a tensor of logits.

    Example: Given tensor([[0.2, -0.8], [-0.9, -3.1], [0.5, 2.3]]), return tensor([0, 0, 1])

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    pred = logits.max(1)[1]
    return pred

def cross_val(bert_model, x, y, device): 
    # Use a logarithmic scale to sample weight decays
    params = {
        'lr': [5e-6,1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'max_epochs': [5, 10, 20, 30],
        'module__drop_rate': [0.1,0.3,0.5,0.6,0.75,0.9],
        'optimizer__weight_decay':[0, 1e-6, 1e-4, 1e-2]
    }
    
    _vals = np.meshgrid(params['lr'], params['max_epochs'],params['module__drop_rate'],params['optimizer__weight_decay'])
    param_set = np.array([_vals[0].ravel(), _vals[1].ravel(),_vals[2].ravel(), _vals[3].ravel()]).T
    skf = StratifiedKFold(shuffle=True,random_state=42)
    
    best_performance = float('-inf')
    
    for lr, max_epoch, drop_rate, w_d in param_set:
        model = StanceDetect(bert_model,2,drop_rate).to(device)
        criterion=nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(model.parameters(),weight_decay=w_d,lr=lr)

        performances = []
        inpt = x['input_ids']
        
        for tr_idx, val_idx in skf.split(inpt,y):
            tr_x,val_x = x[tr_idx].to(device), x[val_idx].to(device)
            tr_y,val_y = y[tr_idx].to(device), y[val_idx].to(device)
            # Currenty x is a dict of inpt_ids, atten_mask
            # Need to be torch.utils.data.dataset.Subset 
            # 
            train_loader = DataLoader(torch.tensor(list(zip(tr_x,tr_y))),batch_size=64)
            val_loader = DataLoader(torch.tensor(list(zip(val_x,val_y))),batch_size=64,collate_fn=basic_collate_fn)
            
            train(model,train_loader,max_epoch,optimizer,criterion)

            model.eval()
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    pred = model(v_x)
                    pred = predictions(pred)
                    correct += torch.sum(torch.eq(pred,v_y).type(torch.IntTensor))
                    total += v_y.size(0)
                performances.append(correct/total)
        perf = np.mean(performances)
        if perf > best_performance:
            best_performance = perf
            params = {'lr':lr,'max_epochs':max_epoch, 'module__drop_rate':drop_rate, 'optimizer__weight_decay':w_d}
    return params, best_performance

def train(model, train, epoch, optimizer,criterion): 

    # train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    # num_itr = 0
    # best_model, best_accuracy = None, 0

    for i in range(epoch):
        model.train()
        for batch in train:
            # num_itr += 1
            #tokenize here before passing into the model - not needed
            optimizer.zero_grad()
            y = batch['labels']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pred = model(input_ids,attention_mask)
            loss = criterion(pred, y)
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
                

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
         for batch in test_loader:
            y = batch['labels']
            y = F.one_hot(y, num_classes=2).float()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            output = model(input_ids,attention_mask).argmax(dim=1)
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

def process_input():
    """ Change input data to be in format [(input_ids, attention_mask)] for training with DistilBert 
    """
    pass

def basic_collate_fn(batch):
    """Collate function for basic setting."""

    inputs = []
    outputs = []

    ############################## START OF YOUR CODE ##############################
    input_ten = []
    for data in batch:
        outputs += [torch.tensor(data['pos_ids'])]
        input_ten += [torch.tensor(data['input_ids'])]
    input_ten = pad_sequence(input_ten, batch_first=True)
    outputs = pad_sequence(outputs, batch_first=True)
    
    atten_masks = torch.zeros_like(input_ten)
    atten_masks[input_ten != 0] = 1
    inputs = {'input_ids': input_ten, 'attention_mask': atten_masks}
    ############################### END OF YOUR CODE ###############################

    return inputs, outputs

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, y_train, x_test, y_test = clustering.cluster_then_label()
    print("done")
    y_test = y_test.to(device)
    
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    x_tokenized_inputs = tokenizer(x_train.tolist(), padding=True, truncation=True, max_length=300, return_tensors='pt')
    test_inputs = tokenizer(x_test.tolist(), padding=True, truncation=True, max_length=300, return_tensors='pt').to(device)

    # patience = 5
    # curr_count_to_patience = 0
    # global_min_loss = stats[0][1]
        
    # while curr_count_to_patience < patience:

    # param, cv_perf = cross_val(bert_model, x_tokenized_inputs, y_train, device)
    # print(f"cv_performance: {cv_perf}")


    # model = StanceDetect(bert_model, 2, param["module__drop_rate"]).to(device)

    model = StanceDetect(bert_model, 2, 0.75).to(device)
    
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=param["lr"] , weight_decay=param["optimizer__weight_decay"])  

    optimizer = torch.optim.Adam(params=model.parameters(),  lr=5e-5, weight_decay=1e-5)  


    x_tokenized_inputs['labels'] = y_train
    full_set = BinaryDataset(x_tokenized_inputs.to(device))
    full_set = DataLoader(full_set, batch_size=64)

    criterion = nn.CrossEntropyLoss()
    
    train(model,full_set, 25, optimizer, criterion)
    # train(bert_model, full_set, 25, optimizer, criterion)
    test_inputs['labels'] = y_test
    test_inputs = BinaryDataset(test_inputs.to(device))
    test_inputs = DataLoader(test_inputs,batch_size=64)
    accuracy = test(model, test_inputs, device)

    print(accuracy)

if __name__ == "__main__":
    print("hey")
    main()
    print("done")



