#training.py
import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AdamW,DistilBertModel,DistilBertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import itertools

import os
import clustering, utils
from utils import config
from model import StanceDetect, BinaryDataset



def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)

def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    try:
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats

def restore_best(model, best_epoch, checkpoint_dir, cuda=True, force=False, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        if best_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if best_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, []
    else:
        if best_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(best_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    try:
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, stats

def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

def evaluate_epoch(
   # axes,
    tr_loader,
    val_loader,
    model,
    criterion,
    stats,
    te_loader=None,
    include_test=False,
    # update_plot=True,
    # multiclass=False,
):
    """Evaluate the `model` on the train and validation set."""
    model.eval()
    def _get_metrics(loader):
        y_true, y_pred = [], []
        correct, total, true_posit, false_posit, false_neg = 0, 0, 0, 0, 0
        running_loss = []
        for batch in loader:
            with torch.no_grad():
                y = batch['labels']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                output = model(input_ids,attention_mask) #output is softmaxed rn, shape (64, 2)
                predicted = output.argmax(1) #class predictionss
                
                y_true.append(y)
                y_pred.append(predicted)                
                total += y.size(0)
                correct += (predicted == y).sum().item()

                predicted = predicted / y
                true_posit += torch.sum(predicted == 1).item()
                false_posit += torch.sum(predicted == float('inf')).item()
                false_neg += torch.sum(predicted == 0).item()
                running_loss.append(criterion(output, y).item())
                
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        loss = np.mean(running_loss)
        acc = correct / total
        recall = true_posit / (true_posit + false_neg)
        precision = true_posit / (true_posit + false_posit)
        return acc, loss, recall, precision

    train_acc, train_loss, train_rec, train_prec = _get_metrics(tr_loader)
    val_acc, val_loss, val_rec, val_prec = _get_metrics(val_loader)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_rec, 
        val_prec,
        train_acc,
        train_loss,
        train_rec, 
        train_prec
    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    # log_training(epoch, stats)
    # if update_plot:
    #     utils.update_training_plot(axes, epoch, stats)

# def log_training(epoch, stats):
#     """Print the train, validation, test accuracy/loss/auroc.

#     Each epoch in `stats` should have order
#         [val_acc, val_loss, val_auc, train_acc, ...]
#     Test accuracy is optional and will only be logged if stats is length 9.
#     """
#     splits = ["Validation", "Train", "Test"]
#     metrics = ["Accuracy", "Loss", "Recall", "Precision"]
#     print("Epoch {}".format(epoch))
#     for j, split in enumerate(splits):
#         for i, metric in enumerate(metrics):
#             idx = len(metrics) * j + i
#             if idx >= len(stats[-1]):
#                 continue
#             print(f"\t{split} {metric}:{round(stats[-1][idx],4)}")

def search_hyperparams(bert_model, tr_loader, val_loader, device): 
    params = {
        'lr': [5e-6,1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'module__drop_rate': [0.1,0.3,0.5,0.75],
        'optimizer__weight_decay':[0, 1e-6, 1e-4, 1e-2]
    }
    
    _vals = np.meshgrid(params['lr'],params['module__drop_rate'],params['optimizer__weight_decay'])
    param_set = np.array([_vals[0].ravel(), _vals[1].ravel(),_vals[2].ravel()]).T
    
    best_performance = float('inf')
    stats = []
    for lr, drop_rate, w_d in param_set:
        model = StanceDetect(bert_model,2,drop_rate).to(device)
        criterion=nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(model.parameters(),weight_decay=w_d,lr=lr)         

        evaluate_epoch(#axes,
                   tr_loader, val_loader, model, criterion, stats)
        
        train(model, optimizer, criterion, tr_loader, val_loader, stats, 0, True)

        
        evaluate_epoch(#axes,
                   tr_loader, val_loader, model, criterion, stats)
        
        if stats[-1][1] < best_performance:
            best_performance = stats[-1][1]
            params = {'lr':lr, 'module__drop_rate':drop_rate, 'optimizer__weight_decay':w_d}
    return params, best_performance



def train_epoch(model, train, optimizer,criterion):
    model.train()
    for batch in train:
        optimizer.zero_grad()
        y = batch['labels']
        y = F.one_hot(y, num_classes=2).float()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pred = model(input_ids,attention_mask)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()


def train(model, optimizer, criterion, train_loader, val_loader, stats, start_epoch, cv):
    patience = 5
    curr_count_to_patience = 0
    global_min_loss = stats[0][1]

    epoch = start_epoch
    while curr_count_to_patience < patience:
        train_epoch(model,train_loader, optimizer, criterion)
        evaluate_epoch(
            # axes,
            train_loader,
            val_loader,
            model,
            criterion,
            epoch + 1,
            stats,
        )
        # Save model parameters
        if not cv:
            save_checkpoint(model, epoch + 1, config("DistilBert_FineTune.checkpoint"), stats)

        # update early stopping parameters
        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )

        epoch += 1
    return epoch



def early_stopping(stats, curr_count_to_patience, global_min_loss):
    """Calculate new patience and validation loss.

    Increment curr_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0
    Increment curr_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_patience and global_min_loss
    """
    if stats[-1][1] >= global_min_loss:
      curr_count_to_patience += 1
    else:
      global_min_loss = stats[-1][1]
      curr_count_to_patience = 0
    return curr_count_to_patience, global_min_loss



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, y_train, x_test, y_test = clustering.cluster_then_label()
    print("device: ",device)
    print("device: ",device)
    y_test = y_test.to(device)
    
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    tr_idx, val_idx = train_test_split(np.arange(y_train.size(0)),shuffle=True,test_size=0.1,stratify=y_train)
    x_train, x_val = x_train[tr_idx], x_train[val_idx]
    y_train, y_val = y_train[tr_idx], y_train[val_idx]
    
    train_inputs = tokenizer(x_train.tolist(), padding=True, truncation=True, max_length=300, return_tensors='pt')
    val_inputs = tokenizer(x_val.tolist(), padding=True, truncation=True, max_length=300, return_tensors='pt')
    test_inputs = tokenizer(x_test.tolist(), padding=True, truncation=True, max_length=300, return_tensors='pt')
    
    train_inputs['labels'] = y_train
    train_loader = BinaryDataset(train_inputs.to(device))
    train_loader = DataLoader(train_loader, batch_size=64)
    
    val_inputs['labels'] = y_val
    val_loader = BinaryDataset(val_inputs.to(device))
    val_loader = DataLoader(val_loader, batch_size=64)
    
    test_inputs['labels'] = y_test
    test_loader = BinaryDataset(test_inputs.to(device))
    test_loader = DataLoader(test_loader,batch_size=64)
    
    #####################################   
    param, perf = search_hyperparams(bert_model, train_loader, device)
    print(f"hyperparam search best val loss: {perf}")
    model = StanceDetect(bert_model, 2, param["module__drop_rate"]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=param["lr"] , weight_decay=param["optimizer__weight_decay"]) 
    #####################################   
    
    model, start_epoch, stats = restore_checkpoint(model, config("DistilBert_FineTune.checkpoint"), torch.cuda.is_available())
    model.to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(),  lr=5e-5, weight_decay=1e-5)  
    criterion = CrossEntropyLoss()
    
    evaluate_epoch(#axes,
                   train_loader, val_loader, test_loader, model, criterion, stats)
    
    train(model, optimizer, criterion, train_loader, val_loader, stats, start_epoch)
    # utils.save_dbert_training_plot()
    
    best_epoch = np.array(stats)[:,1].argmin() + 1
    model, stats = restore_best(model, best_epoch, config("DistilBert_FineTune.checkpoint"), torch.cuda.is_available())
    
    evaluate_epoch(#axes,
                   train_loader, val_loader, model, criterion, stats, test_loader, include_test=True)
    stats = np.array(stats[0:-1])
    utils.make_training_plot(stats)
    print(pd.DataFrame(np.array(stats[-1]).reshape(3,4), columns=['Accuracy','Loss','Recall','Precision'],index=["Val","Train","Test"]).reindex(["Train","Val","Test"]))


if __name__ == "__main__":
    print("hey")
    main()
    print("done")
