    
#clustering.py
#Notes (please read fully): First code block is most same from data_process.py. 
#Changed a few important things: 
#1). Dropped rows with NaN value for text in labelled and unlabelled
#2). Dropped Stop Words and Punctuations

#3). For each article, I averaged the GloVe embeddings together so each article has size 1 x 200
    #4). Much easier to process and use for clustering and avoids padding issues
    
#5). VERY IMPORTANT: We have about 24K datapoints - by both space and time complexity, Spectral Clustering is WAY TOO ineffective.
    #6). Therefore, I have implemented k-Means below - this could be used as our baseline model as its not great but a good starting point
    #7). If we are planning to use as anything more than baseline, consider using HDBSCAN (Hierarchical DBSCAN) clustering
        #8). HDBSCAN is faster than Spectral and considered better for text classification than K-means
    
    
import pandas as pd
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import gensim.downloader
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import torch
from sklearn.cluster import AgglomerativeClustering
# import data_process # For later, integrate common functions to reduce redundancy

nltk.download('stopwords')
nltk.download('punkt')
glove = gensim.downloader.load('glove-wiki-gigaword-200')

def process_fake_news():
    filename = "f-n.csv"
    df = pd.read_csv(filename)
    df = df.dropna(subset=['text'])
    return df[["text", "label"]]

def process_unlabelled_data():
    filename = "p-b.csv"
    df = pd.read_csv(filename)
    df = df.dropna(subset=['text'])
    return df[["text"]]

def get_glove_feature(df,glove):
    features = []
    stop_words = set(stopwords.words('english'))
    for i, row in df.iterrows():
        text = row["text"]
        text = str(text).lower() 
        words = word_tokenize(text)
        
        words = [word for word in words if word not in stop_words and word.isalnum()]
        
        feature = [glove[word] for word in words if word in glove]
        
        features.append(np.mean(feature, axis=0) if feature else np.zeros(200))
    return np.array(features)

def split(df_x, df_y):
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.75, random_state=random_state)
    return X_train, X_test, y_train, y_test

def process(glove):
    df = process_fake_news()
    unlab = process_unlabelled_data()
    df_features = get_glove_feature(df,glove)
    train_unlabel = get_glove_feature(unlab,glove)
    train_label, X_test, y_label, y_test = split(df_features, df["label"])
    
    return train_label, X_test, y_label, y_test, train_unlabel

def proces_normal():
    df = process_fake_news()
    unlab = process_unlabelled_data()
    train_label, X_test, _, _ = split(df, df["label"])
    
    return train_label["text"], X_test["text"], unlab.squeeze()

#Note: K-Means is very sensitive to the scale of the features, so used L2 normalization. L2 normalization will not distort direction so semantic value from GloVe will still be there
def normalize(features):
    normalizer = Normalizer()
    return normalizer.fit_transform(features)

def clustering_Hierarchical(normalized_data):
    clustering_model = AgglomerativeClustering(n_clusters=2, linkage='ward')  
    clustering_model.fit(normalized_data)
    return clustering_model.labels_

def clustering(normalized_data):
    # Change to hierarchical clustering, compare different linkage methods for best
    kmeans = KMeans(n_clusters=2,n_init='auto') #Note: for the 3 categories - change accordingly as needed
    kmeans.fit(normalized_data)
    return kmeans.labels_

def clustering_Hierarchical(normalized_data):
    clustering_model = AgglomerativeClustering(n_clusters=2, linkage='ward')  
    clustering_model.fit(normalized_data)
    return clustering_model.labels_

from sklearn.mixture import GaussianMixture

def clustering_GMM(normalized_data):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(normalized_data)
    labels = gmm.predict(normalized_data)
    return labels

from sklearn.cluster import DBSCAN, SpectralClustering

def clustering_DBSCAN(normalized_data):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(normalized_data)
    return labels

def clustering_Spectral(normalized_data):
    Spectral = SpectralClustering(n_clusters=2,affinity='nearest_neighbors', random_state=42)
    labels = Spectral.fit_predict(normalized_data)
    return labels
    
#For now we are clustering both labelled and unlabelled data together as this si the standard.
#A possible future direction is to first cluster labelled data and get the cluster boundaries, before labelling unlabelled data based on where it falls in the boundaries.
#This may result in improvement

def normalize_and_cluster(train_label, train_unlabel):
    full_data = np.vstack((train_label, train_unlabel)) #Note: combined the labeled and unlabeled data
    normalized_data = normalize(full_data)
    labels = clustering(normalized_data)

    return labels


def actual_label(labels, train_label, y_label):
    """
    To give the unlabeled data labels based of the labeled data set?    
    """
    cluster0 = {}
    cluster0[0] = 0
    cluster0[1] = 0
    cluster1 = {}
    cluster1[0] = 0
    cluster1[1] = 0
    i = 0
    size = train_label.shape[0]
    for label in labels:
        if i>=size:
            break
        if label == 0:
            actual_label = y_label.iloc[i]
            if actual_label == 0:               #dont need this
                cluster0[actual_label] += 1
            else:                               #dont need this
                cluster0[actual_label] += 1
        else:
            actual_label = y_label.iloc[i]
            if actual_label == 0:               #dont need this
                cluster1[actual_label] += 1
            else:                               #dont need this
                cluster1[actual_label] += 1
        i += 1
    ratio0 = cluster0[0]/cluster1[0]
    ratio1 = cluster0[1]/cluster1[1]
    y_unlabel = labels[size:]
    if ratio1 > ratio0:
        for label in y_unlabel:
            if label == 0:
                label = 1
            else:
                label = 0
    return y_unlabel

    
def cluster_then_label():
    glove = gensim.downloader.load('glove-wiki-gigaword-200')
    x_train_label, _, y_label, y_test, train_unlabel = process(glove)
    x_train_actual, x_test, train_unlab_actual = proces_normal()
    labels = normalize_and_cluster(x_train_label, train_unlabel)
    y_unlabel = actual_label(labels, x_train_label, y_label)
    
    x_train = np.concatenate((x_train_actual, train_unlab_actual))
    y_train = np.concatenate((y_label, y_unlabel))
    
    return x_train, torch.from_numpy(y_train), x_test, torch.from_numpy(y_test.to_numpy())


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
        return {key: val[idx].clone().detach() for key, val in self.data.items()}



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

from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def train(model, train, epoch, optimizer,criterion): 

    # train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    # num_itr = 0
    # best_model, best_accuracy = None, 0

    losses = []
    acc = []
    prec = []
    rec = []

    for i in range(epoch):
        model.train()

        train_loss, train_accuracy, train_precision, train_recall = 0.0, 0.0, 0.0, 0.0
        total = 0.0

        for batch in train:
            # num_itr += 1
            #tokenize here before passing into the model - not needed
            optimizer.zero_grad()
            y = batch['labels']
            y_copy = batch['labels']
            y = F.one_hot(y, num_classes=2).float()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pred = model(input_ids,attention_mask)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step() 

            train_loss += loss.item()

            # Convert predictions to labels (assuming binary classification)
            y_pred = pred.argmax(dim=1)

            # Update metrics
            train_accuracy += accuracy_score(y_copy.cpu(), y_pred.cpu())
            train_precision += precision_score(y_copy.cpu(), y_pred.cpu())
            train_recall += recall_score(y_copy.cpu(), y_pred.cpu())
            total += 1
        
        avg_train_loss = train_loss / total
        avg_train_accuracy = train_accuracy / total
        avg_train_precision = train_precision / total
        avg_train_recall = train_recall / total

        losses.append(avg_train_loss)
        acc.append(avg_train_accuracy)
        prec.append(avg_train_precision)
        rec.append(avg_train_recall)
        print("epoch done")
        print("loss:")
        print(avg_train_loss)
        print("accuracy:")
        print(avg_train_accuracy)
        print("rec:")
        print(avg_train_recall)
        print("prec:")
        print(avg_train_precision)

    print("all losses: ")
    print(losses)
    print("all rec: ")
    print(rec)
    print("all acc: ")
    print(acc)
    print("all prec: ")
    print(prec)
    
    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(10, 6))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses, label='Loss', marker='o')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, acc, label='Accuracy', marker='o', color='green')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs, prec, label='Precision', marker='o', color='red')
    plt.title('Precision per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, rec, label='Recall', marker='o', color='purple')
    plt.title('Recall per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()
                

                

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    train_accuracy, train_precision, train_recall = 0.0, 0.0, 0.0, 0.0
    sum = 0.0
    with torch.no_grad():
        for batch in test_loader:
            y = batch['labels']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            output = model(input_ids,attention_mask).argmax(dim=1)
            # pred = predictions(output.data)
            correct += torch.sum(torch.eq(output,y).type(torch.IntTensor))
            total += y.size(dim=0)
            train_accuracy += accuracy_score(y.cpu(), output.cpu())
            train_precision += precision_score(y.cpu(), output.cpu())
            train_recall += recall_score(y.cpu(), output.cpu())
            sum += 1

    avg_train_accuracy = train_accuracy / sum
    avg_train_precision = train_precision / sum
    avg_train_recall = train_recall / sum

    print("testing done")
    print("average accuracy:")
    print(avg_train_accuracy)
    print("rec:")
    print(avg_train_recall)
    print("prec:")
    print(avg_train_precision)

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
    print(device)
    x_train, y_train, x_test, y_test = cluster_then_label()
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
    print("final accuracy: ")
    print(accuracy)

if __name__ == "__main__":
    print("hey")
    main()
    print("done")


