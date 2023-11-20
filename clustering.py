#Venkat-Notes (please read fully): First code block is most same from data_process.py. 
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

nltk.download('stopwords')
nltk.download('punkt')

def process_fake_news():
    filename = "fake-news.csv"
    df = pd.read_csv(filename)
    df = df.dropna(subset=['text'])
    return df[["text", "label"]]

def process_unlabelled_data():
    filename = "political-bias.csv"
    df = pd.read_csv(filename)
    df = df.dropna(subset=['text'])
    return df[["text"]]

def get_glove_feature(df):
    features = []
    glove = gensim.downloader.load('glove-wiki-gigaword-200')
    stop_words = set(stopwords.words('english'))
    for i, row in df.iterrows():
        text = row["text"]
        text = str(text).lower() 
        words = word_tokenize(text)
        
        words = [word for word in words if word not in stop_words and word.isalpha()]
        
        feature = [glove[word] for word in words if word in glove]
        
        features.append(np.mean(feature, axis=0) if feature else np.zeros(200))
    return np.array(features)

def split(df):
    random_state = 42
    df_train, df_test = train_test_split(df, train_size=0.75, random_state=random_state)
    return df_train, df_test

def process():
    df = process_fake_news()
    unlab = process_unlabelled_data()
    df_features = get_glove_feature(df)
    unlab_features = get_glove_feature(unlab)
    train_label, test = split(df_features)
    train_unlabel = unlab_features
    return train_label, train_unlabel, test

train_label, train_unlabel, test = process()

#Venkat-Note: K-Means is very sensitive to the scale of the features, so used L2 normalization. L2 normalization will not distort direction so semantic value from GloVe will still be there
def normalize(features):
    normalizer = Normalizer()
    return normalizer.fit_transform(features)

def kmeans_clustering(normalized_data):
    kmeans = KMeans(n_clusters=3) #Venkat-Note: for the 3 categories - change accordingly as needed
    kmeans.fit(normalized_data)
    return kmeans.labels_

def normalize_and_cluster(train_label, train_unlabel):
    full_data = np.vstack((train_label, train_unlabel)) #Venkat-Note: combined the label and unlabeled data
    normalized_data = normalize(full_data)
    labels = kmeans_clustering(normalized_data)

    return labels

labels = normalize_and_cluster(train_label, train_unlabel)