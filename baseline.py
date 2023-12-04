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
from sklearn.cluster import AgglomerativeClustering
# import data_process # For later, integrate common functions to reduce redundancy

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

nltk.download('stopwords')
nltk.download('punkt')
glove = gensim.downloader.load('glove-wiki-gigaword-200')

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

def process():
    df = process_fake_news()
    unlab = process_unlabelled_data()
    df_features = get_glove_feature(df)
    train_unlabel = get_glove_feature(unlab)
    train_label, X_test, y_label, y_test = split(df_features, df["label"])
    
    return train_label, X_test, y_label, y_test, train_unlabel


#Note: K-Means is very sensitive to the scale of the features, so used L2 normalization. L2 normalization will not distort direction so semantic value from GloVe will still be there
def normalize(features):
    normalizer = Normalizer()
    return normalizer.fit_transform(features)

def clustering(normalized_data):
    # Change to hierarchical clustering, compare different linkage methods for best
    kmeans = KMeans(n_clusters=2) #Note: for the 3 categories - change accordingly as needed
    kmeans.fit(normalized_data)
    return kmeans.labels_
    #accuracy: 0.5875

#For now we are clustering both labelled and unlabelled data together as this si the standard.
#A possible future direction is to first cluster labelled data and get the cluster boundaries, before labelling unlabelled data based on where it falls in the boundaries.
#This may result in improvement
def clustering_Hierarchical(normalized_data):
    clustering_model = AgglomerativeClustering(n_clusters=2, linkage='average')  
    clustering_model.fit(normalized_data)
    return clustering_model.labels_
    #accuracy : 0.565

from sklearn.mixture import GaussianMixture

def clustering_GMM(normalized_data):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(normalized_data)
    labels = gmm.predict(normalized_data)
    return labels
    #accuracy: 0.5702

from sklearn.cluster import DBSCAN, SpectralClustering

def clustering_DBSCAN(normalized_data):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(normalized_data)
    return labels
    #accuracy: 0.5080

def clustering_Spectral(normalized_data):
    Spectral = SpectralClustering(n_clusters=2,affinity='nearest_neighbors', random_state=42)
    labels = Spectral.fit_predict(normalized_data)
    return labels
    #accuracy: 


# def cluster

def normalize_and_cluster(train_label, train_unlabel):
    full_data = np.vstack((train_label, train_unlabel)) #Note: combined the labeled and unlabeled data
    normalized_data = normalize(full_data)
    labels = clustering_Spectral(normalized_data)

    return labels


def actual_label(labels, train_label, y_label):
    """
    To give the unlabeled data labels based of the labeled data set?    
    """
    cluster0 = {}
    cluster0["lab_0"] = 0
    cluster0["lab_1"] = 0
    cluster1 = {}
    cluster1["lab_0"] = 0
    cluster1["lab_1"] = 0
    i = 0
    size = train_label.shape[0]
    for label in labels:
        if i>=size:
            break
        if label == 0:
            actual_label = y_label.iloc[i]
            if actual_label == 0:
                cluster0["lab_0"] += 1
            else:
                cluster0["lab_1"] += 1
        else:
            actual_label = y_label.iloc[i]
            if actual_label == 0:
                cluster1["lab_0"] += 1
            else:
                cluster1["lab_1"] += 1
        i += 1
    ratio0 = cluster0["lab_0"]/cluster1["lab_0"]
    ratio1 = cluster0["lab_1"]/cluster1["lab_1"]
    y_unlabel = labels[size:]
    if ratio1 > ratio0:
        for label in y_unlabel:
            if label == 0:
                label = 1
            else:
                label = 0
    return y_unlabel

def cluster_then_label():
    x_train_label, x_test, y_label, y_test, train_unlabel = process()
    labels = normalize_and_cluster(x_train_label, train_unlabel)
    y_unlabel = actual_label(labels, x_train_label, y_label)
    
    x_train = np.vstack((x_train_label, train_unlabel))
    y_train = np.concatenate((y_label, y_unlabel))
    return x_train, y_train, x_test, y_test

#if __name__ == "__main__":
 #   cluster_then_label()

x_train, y_train, x_test, y_test = cluster_then_label()

def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(200,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = create_model()
model.summary()
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
