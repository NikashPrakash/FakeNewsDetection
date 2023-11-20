import pandas as pd
import json
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import gensim.downloader

def process_fake_news():
    filename = "fake-news.csv"
    df = pd.read_csv(filename)
    return df[["text", "label"]]

def process_unlabelled_data():
    filename = "political-bias.csv"
    df = pd.read_csv(filename)
    return df[["text"]]

def get_glove_feature(df):
    features = []
    glove = gensim.downloader.load('glove-wiki-gigaword-200')
    for i, row in df.iterrows():
        text = row["text"]
        text = text.lower()
        words = word_tokenize(text)
        feature = []
        for word in words:
            if word in glove:  
                feature.append(glove[word])
        features.append(feature)
    features = np.array(features)
    return features

def split(df):
    random_state = 42
    df_train, df_test = train_test_split(df, train_size=0.75, random_state=random_state)


# input for transformer or RNN is an array of the glove embedding of each words in the text to be analyzed



# def big_news():
#     with open('FILL IN FILE NAME(S)') as fp:
#         train = [json.loads(line) for line in fp]
        
def process():
    df = process_fake_news()
    #df = big_news()
    unlab = process_unlabelled_data()
    df,unlab = get_glove_feature(df), get_glove_feature(unlab)
    train_label, test = split(df)
    train_unlabel = unlab
    return train_label, train_unlabel, test