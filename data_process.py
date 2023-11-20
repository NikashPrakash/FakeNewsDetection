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

def get_glove_feature(df):
    features = []
    for i, row in df.iterrows():
        text = row["text"]
        text = text.lower()
        glove = gensim.downloader.load('glove-wiki-gigaword-200')
        words = word_tokenize(text)
        feature = []
        for word in words:
            if word in glove:  
                feature.append(glove[word])
        features.append(feature)
    features = np.array(features)
    return features

    def feature_gen(headline):
        tokens = word_tokenize(headline.lower())
        indices = [self.glove.key_to_index[token] for token in tokens if token in self.glove.key_to_index]
        if indices:  
            vectors = matrix[indices]
            mean_vector = np.mean(vectors, axis=0)
            return mean_vector
    # return np.array(pd.DataFrame(df['headline'].apply(feature_gen).tolist()))


def split(df):
    random_state = 42
    df_train, df_test = train_test_split(df, train_size=0.75, random_state=random_state)


# input for transformer or RNN is an array of the glove embedding of each words in the text to be analyzed



def big_news():
    with open('FILL IN FILE NAME(S)') as fp:
        train = [json.loads(line) for line in fp]
        
def process():
    df = process_fake_news()
    #df = big_news()
    df = get_glove_feature(df)
    train, test = split(df)
    return train, test