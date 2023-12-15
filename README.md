# Fake News Detection

#### -- Project Status: On-Hold

## Project Intro/Objective
Fake news detection is something that all of us are deeply passionate about. We feel that news has become more commercialized and is being used to push agendas rather than speak the truth. We think this way for both aisles of the political spectrum. However, we also acknowledge the importance of having a free and effective media ecosystem to promote democracy and growth. We believe that one way to help solve this issue is to identify fake news so that news companies can be held accountable and democracy can be strengthened. Natural Language Processing is one of the main ways for us to detect such biases, as we can do this automatically.

### Partners
* Devang Chowdhuary, Venkat Kannan


### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Natural Language Processing

#### Methodology

1. Collect unlabelled datapoints that correspond to our task.
2. Preprocess the data. To do this, we will get in the text such as glove embeddings
3. Cluster-than-label SSL procedure
   1. Normalize/preprocess data -> Glove Embeddings -> K-means (k=2)
   2. Label the clusters based on the most prevalent label in that cluster (based on already labeled data ins the cluster). This will mean that each cluster's label will correspond to a label in the labelled dataset.
   4. Label each unlabelled datapoint in a cluster based on which label the cluster corresponds to.
4. Randomized hyperparameter search with dedicated train/val split
5. Fine-tune DistilBert with extra classification layers to classify new inputs into a label.
6. Test on our test data set and use evaluation metrics to evalute performance.

### Technologies
* Python
* Pytorch
* Natural Language Processing
* High Performance Computing - U of M Great Lakes
* Pandas, Numpy

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept at [here](https://github.com/Media-Bias-Group/MBIB#introducing-mbib---the-first-media-bias-identification-benchmark-task-and-dataset-collection) within this repo.    
3. Data processing/transformation scripts are being kept [here](https://github.com/NikashPrakash/FakeNewsDetection/blob/main/clustering.py)
4. Run training.py

