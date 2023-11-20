# Media Bias Detection

## Methodology

#### SSL Method employed - Cluster-than-label SSL along woth contrastive learning and word embeddings

1. Collect unlabelled datapoints that correspond to our task.
2. We will preprocess the data. To do this, we will get in the text such as glove embeddings. {{After this, we may also add additonal features such as author, etc}}
3. We will then use cluster-than-label SSL procedure
   1. First we will use hierarchical, **spectral** (normalize data first), k means, or other cluster methods to create clusters. {{We will use the method with highest performance / most clear set of clusters.}}
   2. We will then label the clusters based on the most prevalent label in that cluster (based on already labeled data ins the cluster). This will mean that each cluster's label will correspond to a label in the labelled dataset.
   3. {{We may use methods such as contrastive learning to seperate the data more clearly to have more distinct boundaries}}
   4. We will then label each unlabelled datapoint in a cluster based on which label the cluster corresponds to.
4. We will train either a RNN or transformer to classify new inputs into a label.
5. We will test on our test data set and use evaluation metrics to evalute performance.

<!-- input for cluster: -->

 
## Model Architecture

#### Baseline Model

#### RNN or Transformer

##### Transformer
1. DistilBert Layer
2. Linear Layer 1: input_dim=hidden_dim[0], output=hidden_dim[1] --
3. ReLU
4. Linear Layer 2: input_dim=hidden_dim[1], output dim = (2,3)