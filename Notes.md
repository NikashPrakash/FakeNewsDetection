Cluster-Then-Label is another approach to Semi-Supervised Learning (SSL). As the name suggests, it involves two broad steps: clustering and then labelling. 

Here's a basic overview of how it works:

Clustering: This involves using all the data - labelled and unlabelled - to find natural groupings or clusters among the data points. This is typically done using unsupervised learning techniques like K-Means, DBSCAN, Hierarchical Clustering, etc. The assumption here is that the data points in the same cluster are more likely to be of the same class.

Label Propagation: Once the clusters are identified, we assign the labels. The labels for each cluster can be determined in different ways:

    
You can use the labels of the already labeled data within each cluster. The most common label within the labeled data in the cluster would be the label for the entire cluster.
  
In another approach, the centroid or representative point of each cluster is computed and then the nearest labeled point to that centroid is used to label the whole cluster.

After the clusters have been labeled, you have a "labeled" dataset and can proceed as you would in a supervised learning problem.

This method allows us to utilize the structure found in unlabeled data, assuming that the structure is relevant for prediction tasks. However, it makes a strong assumption that each cluster belongs to only one class. This assumption may not hold true in many real-world data distributions, which may limit the method's effectiveness.
