# Exploring unsupervised graph-based learning methods, clustering methods and neural networks for clothing image classification and academic paper clustering

---
## Overview
- Clustering of academic papers using K-Means, comparing different scoring metrics and analyzing optimal number of clusters and randomness in K-Means
- Graph-based learning methods for clustering including community detection and centrality measures
- Image classification of clothing items using unsupervised methods
- Used PCA to visualise clustering of clothing items and centroids
- Comparison of kNN, hierachical clustering and neural networks for supervised image classification
- Comparison of MLP neural networks and CNNs
- Alteration of CNNs using dropout, other layers and alterations to kernel sizes to increase accuracy
- K-Fold Stratified Cross-Validation used throughout

Community Detection using CNM  |  Clustering visualised with PCA
:-------------------------:|:-------------------------:
![](https://github.com/leonjwu/Data-Science/blob/master/Graph-Cluster-NN/Project%203_files/Project%203_73_0.png)  |  ![](https://github.com/leonjwu/Data-Science/blob/master/Graph-Cluster-NN/Project%203_files/Project%203_121_0.png)
---

---
---
# Unsupervised learning: text documents with citation graph
---
---


## Introduction
---
This first dataset is comprised of journal papers, represented as a vector of p features, with each feature a representing whether or not a word exists in the each paper, and also as an undirected graph in the form of an adjacency matrix. This project will explore the closeness and importance of documents using various unsupervised and supervised learning techniques.

## 1.1 Clustering of the feature matrix
---
- The aim of unsupervised clustering of data is to find groups that partition the samples, such that points in a group are more similar to eachother than points from different groups.
- More formally, let N be the number of sample and K the number of clusters. The aim is to minimize
$$ Trace \Big[ (H^TH)^{-1} [H^TDH] \Big]
$$
where H is the NxK assigment matrix mapping each sample to one of k clusters, and D is the NxN distance (or measure of 'closeness') matrix between all the samples. The elements on the diagonals of the KxK matrix $[H^TDH]$ correspond to the sums of the distances between all points within the same cluster, so that the sum of these distances is minimized. $(H^TH)^{-1}$ is used to normalize the diagonals of the matrix $[H^TDH]$ using the sizes of the groups.


### K-means Algorithm
- Complete enumeration in order to find this global minimum is not computationally feasible for any sufficiently large dataset.
- The K-means algorithm uses a heuristic to reduce the computational cost of finding a solution (but not necessarily the global solution) to this problem.
- The algorithm starts by randomly assinging each sample to one of the k clusters. At each iteration until convergence, each sample is reassigned to the cluster whose centroid is closest to the sample.
- The algorithm will terminate when some condition is met, such as when the clustering doesn't change between iterations or after a set number of iterations is reached.


### Measures of 'closeness' between samples
- There needs to be decisions made about how to define '**closeness**' between points, and **how many groups** to split samples into.
- Measures of similarity/dissimilarity to consider:
    - Euclidean
    - Manhattan Distance
    - Cosine similarity (measure of vector alignment)
    - Correlation

- The **euclidean distance** will be used in this example, although the other distances could be considered. The sklearn implementation of K-Means uses the Euclidean distance so it is also convenient to use.

### Choosing the number of clusters, k
- A range of values for k will be tested. The aim is to find the value of k for which the clustering is optimal by some metric/metrics.


```python
# Load data from my drive into dataframes
drive.mount("/content/drive/")
feat_mat = pd.read_csv("/content/drive/My Drive/A50/CW3/feature_matrix.csv", header=None, dtype=int)

feat_mat.head()
```

    Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount("/content/drive/", force_remount=True).





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>...</th>
      <th>1393</th>
      <th>1394</th>
      <th>1395</th>
      <th>1396</th>
      <th>1397</th>
      <th>1398</th>
      <th>1399</th>
      <th>1400</th>
      <th>1401</th>
      <th>1402</th>
      <th>1403</th>
      <th>1404</th>
      <th>1405</th>
      <th>1406</th>
      <th>1407</th>
      <th>1408</th>
      <th>1409</th>
      <th>1410</th>
      <th>1411</th>
      <th>1412</th>
      <th>1413</th>
      <th>1414</th>
      <th>1415</th>
      <th>1416</th>
      <th>1417</th>
      <th>1418</th>
      <th>1419</th>
      <th>1420</th>
      <th>1421</th>
      <th>1422</th>
      <th>1423</th>
      <th>1424</th>
      <th>1425</th>
      <th>1426</th>
      <th>1427</th>
      <th>1428</th>
      <th>1429</th>
      <th>1430</th>
      <th>1431</th>
      <th>1432</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1433 columns</p>
</div>



#### Calinski-Harabasz Score
- The Calinski-Harabasz score takes into account the dispersion within clusters and dispersion between clusters, so that tight clusters that are far away from eachother score well. 
- A higher score is better - more will be said about this score later.

#### Davies-Bouldin Score
- The Davies-Bouldin score is similar to the Calinski-Harabasz score except it tends to score clusters equally-distanced from eachother highly, in comparison to clusters of approximately the same size as in the Calinsky-Harabasz. A lower score is better.


#### Inertia
- Inertia is the cost function that k-means uses, ie. the sum of squared distances of all samples from their closest cluster centroid.


```python
# Toggle on/off printing
verbose = True

# Setup list of numbers of clusters and metrics
k_values = np.linspace(2, 30, 28, dtype=int)
CH = []
DB = []
inertia = []
X = feat_mat

for k in k_values:
    if verbose: print(f'Running k-means for k={k}')
    # Use K-means algorithm to cluster papers with k clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)

    # Store Calinski-Harabasz score
    CH.append(CH_score(X, kmeans.labels_))
    # Store Davies-Bouldin Score
    DB.append(DB_score(X, kmeans.labels_))
    # Store Inertia
    inertia.append(kmeans.inertia_)

# Compute Optimal value for k
k_optimal = k_values[next(i for i,x in enumerate(CH) if x<7)]
```

    Running k-means for k=2
    Running k-means for k=3
    Running k-means for k=4
    Running k-means for k=5
    Running k-means for k=6
    Running k-means for k=7
    Running k-means for k=8
    Running k-means for k=9
    Running k-means for k=10
    Running k-means for k=11
    Running k-means for k=12
    Running k-means for k=13
    Running k-means for k=14
    Running k-means for k=15
    Running k-means for k=16
    Running k-means for k=17
    Running k-means for k=18
    Running k-means for k=19
    Running k-means for k=20
    Running k-means for k=21
    Running k-means for k=22
    Running k-means for k=23
    Running k-means for k=24
    Running k-means for k=25
    Running k-means for k=26
    Running k-means for k=27
    Running k-means for k=28
    Running k-means for k=30



```python
# Begin plot
fig, ax1 = plt.subplots(1, 1, figsize=(15,8))
ax2 = ax1.twinx()

# Plot graphs against values of k
ax1.plot(k_values, CH, marker='o', label='Calinski-Harabasz Score')
ax1.plot(k_values, DB, marker='o', label='Davies-Bouldin Score')
ax2.plot(k_values, inertia, marker='o', color='purple', label='Inertia')
ax1.set_title(f'Evaluation of Clusterings Using Different Metrics against k', fontsize=19)

# Plot vertical line for optimal k value
ax1.axvline(x=k_optimal, linestyle='dashed', color='g', label=f'Optimal* k={k_optimal}')
ax1.legend(loc='upper left')
ax2.legend()

ax1.set_xlabel('k')
ax1.set_ylabel('Score')
ax2.set_ylabel('Inertia')
ax2.grid(False)

plt.show()
```


![png](Project%203_files/Project%203_15_0.png)


- As k increases, we see that the Davies-Bouldin score decreases signifying that the clustering is better, ie the clusters are tighter and/or more well separated between clusters. For $k > 20$, the score don't reduce much after this, which suggests splitting into more clusters doesn't provide a significantly better clustering.

- Also, the Calinski-Harabasz score decreases, which means that actually the clustering gets worse as k increases according to this metric. However this score is expected to decrease as k increases, and this score will be discussed more later on.


```python
CH_grad = np.gradient(CH)
k_optimal_alt = np.argmax(CH_grad) + 2
# Begin plot
fig, ax1 = plt.subplots(1, 1, figsize=(15,8))

# Plot graph against values of k
ax1.plot(k_values, CH_grad, marker='o', label='Calinski-Harabasz Score Gradient')
ax1.set_title(f'Evaluation of Clusterings Using CH score', fontsize=19)

ax1.set_xlabel('k')
ax1.set_ylabel('CH Score Gradient')

plt.show()
```


![png](Project%203_files/Project%203_17_0.png)


- The above figure would suggest that a value of 9 (where the gradient of the CH score rapidly increases) or 20 (where the gradient is at its maximum) would be valid choices for an optimum value of k, as alluded to in the original paper by Calinski and Harabasz.

## Analysis of the Optimal Clustering Performance
- Note that there is no 'ground-truth' labels in order to test the clustering against, so no metrics that use these labels can be used. However, there are some measures of performance that don't use ground truth labels:

### Silhouette Coefficient
- The Silhouette coefficient is bounded between -1 and 1, with a higher score indicating a better clustering with dense and well-separated clusters. It uses the distance between samples and other points in the same cluster, as well as the distance between samples and other points in the next nearest cluster.

- The silhouette plot shows how close each point in each cluster to points in nearby clusters. It provide a way to visualize the quality of clusters. The aim is to find a clustering that provides clusters with similar and high silhouette scores. For this we can use the silhouette score, which is the average silhouette coefiicient across all k clusters.


```python
k_plot_list = [12, k_optimal, k_optimal+5]
fig, axes = plt.subplots(1, len(k_plot_list), figsize=(15,8))
X = feat_mat

for j, k in enumerate(k_plot_list):
    # Initialize the silhouette plot
    axes[j].set_xlim([-0.2, 1])
    axes[j].set_ylim([0, len(X) + (k + 1) * 10])

    # Cluster using k-means with
    kmeans = KMeans(n_clusters=k, n_init=1, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.predict(X)

    # Compute average silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        axes[j].fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    axes[j].set_title(f"k = {k}, Score = {round(silhouette_avg, 3)}", fontsize=15)
    axes[j].set_xlabel("Silhouette coefficient")
    axes[j].set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    axes[j].axvline(x=silhouette_avg, color="red", linestyle="--")

    axes[j].set_yticks([])  # Clear the yaxis labels / ticks
    axes[j].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

fig.suptitle('Silhouette Plots for varying k', fontsize=18)
# Reference: sklearn's example for silhouette plots
```




    Text(0.5, 0.98, 'Silhouette Plots for varying k')




![png](Project%203_files/Project%203_21_1.png)


- From the silhouette plots, we can see that actually the silhouette scores are quite low (values around 0 (or even below 0) suggest that the clusters are not very good), suggesting that the clusters are not very well-separated. Varying k also doesn't seem to yield significantly between clusterings.

- The plots themselves show a lot of variation in the size of the plots and also their are many clusters with a score below the average score. This suggests that perhaps the clusters are not that well-defined and it is hard to cluster this data well, or maybe that the threshold=7 for the CH score is not optimal.

#### Distribution of Cluster Sizes
- The distribution of cluster sizes is now plotted as a histogram:


```python
# Train optimal model
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X)

# Plot cluster size distribution
cluster_sizes = np.zeros(k_optimal)
cluster_labels = kmeans.predict(X)
for i in range(k_optimal):
    cluster_sizes[i] = int(cluster_labels[cluster_labels == i].shape[0])

fig, ax = plt.subplots(figsize=(15,8))
ax.hist(cluster_sizes)
fig.suptitle('Distribution of Cluster Sizes')
ax.set_xlabel('Cluster Size')
ax.set_ylabel('Frequency')
print(sorted(cluster_sizes))
```

    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 14.0, 61.0, 71.0, 90.0, 170.0, 209.0, 209.0, 299.0, 365.0, 423.0, 541.0]



![png](Project%203_files/Project%203_24_1.png)


- There are many small cluster with fewer than 10 items in each, which suggests that maybe there should be fewer clusters. It could also suggest that these are outliers, and a clustering method that handles outliers better than K-Means should be considered; K-Medoids for example.

#### Within Groups / Across Group Similarity
- The within and across group dispersion is calcualted as in the Calinski-Harabasz score, which uses the mean of the clusters and of all the points for its calculations:


```python
extra_disp, intra_disp = 0., 0.
mean = np.mean(X, axis=0)
for k in range(k_optimal):
    cluster_k = np.array(X[cluster_labels == k])
    cluster_k = cluster_k[0, :]
    mean_k = np.mean(cluster_k, axis=0)
    extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
    intra_disp += np.sum((cluster_k - mean_k) ** 2)
```

### Final model metrics


```python
# Populate table (training)
df_table = pd.DataFrame(columns=['', f'k_optimal={k_optimal}'])
df_table.loc[0] = ['CH Score', round(CH_score(X, kmeans.labels_), 3)]
df_table.loc[1] = ['DB Score', round(DB_score(X, kmeans.labels_), 3)]
df_table.loc[2] = ['Silhouette Score', round(silhouette_score(X, kmeans.labels_), 3)]
df_table.loc[3] = ['Within Group Dispersion', round(intra_disp, 1)]
df_table.loc[4] = ['Across Group Dispersion', round(extra_disp, 1)]
df_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>k_optimal=27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CH Score</td>
      <td>6.794</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DB Score</td>
      <td>3.217</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Silhouette Score</td>
      <td>-0.059</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Within Group Dispersion</td>
      <td>558.300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Across Group Dispersion</td>
      <td>33741.700</td>
    </tr>
  </tbody>
</table>
</div>



### Final Thoughts:
- **From the low scoring silhouette plots, the skewed distribution of cluster sizes and the CH score gradient plot, it seems as if a more optimal clustering might be less than what was calculated as the 'optimal' clustering by choosing the first clustering with CH score < 7.**

### Randomness in K-means clustering
- As mentioned before, the K-means is not likely to converge to the global minimum, and so it will likely converge to just a local minimum. The algorithm is sensitive to the initial random clustering of the points, and will likely yield different results each time it is run.

- **In order to account for this, the algorithm can be run many times, and the best clustering is then chosen. This is acheived using the n_init parameter in sklearn's K-Means implementation. In the above case, n_init is set to 10 be default.**

- If the outcomes are similar, it could be an indication that your outcomes are 'good', whereas if there is a lot of variance in the outcomes, it could indicate that k-means might not be the best algorithm with which to cluster this datset.

- 'k-means++' can also be used, which chooses the initial clustering in a better way. This is likely to decrease the variance of the models for different random seeds. Note that this alteration is used by default by sklearn.

**- A graph is now plotted to show how the value of n_init affects the best model chosen:**



```python
# Toggle on/off printing
verbose = True

# Setup list of numbers of clusters and metrics
n_init_values = np.linspace(1, 21, 10, dtype=int)
CH = []
inertia = []
CH2 = []
inertia2 = []

init = 'random'
for n_init in n_init_values:
    if verbose: print(f'Running k-means for n_init={n_init}')
    # Use K-means algorithm
    kmeans = KMeans(n_clusters=k_optimal, init=init, n_init=n_init, random_state=42)
    kmeans.fit(X)

    # Store Calinski-Harabasz score
    CH.append(CH_score(X, kmeans.labels_))
    # Store Inertia
    inertia.append(kmeans.inertia_)
    
init = 'k-means++'
for n_init in n_init_values:
    if verbose: print(f'Running k-means for n_init={n_init}')
    # Use K-means algorithm
    kmeans = KMeans(n_clusters=k_optimal, init=init, n_init=n_init, random_state=42)
    kmeans.fit(X)

    # Store Calinski-Harabasz score
    CH2.append(CH_score(X, kmeans.labels_))
    # Store Inertia
    inertia2.append(kmeans.inertia_)
    
# Begin plot
fig, ax1 = plt.subplots(1, 1, figsize=(15,8))
ax2 = ax1.twinx()

# Plot graphs against values of n_init
ax1.plot(n_init_values, CH, marker='o', label='Calinski-Harabasz Score, init=random')
ax1.plot(n_init_values, CH2, marker='o', label='Calinski-Harabasz Score, init=k-means++')
ax2.plot(n_init_values, inertia, marker='o', color='purple', label='Inertia, init=random')
ax2.plot(n_init_values, inertia2, marker='o', color='green', label='Inertia, init=k-means++')
ax1.set_title(f'Evaluation of Clusterings against n_init, k = {k_optimal}', fontsize=19)

ax1.set_xlabel('n_init')
ax1.set_ylabel('Score')
ax2.set_ylabel('Inertia')
ax1.legend()
ax2.legend()
ax2.grid(False)

plt.show()
```

    Running k-means for n_init=1
    Running k-means for n_init=3
    Running k-means for n_init=5
    Running k-means for n_init=7
    Running k-means for n_init=9
    Running k-means for n_init=12
    Running k-means for n_init=14
    Running k-means for n_init=16
    Running k-means for n_init=18
    Running k-means for n_init=21
    Running k-means for n_init=1
    Running k-means for n_init=3
    Running k-means for n_init=5
    Running k-means for n_init=7
    Running k-means for n_init=9
    Running k-means for n_init=12
    Running k-means for n_init=14
    Running k-means for n_init=16
    Running k-means for n_init=18
    Running k-means for n_init=21



![png](Project%203_files/Project%203_33_1.png)


- From the figure, it can be seen that above around n_init = 13 the extra attempts don't yield a significantly better clustering. Hence the used value of n_init=15 used below is sufficently high for a good clustering without too high a computational cost.

- A graph is also plotted to show how the best clusters are scored relative to n_init and k:


```python
# Setup list of numbers of clusters and metrics
k_values = np.linspace(2, 30, 10, dtype=int)
n_init_values = [1, 5, 10]

 # Begin plot
fig, ax1 = plt.subplots(1, figsize=(15,8))
for n_init in n_init_values:
    CH = []
    for k in k_values:
        if verbose: print(f'Running k-means for k={k}')
        # Use K-means algorithm to cluster papers with k clusters
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        kmeans.fit(X)

        # Store Calinski-Harabasz score
        CH.append(CH_score(X, kmeans.labels_))

    # Plot graphs against values of k
    ax1.plot(k_values, CH, marker='o', label=f'n_init={n_init}')

ax1.set_title(f'Calinski-Harabasz Score against k', fontsize=19)
ax1.legend(loc='upper left')
ax1.set_xlabel('k')
ax1.set_ylabel('Score')

plt.show()
```

    Running k-means for k=2
    Running k-means for k=5
    Running k-means for k=8
    Running k-means for k=11
    Running k-means for k=14
    Running k-means for k=17
    Running k-means for k=20
    Running k-means for k=23
    Running k-means for k=26
    Running k-means for k=30
    Running k-means for k=2
    Running k-means for k=5
    Running k-means for k=8
    Running k-means for k=11
    Running k-means for k=14
    Running k-means for k=17
    Running k-means for k=20
    Running k-means for k=23
    Running k-means for k=26
    Running k-means for k=30
    Running k-means for k=2
    Running k-means for k=5
    Running k-means for k=8
    Running k-means for k=11
    Running k-means for k=14
    Running k-means for k=17
    Running k-means for k=20
    Running k-means for k=23
    Running k-means for k=26
    Running k-means for k=30



![png](Project%203_files/Project%203_36_1.png)


- Repetitions of k-means for low k is more effective at finding a good clustering, so it may be more sensible to observe the n_init=1 graph, or at least be aware of the correlation between n_init and the CH score for a given number of clusters. Given enough repeats, high values of k can also be optimized in the same way. This is seen for low k: The distance between the three plotted lines is greater than that for large k.

### Outlier Sensitivity 
- K-means senstive to outliers because of how centroids are calculated as the mean of all the samples in the respective cluster. This can be combatted using data exploration (this is hard with many dimensions), or extensions to kmeans.

- K-medoids can be used, which assigns one of the points within the cluster to be the 'center', choosing the point which is most representative of the cluster.

## 1.2 Analysis of the citation graph
---

#### Graphs
Each of the samples is represented as a node in a graph. The edges that connect nodes represent  the similarities that exist between the nodes or samples. A graph can be constructed from the adjacency matrix. 

- The aim of clustering graphs is to find the relabelling of nodes such that the adjancancy matrix is as blcok diagonal as possible.

- The distances between nodes for graphs can be calculated in multiple ways: 
    - Geodisic distance between nodes (undirected graph) (minimum number of edges for all paths connected two nodes)
    - Average number of edges across all paths between all nodes
<!-- 
In order to make a graph that is more sparse than a fully connected graph, a few things methods can be considered:
Introduce a **tolerance**, such that any weight between nodes under this tolerance is set to 0 and the nodes are not connected. In practice this method is shown not to be very good.
Construct a **geometric graph**, using an $\epsilon$-ball, or KNN methodology. DO THESE:
The $\epsilon$-ball uses a local threshold for each node to connect other nodes to it. With this methodology, points which are quite close, but not within epsilon, of eachother in (other space) can be very far from eachother in the graph.
The KNN methodoloy links each node to its k nearest neighbours, such that the degree of all nodes is k. This methodology doesn't capture 'gaps' in the data, since two points on the edges of gaps will be connected?? since fixed k... -->


```python
# Load data from my drive into dataframes
drive.mount("/content/drive/")
adj_mat = pd.read_csv("/content/drive/My Drive/A50/CW3/adjacency_matrix.csv", header=None, dtype=int)

adj_mat.head()
```

    Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount("/content/drive/", force_remount=True).





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>...</th>
      <th>2445</th>
      <th>2446</th>
      <th>2447</th>
      <th>2448</th>
      <th>2449</th>
      <th>2450</th>
      <th>2451</th>
      <th>2452</th>
      <th>2453</th>
      <th>2454</th>
      <th>2455</th>
      <th>2456</th>
      <th>2457</th>
      <th>2458</th>
      <th>2459</th>
      <th>2460</th>
      <th>2461</th>
      <th>2462</th>
      <th>2463</th>
      <th>2464</th>
      <th>2465</th>
      <th>2466</th>
      <th>2467</th>
      <th>2468</th>
      <th>2469</th>
      <th>2470</th>
      <th>2471</th>
      <th>2472</th>
      <th>2473</th>
      <th>2474</th>
      <th>2475</th>
      <th>2476</th>
      <th>2477</th>
      <th>2478</th>
      <th>2479</th>
      <th>2480</th>
      <th>2481</th>
      <th>2482</th>
      <th>2483</th>
      <th>2484</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2485 columns</p>
</div>



- The citation graph is now plotted from the adjacency matrix:


```python
# convert graph from adjacency matrix
g1 = nx.convert_matrix.from_pandas_adjacency(adj_mat)

# plot the graph
plt.figure(figsize=(15,15))
nx.draw(g1, node_size=30)
plt.title('Citation Graph', fontsize=20)
plt.show()
```


![png](Project%203_files/Project%203_43_0.png)


## Node Centrality Measures

#### Degree
- The degree of a node is the number of edges it has. If a node has a high degree, it could be considered important, since in this case, the paper is cited a lot, or uses a lot of citations (or both).
- It would be useful to examine the directed graph, since then you could distinguish the **number of citations a paper uses** and the **number of citations that uses this paper**, since for each node you could compute the number of edges both coming in to and out of the node using the direction. However in this case we don't have this information.
- The degree distribution is plotted (Note the y axis log scale):


```python
# get degrees of all nodes
degree = g1.degree()
print(degree)
sorted_degree = sorted(degree, key=itemgetter(1), reverse=True)

# plot histograms of degrees
fig, ax = plt.subplots(figsize=(8, 6))
plt.hist([deg[1] for deg in g1.degree()], bins=20)
fig.suptitle('Histogram of Node Degrees')
ax.set_xlabel('Degree')
ax.set_ylabel('Frequency')
plt.yscale('log')
plt.show()
```

    [(0, 3), (1, 3), (2, 5), (3, 5), (4, 3), (5, 4), (6, 3), (7, 2), (8, 2), (9, 2), (10, 2), (11, 5), (12, 4), (13, 4), (14, 5), (15, 5), (16, 1), (17, 5), (18, 2), (19, 5), (20, 7), (21, 4), (22, 4), (23, 1), (24, 2), (25, 6), (26, 4), (27, 9), (28, 1), (29, 3), (30, 8), (31, 3), (32, 4), (33, 7), (34, 3), (35, 4), (36, 2), (37, 6), (38, 6), (39, 2), (40, 2), (41, 9), (42, 2), (43, 1), (44, 6), (45, 5), (46, 3), (47, 2), (48, 12), (49, 4), (50, 1), (51, 1), (52, 10), (53, 3), (54, 5), (55, 1), (56, 1), (57, 3), (58, 10), (59, 3), (60, 3), (61, 7), (62, 2), (63, 3), (64, 2), (65, 12), (66, 9), (67, 3), (68, 2), (69, 3), (70, 2), (71, 3), (72, 4), (73, 2), (74, 2), (75, 3), (76, 4), (77, 3), (78, 36), (79, 10), (80, 4), (81, 6), (82, 4), (83, 11), (84, 21), (85, 1), (86, 2), (87, 1), (88, 2), (89, 3), (90, 8), (91, 6), (92, 4), (93, 4), (94, 4), (95, 32), (96, 5), (97, 6), (98, 4), (99, 3), (100, 3), (101, 1), (102, 2), (103, 19), (104, 6), (105, 5), (106, 7), (107, 5), (108, 2), (109, 6), (110, 4), (111, 2), (112, 3), (113, 2), (114, 5), (115, 9), (116, 1), (117, 3), (118, 1), (119, 4), (120, 3), (121, 8), (122, 2), (123, 3), (124, 5), (125, 3), (126, 5), (127, 2), (128, 5), (129, 4), (130, 3), (131, 4), (132, 4), (133, 6), (134, 12), (135, 7), (136, 6), (137, 4), (138, 4), (139, 2), (140, 7), (141, 2), (142, 5), (143, 1), (144, 2), (145, 4), (146, 2), (147, 2), (148, 2), (149, 3), (150, 5), (151, 4), (152, 4), (153, 7), (154, 3), (155, 4), (156, 1), (157, 3), (158, 10), (159, 4), (160, 2), (161, 3), (162, 2), (163, 6), (164, 2), (165, 4), (166, 2), (167, 6), (168, 1), (169, 2), (170, 5), (171, 2), (172, 3), (173, 3), (174, 1), (175, 2), (176, 7), (177, 4), (178, 5), (179, 1), (180, 2), (181, 2), (182, 1), (183, 1), (184, 10), (185, 5), (186, 1), (187, 1), (188, 3), (189, 2), (190, 3), (191, 11), (192, 3), (193, 3), (194, 1), (195, 3), (196, 2), (197, 1), (198, 2), (199, 5), (200, 4), (201, 5), (202, 9), (203, 5), (204, 1), (205, 2), (206, 1), (207, 11), (208, 1), (209, 1), (210, 13), (211, 5), (212, 1), (213, 4), (214, 5), (215, 4), (216, 2), (217, 4), (218, 3), (219, 3), (220, 7), (221, 4), (222, 3), (223, 2), (224, 4), (225, 1), (226, 4), (227, 7), (228, 1), (229, 4), (230, 2), (231, 3), (232, 4), (233, 1), (234, 4), (235, 3), (236, 2), (237, 5), (238, 5), (239, 5), (240, 1), (241, 2), (242, 2), (243, 2), (244, 3), (245, 5), (246, 1), (247, 12), (248, 2), (249, 8), (250, 2), (251, 2), (252, 2), (253, 3), (254, 1), (255, 3), (256, 5), (257, 1), (258, 1), (259, 2), (260, 2), (261, 2), (262, 5), (263, 3), (264, 5), (265, 3), (266, 2), (267, 5), (268, 3), (269, 4), (270, 2), (271, 78), (272, 5), (273, 3), (274, 17), (275, 4), (276, 3), (277, 1), (278, 2), (279, 5), (280, 10), (281, 2), (282, 4), (283, 1), (284, 2), (285, 3), (286, 1), (287, 5), (288, 3), (289, 2), (290, 3), (291, 5), (292, 3), (293, 3), (294, 3), (295, 2), (296, 5), (297, 2), (298, 3), (299, 2), (300, 2), (301, 4), (302, 3), (303, 1), (304, 2), (305, 5), (306, 2), (307, 2), (308, 5), (309, 1), (310, 2), (311, 4), (312, 4), (313, 3), (314, 14), (315, 1), (316, 3), (317, 4), (318, 3), (319, 1), (320, 5), (321, 4), (322, 1), (323, 3), (324, 2), (325, 1), (326, 1), (327, 1), (328, 11), (329, 2), (330, 4), (331, 2), (332, 7), (333, 5), (334, 7), (335, 4), (336, 3), (337, 7), (338, 4), (339, 3), (340, 4), (341, 2), (342, 6), (343, 3), (344, 3), (345, 4), (346, 2), (347, 3), (348, 3), (349, 3), (350, 8), (351, 4), (352, 6), (353, 1), (354, 5), (355, 3), (356, 2), (357, 4), (358, 2), (359, 5), (360, 1), (361, 6), (362, 7), (363, 7), (364, 4), (365, 4), (366, 4), (367, 6), (368, 4), (369, 2), (370, 4), (371, 3), (372, 1), (373, 1), (374, 21), (375, 2), (376, 5), (377, 1), (378, 1), (379, 2), (380, 2), (381, 5), (382, 2), (383, 1), (384, 8), (385, 2), (386, 5), (387, 16), (388, 1), (389, 2), (390, 2), (391, 5), (392, 3), (393, 18), (394, 5), (395, 6), (396, 2), (397, 6), (398, 11), (399, 8), (400, 4), (401, 8), (402, 2), (403, 3), (404, 4), (405, 4), (406, 4), (407, 2), (408, 3), (409, 4), (410, 3), (411, 11), (412, 3), (413, 10), (414, 4), (415, 6), (416, 4), (417, 3), (418, 2), (419, 4), (420, 4), (421, 5), (422, 5), (423, 3), (424, 3), (425, 3), (426, 5), (427, 3), (428, 8), (429, 4), (430, 1), (431, 5), (432, 3), (433, 2), (434, 6), (435, 4), (436, 9), (437, 4), (438, 7), (439, 4), (440, 4), (441, 12), (442, 2), (443, 3), (444, 9), (445, 1), (446, 7), (447, 3), (448, 1), (449, 3), (450, 1), (451, 6), (452, 2), (453, 1), (454, 2), (455, 3), (456, 3), (457, 3), (458, 11), (459, 4), (460, 18), (461, 1), (462, 3), (463, 3), (464, 2), (465, 5), (466, 3), (467, 5), (468, 6), (469, 2), (470, 2), (471, 4), (472, 12), (473, 1), (474, 1), (475, 3), (476, 5), (477, 15), (478, 4), (479, 2), (480, 1), (481, 3), (482, 3), (483, 2), (484, 2), (485, 2), (486, 4), (487, 2), (488, 2), (489, 3), (490, 5), (491, 6), (492, 4), (493, 2), (494, 8), (495, 2), (496, 2), (497, 5), (498, 4), (499, 1), (500, 2), (501, 4), (502, 8), (503, 3), (504, 1), (505, 2), (506, 1), (507, 5), (508, 3), (509, 1), (510, 4), (511, 5), (512, 3), (513, 12), (514, 2), (515, 7), (516, 3), (517, 2), (518, 2), (519, 2), (520, 3), (521, 5), (522, 3), (523, 1), (524, 5), (525, 12), (526, 4), (527, 3), (528, 8), (529, 6), (530, 4), (531, 5), (532, 5), (533, 4), (534, 4), (535, 2), (536, 5), (537, 3), (538, 2), (539, 3), (540, 6), (541, 2), (542, 33), (543, 2), (544, 1), (545, 1), (546, 2), (547, 7), (548, 5), (549, 2), (550, 6), (551, 3), (552, 5), (553, 8), (554, 7), (555, 1), (556, 2), (557, 3), (558, 2), (559, 4), (560, 2), (561, 2), (562, 2), (563, 3), (564, 4), (565, 4), (566, 6), (567, 2), (568, 4), (569, 4), (570, 4), (571, 3), (572, 1), (573, 2), (574, 1), (575, 3), (576, 2), (577, 1), (578, 2), (579, 7), (580, 1), (581, 6), (582, 2), (583, 4), (584, 6), (585, 12), (586, 1), (587, 5), (588, 1), (589, 2), (590, 7), (591, 7), (592, 2), (593, 1), (594, 3), (595, 3), (596, 12), (597, 1), (598, 3), (599, 2), (600, 16), (601, 2), (602, 1), (603, 5), (604, 4), (605, 3), (606, 1), (607, 3), (608, 2), (609, 4), (610, 2), (611, 2), (612, 1), (613, 2), (614, 3), (615, 2), (616, 1), (617, 1), (618, 4), (619, 2), (620, 2), (621, 2), (622, 1), (623, 2), (624, 15), (625, 2), (626, 5), (627, 1), (628, 2), (629, 5), (630, 2), (631, 9), (632, 4), (633, 2), (634, 5), (635, 3), (636, 4), (637, 9), (638, 3), (639, 8), (640, 2), (641, 5), (642, 1), (643, 4), (644, 2), (645, 4), (646, 9), (647, 3), (648, 3), (649, 1), (650, 4), (651, 1), (652, 5), (653, 5), (654, 3), (655, 5), (656, 2), (657, 3), (658, 5), (659, 3), (660, 2), (661, 5), (662, 5), (663, 1), (664, 4), (665, 5), (666, 29), (667, 12), (668, 9), (669, 10), (670, 7), (671, 5), (672, 7), (673, 3), (674, 5), (675, 3), (676, 6), (677, 4), (678, 8), (679, 3), (680, 4), (681, 11), (682, 3), (683, 1), (684, 5), (685, 3), (686, 2), (687, 3), (688, 3), (689, 11), (690, 3), (691, 4), (692, 2), (693, 6), (694, 2), (695, 2), (696, 3), (697, 4), (698, 3), (699, 5), (700, 4), (701, 3), (702, 1), (703, 2), (704, 4), (705, 1), (706, 2), (707, 3), (708, 3), (709, 1), (710, 5), (711, 8), (712, 2), (713, 1), (714, 5), (715, 2), (716, 1), (717, 3), (718, 5), (719, 2), (720, 5), (721, 3), (722, 5), (723, 4), (724, 5), (725, 3), (726, 4), (727, 5), (728, 4), (729, 1), (730, 4), (731, 4), (732, 2), (733, 2), (734, 1), (735, 4), (736, 2), (737, 4), (738, 4), (739, 5), (740, 1), (741, 2), (742, 5), (743, 2), (744, 4), (745, 6), (746, 7), (747, 8), (748, 2), (749, 5), (750, 2), (751, 2), (752, 5), (753, 1), (754, 1), (755, 4), (756, 4), (757, 3), (758, 1), (759, 3), (760, 7), (761, 9), (762, 4), (763, 4), (764, 8), (765, 3), (766, 6), (767, 3), (768, 2), (769, 2), (770, 4), (771, 4), (772, 3), (773, 3), (774, 3), (775, 2), (776, 3), (777, 4), (778, 3), (779, 4), (780, 1), (781, 3), (782, 7), (783, 2), (784, 2), (785, 5), (786, 2), (787, 5), (788, 3), (789, 4), (790, 2), (791, 5), (792, 3), (793, 1), (794, 2), (795, 9), (796, 8), (797, 5), (798, 3), (799, 5), (800, 3), (801, 6), (802, 1), (803, 5), (804, 6), (805, 6), (806, 3), (807, 2), (808, 2), (809, 2), (810, 1), (811, 4), (812, 2), (813, 7), (814, 2), (815, 5), (816, 3), (817, 5), (818, 2), (819, 3), (820, 2), (821, 6), (822, 3), (823, 5), (824, 1), (825, 8), (826, 4), (827, 2), (828, 6), (829, 3), (830, 3), (831, 7), (832, 4), (833, 3), (834, 2), (835, 2), (836, 1), (837, 6), (838, 5), (839, 1), (840, 1), (841, 2), (842, 4), (843, 1), (844, 3), (845, 1), (846, 2), (847, 3), (848, 2), (849, 1), (850, 3), (851, 1), (852, 1), (853, 3), (854, 3), (855, 2), (856, 1), (857, 3), (858, 2), (859, 5), (860, 5), (861, 3), (862, 3), (863, 1), (864, 1), (865, 2), (866, 1), (867, 1), (868, 4), (869, 2), (870, 4), (871, 1), (872, 3), (873, 2), (874, 2), (875, 3), (876, 2), (877, 8), (878, 3), (879, 3), (880, 4), (881, 7), (882, 19), (883, 6), (884, 4), (885, 1), (886, 3), (887, 5), (888, 5), (889, 2), (890, 4), (891, 17), (892, 1), (893, 2), (894, 1), (895, 2), (896, 6), (897, 4), (898, 2), (899, 5), (900, 1), (901, 3), (902, 5), (903, 1), (904, 1), (905, 4), (906, 3), (907, 2), (908, 2), (909, 4), (910, 3), (911, 6), (912, 3), (913, 6), (914, 2), (915, 4), (916, 6), (917, 5), (918, 2), (919, 3), (920, 4), (921, 4), (922, 2), (923, 2), (924, 2), (925, 5), (926, 34), (927, 1), (928, 13), (929, 5), (930, 3), (931, 1), (932, 1), (933, 5), (934, 1), (935, 8), (936, 3), (937, 2), (938, 4), (939, 1), (940, 1), (941, 2), (942, 1), (943, 3), (944, 3), (945, 7), (946, 3), (947, 4), (948, 3), (949, 4), (950, 6), (951, 23), (952, 2), (953, 5), (954, 5), (955, 3), (956, 4), (957, 10), (958, 6), (959, 1), (960, 2), (961, 6), (962, 2), (963, 1), (964, 1), (965, 1), (966, 3), (967, 4), (968, 3), (969, 3), (970, 7), (971, 1), (972, 7), (973, 3), (974, 1), (975, 7), (976, 4), (977, 30), (978, 3), (979, 3), (980, 2), (981, 1), (982, 4), (983, 1), (984, 4), (985, 7), (986, 2), (987, 2), (988, 4), (989, 1), (990, 4), (991, 4), (992, 2), (993, 2), (994, 4), (995, 7), (996, 3), (997, 3), (998, 6), (999, 6), (1000, 6), (1001, 5), (1002, 5), (1003, 4), (1004, 3), (1005, 2), (1006, 5), (1007, 5), (1008, 17), (1009, 3), (1010, 1), (1011, 3), (1012, 5), (1013, 4), (1014, 4), (1015, 5), (1016, 2), (1017, 2), (1018, 7), (1019, 2), (1020, 4), (1021, 4), (1022, 13), (1023, 5), (1024, 6), (1025, 1), (1026, 2), (1027, 3), (1028, 2), (1029, 1), (1030, 4), (1031, 3), (1032, 5), (1033, 2), (1034, 21), (1035, 1), (1036, 4), (1037, 3), (1038, 7), (1039, 4), (1040, 1), (1041, 3), (1042, 4), (1043, 1), (1044, 2), (1045, 2), (1046, 6), (1047, 2), (1048, 3), (1049, 4), (1050, 7), (1051, 4), (1052, 4), (1053, 1), (1054, 2), (1055, 4), (1056, 2), (1057, 11), (1058, 2), (1059, 4), (1060, 4), (1061, 9), (1062, 3), (1063, 6), (1064, 1), (1065, 2), (1066, 2), (1067, 1), (1068, 2), (1069, 2), (1070, 2), (1071, 23), (1072, 10), (1073, 2), (1074, 3), (1075, 6), (1076, 2), (1077, 5), (1078, 4), (1079, 4), (1080, 2), (1081, 3), (1082, 4), (1083, 6), (1084, 2), (1085, 7), (1086, 2), (1087, 2), (1088, 1), (1089, 5), (1090, 3), (1091, 3), (1092, 3), (1093, 2), (1094, 3), (1095, 4), (1096, 5), (1097, 2), (1098, 1), (1099, 6), (1100, 3), (1101, 3), (1102, 5), (1103, 7), (1104, 5), (1105, 3), (1106, 2), (1107, 3), (1108, 3), (1109, 2), (1110, 1), (1111, 3), (1112, 6), (1113, 4), (1114, 4), (1115, 2), (1116, 4), (1117, 3), (1118, 1), (1119, 5), (1120, 2), (1121, 2), (1122, 26), (1123, 2), (1124, 4), (1125, 1), (1126, 3), (1127, 15), (1128, 2), (1129, 2), (1130, 4), (1131, 4), (1132, 3), (1133, 2), (1134, 3), (1135, 6), (1136, 3), (1137, 3), (1138, 3), (1139, 1), (1140, 5), (1141, 1), (1142, 1), (1143, 3), (1144, 3), (1145, 5), (1146, 5), (1147, 2), (1148, 2), (1149, 3), (1150, 2), (1151, 8), (1152, 16), (1153, 3), (1154, 2), (1155, 2), (1156, 1), (1157, 6), (1158, 3), (1159, 3), (1160, 3), (1161, 3), (1162, 3), (1163, 8), (1164, 2), (1165, 5), (1166, 4), (1167, 8), (1168, 1), (1169, 2), (1170, 3), (1171, 1), (1172, 3), (1173, 4), (1174, 2), (1175, 4), (1176, 8), (1177, 3), (1178, 7), (1179, 2), (1180, 4), (1181, 2), (1182, 2), (1183, 3), (1184, 3), (1185, 3), (1186, 3), (1187, 2), (1188, 2), (1189, 8), (1190, 3), (1191, 3), (1192, 9), (1193, 6), (1194, 1), (1195, 5), (1196, 3), (1197, 2), (1198, 2), (1199, 5), (1200, 3), (1201, 16), (1202, 4), (1203, 3), (1204, 5), (1205, 3), (1206, 2), (1207, 4), (1208, 11), (1209, 2), (1210, 4), (1211, 1), (1212, 2), (1213, 1), (1214, 2), (1215, 6), (1216, 2), (1217, 3), (1218, 3), (1219, 3), (1220, 3), (1221, 4), (1222, 5), (1223, 3), (1224, 8), (1225, 4), (1226, 3), (1227, 3), (1228, 4), (1229, 3), (1230, 2), (1231, 7), (1232, 5), (1233, 2), (1234, 4), (1235, 2), (1236, 4), (1237, 4), (1238, 1), (1239, 10), (1240, 1), (1241, 2), (1242, 4), (1243, 2), (1244, 2), (1245, 168), (1246, 5), (1247, 8), (1248, 2), (1249, 3), (1250, 2), (1251, 1), (1252, 4), (1253, 3), (1254, 10), (1255, 3), (1256, 3), (1257, 8), (1258, 2), (1259, 3), (1260, 1), (1261, 3), (1262, 5), (1263, 2), (1264, 3), (1265, 2), (1266, 5), (1267, 2), (1268, 2), (1269, 3), (1270, 3), (1271, 1), (1272, 6), (1273, 3), (1274, 2), (1275, 1), (1276, 3), (1277, 3), (1278, 6), (1279, 6), (1280, 1), (1281, 2), (1282, 5), (1283, 2), (1284, 5), (1285, 5), (1286, 3), (1287, 1), (1288, 4), (1289, 2), (1290, 5), (1291, 5), (1292, 4), (1293, 1), (1294, 5), (1295, 19), (1296, 4), (1297, 5), (1298, 18), (1299, 7), (1300, 1), (1301, 3), (1302, 3), (1303, 7), (1304, 1), (1305, 2), (1306, 2), (1307, 2), (1308, 3), (1309, 4), (1310, 7), (1311, 3), (1312, 2), (1313, 2), (1314, 4), (1315, 3), (1316, 2), (1317, 3), (1318, 1), (1319, 1), (1320, 2), (1321, 23), (1322, 2), (1323, 8), (1324, 4), (1325, 2), (1326, 2), (1327, 1), (1328, 5), (1329, 1), (1330, 3), (1331, 2), (1332, 3), (1333, 2), (1334, 1), (1335, 2), (1336, 1), (1337, 3), (1338, 1), (1339, 2), (1340, 3), (1341, 2), (1342, 4), (1343, 7), (1344, 9), (1345, 2), (1346, 1), (1347, 2), (1348, 3), (1349, 7), (1350, 6), (1351, 2), (1352, 5), (1353, 4), (1354, 4), (1355, 5), (1356, 4), (1357, 3), (1358, 4), (1359, 2), (1360, 5), (1361, 5), (1362, 22), (1363, 4), (1364, 1), (1365, 3), (1366, 4), (1367, 4), (1368, 1), (1369, 3), (1370, 1), (1371, 2), (1372, 1), (1373, 3), (1374, 3), (1375, 2), (1376, 4), (1377, 4), (1378, 4), (1379, 7), (1380, 4), (1381, 4), (1382, 13), (1383, 2), (1384, 6), (1385, 3), (1386, 2), (1387, 4), (1388, 1), (1389, 4), (1390, 3), (1391, 1), (1392, 2), (1393, 2), (1394, 2), (1395, 3), (1396, 7), (1397, 4), (1398, 2), (1399, 3), (1400, 2), (1401, 3), (1402, 4), (1403, 7), (1404, 8), (1405, 5), (1406, 7), (1407, 5), (1408, 4), (1409, 2), (1410, 4), (1411, 5), (1412, 6), (1413, 4), (1414, 3), (1415, 8), (1416, 1), (1417, 3), (1418, 30), (1419, 4), (1420, 2), (1421, 5), (1422, 1), (1423, 2), (1424, 3), (1425, 3), (1426, 2), (1427, 5), (1428, 4), (1429, 4), (1430, 1), (1431, 4), (1432, 8), (1433, 2), (1434, 3), (1435, 2), (1436, 2), (1437, 2), (1438, 2), (1439, 5), (1440, 4), (1441, 4), (1442, 3), (1443, 3), (1444, 1), (1445, 4), (1446, 2), (1447, 3), (1448, 1), (1449, 2), (1450, 7), (1451, 1), (1452, 5), (1453, 6), (1454, 3), (1455, 10), (1456, 3), (1457, 3), (1458, 3), (1459, 2), (1460, 3), (1461, 3), (1462, 2), (1463, 1), (1464, 5), (1465, 1), (1466, 1), (1467, 2), (1468, 5), (1469, 6), (1470, 1), (1471, 10), (1472, 1), (1473, 4), (1474, 3), (1475, 3), (1476, 3), (1477, 4), (1478, 3), (1479, 7), (1480, 2), (1481, 1), (1482, 3), (1483, 1), (1484, 9), (1485, 1), (1486, 5), (1487, 3), (1488, 6), (1489, 2), (1490, 4), (1491, 42), (1492, 17), (1493, 3), (1494, 1), (1495, 4), (1496, 17), (1497, 3), (1498, 3), (1499, 2), (1500, 3), (1501, 1), (1502, 6), (1503, 4), (1504, 5), (1505, 2), (1506, 1), (1507, 3), (1508, 3), (1509, 3), (1510, 3), (1511, 1), (1512, 2), (1513, 4), (1514, 2), (1515, 1), (1516, 4), (1517, 3), (1518, 5), (1519, 5), (1520, 4), (1521, 12), (1522, 3), (1523, 2), (1524, 2), (1525, 3), (1526, 4), (1527, 4), (1528, 3), (1529, 4), (1530, 4), (1531, 6), (1532, 3), (1533, 2), (1534, 4), (1535, 5), (1536, 6), (1537, 1), (1538, 3), (1539, 4), (1540, 5), (1541, 5), (1542, 2), (1543, 3), (1544, 2), (1545, 9), (1546, 10), (1547, 4), (1548, 2), (1549, 3), (1550, 1), (1551, 4), (1552, 3), (1553, 2), (1554, 2), (1555, 19), (1556, 5), (1557, 1), (1558, 3), (1559, 1), (1560, 4), (1561, 3), (1562, 1), (1563, 74), (1564, 10), (1565, 17), (1566, 4), (1567, 7), (1568, 2), (1569, 1), (1570, 6), (1571, 5), (1572, 5), (1573, 5), (1574, 3), (1575, 4), (1576, 4), (1577, 3), (1578, 4), (1579, 4), (1580, 4), (1581, 4), (1582, 4), (1583, 8), (1584, 3), (1585, 5), (1586, 5), (1587, 16), (1588, 3), (1589, 3), (1590, 5), (1591, 5), (1592, 2), (1593, 5), (1594, 5), (1595, 3), (1596, 4), (1597, 8), (1598, 4), (1599, 6), (1600, 5), (1601, 9), (1602, 14), (1603, 4), (1604, 15), (1605, 9), (1606, 2), (1607, 7), (1608, 3), (1609, 2), (1610, 3), (1611, 4), (1612, 6), (1613, 4), (1614, 3), (1615, 3), (1616, 4), (1617, 3), (1618, 6), (1619, 3), (1620, 6), (1621, 4), (1622, 2), (1623, 7), (1624, 4), (1625, 5), (1626, 2), (1627, 17), (1628, 3), (1629, 2), (1630, 3), (1631, 4), (1632, 4), (1633, 3), (1634, 8), (1635, 6), (1636, 1), (1637, 5), (1638, 12), (1639, 5), (1640, 7), (1641, 6), (1642, 4), (1643, 5), (1644, 8), (1645, 2), (1646, 5), (1647, 5), (1648, 3), (1649, 6), (1650, 3), (1651, 5), (1652, 3), (1653, 4), (1654, 2), (1655, 1), (1656, 4), (1657, 6), (1658, 3), (1659, 5), (1660, 8), (1661, 4), (1662, 3), (1663, 5), (1664, 3), (1665, 4), (1666, 5), (1667, 5), (1668, 3), (1669, 5), (1670, 3), (1671, 3), (1672, 44), (1673, 2), (1674, 5), (1675, 2), (1676, 2), (1677, 3), (1678, 2), (1679, 2), (1680, 4), (1681, 3), (1682, 3), (1683, 5), (1684, 2), (1685, 3), (1686, 8), (1687, 4), (1688, 8), (1689, 4), (1690, 4), (1691, 5), (1692, 4), (1693, 2), (1694, 1), (1695, 3), (1696, 5), (1697, 4), (1698, 4), (1699, 6), (1700, 3), (1701, 5), (1702, 10), (1703, 2), (1704, 3), (1705, 2), (1706, 5), (1707, 3), (1708, 7), (1709, 7), (1710, 4), (1711, 6), (1712, 2), (1713, 1), (1714, 1), (1715, 2), (1716, 5), (1717, 3), (1718, 2), (1719, 2), (1720, 1), (1721, 1), (1722, 4), (1723, 2), (1724, 4), (1725, 3), (1726, 2), (1727, 3), (1728, 4), (1729, 10), (1730, 4), (1731, 4), (1732, 1), (1733, 5), (1734, 6), (1735, 3), (1736, 2), (1737, 2), (1738, 3), (1739, 4), (1740, 4), (1741, 5), (1742, 5), (1743, 1), (1744, 6), (1745, 3), (1746, 2), (1747, 2), (1748, 3), (1749, 9), (1750, 2), (1751, 3), (1752, 3), (1753, 2), (1754, 14), (1755, 2), (1756, 4), (1757, 2), (1758, 4), (1759, 5), (1760, 2), (1761, 6), (1762, 10), (1763, 8), (1764, 3), (1765, 2), (1766, 5), (1767, 8), (1768, 7), (1769, 6), (1770, 2), (1771, 4), (1772, 10), (1773, 4), (1774, 31), (1775, 2), (1776, 3), (1777, 3), (1778, 2), (1779, 6), (1780, 6), (1781, 3), (1782, 2), (1783, 2), (1784, 3), (1785, 3), (1786, 7), (1787, 13), (1788, 5), (1789, 4), (1790, 2), (1791, 3), (1792, 3), (1793, 4), (1794, 2), (1795, 5), (1796, 5), (1797, 2), (1798, 1), (1799, 2), (1800, 4), (1801, 5), (1802, 3), (1803, 2), (1804, 6), (1805, 6), (1806, 4), (1807, 4), (1808, 3), (1809, 2), (1810, 15), (1811, 4), (1812, 10), (1813, 3), (1814, 9), (1815, 8), (1816, 4), (1817, 6), (1818, 7), (1819, 5), (1820, 4), (1821, 3), (1822, 3), (1823, 2), (1824, 5), (1825, 2), (1826, 12), (1827, 2), (1828, 3), (1829, 3), (1830, 3), (1831, 3), (1832, 3), (1833, 15), (1834, 3), (1835, 7), (1836, 3), (1837, 2), (1838, 5), (1839, 5), (1840, 5), (1841, 3), (1842, 2), (1843, 4), (1844, 5), (1845, 2), (1846, 65), (1847, 4), (1848, 2), (1849, 2), (1850, 1), (1851, 2), (1852, 2), (1853, 2), (1854, 2), (1855, 4), (1856, 5), (1857, 3), (1858, 8), (1859, 4), (1860, 4), (1861, 16), (1862, 3), (1863, 5), (1864, 4), (1865, 1), (1866, 3), (1867, 1), (1868, 3), (1869, 5), (1870, 4), (1871, 3), (1872, 2), (1873, 2), (1874, 2), (1875, 3), (1876, 5), (1877, 5), (1878, 7), (1879, 2), (1880, 2), (1881, 5), (1882, 3), (1883, 3), (1884, 6), (1885, 12), (1886, 4), (1887, 4), (1888, 1), (1889, 3), (1890, 3), (1891, 2), (1892, 1), (1893, 4), (1894, 40), (1895, 3), (1896, 2), (1897, 2), (1898, 2), (1899, 3), (1900, 4), (1901, 4), (1902, 2), (1903, 2), (1904, 2), (1905, 32), (1906, 5), (1907, 2), (1908, 3), (1909, 2), (1910, 2), (1911, 3), (1912, 4), (1913, 2), (1914, 9), (1915, 4), (1916, 8), (1917, 2), (1918, 6), (1919, 1), (1920, 1), (1921, 2), (1922, 5), (1923, 2), (1924, 3), (1925, 4), (1926, 3), (1927, 3), (1928, 3), (1929, 3), (1930, 8), (1931, 3), (1932, 3), (1933, 4), (1934, 8), (1935, 5), (1936, 6), (1937, 4), (1938, 3), (1939, 10), (1940, 2), (1941, 2), (1942, 10), (1943, 3), (1944, 5), (1945, 4), (1946, 2), (1947, 2), (1948, 3), (1949, 4), (1950, 3), (1951, 1), (1952, 3), (1953, 6), (1954, 5), (1955, 6), (1956, 4), (1957, 1), (1958, 4), (1959, 4), (1960, 2), (1961, 3), (1962, 3), (1963, 1), (1964, 2), (1965, 2), (1966, 4), (1967, 2), (1968, 6), (1969, 4), (1970, 1), (1971, 6), (1972, 6), (1973, 5), (1974, 3), (1975, 2), (1976, 8), (1977, 3), (1978, 2), (1979, 5), (1980, 3), (1981, 10), (1982, 6), (1983, 7), (1984, 5), (1985, 8), (1986, 4), (1987, 2), (1988, 4), (1989, 12), (1990, 3), (1991, 4), (1992, 7), (1993, 2), (1994, 3), (1995, 4), (1996, 3), (1997, 3), (1998, 4), (1999, 5), (2000, 4), (2001, 1), (2002, 4), (2003, 6), (2004, 4), (2005, 7), (2006, 6), (2007, 4), (2008, 1), (2009, 3), (2010, 2), (2011, 3), (2012, 4), (2013, 17), (2014, 4), (2015, 5), (2016, 5), (2017, 3), (2018, 3), (2019, 2), (2020, 1), (2021, 8), (2022, 5), (2023, 2), (2024, 1), (2025, 5), (2026, 5), (2027, 8), (2028, 4), (2029, 7), (2030, 1), (2031, 5), (2032, 8), (2033, 19), (2034, 3), (2035, 2), (2036, 4), (2037, 5), (2038, 4), (2039, 1), (2040, 6), (2041, 2), (2042, 1), (2043, 1), (2044, 5), (2045, 4), (2046, 2), (2047, 6), (2048, 5), (2049, 10), (2050, 3), (2051, 3), (2052, 4), (2053, 5), (2054, 3), (2055, 3), (2056, 3), (2057, 3), (2058, 1), (2059, 2), (2060, 2), (2061, 4), (2062, 3), (2063, 3), (2064, 4), (2065, 7), (2066, 5), (2067, 2), (2068, 3), (2069, 5), (2070, 6), (2071, 6), (2072, 6), (2073, 3), (2074, 4), (2075, 2), (2076, 3), (2077, 3), (2078, 1), (2079, 3), (2080, 4), (2081, 3), (2082, 11), (2083, 1), (2084, 7), (2085, 6), (2086, 2), (2087, 5), (2088, 2), (2089, 2), (2090, 4), (2091, 7), (2092, 14), (2093, 6), (2094, 4), (2095, 3), (2096, 3), (2097, 1), (2098, 1), (2099, 2), (2100, 1), (2101, 1), (2102, 4), (2103, 3), (2104, 2), (2105, 2), (2106, 3), (2107, 1), (2108, 3), (2109, 4), (2110, 2), (2111, 6), (2112, 7), (2113, 6), (2114, 1), (2115, 1), (2116, 1), (2117, 3), (2118, 3), (2119, 3), (2120, 3), (2121, 2), (2122, 3), (2123, 5), (2124, 3), (2125, 6), (2126, 3), (2127, 2), (2128, 4), (2129, 3), (2130, 3), (2131, 3), (2132, 4), (2133, 5), (2134, 9), (2135, 2), (2136, 3), (2137, 2), (2138, 5), (2139, 3), (2140, 3), (2141, 1), (2142, 2), (2143, 1), (2144, 2), (2145, 2), (2146, 3), (2147, 4), (2148, 6), (2149, 5), (2150, 3), (2151, 4), (2152, 8), (2153, 2), (2154, 2), (2155, 2), (2156, 2), (2157, 2), (2158, 1), (2159, 1), (2160, 3), (2161, 2), (2162, 3), (2163, 2), (2164, 2), (2165, 1), (2166, 2), (2167, 4), (2168, 3), (2169, 4), (2170, 4), (2171, 1), (2172, 6), (2173, 3), (2174, 2), (2175, 3), (2176, 3), (2177, 2), (2178, 9), (2179, 5), (2180, 1), (2181, 2), (2182, 4), (2183, 2), (2184, 1), (2185, 2), (2186, 4), (2187, 1), (2188, 3), (2189, 3), (2190, 3), (2191, 7), (2192, 3), (2193, 6), (2194, 3), (2195, 3), (2196, 1), (2197, 3), (2198, 5), (2199, 5), (2200, 5), (2201, 11), (2202, 4), (2203, 1), (2204, 1), (2205, 2), (2206, 3), (2207, 7), (2208, 3), (2209, 14), (2210, 2), (2211, 2), (2212, 3), (2213, 3), (2214, 3), (2215, 1), (2216, 1), (2217, 4), (2218, 7), (2219, 1), (2220, 2), (2221, 3), (2222, 3), (2223, 3), (2224, 3), (2225, 10), (2226, 1), (2227, 7), (2228, 3), (2229, 6), (2230, 6), (2231, 2), (2232, 5), (2233, 4), (2234, 3), (2235, 2), (2236, 16), (2237, 5), (2238, 3), (2239, 1), (2240, 2), (2241, 5), (2242, 2), (2243, 4), (2244, 2), (2245, 2), (2246, 1), (2247, 7), (2248, 2), (2249, 3), (2250, 2), (2251, 2), (2252, 3), (2253, 3), (2254, 2), (2255, 3), (2256, 2), (2257, 1), (2258, 6), (2259, 4), (2260, 1), (2261, 1), (2262, 2), (2263, 4), (2264, 2), (2265, 5), (2266, 1), (2267, 3), (2268, 1), (2269, 2), (2270, 5), (2271, 2), (2272, 4), (2273, 4), (2274, 4), (2275, 2), (2276, 5), (2277, 2), (2278, 6), (2279, 4), (2280, 8), (2281, 3), (2282, 5), (2283, 1), (2284, 12), (2285, 11), (2286, 2), (2287, 2), (2288, 2), (2289, 4), (2290, 2), (2291, 2), (2292, 2), (2293, 1), (2294, 4), (2295, 12), (2296, 3), (2297, 5), (2298, 2), (2299, 1), (2300, 2), (2301, 2), (2302, 2), (2303, 6), (2304, 2), (2305, 2), (2306, 2), (2307, 7), (2308, 1), (2309, 1), (2310, 3), (2311, 2), (2312, 6), (2313, 3), (2314, 1), (2315, 1), (2316, 1), (2317, 2), (2318, 3), (2319, 5), (2320, 4), (2321, 2), (2322, 4), (2323, 1), (2324, 3), (2325, 3), (2326, 2), (2327, 3), (2328, 5), (2329, 1), (2330, 6), (2331, 3), (2332, 1), (2333, 1), (2334, 3), (2335, 3), (2336, 5), (2337, 1), (2338, 1), (2339, 1), (2340, 1), (2341, 2), (2342, 5), (2343, 3), (2344, 2), (2345, 7), (2346, 4), (2347, 3), (2348, 1), (2349, 3), (2350, 2), (2351, 3), (2352, 2), (2353, 1), (2354, 2), (2355, 6), (2356, 1), (2357, 1), (2358, 1), (2359, 5), (2360, 2), (2361, 1), (2362, 5), (2363, 4), (2364, 1), (2365, 2), (2366, 1), (2367, 3), (2368, 1), (2369, 4), (2370, 2), (2371, 1), (2372, 1), (2373, 1), (2374, 1), (2375, 1), (2376, 5), (2377, 2), (2378, 5), (2379, 3), (2380, 1), (2381, 3), (2382, 1), (2383, 3), (2384, 1), (2385, 2), (2386, 5), (2387, 1), (2388, 5), (2389, 1), (2390, 1), (2391, 2), (2392, 1), (2393, 1), (2394, 3), (2395, 2), (2396, 2), (2397, 3), (2398, 4), (2399, 2), (2400, 3), (2401, 1), (2402, 2), (2403, 1), (2404, 1), (2405, 1), (2406, 3), (2407, 2), (2408, 2), (2409, 1), (2410, 2), (2411, 1), (2412, 4), (2413, 6), (2414, 1), (2415, 3), (2416, 2), (2417, 1), (2418, 1), (2419, 1), (2420, 3), (2421, 2), (2422, 1), (2423, 1), (2424, 2), (2425, 1), (2426, 5), (2427, 2), (2428, 2), (2429, 1), (2430, 1), (2431, 1), (2432, 1), (2433, 1), (2434, 1), (2435, 2), (2436, 1), (2437, 1), (2438, 1), (2439, 1), (2440, 2), (2441, 4), (2442, 1), (2443, 1), (2444, 4), (2445, 1), (2446, 3), (2447, 3), (2448, 2), (2449, 1), (2450, 3), (2451, 3), (2452, 2), (2453, 3), (2454, 1), (2455, 2), (2456, 1), (2457, 1), (2458, 2), (2459, 6), (2460, 5), (2461, 3), (2462, 14), (2463, 1), (2464, 3), (2465, 3), (2466, 3), (2467, 2), (2468, 3), (2469, 5), (2470, 2), (2471, 5), (2472, 2), (2473, 1), (2474, 2), (2475, 4), (2476, 2), (2477, 2), (2478, 1), (2479, 2), (2480, 1), (2481, 1), (2482, 2), (2483, 4), (2484, 4)]



![png](Project%203_files/Project%203_46_1.png)


- It can be seen that few papers have a very high degree, which are likely to be papers that are important and are cited a lot by other papers. This is what we expect, since most papers are not well cited, and only a few are cited very often.

- Another graph is now drawn, coloured according to each node's degree:


```python
# Plot the graph, colouring nodes by degree
# Fix layout for consistency
pos=nx.spring_layout(g1)
# Degree is scaled to increase node visibility by colours
colors=[deg[1]**0.1 for deg in g1.degree()]
cmap=plt.cm.inferno
vmin = min(colors)
vmax = max(colors)
fig, ax = plt.subplots(figsize=(18,18))
nx.draw_networkx(g1, ax=ax, pos=pos, node_color=colors, cmap=cmap,
           node_size=30, with_labels=False, alpha=0.5, vmin=vmin, vmax=vmax)

# make colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin**10, vmax=vmax**10))
sm._A = []
plt.colorbar(sm)
plt.title('Citation graph coloured by node degrees', fontsize=20)
plt.show()
```


![png](Project%203_files/Project%203_48_0.png)


- It can be seen that the nodes with high degrees are mostly in the center of the figure, with some highly visible nodes with very high degrees in the lighter colours.

#### Betweeness centrality
- The betweeness centrality ranks highly nodes that lie in a lot of paths between nodes. It is calculated by computing all minimum paths between any two nodes in the graph and counting the number of the paths that go through each node.
- Similarly, a histogram is plotted:


```python
# Compute betweeness centrality for each node
bet_cen = nx.betweenness_centrality(g1)
sorted_bet_cen = sorted(bet_cen.items(), key=itemgetter(1), reverse=True)
nsorted_bet_cen = sorted(bet_cen.items(), key=itemgetter(0), reverse=False)

fig, ax = plt.subplots(figsize=(8, 6))
plt.hist([x[1] for x in nsorted_bet_cen], bins=20)
fig.suptitle('Histogram of Node Betweeness Centrality')
ax.set_xlabel('Betweeness Centrality')
ax.set_ylabel('Frequency')
plt.yscale('log')
plt.show()

# Set as atrribute
# nx.set_node_attributes(g1, bet_cen, 'bet_cen')
```


![png](Project%203_files/Project%203_51_0.png)



```python
# Plot graph, colouring nodes by betweeness centrality
pos=nx.spring_layout(g1)
# Scale nodes to increase visibility by colouring
colors=[x**0.2 for i,x in bet_cen.items()]
cmap=plt.cm.inferno
vmin = min(colors)
vmax = max(colors)
fig, ax = plt.subplots(figsize=(18,18))
nx.draw_networkx(g1, ax=ax, pos=pos, node_color=colors, cmap=cmap,
           node_size=30, with_labels=False, alpha=0.5, vmin=vmin, vmax=vmax)
# make colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin**5, vmax=vmax**5))
sm._A = []
plt.colorbar(sm)
plt.title('Citation graph coloured by node betweenness centrality', fontsize=20)
plt.show()
```


![png](Project%203_files/Project%203_52_0.png)


- The nodes with a high betweenness centrality score differ slightly from the degree and pagerank score, since it scores highly nodes that are passed through often, which might mean nodes that have a low degree can have a high betweeness centrality score.

- This can be seen in this graph, where some nodes near the extremeties

#### Pagerank
- Pagerank: The pagerank algorithm assigns a centrality measure to each node by performing a kind of random walk, which at every step follows the graph with probabiltiy $\alpha$, and randomly jumps to a different node with probability $(1-\alpha)$,  $0<\alpha<1$. The centrality measure is then the stationary vector of this process.



```python
# Make histogram of pagerank
pagerank = nx.pagerank(g1, alpha=0.85)
sorted_pagerank = sorted(pagerank.items(), key=itemgetter(1), reverse=True)
nsorted_pagerank = sorted(pagerank.items(), key=itemgetter(0), reverse=False)

fig, ax = plt.subplots(figsize=(8, 6))
plt.hist([x[1] for x in nsorted_pagerank], bins=20)
fig.suptitle('Histogram of Node Pagerank')
ax.set_xlabel('Pagerank')
ax.set_ylabel('Frequency')
plt.yscale('log')
plt.show()

# Set as atrribute
# nx.set_node_attributes(g1, pagerank, 'pagerank')
```


![png](Project%203_files/Project%203_55_0.png)



```python
# Plot graph, colouring nodes by pagerank
pos=nx.spring_layout(g1)
colors=[x**0.1 for i,x in pagerank.items()]
cmap=plt.cm.inferno
vmin = min(colors)
vmax = max(colors)
fig, ax = plt.subplots(figsize=(18,18))
nx.draw_networkx(g1, ax=ax, pos=pos, node_color=colors, cmap=cmap,
           node_size=30, with_labels=False, alpha=0.5, vmin=vmin, vmax=vmax)
# Make colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin**10, vmax=vmax**10))
sm._A = []
plt.colorbar(sm)
plt.title('Citation graph coloured by node pagerank', fontsize=20)
plt.show()
```


![png](Project%203_files/Project%203_56_0.png)


- As is shown below, the pagerank scores and very similar to the degree scores, and therefore so is the resulting graph plotted. Nodes which are visited often from a random walk score highly as represented in the graph.

### Correlations between centrality measures


```python
# Plot correlations between all centrality measures
fig, axes = plt.subplots(1, 3, figsize=(17, 6))

axes[0].scatter([x[1] for x in nsorted_bet_cen], [x[1] for x in nsorted_pagerank])
axes[0].set_title(f'correlation = {round(np.corrcoef([x[1] for x in nsorted_bet_cen], [x[1] for x in nsorted_pagerank])[0][1], 3)}')
axes[0].set_xlabel('Betweenness Centrality')
axes[0].set_ylabel('Pagerank')

axes[1].scatter([x[1] for x in nsorted_bet_cen], [x[1] for x in degree])
axes[1].set_title(f'correlation = {round(np.corrcoef([x[1] for x in nsorted_bet_cen], [x[1] for x in degree])[0][1], 3)}')
axes[1].set_xlabel('Betweenness Centrality')
axes[1].set_ylabel('Degree')

axes[2].scatter([x[1] for x in nsorted_pagerank], [x[1] for x in degree])
axes[2].set_title(f'correlation = {round(np.corrcoef([x[1] for x in nsorted_pagerank], [x[1] for x in degree])[0][1], 3)}')
axes[2].set_xlabel('Pagerank')
axes[2].set_ylabel('Degree')

fig.suptitle('Correlations between all centrality measures', fontsize=20)
plt.show()
```


![png](Project%203_files/Project%203_59_0.png)


- A high betweeness centrality means that the node connects many nodes together. **These nodes are often BETWEEN groups, not within groups, which differ from the other measures of centrality.**

- This explains the lower correlation for betweenness centrality with the other mesures (compared to the very high correlation for pagerank and degree, which are expected to be highly correlated).

- Since pagerank and degree are highly correlated, it is likely that, for example, websites that want to be ranked highly (by a simple pagerank algorithm) should aim to connect to as many other websites as possible, as would be expected. 

- A paper with a high pagerank means that that paper is very important, not only because it is cited many times (high degree) but also because it is likely that other important papers cite it, which is reflected in the high pagerank score, since a random walk is likely to visit this paper from other highly visited papers.

- The nodes which are highly central according to each of the measures is now displayed as a table:



```python
a = [int(x[0]) for x in sorted_degree[:30]]
b = [int(x[0]) for x in sorted_bet_cen[:30]]
c = [int(x[0]) for x in sorted_pagerank[:30]]

df_table = pd.DataFrame(np.zeros((30, 3)), columns=['Degree', 'Bet-Cen', 'Pagerank'])
df_table.loc[:30, 'Degree'] = pd.Series(a)
df_table.loc[:30, 'Bet-Cen'] = pd.Series(b)
df_table.loc[:30, 'Pagerank'] = pd.Series(c)
print('-------------------------------------------')
print('-------------------------------------------')
print('Nodes that have top 30 scores for all 3 centrality measures:')
print(list(set(a) & set(b) & set(c)))
print('-------------------------------------------')
print('-------------------------------------------')
print('Top 30 nodes for each measure:')
print('-------------------------------------------')
df_table.astype(int)
```

    -------------------------------------------
    -------------------------------------------
    Nodes that have top 30 scores for all 3 centrality measures:
    [1122, 1894, 542, 1672, 1034, 1418, 78, 271, 977, 1905, 882, 1846, 666, 1563, 1245, 926, 95]
    -------------------------------------------
    -------------------------------------------
    Top 30 nodes for each measure:
    -------------------------------------------





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Degree</th>
      <th>Bet-Cen</th>
      <th>Pagerank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1245</td>
      <td>1245</td>
      <td>1245</td>
    </tr>
    <tr>
      <th>1</th>
      <td>271</td>
      <td>1846</td>
      <td>1563</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1563</td>
      <td>1894</td>
      <td>1846</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1846</td>
      <td>1563</td>
      <td>271</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1672</td>
      <td>271</td>
      <td>1672</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1491</td>
      <td>977</td>
      <td>1894</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1894</td>
      <td>926</td>
      <td>1491</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78</td>
      <td>1672</td>
      <td>78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>926</td>
      <td>78</td>
      <td>542</td>
    </tr>
    <tr>
      <th>9</th>
      <td>542</td>
      <td>95</td>
      <td>926</td>
    </tr>
    <tr>
      <th>10</th>
      <td>95</td>
      <td>1989</td>
      <td>1774</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1905</td>
      <td>2093</td>
      <td>1321</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1774</td>
      <td>882</td>
      <td>666</td>
    </tr>
    <tr>
      <th>13</th>
      <td>977</td>
      <td>666</td>
      <td>1905</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1418</td>
      <td>542</td>
      <td>95</td>
    </tr>
    <tr>
      <th>15</th>
      <td>666</td>
      <td>1905</td>
      <td>1418</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1122</td>
      <td>1122</td>
      <td>977</td>
    </tr>
    <tr>
      <th>17</th>
      <td>951</td>
      <td>585</td>
      <td>1122</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1071</td>
      <td>1052</td>
      <td>951</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1321</td>
      <td>1034</td>
      <td>1071</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1362</td>
      <td>376</td>
      <td>374</td>
    </tr>
    <tr>
      <th>21</th>
      <td>84</td>
      <td>1418</td>
      <td>2033</td>
    </tr>
    <tr>
      <th>22</th>
      <td>374</td>
      <td>891</td>
      <td>1555</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1034</td>
      <td>472</td>
      <td>103</td>
    </tr>
    <tr>
      <th>24</th>
      <td>103</td>
      <td>1754</td>
      <td>1034</td>
    </tr>
    <tr>
      <th>25</th>
      <td>882</td>
      <td>1706</td>
      <td>84</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1295</td>
      <td>1000</td>
      <td>600</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1555</td>
      <td>600</td>
      <td>1298</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2033</td>
      <td>1861</td>
      <td>1362</td>
    </tr>
    <tr>
      <th>29</th>
      <td>393</td>
      <td>411</td>
      <td>882</td>
    </tr>
  </tbody>
</table>
</div>



### Central Nodes
- A reduced graph showing only the highly central nodes according to the three centrality measures are now drawn:


```python
central_nodes = list(set(a) | set(b) | set(c))
# Function to calculate node size depending on centrality
def get_size(node, central_nodes):
    if node in central_nodes:
        return 200
    else:
        return 10

def get_color(node, central_nodes):
    if node in central_nodes:
        return 'red'
    else:
        return 'black'

# Plot the graph showing most central nodes
# Fix layout for consistency
pos=nx.spring_layout(g1)
cmap=plt.cm.inferno
vmin = min(colors)
vmax = max(colors)
nodelist = g1.nodes()
node_size = [get_size(x, central_nodes) for x in nodelist]
node_color = [get_color(x, central_nodes) for x in nodelist]
fig, ax = plt.subplots(figsize=(18,18))
nx.draw_networkx(g1, ax=ax, pos=pos, cmap=cmap, nodelist=nodelist, node_color=node_color,
           node_size=node_size, with_labels=False, alpha=0.4, vmin=vmin, vmax=vmax)

plt.title('Citation graph showing most central nodes', fontsize=20)
plt.show()
```


![png](Project%203_files/Project%203_64_0.png)


- These most central nodes according to the three measures are likely important papers that many other papers have cited (or less likely, that the paper itself cited many other papers).

## 1.3 Community detection on the citation graph
---
- The Clauset-Newman-Moore greedy modularity maximisation algorithm will be used to both choose the optimal number of clusters and the corresponding clustering of the feature matrix.


```python
CNM_cluster = [list(x) for x in CNM(g1)]

# Construct dictionary assinging nodes to cluster
# CNM_cluster_dict = {}
CNM_labels = np.zeros(len(adj_mat[0]), dtype=int)
for i, x in enumerate(CNM_cluster):
    # CNM_cluster_dict.update(dict.fromkeys(x, i))
    for y in x:
        CNM_labels[y] = i

fig, ax = plt.subplots(figsize=(18,18))
nx.draw_networkx(g1, ax=ax, node_color=CNM_labels, node_size=20, with_labels=False, alpha=0.8, cmap='gist_rainbow')
plt.title('Graph clustered using Clauset-Newman-Moore algorithm')
plt.show()

k_star = len(set(CNM_labels))
print('k* = ' + str(k_star))
```


![png](Project%203_files/Project%203_67_0.png)


    k* = 29


- The network nicely displays the clusterings given by the CNM algorithm using different colours. The distinct clusters can be easily seen at the extremeties of the graph.

### Distribution of Central Nodes
The distribution of central nodes across the k* communities will now be analyzed.
- First the frequnecy of the top 30 central nodes by degree and pagerank are visualised across communities:


```python
# Calculate frequency of central nodes across communities
a_dist = np.zeros(k_star)
c_dist = np.zeros(k_star)
for node in a:
    a_dist[CNM_labels[node]] += 1
for node in c:
    c_dist[CNM_labels[node]] += 1

# Plot frequency distributions across communities
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].bar(range(k_star), a_dist)
axes[0].set_xlabel('Community')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Distribution of top 30 nodes by degree across {k_star} CNM communities')

axes[1].bar(range(k_star), c_dist)
axes[1].set_xlabel('Community')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Distribution of top 30 nodes be pagerank across {k_star} CNM communities')

plt.show()
```


![png](Project%203_files/Project%203_70_0.png)


- The the Clauset-Newman-Moore greedy modu-
larity maximisation algorithm (CNM) works by greedily searching which communities to join that maximises modularity at each step. 

- Networks with high modularity have many connections between nodes within the same community, but few connections between nodes in different communities.

- From the frequency plots, we can see that all the highly central nodes according to degree and pagerank are distributed in the first communities, increasing in frequency as the community label decreases.

- This can be explained by the greedy CNM algorithm: the algorithm is likely to join communities with closeby central nodes, since these communities are likely to have many connections between them, whilst not necessarily having many connections to other communities. 

- Now the central nodes are visualised on the graph itself:



```python
# Plot graph showing bigger central nodes
node_size = [get_size(x, list(set(a) | set(c))) for x in g1.nodes()]
fig, ax = plt.subplots(figsize=(18,18))
nx.draw_networkx(g1, ax=ax, node_color=CNM_labels, node_size=node_size, with_labels=False, alpha=0.8, cmap='gist_rainbow')
plt.title('Graph clustered using Clauset-Newman-Moore algorithm, showing central nodes')
plt.show()
```


![png](Project%203_files/Project%203_73_0.png)


- From the graph, it can be seen that there are some highly central and dense communities containing many central nodes and connections, constrasted with sparse communities that have little to no central nodes, often displayed at the extremeties of the graph.

- This again is a reflection of the CNM algorithm, and better expresses how it creates unbalanced communities in terms of density and centrality of nodes.

## 1.4 Comparing feature and graph clusterings
---
The ARI and AMI scores are computed to compare the k-means and the CNM clusterings.

- The ARI score represents how far away the clustering is from a purely random assignment.
- The AMI score evaluates the information contained in one clustering from the other.

- Note that the clusterings that are being compared are for two different datasets, so that there may be two documents that are very close to eachother in content, but do not cite eachother. This is likely to lead to different clusterings regardless of the method of clustering from the start.




```python
AMI = adjusted_mutual_info_score(CNM_labels, kmeans.labels_)
ARI = adjusted_rand_score(CNM_labels, kmeans.labels_)

print(AMI)
print(ARI)
```

    0.15315991285605193
    0.06733446498800282


- The AMI and ARI scores can be used both to evaluate clustering quality and to evaulate similarities between different clusterings.
- The low ARI score suggests the two clusterings are not very similar to eachother, since a value close to 0 indicates a random labeling of the clusters.

- Similarly the low AMI score suggests that the samples are split among very different clusters.

- The two clusterings are now plot on graphs:



```python
# Fit the kmeans algorithm
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=1, random_state=42)
kmeans.fit(X)

pos=nx.spring_layout(g1)

# Plot graph with two different clusterings
fig, axes = plt.subplots(2, 1, figsize=(18,24))
nx.draw_networkx(g1, ax=axes[0], node_color=kmeans.labels_, node_size=20, with_labels=False, alpha=0.8, cmap='gist_rainbow')
axes[0].set_title(f'Graph clustered using K-means clustering, with {k_optimal} clusters')

nx.draw_networkx(g1, ax=axes[1], node_color=CNM_labels, node_size=20, with_labels=False, alpha=0.8, cmap='gist_rainbow')
axes[1].set_title(f'Graph clustered using Clauset-Newman-Moore algorithm, with {k_star} clusters')
plt.show()
```


![png](Project%203_files/Project%203_78_0.png)


- The clustering from k-means looks less clustered than the CNM clustering in this layout. This suggests that maybe the CNM clustering is a better choice in this scenario. The K-means clusters seem to yield strange results, where some sparsely connected nodes at the extremeties of the graph are not in the same cluster when it looks like they should be.
- Note that the k-means also has fewer clusters than with the CNM clustering.

- A heatmap between the two clusterings is now plotted in order to visualise the similiarities/differences in a different way:


```python
fig = plt.figure(figsize=(15,11))
sns.heatmap(confusion_matrix(kmeans.labels_, CNM_labels))
plt.title('Heatmap between kmeans and CNM clustering', fontsize=20)
plt.xlabel('K-means clusters')
plt.ylabel('CNM clusters')
plt.show()
```


![png](Project%203_files/Project%203_81_0.png)


- Again, this heatmap does not show at all a strong correspondance between the clusters. If there were, we would expect hot points for each row and column, but there is little evidence of this in theis heatmap. 
- However the 0th kmeans cluster and the 6th CNM cluster seem to be very similar.


```python

```

---
---
# Classification of a set of images
---
---

## Introduction
---
The next task will focus on the classification and clustering of 28x28 greyscale images of fashion items using various techniques for supervised and unsupervised learning.

## Data Preparation

The data is split up into a training and test set. Some of the images are also displayed so we know what kind of images we are dealing with.

- BALANCE THIS SPLIT?! + RANDOM SAMPLE IT?


```python
# Load in the fashion dataset from sklearn
mnist = fetch_openml('Fashion-MNIST', cache=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Split into training and test sets using stratified random sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

# Create mapping of labels
fashion_map = ['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boots']
fashion_colors = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds'] * 2

# Display some labelled images in the dataset
dim = 8
fig, axes = plt.subplots(dim, dim, figsize=(12,15))

for i in range(dim):
    for j in range(dim):
        axes[i, j].imshow(X[j*dim+i].reshape(28, 28) , cmap=fashion_colors[y[j*dim+i]])
        axes[i, j].set_title(fashion_map[y[j*dim+i]], fontsize=14)
        axes[i, j].axis('off')
plt.show()
```


![png](Project%203_files/Project%203_88_0.png)


## Scale/Orientation of Data
- The displayed images are in 5 colours just to make them stand out more, but in the data they are just greyscale images. Also, note that all clothing items are of the same orientation, including all the shoes which point to the left. This makes it easier for a classifier to classify these images, but will probably not work well for data with different orientations. 

- If we wanted to train a classifier that could recognize the clothing items irrespective of their orientation/scale, we could rescale/mirror/rotate the original data in order to create a much bigger dataset with all the augmented images included. The hope then is that the model would be able to reognize items of clothing irrepective of their orientation/scale. We might however expect this model's accuracy to be loewr since this is a harder task.

## 2.1 Unsupervised clustering of the image dataset
---

The following introduction to clustering and k-means is from part 1, and is included for completeness.

- The aim of unsupervised clustering of data is to find groups that partition the samples, such that points in a group are more similar to eachother than points from different groups.
- More formally, let N be the number of sample and K the number of clusters. The aim is to minimize
$$ Trace \Big[ (H^TH)^{-1} [H^TDH] \Big]
$$
where H is the NxK assigment matrix mapping each sample to one of k clusters, and D is the NxN distance (or measure of 'closeness') matrix between all the samples. The elements on the diagonals of the KxK matrix $[H^TDH]$ correspond to the sums of the distances between all points within the same cluster, so that the sum of these distances is minimized. $(H^TH)^{-1}$ is used to normalize the diagonals of the matrix $[H^TDH]$ using the sizes of the groups.


### K-means Algorithm
- Complete enumeration in order to find this global minimum is not computationally feasible for any sufficiently large dataset.
- The K-means algorithm uses a heuristic to reduce the computational cost of finding a solution (but not necessarily the global solution) to this problem.
- The algorithm starts by randomly assinging each sample to one of the k clusters. At each iteration until convergence, each sample is reassigned to the cluster whose centroid is closest to the sample.
- The algorithm will terminate when some condition is met, such as when the clustering doesn't change between iterations or after a set number of iterations is reached.


### Measures of 'closeness' between samples
- There needs to be decisions made about how to define '**closeness**' between points, and **how many groups** to split samples into.
- Measures of similarity/dissimilarity to consider:
    - Euclidean
    - Manhattan Distance
    - Cosine similarity (measure of vector alignment)
    - Correlation

- The **euclidean distance** will be used in this example, although the other distances could be considered. The sklearn implementation of K-Means uses the Euclidean distance so it is also convenient to use.

- In this example, the samples are 784 dimensional vectors.

### Choosing the number of clusters, k
- A range of values for k will be tested. The aim is to find the value of k for which the clustering is optimal by some metric/metrics.

- As discussed in the 'Randomness in K-Means Clustering' section, n_init would ideally be set to something like 10, so that the K-means algorithm runs this many times and the best model is chosen (in terms of inertia). However computing time is very large so I have chosen to set n_init to 1. 


```python
# Setup list of numbers of clusters and metrics
k_values = np.linspace(2, 30, 6, dtype=int)
CH = []
DB = []
sil = []

for k in k_values:
    if verbose: print(f'Running k-means for k={k}')
    # Use K-means algorithm to cluster papers with k clusters
    kmeans = KMeans(n_clusters=k, n_init=1, random_state=42)
    kmeans.fit(X_train)

    # Store Calinski-Harabasz score
    CH.append(CH_score(X_train, kmeans.labels_))
    # Store Davies-Bouldin Score
    DB.append(DB_score(X_train, kmeans.labels_))
    # Store silhouette score
    sil.append(silhouette_score(X_train, kmeans.labels_))
```

    Running k-means for k=2
    Running k-means for k=7
    Running k-means for k=13
    Running k-means for k=18
    Running k-means for k=24
    Running k-means for k=30


#### Calinski-Harabasz Score
- The Calinski-Harabasz score takes into account the dispersion within clusters and dispersion between clusters, so that tight clusters that are far away from eachother score well. 
- A higher score is better - more will be said about this score later.

#### Davies-Bouldin Score
- The Davies-Bouldin score is similar to the Calinski-Harabasz score except it tends to score clusters equally-distanced from eachother highly, in comparison to clusters of approximately the same size as in the Calinsky-Harabasz. A lower score is better.


#### Inertia
- Inertia is the cost function that k-means uses, ie. the sum of squared distances of all samples from their closest cluster centroid.


```python
# Plot scores against k
fig, ax1 = plt.subplots(1, 1, figsize=(15,8))
ax2 = ax1.twinx()

# Plot graphs against values of k
ax1.plot(k_values, CH, marker='o', color='purple', label='Calinski-Harabasz Score')
ax2.plot(k_values, DB, marker='o', label='Davies-Bouldin Score')
ax2.plot(k_values, sil, marker='o', label='Silhouette Score')
ax1.set_title(f'Evaluation of K-Means Clusterings Using Different Metrics against k', fontsize=19)

# Plot vertical line for optimal k value
ax1.axvline(x=k_optimal, linestyle='dashed', color='g', label=f'Optimal* k={k_optimal}')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.set_xlabel('k')
ax1.set_ylabel('CH Score')
ax2.set_ylabel('Silhouette/DB Score')
ax2.grid(False)

plt.show()
```


![png](Project%203_files/Project%203_100_0.png)


- The DB score rises slightly as k increases, with a maximum value reached somewhere around k = 10 to 14. This could be some evidence in favour of the correct number of clusterings for this dataset, 10.

- The other scores don't give much away in terms of finding an optimal k value. The silhouette score remains around 0, and is discussed in the next section in more detail.

### Silhouette Coefficient
- The Silhouette coefficient is bounded between -1 and 1, with a higher score indicating a better clustering with dense and well-separated clusters. It uses the distance between samples and other points in the same cluster, as well as the distance between samples and other points in the next nearest cluster.

- The silhouette plot shows how close each point in each cluster to points in nearby clusters. It provide a way to visualize the quality of clusters. The aim is to find a clustering that provides clusters with similar and high silhouette scores. For this we can use the silhouette score, which is the average silhouette coefiicient across all k clusters.


```python
k_plot_list = [6, 10, 14]
fig, axes = plt.subplots(1, len(k_plot_list), figsize=(15,8))

for j, k in enumerate(k_plot_list):
    if verbose print(k)
    # Initialize the silhouette plot
    axes[j].set_xlim([-0.2, 0.5])
    axes[j].set_ylim([0, len(X_train) + (k + 1) * 10])

    # Cluster using k-means with
    kmeans = KMeans(n_clusters=k, n_init=1, random_state=42)
    kmeans.fit(X_train)
    cluster_labels = kmeans.predict(X_train)

    # Compute average silhouette score
    silhouette_avg = silhouette_score(X_train, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_train, cluster_labels)

    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        axes[j].fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        axes[j].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    axes[j].set_title(f"k = {k}, Score = {round(silhouette_avg, 3)}", fontsize=15)
    axes[j].set_xlabel("Silhouette coefficient")
    axes[j].set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    axes[j].axvline(x=silhouette_avg, color="red", linestyle="--")
    
    axes[j].set_yticks([])  # Clear the yaxis labels / ticks
    axes[j].set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])

fig.suptitle('Silhouette Plots for varying k', fontsize=18)
```




    Text(0.5, 0.98, 'Silhouette Plots for varying k')




![png](Project%203_files/Project%203_103_1.png)


- For k=10 the silhouette plot show reasonably well-balanced sizes of clusters, with not too much overlap.  k=14 shows some small classes, with more overlapping than in k=10. k=6 has a cluster that has a very low silhouette score. 
- As a result, and from the silhouette plots, there is some reason to say that k=10 is roughly optimal for this dataset.

### Randomness in K-means clustering
- As mentioned before, the K-means is not likely to converge to the global minimum, and so it will likely converge to just a local minimum. The algorithm is sensitive to the initial random clustering of the points, and will likely yield different results each time it is run.

- **In order to account for this, the algorithm can be run many times, and the best clustering is then chosen. This is acheived using the n_init parameter in sklearn's K-Means implementation. In this case, I set n_init to 10.**

- If the outcomes are similar, it could be an indication that your outcomes are 'good', whereas if there is a lot of variance in the outcomes, it could indicate that k-means might not be the best algorithm with which to cluster this datset.

- 'k-means++' can also be used, which chooses the initial clustering in a better way. This is likely to decrease the variance of the models for different random seeds. Note that this alteration is used by default by sklearn.

#### Distribution of Cluster Sizes


```python
# Train model for k=10
k_optimal = 10
print(k_optimal)
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=1, random_state=42)
kmeans.fit(X_train)

# Plot cluster size distribution
cluster_sizes = np.zeros(k_optimal)
cluster_labels = kmeans.predict(X_train)
for i in range(k_optimal):
    cluster_sizes[i] = int(cluster_labels[cluster_labels == i].shape[0])

fig, ax = plt.subplots(figsize=(15,8))
ax.hist(cluster_sizes)
fig.suptitle('Distribution of Cluster Sizes')
ax.set_xlabel('Cluster Size')
ax.set_ylabel('Frequency')
print(cluster_sizes)
```

    10
    remove
    [7817. 6545. 3366. 7385. 2651. 8904. 7191. 4292. 6492. 5357.]



![png](Project%203_files/Project%203_107_1.png)


- The clustering results in a modest range of cluster sizes, which makes sense with the given data. There are no outlying clusters with very small or big size, which could suggest this clustering for k=10 is reasonable for the data.

#### Within Groups / Across Group Similarity
- The within and across group dispersion is calcualted as in the Calinski-Harabasz score, which uses the mean of the clusters and of all the points for its calculations:


```python
extra_disp, intra_disp = 0., 0.
mean = np.mean(X, axis=0)
for k in range(k_optimal):
    cluster_k = np.array(X_train[cluster_labels == k])
    cluster_k = cluster_k[0, :]
    mean_k = np.mean(cluster_k, axis=0)
    extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
    intra_disp += np.sum((cluster_k - mean_k) ** 2)
```

### Final model metrics


```python
# Populate table (training)
df_table = pd.DataFrame(columns=['', f'k_optimal={k_optimal}'])
df_table.loc[0] = ['CH Score', round(CH_score(X_train, kmeans.labels_), 3)]
df_table.loc[1] = ['DB Score', round(DB_score(X_train, kmeans.labels_), 3)]
df_table.loc[2] = ['Silhouette Score', round(silhouette_score(X_train, kmeans.labels_), 3)]
df_table.loc[3] = ['Within Group Dispersion', round(intra_disp, 1)]
df_table.loc[4] = ['Across Group Dispersion', round(extra_disp, 1)]
df_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>k_optimal=10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CH Score</td>
      <td>7.538723e+03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DB Score</td>
      <td>2.017000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Silhouette Score</td>
      <td>1.380000e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Within Group Dispersion</td>
      <td>6.490045e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Across Group Dispersion</td>
      <td>1.983541e+10</td>
    </tr>
  </tbody>
</table>
</div>



### Outlier Sensitivity 
- K-means senstive to outliers because of how centroids are calculated as the mean of all the samples in the respective cluster. This can be combatted using data exploration (this is hard with many dimensions), or extensions to kmeans.

- K-medoids can be used, which assigns one of the points within the cluster to be the 'center', choosing the point which is most representative of the cluster.

## Visualization of Centroids and Clustering

The k-means clustering can be visualsed using PCA. 

- The samples and centroids will be reduced to two dimensions using the PCA tranform fit on the samples.

- In this first implementation, a kmeans clustering is fit onto the **PCA reduced data** in two dimensions. 

- Only 10000 points are plotted so that the figure isn't too cluttered.

- The centroids are mapped onto the correct labels by using the majority label from the predicted labels of that cluster for the centroid.


```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

fashion_kmeans_labels = np.zeros(10, dtype=int)

# Scale data
Xscaler = sklearn.preprocessing.StandardScaler()
Xscaler.fit(X_train)
data = Xscaler.transform(X_train)

n_samples, n_features = data.shape
n_labels = len(np.unique(y_train))
labels = y_train

# Use PCA on data in order to visualize
pca = PCA(n_components=2).fit(data)
print('PCA explained variance in first 2 dimensions:')
print(sum(pca.explained_variance_ratio_[:2]))

reduced_data = pca.transform(data)
kmeans = KMeans(n_clusters=10, n_init=1, random_state=42)

# Fit k-means on REDUCED data
kmeans.fit(reduced_data)

n = 60000

# Plot the centroids as images
def imscatter(x, y, image, ax=None, zoom=1):
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

# Transform high-dimensional centroids into 2D from pca fit
centroids = kmeans.cluster_centers_
# centroids_reduced = pca.transform(Xscaler.transform(centroids))
centroids_reduced = centroids

# Save images
predictions = kmeans.predict(reduced_data)
for i, centroid in enumerate(centroids):
    # Draw centroid
    # plt.figure(frameon=False)
    matrix = centroid.copy()

    # Find what it represents by finding the majority label
    j = Counter(y_train[np.where(predictions==i)]).most_common()[0][0]
```

    PCA explained variance in first 2 dimensions:
    0.36443158984184265


The Variance described by the first two dimensions of the PCA transform is reasonably high, so that it is justifiable to fit a clustering on just these two dimensions.


```python
# Step size of the mesh
h = .2

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(figsize=(15,12))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

color_map = [fashion_colors[i][:-1] for i in y_train[:n]]
plt.scatter(reduced_data[:n, 0], reduced_data[:n, 1], c='black', s=5, alpha=0.5)

# plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

plt.title('K-means clustering on PCA-reduced data\n'
          'Centroids are marked with white cross', fontsize=20)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
ax.set_facecolor('white')
plt.show()
```


![png](Project%203_files/Project%203_118_0.png)


- Now, a K-means clustering is fit to the **full** (non-reduced) data, and **PCA is used only to visualize the clustering**.


```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

fashion_kmeans_labels = np.zeros(10, dtype=int)

# Scale data
Xscaler = sklearn.preprocessing.StandardScaler()
Xscaler.fit(X_train)
data = Xscaler.transform(X_train)

n_samples, n_features = data.shape
n_labels = len(np.unique(y_train))
labels = y_train

# Use PCA on data in order to visualize
pca = PCA(n_components=2).fit(data)

reduced_data = pca.transform(data)
kmeans = KMeans(n_clusters=10, n_init=1, random_state=42)

# Fit k-means on FULL data
kmeans.fit(X_train)

# Step size of the mesh
h = .2


n = 60000

# Plot the centroids as images
def imscatter(x, y, image, ax=None, zoom=1):
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

# Transform high-dimensional centroids into 2D from pca fit
centroids = kmeans.cluster_centers_
centroids_reduced = pca.transform(Xscaler.transform(centroids))

# Save images
predictions = kmeans.predict(X_train)
for i, centroid in enumerate(centroids):
    # Draw centroid
    plt.figure(frameon=False)
    matrix = centroid.reshape(28, 28).copy()

    # Find what it represents by finding the majority label
    j = Counter(y_train[np.where(predictions==i)]).most_common()[0][0]
    print(j, fashion_colors[j])

    # Set white pixels to nan for transparency
    matrix[matrix == 0] = np.nan
    plt.imshow(matrix, cmap=fashion_colors[j])
    plt.axis('off')
    plt.savefig(str(j)+'.png', transparent=True)

    # Map to actual label
    fashion_kmeans_labels[i] = j
```


```python
fig, ax = plt.subplots(figsize=(15,12))
color_map = [fashion_colors[fashion_kmeans_labels[i]][:-1] for i in kmeans.predict(X_train)]
plt.scatter(reduced_data[:n, 0], reduced_data[:n, 1], c=color_map, s=5, alpha=0.5) 

# plot centroids
for i, centroid in enumerate(centroids):
    j = Counter(y_train[np.where(predictions==i)]).most_common()[0][0]
    # Plot on graph
    image1 = plt.imread(str(j) + '.png')
    imscatter(centroids_reduced[i, 0], centroids_reduced[i, 1], image1, ax=ax, zoom=0.2)

# for i in range(10):
#     image1 = plt.imread(str(i) + '.png')
#     imscatter(centroids_reduced[i, 0], centroids_reduced[i, 1], image1, ax=ax, zoom=0.2)


# image1 = plt.imread('image1.png')
# imscatter(centroids[:, 0], centroids[:, 1], image1, ax=ax, zoom=0.3)
# my_labels = kmeans.predict(centroids)
#
# for i, txt in enumerate(my_labels):
#     plt.annotate(fashion_map[txt], (centroids[i, 0], centroids[i, 1]))

plt.title('K-means clustering on the FULL fashion dataset (PCA used for display only)\n'
          'Centroids are marked with kmeans centroids', fontsize=20)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
ax.set_facecolor('white')
plt.show()
```


![png](Project%203_files/Project%203_121_0.png)


- Using PCA, the high-dimensional clusters can be visualized. There is some nice separation of the clusters, especially for the easier-to-classify classes such as the Trousers, which are quite different from the other fashion items.

- Note the left facing shoes and general homogeneity of the orientation and scale of the images, which is reflected in the cetroids since they are quite well-defined and clear. For a less contrived data set with more 'messy' data, we would expect the centroids to be more blurry. 

- Data augmentation such as adding reflected/rotated/scaled images into the training data would probably be beneficial with other more 'messy' datasets, but is likely to be detrimental in this case since all the data is nicely scaled/oriented.

### Visualization of Centroids
- The real labels are assigned to each cluster by taking the majority prediction of each cluster.


```python
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
for i in range(2):
    for j in range(5):
        matrix = centroids[5*i+j].reshape(28, 28).copy()
        k = Counter(y_train[np.where(predictions==5*i+j)]).most_common()[0][0]
        cmap = fashion_colors[k]
        axes[i, j].imshow(matrix, cmap=cmap)
        axes[i, j].axis('off')
        axes[i, j].set_title(fashion_map[k])
        print(5*i+j)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9



![png](Project%203_files/Project%203_124_1.png)


- Using this method, we get some overlapping/missing classes. This is natural considering the only supervision given to the model was to assign 10 clusters to the data.
- The resulting centroids and labels mostly bear a good resemblance to the actual 10 classes, which is impressive since this method is unsupervised.

## kNN
- The kNN classifier **uses the unsupervised kmeans clustering** in order to classify the data. 
- **An optimization of k was attempted but the computational time is very large. Ideally, I would have used K-Fold Cross Validation to maximise the average validation accuracy across folds by varying the value of k between, say, 1 and 20.**


```python
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, [fashion_kmeans_labels[x] for x in predictions])
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')




```python
# Calculate metrics
y_pred = neigh.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Populate metrics table
df_table = pd.DataFrame(columns=['Test Set Metrics', 'kNN Classifier'])
df_table.loc[0] = ['Accuracy', round(accuracy, 3)]
df_table.loc[1] = ['Precision', round(precision, 3)]
df_table.loc[2] = ['Recall', round(recall, 3)]
df_table.loc[3] = ['F1 Score', round(f1, 3)]
df_table
```

    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Set Metrics</th>
      <th>kNN Classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.554</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.521</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.555</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1 Score</td>
      <td>0.516</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(15,11))
labels = [fashion_map[int(x)] for x in fashion_kmeans_labels]
sns.heatmap(confusion_matrix(y_test, y_pred), xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for kNN Classifier', fontsize=20)
plt.xlabel('Real Labels')
plt.ylabel('Predicted Labels')
plt.show()
```


![png](Project%203_files/Project%203_129_0.png)


- The kNN classifiers achieves a reasonable accuracy score on the test set, predicting much better than average. This was expected after seeing the the kmeans classification also gave a reasonable but far from perfect clustering of the data.
- The classifier seems particularly good at classifying trousers / ankle boots because they distinctly different to the other items.

## Alternative Clustering Methods

### Hierachical Clustering
- Hierachical clustering aims to describe clustering of data as a binary tree (or a dendogram).
- Because of how the binary splits are generated, the outcome is hierachical, in that lower branches originating from one split are in the same cluster that corresponds to that split.
- An agglomerative scheme will be used, which goes from the bottom of the tree upwards. That is, starting from the N samples in N clusters, the number of clusters is reduced from N to 1.


```python
# Setup list of numbers of clusters and metrics
k_values = np.linspace(2, 30, 8, dtype=int)
CH = []
DB = []
sil = []

for k in k_values:
    if verbose: print(f'Running hierachical clustering for k={k}')
    # Use algorithm to cluster clothing with k clusters
    model = AgglomerativeClustering(n_clusters=k)
    model = model.fit(X_train[:5000, :])

    # Store Calinski-Harabasz score
    CH.append(CH_score(X_train[:5000, :], model.labels_))
    # Store Davies-Bouldin Score
    DB.append(DB_score(X_train[:5000, :], model.labels_))
    # Store Silhouette Score
    sil.append(silhouette_score(X_train[:5000, :], model.labels_))
```

    Running hierachical clustering for k=2
    Running hierachical clustering for k=6
    Running hierachical clustering for k=10
    Running hierachical clustering for k=14
    Running hierachical clustering for k=18
    Running hierachical clustering for k=22
    Running hierachical clustering for k=26
    Running hierachical clustering for k=30



```python
# Plot scores against k
fig, ax1 = plt.subplots(1, 1, figsize=(15,8))
ax2 = ax1.twinx()

# Plot graphs against values of k
ax1.plot(k_values, CH, marker='o', color='purple', label='Calinski-Harabasz Score')
ax2.plot(k_values, DB, marker='o', label='Davies-Bouldin Score')
ax2.plot(k_values, sil, marker='o', label='Silhouette Score')
ax1.set_title(f'Evaluation of Hierachical Clusterings Using Different Metrics against k', fontsize=19)

# Plot vertical line for optimal k value
ax1.axvline(x=10, linestyle='dashed', color='g', label=f'k=10')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.set_xlabel('k')
ax1.set_ylabel('CH Score')
ax2.set_ylabel('DB/Silhouette Score')
ax2.grid(False)

plt.show()
```


![png](Project%203_files/Project%203_134_0.png)


- As before, the CH and Silhouette scores are not very informative here of an optimal value for k. However for the DB score, there is a slight decrease at k=10, again suggesting that 10 could be the optimal number of clusters, matching the number of real labels.

- The dendrogram of the hierachical clustering is now plotted:


```python
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

plt.figure(figsize=(15,8))
model = model.fit(X_train[:1000])
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
```


![png](Project%203_files/Project%203_136_0.png)


- The resulting dendrogram illustrates the process of hierachical clustering. Each cluster is represented as a branch, and branches below  other branches belong to the clustering of the upper branch, hence 'hierachical'.

---
## 2.2 Supervised classification of the training set
---

# Balancing Data
- Classification methods are likely to work better if the models are trained on balanced datasets (that is, that the classes that the model is trying to predict are balanced in frequency). Without balancing, the model is likely to missclassify less frequent classes, since it has less information about these classes and is penalized less for missclassifying them since there are fewer of them (in the case that we are using accuracy as a performance metric).

- The size of the classes also has to be decided. If the size is too large (so many classes will be oversampled), the model will tend to overfit, since repeated values of smaller classes are trained on. If the size is too small (so many classes will be undersampled), then useful data might be removed, and the model might not describe the data as well, and will perform poorly on both in-sample and out-of-sample data. 
- The training datasets should be **balanced inside the cross-validation splits**, so that each training set for each fold is balanced separately, and the validation set for each fold is not balanced, so that the evaulation is more realistic. This is important since **if the dataset is balanced before cross validation, then the validation set will be balanced**. However this **is not the desired validation set**, since we should assume that real out-of-sample data has a similar distribution of classes to the unbalanced training data.



```python
def balance_data(X, y, size=1400):
    # Combine X and y
    temp = X.join(y, how='outer')

    # Divide by class
    class_range = range(10)
    classes = [temp[temp['rating'] == i] for i in class_range]
    min_size = min([len(i) for i in train_classes])
    max_size = max([len(i) for i in train_classes])

    train_classes_bal = []
    for train_class in train_classes:
        if size > len(train_class):
            # Oversample with replacement
            replace = True
        else:
            # Undersample without replacement
            replace = False

        # Balance classes
        # Set random state for reproducibility of results
        train_classes_bal.append(train_class.sample(size, replace=replace, random_state=42).reset_index().drop(columns=['index']))
        train_balanced = pd.concat([i for i in train_classes_bal])

    # Split into X_train and y_train
    X_train = train_balanced[predictors]
    y_train = train_balanced['rating']

    return X_train, y_train
```

## 2.2.1 MLP neural network supervised classification
---

- An MLP Neural Network will be trained as a classifier for the fashion image dataset using the architecture as in the assignment:


```python
# Define the MLP neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0):
        self.dropout = dropout
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        out = F.log_softmax(out, dim=1)
        return out
```


```python
# Setup parameters
hidden_size = 100
input_size = X_train.shape[1]
num_classes = y_train.shape[0]
learning_rate = 0.005
batch_size = 128
num_epochs = 30

# Create new instance of the NN
net = NeuralNet(input_size=X_train.shape[1], hidden_size=100, num_classes=y_train.shape[0])

# Define Loss as Negative Log-Likelihood
criterion = nn.NLLLoss()

# Use Stochastic Gradient Descent for the optimizer
optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate) 

# Create a tensor data set
# X_train, X_val = torch.from_numpy(X_train).float(), torch.from_numpy(X_val).float()
# y_train, y_val = torch.from_numpy(np.array(y_train)).float(), torch.from_numpy(np.array(y_val)).float()
X_train_tens = torch.from_numpy(X_train).float()
y_train_tens = torch.from_numpy(np.array(y_train)).float()

X_test_tens = torch.from_numpy(X_test).float()
y_test_tens = torch.from_numpy(np.array(y_test)).float()

# Load data
train = torch.utils.data.TensorDataset(X_train_tens, y_train_tens)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

test = torch.utils.data.TensorDataset(X_test_tens, y_test_tens)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

# Train the model
total_step = len(train_loader)
loss_values = []

for epoch in range(num_epochs+1):
    print(epoch)
    ###################
    # train the model #
    ###################
    net.train()
    train_loss = 0.0
    
    for i, (cars, labels) in enumerate(train_loader, 0):
        
        # reshape to long vector
        cars = cars.reshape(-1, 784)
        labels = labels.squeeze_().type(torch.LongTensor)
            
        # forward pass
        outputs = net(cars)
        loss = criterion(outputs, labels)
            
        # backward and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Compute training loss
        train_loss += loss.item()

    # Store loss at each epoch
    loss_values.append(train_loss)

# Calculate test metrics after fully training nn
net.eval()
correct = 0
total = 0
y_test_pred = []
y_test = []
for cars, labels in test_loader:
    cars = cars.reshape(-1, 784)
    labels = labels.squeeze_().type(torch.LongTensor)
    outputs = net(cars)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    y_test_pred += predicted.tolist()
    y_test += labels.tolist()

# Calculate metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average='macro')
f1 = f1_score(y_test, y_test_pred, average='macro')

```

Note that Shuffle = True within the DataLoaders. This is to ensure that at each epoch, the neural network is fed in the data in different orders.


```python
# Plot heatmap
fig = plt.figure(figsize=(15,11))
labels = [fashion_map[int(x)] for x in fashion_kmeans_labels]
sns.heatmap(confusion_matrix(y_test, y_test_pred), xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for MLP Classifier', fontsize=20)
plt.xlabel('Real Labels')
plt.ylabel('Predicted Labels')
plt.show()
```


![png](Project%203_files/Project%203_146_0.png)



```python
# Populate metrics table
df_table = pd.DataFrame(columns=['Test Set Metrics', 'MLP Classifier'])
df_table.loc[0] = ['Accuracy', round(accuracy, 3)]
df_table.loc[1] = ['Precision', round(precision, 3)]
df_table.loc[2] = ['Recall', round(recall, 3)]
df_table.loc[3] = ['F1 Score', round(f1, 3)]
df_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Set Metrics</th>
      <th>MLP Classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.878</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.887</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1 Score</td>
      <td>0.880</td>
    </tr>
  </tbody>
</table>
</div>



## 2.2.2 Convolutional neural network (CNN) supervised classification
---
- Convolutional neural networks are specifically for image classification, since they can preserve the spatial relationship between pixels. They are designed to identify low-level features of images, which combine into medium and high-level features in order to classify the images. 
- Filters are applied to the images in order to create a new 'convolved' matrix, which maps the image onto a smaller grid which preserves spatial relationships between pixels.
- The type and size of kernels used on the images will result in different convolved outputs of the images. These kernels or filters are learned by the network in order to learn different features of the image set, such as certain outlines of clothing that frequently occur in images.
- Pooling reduces feature size down. For example, max pooling takes the max of the filter. Note that clearly pooling also preserves spatial relationships.

Note that the padding is 0 in all convolutions.


```python
# Define the CNN neural network
class ConvNet(nn.Module):
    def __init__(self, input_size, dropout=0, upscale=False, batch_norm=False, fc=False):
        self.input_size = input_size
        self.dropout = dropout
        self.upscale = upscale
        self.batch_norm = batch_norm
        self.fc = fc
        super(ConvNet, self).__init__()

        # add the layers
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=2, dilation=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        if self.upscale:
            self.layer1 = nn.Sequential(
                nn.Conv2d(6, 6, kernel_size=5, stride=1, dilation=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, dilation=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        if self.batch_norm:
            self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, dilation=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(16)
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, dilation=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        if self.dropout:
            self.dropout = dropout
        if self.fc: 
            self.fc0 = nn.Linear(4*4*16, 240)
            self.fc1 = nn.Linear(240, 120)
        else:
            self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    # forward 
    def forward(self, x):
        out = x
        if self.upscale:
            out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        if self.training:
            out = F.dropout(out, p=self.dropout)
        if self.fc:
            out = F.relu(self.fc0(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out
```


```python
def eval_CNN(X, y, X_test, y_test, k=5, dropout=0, upscale=False, batch_norm=False, fc=False, num_epochs=30):
    '''
    k=fold Stratified Cross Validation of CNN
    '''
    if upscale:
        p = 120
    else:
        p = 28
     
    # Set up CV
    if k is not None:
        skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    else:
        # Unused skf
        skf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)

    # Setup parameters
    hidden_size = 100
    input_size = p
    num_classes = y.shape[0]
    learning_rate = 0.005
    batch_size = 128
    num_epochs = num_epochs

    # Create new instance of the NN
    net = ConvNet(input_size=input_size, dropout=dropout, upscale=upscale, batch_norm=batch_norm, fc=fc)

    # Define Loss as Negative Log-Likelihood
    criterion = nn.NLLLoss()

    # Use Stochastic Gradient Descent for the optimizer
    optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate) 

    # Use stratified k-fold cross-validation to evaluate models
    if k is not None:
        skf_split = skf.split(X, y)
    else:
        skf_split = [(0, 0)]

    accuracy_train = []
    accuracy_val = []

    for train_index, val_index in skf_split:
        if k is not None:
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
        else:
            X_train, X_val = X, X
            y_train, y_val = y, y

        # Balance training set
        # X_train, y_train = balance_data(X_train, y_train)

        # Create a tensor data set
        X_train = X_train[:, None, :, :]
        X_val = X_val[:, None, :, :]
        X_test = X_test[:, None, :, :]

        X_train_tens, X_val_tens, X_test_tens = torch.from_numpy(X_train).float(), torch.from_numpy(X_val).float(), torch.from_numpy(X_test).float()
        y_train_tens, y_val_tens, y_test_tens = torch.from_numpy(np.array(y_train)).float(), torch.from_numpy(np.array(y_val)).float(), torch.from_numpy(np.array(y_test)).float()

        # Load data
        train = torch.utils.data.TensorDataset(X_train_tens, y_train_tens)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        val = torch.utils.data.TensorDataset(X_val_tens, y_val_tens)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

        test = torch.utils.data.TensorDataset(X_test_tens, y_test_tens)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

        # Train the model
        total_step = len(train_loader)
        loss_values = []

        for epoch in range(num_epochs+1):
            net.train()
            train_loss = 0.0
            
            for i, (image, labels) in enumerate(train_loader, 0):
                
                # reshape to long vector
                labels = labels.squeeze_().type(torch.LongTensor)
                    
                # forward pass
                outputs = net(image)
                loss = criterion(outputs, labels)
                    
                # backward and optimise
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Compute training loss
                train_loss += loss.item()

            # Store loss at each epoch
            loss_values.append(train_loss)

        # Calculate training accuracy after fully training nn
        net.eval()
        correct = 0
        total = 0
        for image, labels in train_loader:
            labels = labels.squeeze_().type(torch.LongTensor)
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy_train.append(correct/total)

        # Calculate validation accuracy after fully training nn
        net.eval()
        correct = 0
        total = 0
        for image, labels in val_loader:
            labels = labels.squeeze_().type(torch.LongTensor)
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy_val.append(correct/total)

    # Calculate test accuracy after (when not using k-fold CV)
    net.eval()
    correct = 0
    total = 0
    y_test_pred = []
    y_test = []
    for image, labels in test_loader:
        labels = labels.squeeze_().type(torch.LongTensor)
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_test_pred += predicted.tolist()
        y_test += labels.tolist()

    if k is not None:
        accuracy_test = None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')

    print('Finished')
    return accuracy_train, accuracy_val, accuracy, precision, recall, f1

```

- The unoptimized CNN is now trained on the images:


```python
# Train the CNN
accuracy_train, accuracy_val, accuracy, precision, recall, f1 = eval_CNN(X_train_mat, y_train, X_test_mat, y_test, k=None, 
        dropout=False, upscale=False, batch_norm=False, fc=False, num_epochs=30)
```


```python
# Populate metrics table
df_table = pd.DataFrame(columns=['Test Set Metrics', 'CNN Classifier'])
df_table.loc[0] = ['Accuracy', round(accuracy, 3)]
df_table.loc[1] = ['Precision', round(precision, 3)]
df_table.loc[2] = ['Recall', round(recall, 3)]
df_table.loc[3] = ['F1 Score', round(f1, 3)]
df_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Set Metrics</th>
      <th>CNN Classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.877</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.868</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1 Score</td>
      <td>0.877</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(15,11))
labels = [fashion_map[int(x)] for x in fashion_kmeans_labels]
sns.heatmap(confusion_matrix(y_test, y_test_pred), xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for CNN Classifier', fontsize=20)
plt.xlabel('Real Labels')
plt.ylabel('Predicted Labels')
plt.show()
```


![png](Project%203_files/Project%203_155_0.png)


## 2.2.3 Comparisons of the classifiers
---


- From the metrics derived above for both the MLP and the CNN, the CNN performs slightly better without optimization of any of the classifiers. 

- There are a few reasons why the CNN classifier would be expected to perform better than the MLP Classifer.

- Firstly, the CNN's architecture allows it to preserve spatial correlations between pixels in the image, since the inputs are 2D and the convoluational layers help to learn more meaningful features from the image.

- The default CNN also has fewer parameters than the defaul MLP classifier, which in turn means that it is less prone to overfitting. In the MLP classifier, each pixel in the input vector becomes a parameter.

- In the CNN, pooling provides two main benefits:
    - It uses down-sampling to reduce the number of parameters
    - It decreases the sensitivity to changes in scale and orientation of the image.
    
This is achieved using a sliding window across the image, which maps the image onto a smaller image with a particular function. For example, a 2x2 window could map 4 pixels onto just 1 pixel by taking the maximum greyscale value of the 4 pixels; this would be called max pooling, and the stride is of size 2.

- Most of these factors point to CNN being a better model choice for image classification.


## Supervised vs Unsupervised Learning
- Since unsupervised learning doesn't use ground-truth labels and so can't train on these labels, the accuracy from these methods is clearly going to be lower.
- Unsupervised learning methods however provide a powerful way to cluster, model or classify data without much input. In the case of the unsupervised kNN classifier that used the kmeans clustering, it achieved an accuracy of nearly 60% which is quite impressive.

### Interpretability of Results
- The MLP and CNN classifiers end up with a lot of trained parameters. This drastically reduces the interpretibility of the networks, so that there are basically 'black boxes'.

- In contrast to this, the unsupervised kmeans clustering (and as an extension, the kNN classifier) was able to generate 10 centroids that intuitively matched roughly to the 10 labels in the data, so it is much more informative of how it is trying to classify the data. 

- In the case of the CNN, the features it has learned may be intuitive, such as learning specific shapes of edges of different types of clothing. It would be interesting to visualise these features as an extension.

### Upscaling of images
- The images are upscaled to 'artificially' increase the resolution by interpolating across pixels to find the greyscale values inbetween. 

- The purpose of this is to allow the addition of an extra convolutional layer. In this way, the hope is that the model learns more features from the larger, 'higher' resolution images (since the size of the kernel stays the same).

- The images are also fed into the CNN as 2D images this time.


```python
X_train_mat = np.array([x.reshape(28,28) for x in X_train])
X_test_mat = np.array([x.reshape(28,28) for x in X_test])

X_train_upscaled = np.array([scipy.ndimage.zoom(x,120/28) for x in X_train_mat])
X_test_upscaled = np.array([scipy.ndimage.zoom(x,120/28) for x in X_test_mat])
```


```python
fig, axes = plt.subplots(2, 5, figsize=(18,9))

for i in range(5):
    axes[0, i].imshow(X_train_mat[i], cmap='Greys')
    axes[1, i].imshow(X_train_upscaled[i], cmap='Greys')
    axes[0, i].axis('off')
    axes[1, i].axis('off')
plt.show()
```


![png](Project%203_files/Project%203_161_0.png)



```python
# Perform a grid search to optimize CNN using k-Fold Stratified Cross Validation
dropouts = [0.1, 0.3]
upscales = [True, False]
batch_norms = [True, False]
fcs = [True, False]

max_acc = 0
best_model = None
for dropout in dropouts:
  for upscale in upscales:
      for batch_norm in batch_norms:
          for fc in fcs:
            model = (dropout, upscale, batch_norm, fc)
            print(model)
            # Use upscaled images if needed
            if upscale:
                X_train_active = X_train_upscaled
                X_test_active = X_test_upscaled
            else:
                X_train_active = X_train_mat
                X_test_active = X_test_mat

            accuracy_train, accuracy_val, accuracy, precision, recall, f1 = eval_CNN(X_train_active, y_train, X_test_active, y_test, k=5, 
                    dropout=dropout, upscale=upscale, batch_norm=batch_norm, fc=fc, num_epochs=30)

            # check if average validation score is higher than best model
            if accuracy_val[-1] > max_acc:
                best_model = model
                max_acc = accuracy_val[-1]
```

    (0.1, True, True, True)
    Finished
    (0.1, True, True, False)
    Finished
    (0.1, True, False, True)
    Finished
    (0.1, True, False, False)
    Finished
    (0.1, False, True, True)
    Finished
    (0.1, False, True, False)
    Finished
    (0.1, False, False, True)
    Finished
    (0.1, False, False, False)
    Finished
    (0.3, True, True, True)
    Finished
    (0.3, True, True, False)
    Finished
    (0.3, True, False, True)
    Finished
    (0.3, True, False, False)
    Finished
    (0.3, False, True, True)


    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


    Finished
    (0.3, False, True, False)
    Finished
    (0.3, False, False, True)
    Finished
    (0.3, False, False, False)
    Finished



```python
# Train model with best architecture found through 5-fold Stratified Cross Validation
dropout, upscale, batch_norm, fc = best_model
print(best_model)
accuracy_train, accuracy_val, accuracy, precision, recall, f1 = eval_CNN(X_train_upscaled, y_train, X_test_upscaled, y_test, k=None, dropout=dropout, 
                upscale=upscale, batch_norm=batch_norm, fc=fc, num_epochs=30)
```

    (0.3, True, True, True)



```python
# Populate metrics table
df_table = pd.DataFrame(columns=['Test Set Metrics', 'Optimised CNN Classifier'])
df_table.loc[0] = ['Accuracy', round(accuracy, 3)]
df_table.loc[1] = ['Precision', round(precision, 3)]
df_table.loc[2] = ['Recall', round(recall, 3)]
df_table.loc[3] = ['F1 Score', round(f1, 3)]
df_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Set Metrics</th>
      <th>Optimised CNN Classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.9090</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.8990</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.8476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1 Score</td>
      <td>0.9020</td>
    </tr>
  </tbody>
</table>
</div>



### CNN Improvements
- Including Dropout regularization reduces the overfitting of the CNN model. It achieves this because some of the weights are set to zero randomly, which increases the indepedance of the training of the weights, leading to less overfitting.
- The addition of a convolutional layer was made easier by an upsampling of the images. In this way, the CNN model is able to learn more detailed features from the data, since the kernel size was kept the same and the images are larger in resolution.
- Including the batch normalization was found not to be that impactful on the test accuracy, but it slightly decreases training time.
- Adding another fully connected layer decreases the jump to the first fully connected layer, which was also found to increase the test accuracy.

- The final model uses a dropout of 0.3, and includes the extra fully-connected layer, the upscaling of images and the extra convolutional layer.
