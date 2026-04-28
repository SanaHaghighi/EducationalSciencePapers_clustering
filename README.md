## Feature Extraction

To represent papers numerically, two approaches were used:

- **TF-IDF** for sparse text representation
- **Word2Vec** for dense semantic representation of titles and keywords

![Word to Vector](word-to-vector.png)

Word2Vec was especially useful for capturing semantic similarity between titles and keywords by averaging or combining word vectors.

---

## K-Means++ on TF-IDF

K-Means++ was applied on TF-IDF features, and the elbow method was used to estimate a suitable number of clusters.

![K-Means++ on TF-IDF](kmeans.png)

---

## K-Means++ on Word2Vec

To improve clustering quality, Word2Vec-based document vectors were also used with K-Means++.

![K-Means++ on W2V](kmeans-plus-plus-w2v.png)

This representation reduced SSE and produced more meaningful cluster distributions and visualizations.

---

## Agglomerative Clustering

Hierarchical clustering was also explored to compare its structure with partition-based clustering methods.

![Agglomerative Dendrogram](agglomerative-dendrogram.png)

![Agglomerative 3D Visualization](agglomerative-3d.png)

---

## Evaluation

Clustering performance was evaluated using internal metrics such as SSE and silhouette-based analysis.

![Evaluation](evaluation.png)

---

## Cluster Analysis

Keyword-level inspection was used to better understand the semantic meaning of each cluster.

![Clusters Keywords W2V](clusters-keywords-w2v.png)
