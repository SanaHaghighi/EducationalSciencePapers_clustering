📚 Educational Science Papers Clustering
This project performs semantic clustering of educational science research papers using Natural Language Processing (NLP) and unsupervised learning algorithms.

It aims to discover hidden topic structures and group papers with similar themes based on their textual content.

🧩 Project Overview
Goal: Automatically cluster academic papers in the field of educational sciences.
Input: Research paper titles, abstracts, and keywords.
Output: Cluster labels, evaluation metrics, and 2D/3D visualizations.
📂 Repository Structure
├── DataMiningProject.ipynb # Main Jupyter Notebook

├── dataset.csv # Dataset of educational papers

├── educational science papers clustering.pdf # Final project report

└── images/ # Generated visualizations (optional)

🗃️ Dataset Description
File: dataset.csv
Rows: 160
Columns:
id — unique identifier
title — paper title
text — abstract or main text
author — author name(s)
language — document language
keywords — list of topic keywords
The dataset contains both Persian and English academic papers in educational sciences.

⚙️ Methodology
1. Text Preprocessing
Implemented using Hazm (for Persian text):

Normalization (hazm.Normalizer())
Noise and punctuation removal
Tokenization (hazm.WordTokenizer())
Stopword removal
Lemmatization (hazm.Lemmatizer())
2. Feature Extraction
Vectorization: via TfidfVectorizer
Dimensionality Reduction:
PCA (sklearn.decomposition.PCA)
t-SNE (sklearn.manifold.TSNE)
3. Clustering Algorithms
K-Means (with k-means++ initialization)
Agglomerative Clustering (hierarchical approach)
Both models were experimented with varying cluster numbers using the Elbow Method.

4. Evaluation
Metrics for assessing cluster quality:

Silhouette Score
Davies–Bouldin Index
Calinski–Harabasz Score
Visual evaluation was performed via:

3D scatter plots (PCA/TSNE)
2D cluster visualizations
Hierarchical dendrograms
📊 Visualization Highlights
Generated directly from the notebook:

Elbow plot (for choosing optimal K)
3D PCA / TSNE scatter plots
Cluster distribution plots
Agglomerative dendrograms
Keyword-based summaries for each cluster
