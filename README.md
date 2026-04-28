# 📚 Educational Science Papers Clustering

This project performs semantic clustering of educational science research papers using **Natural Language Processing (NLP)** and **unsupervised learning algorithms**.

The main goal is to discover hidden topic structures and group educational science papers based on their textual similarity.

---

## 🧩 Project Overview

- **Goal:** Automatically cluster academic papers in the field of educational sciences.
- **Input:** Research paper titles, abstracts/texts, authors, languages, and keywords.
- **Output:** Cluster labels, evaluation metrics, and 2D/3D visualizations.
- **Main Approach:** Text preprocessing, TF-IDF vectorization, dimensionality reduction, and clustering.

---

## 📂 Repository Structure
```text
├── DataMiningProject.ipynb                     # Main Jupyter Notebook
├── dataset.csv                                 # Dataset of educational science papers
├── educational science papers clustering.pdf   # Final project report
├── evaluation-code.png                         # Evaluation code/output screenshot
├── word-to-vector.png                          # Word/vector representation visualization
├── kmeans.png                                  # K-Means clustering visualization
├── kmeans-plus-plus-w2v.png                    # K-Means++ / Word2Vec visualization
├── agglomerative-dendrogram.png                # Hierarchical clustering dendrogram
├── agglomerative-3d.png                        # 3D Agglomerative clustering visualization
├── clusters-keywords-w2v.png                   # Cluster keyword visualization
├── evaluation.png                              # Evaluation results visualization
└── cover.png                                   # Project cover image

---text

## 🗃️ Dataset Description

The dataset used in this project is stored in:

text
dataset.csv

The dataset contains **160 educational science papers**.

### Dataset Columns

| Column | Description |
|---|---|
| `id` | Unique identifier for each paper |
| `title` | Paper title |
| `text` | Abstract or main textual content |
| `author` | Author name(s) |
| `language` | Language of the paper |
| `keywords` | Keywords associated with the paper |

The dataset includes academic papers related to educational sciences and may contain both **Persian** and **English** text.

---

## ⚙️ Methodology

The project follows a complete unsupervised text clustering pipeline.

### 1. Text Preprocessing

Text preprocessing is performed before feature extraction to improve clustering quality.

Main preprocessing steps include:

- Persian text normalization
- Removing noise and unwanted characters
- Removing punctuation
- Tokenization
- Stopword removal
- Lemmatization

Persian NLP preprocessing is mainly handled using the **Hazm** library.

Examples of preprocessing tools:

python
from hazm import Normalizer, Lemmatizer, WordTokenizer

---

### 2. Feature Extraction

After preprocessing, textual data is converted into numerical vectors.

The main vectorization method used is:

- **TF-IDF Vectorization**

python
from sklearn.feature_extraction.text import TfidfVectorizer

TF-IDF helps represent each document based on the importance of its words across the whole dataset.

---

### 3. Dimensionality Reduction

To visualize high-dimensional text vectors and improve interpretability, dimensionality reduction techniques are used.

Implemented methods:

- **PCA**
- **t-SNE**

python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

These methods help project text vectors into 2D or 3D space for visualization.

---

### 4. Clustering Algorithms

Two unsupervised clustering algorithms are used in this project:

#### K-Means Clustering

K-Means partitions documents into a fixed number of clusters.

python
from sklearn.cluster import KMeans

The project also uses **k-means++ initialization** for better centroid initialization.

#### Agglomerative Clustering

Agglomerative Clustering is a hierarchical clustering method that builds clusters step by step.

python
from sklearn.cluster import AgglomerativeClustering

A dendrogram is also used to analyze hierarchical relationships between documents.

---

## 📊 Evaluation

The project evaluates clustering quality using both quantitative metrics and visual analysis.

### Evaluation Metrics

- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Score**

These metrics help compare clustering results and assess how well documents are grouped.

---

## 📈 Visualizations

Several visualizations are generated to better understand clustering results:

- Elbow Method plot for selecting the number of clusters
- 2D cluster scatter plots
- 3D PCA/t-SNE visualizations
- Agglomerative dendrogram
- Cluster keyword visualization
- Evaluation metric plots

Example visualization files include:

text
kmeans.png
kmeans-plus-plus-w2v.png
agglomerative-3d.png
agglomerative-dendrogram.png
clusters-keywords-w2v.png
evaluation.png

---

## 🧠 Key Findings

Based on the clustering process:

- The Elbow Method was used to estimate the optimal number of clusters.
- The selected number of clusters is approximately **8**.
- K-Means and Agglomerative Clustering both provide useful groupings of educational science papers.
- Dimensionality reduction techniques such as PCA and t-SNE make cluster structures easier to interpret visually.
- Keyword analysis helps describe the main theme of each cluster.

Possible cluster themes include:

- Educational policy and governance
- Teacher training and professional development
- Technology and digital learning
- Student behavior and motivation
- Curriculum design
- Psychology and cognitive approaches
- Language learning and bilingual education
- Educational assessment and evaluation

---

## 🧪 Example Code

python
import pandas as pd

from hazm import Normalizer, Lemmatizer, WordTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Initialize Persian NLP tools
normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()

# Simple preprocessing function
def preprocess_text(text):
text = normalizer.normalize(str(text))
tokens = tokenizer.tokenize(text)
tokens = [lemmatizer.lemmatize(token) for token in tokens]
return " ".join(tokens)

# Apply preprocessing
df["processed_text"] = df["text"].fillna("").apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["processed_text"])

# Dimensionality reduction
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X.toarray())

# K-Means clustering
kmeans = KMeans(n_clusters=8, init="k-means++", random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=8, linkage="ward")
agg_labels = agg.fit_predict(X_pca)

# Evaluation
score = silhouette_score(X_pca, kmeans_labels)
print("Silhouette Score:", score)

---

## 🔧 Installation

### 1. Clone the Repository

bash
git clone https://github.com/your-username/educational-science-papers-clustering.git
cd educational-science-papers-clustering

### 2. Create a Virtual Environment

#### Windows

bash
python -m venv .venv
.venv\Scripts\activate

#### macOS / Linux

bash
python -m venv .venv
source .venv/bin/activate

### 3. Install Required Packages

bash
pip install numpy pandas scikit-learn gensim hazm nltk matplotlib seaborn tqdm plotly jupyter

### 4. Download NLTK Resources

bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

---

## 🚀 How to Run

### Option 1: Run with Jupyter Notebook

bash
jupyter notebook DataMiningProject.ipynb

Then open the notebook and run all cells.

### Option 2: Convert Notebook to Python Script

bash
jupyter nbconvert --to script DataMiningProject.ipynb
python DataMiningProject.py

---

## ✅ Results

The project produces:

- Cluster labels for each paper
- Evaluation scores for clustering algorithms
- Visual comparison between clustering methods
- 2D and 3D cluster visualizations
- Hierarchical dendrogram
- Keyword summaries for clusters

---

## 🔮 Future Work

Possible improvements for this project:

- Use **Word2Vec**, **Doc2Vec**, or **Sentence-BERT** for richer semantic embeddings
- Improve multilingual preprocessing
- Add automatic cluster labeling using TF-IDF or KeyBERT
- Build an interactive visualization dashboard using Plotly or Streamlit
- Compare more clustering algorithms such as DBSCAN or Gaussian Mixture Models
- Export clustered results to CSV

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Hazm
- NLTK
- Matplotlib
- Seaborn
- Plotly
- Gensim

---

## 📜 License

This project is released under the **MIT License**.

You can modify the license based on your own requirements.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [your-github-profile](https://github.com/your-username)
- Email: your-email@example.com

---

## 📌 Citation

If you use this project, dataset, or methodology, please cite it as:

text
Educational Science Papers Clustering Project, 2026.

---

## ⭐ Acknowledgements

This project was developed as a data mining and natural language processing project focused on clustering educational science research papers.

Special thanks to the open-source Python ecosystem and libraries used in this work.
