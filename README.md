# Research Paper Clustering Project

This project aims to cluster a set of research papers based on the similarity of their abstracts. It demonstrates skills in natural language processing (NLP), text preprocessing, and unsupervised machine learning techniques.

# Requirements

# Data Preprocessing:
- Accept an Excel dataset containing multiple research papers as input.
- Extract the abstract content from each PDF file using a PDF parsing library (e.g., PyPDF2, pdfminer).
- Preprocess the extracted text by:
- Removing stop words using NLTK's stopwords corpus.
- Lemmatizing words to their base form using NLTK's WordNetLemmatizer.
- Handling special characters and formatting issues with regular expressions.

# Text Vectorisation:
- Convert preprocessed text data into a suitable numerical representation:
- Experiment with different vectorisation techniques (e.g., TF-IDF vectors, Word2Vec embeddings).
- Chose Word2Vec embeddings due to their ability to capture semantic information effectively.

# Clustering:
- Implement the K-means clustering algorithm to group similar research papers based on their abstracts.
- Determine the optimal number of clusters using the elbow method and silhouette analysis.
- Evaluate clustering quality using metrics such as silhouette score and Davies-Bouldin index.

# Visualization (Bonus):
- Create a visualization of the clustering results using PCA for dimensionality reduction.
- Display research paper titles or IDs in the visualization to provide context.

# Output Generation:
- Save clustering results to a file indicating which papers belong to each cluster.
- Generate a summary report including:
    >Number of clusters created.
    >Number of papers in each cluster.
    >Key terms or topics associated with each cluster based on word frequencies.


# Setup Instructions

Install Dependencies:

1) Install necessary Python libraries:
pip install PyPDF2 nltk gensim matplotlib plotly pandas scikit-learn

2) Download NLTK resources (run once):
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

3) Clone the Repository:

git clone https://github.com/ManavBijlani21/Outread-Project.git
cd Outread-Project

4) Run the Script:
python assessment-code.py


# Follow prompts to provide the Excel dataset path and observe clustering results.

# Approach

1) Data Preprocessing: Extracted and cleaned abstracts, ensuring text readiness for vectorisation.
2) Text Vectorisation: Employed Word2Vec embeddings to capture semantic similarities effectively.
3) Clustering: Implemented K-means and validated with elbow method and silhouette scores.
4) Visualization: Utilised PCA for visualising clusters in 2D space with research paper titles for context.
5) Output: Generated CSV file with clustering results and summary report detailing cluster insights.

# Notes

- Lemmatization was preferred over stemming for maintaining valid words and semantic accuracy.
- Experimented with different vectorisation techniques and chose Word2Vec for its semantic understanding capabilities.
- Optimal cluster number decision was based on evaluating multiple metrics and practical considerations.
- In the visualizer for clustering, hovering produces name of the research file to provide context. 
