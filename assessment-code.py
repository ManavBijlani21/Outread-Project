import PyPDF2
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import gensim
from gensim.models import Word2Vec
import numpy
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pandas
import plotly.express as px
from collections import defaultdict


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        #Abstract is always on the first page, so no need to even iterate 
        page = reader.pages[0]
        text = page.extract_text()
            
        # Look for the 'Abstract' keyword
        if 'abstract' in text.lower():
            start_idx = text.lower().index('abstract')+ len('abstract')
            abstract_end_idx = text.lower().find('introduction', start_idx)
            if abstract_end_idx == -1:
                # If no introduction is found, take the rest of the text
                abstract = text[start_idx:].strip()
            else:
                abstract = text[start_idx:abstract_end_idx].strip()
            
            return abstract
                
    return None
   

def iterate_pdfs():
    pdf_texts = {}

    for filename in os.listdir('Green Energy Dataset'):
        if filename.lower().endswith('.pdf'):
            #Constructing the path 
            pdf_path = os.path.join('Green Energy Dataset', filename)
            text = extract_text_from_pdf(pdf_path)

            if text :
                pdf_texts[filename] = text
            
            else:
                pdf_texts[filename] = "None"

    return pdf_texts


#Part 1
def data_preprocessing():

    #Consists of respective file names along with their abstract text
    abstract_dict = iterate_pdfs()
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    for key in abstract_dict:
        filtered_abstract = []
        abstract_sentence = abstract_dict[key]
        
        #Removing the special characters using regular expressions library
        abstract_sentence = re.sub(r'[^\w\s]', '', abstract_sentence)

        #Tokenizing the words
        words = nltk.word_tokenize(abstract_sentence)
        words = [w for w in words if not w in stop_words]

        for word in words:
            filtered_abstract.append(lemmatizer.lemmatize(word))
        abstract_dict[key] = filtered_abstract

    return abstract_dict

#Part 2
def text_vectorisation(abstract_dict):
    # Convert the list of lists of words into a list of sentences (documents)
    sentences = list(abstract_dict.values())

    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    #Generating vectors for each abstract
    abstract_vectors = []
    for abstract in sentences:
        valid_word = sum(model.wv[word] for word in abstract if word in model.wv) / len(abstract)
        abstract_vectors.append(valid_word)
    
    return abstract_vectors


def perform_clustering(abstract_vectors, num_clusters=4):
    kmeans = KMeans(n_clusters = num_clusters, random_state=42, n_init = 10)
    kmeans.fit(abstract_vectors)
    labels = kmeans.labels_

    return labels


def determine_optimal_clusters(abstract_vectors):
    wcss = [] # Sum of squares within clusters

    # Calculating WCSS for different values of k
    for k in range(2, 11):
        kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        kmeans.fit(abstract_vectors)
        wcss.append(kmeans.inertia_)  # Sum of squared distances from each point to its assigned cluster center

    #Plotting the elbow curve (This code can be used to determine the optimal number of clusters using elbow method)
    # plt.plot(range(2, 11), wcss)
    # plt.title('Elbow method for optimal k')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('WCSS')
    # plt.show()


def evaluate_clustering(abstract_vectors, labels):
    # Silhouette score
    silhouette_avg = silhouette_score(abstract_vectors, labels)
    print(f'Silhouette score: {silhouette_avg}')

    # Davies-Bouldin Index
    db_index = davies_bouldin_score(abstract_vectors, labels)
    print(f'Davies-Bouldin Index: {db_index}')


#Part 3
def clustering(abstract_vectors):

    # Part 1: Perform clustering with a chosen number of clusters
    num_clusters = 4 
    labels = perform_clustering(abstract_vectors, num_clusters)

    # Part 2: Determine optimal number of clusters
    determine_optimal_clusters(abstract_vectors)
    
    
    # Part 3: Evaluate clustering results
    evaluate_clustering(abstract_vectors, labels)
    
    return labels



#Part 4
def visualisation(labels, abstract_vectors, lets_preprocess):

    #PCA for dimensional reduction
    pca = PCA(n_components = 2)
    pca_result = pca.fit_transform(abstract_vectors)

    #Create a data frame for plotly
    df = pandas.DataFrame({
        'PCA1' : pca_result[:, 0],
        'PCA2' : pca_result[:, 1],
        'Cluster' : labels,
        'Title': list(lets_preprocess.keys())
    })

    #Plotly scatter plot
    fig = px.scatter(
        df, x = 'PCA1', y= 'PCA2', 
        color = 'Cluster', 
        hover_data = {'Title' : True},
        title = 'Clustering Research Papers'
    )

    fig.show()

#Part 5
def save_cluster_results(labels, lets_preprocess, output_file):
    # Create a DataFrame to store cluster results
    results_df = pandas.DataFrame({
        'Paper': list(lets_preprocess.keys()),
        'Cluster': labels
    })

    # Save to CSV
    results_df.to_csv(output_file, index=False)

def generate_summary_report(labels, lets_preprocess, num_clusters):
    # Count number of papers in each cluster
    cluster_counts = defaultdict(int)
    for label in labels:
        cluster_counts[label] += 1

    # Print number of clusters and papers in each cluster
    print(f"Number of clusters: {num_clusters}")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} papers")


# Load titles and filenames from CSV
csv_path = 'Green Energy Papers Database - Sheet1.csv' 
titles_df = pandas.read_csv(csv_path)

lets_preprocess = data_preprocessing()
abstract_vectors = text_vectorisation(lets_preprocess)
get_labels = clustering(abstract_vectors)
visualisation(get_labels, abstract_vectors, lets_preprocess)

# output_file = 'output-file.csv'
# save_cluster_results(get_labels, lets_preprocess, output_file)

generate_summary_report(get_labels, lets_preprocess, 4)