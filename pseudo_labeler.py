import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.text_handling import *
import nltk
from nltk.stem import PorterStemmer


class PseudoLabeler:

    def __init__(self, kmeans_model: KMeans, n_gram: tuple = (1,2), n_pseudo: float = 0.02, use_stemmer: bool = True):
        self.kmeans_model = kmeans_model
        self.n_gram = n_gram
        self.n_pseudo = n_pseudo
        self.use_stemmer = use_stemmer

    def read_posts(self, config):
        df = pd.read_csv(config['DATA']['labeled_data_path']).iloc[50:250]
        self.posts = df['text'].values

    def generate_pseudo_label(self):
        combined_cluster_list = list()
        for cluster in range(self.kmeans_model.n_clusters):
            cluster_posts = list(self.posts[self.kmeans_model.labels_ == cluster])
            combined_cluster_list.append(" <SEP> ".join(cluster_posts))

        if self.use_stemmer:
            stemmer = PorterStemmer()
            combined_cluster_list = [stemmer.stem(cluster_post) for cluster_post in combined_cluster_list]
        X, m = tf_idf_matrix(combined_cluster_list, self.n_gram)
        tmp = generate_pseudo_labeling(X, m.get_feature_names_out(), 0.01)
        x=0