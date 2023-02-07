import pandas as pd
from sklearn.cluster import KMeans
from utils.text_handling import *
from nltk.stem import PorterStemmer
import functools


class PseudoLabeler:
    def __init__(self, kmeans_model: KMeans, n_gram: tuple = (1, 3), n_pseudo: float = 0.02, use_stemmer: bool = True):
        self.kmeans_model = kmeans_model
        self.n_gram = n_gram
        self.n_pseudo = n_pseudo
        self.use_stemmer = use_stemmer
        self.posts = None

    def read_posts(self, config):
        df = pd.read_csv(config['DATA']['posts_to_embed'])
        self.posts = df['text'].values

    def generate_pseudo_label(self):
        combined_cluster_list = list()
        for cluster in range(self.kmeans_model.n_clusters):
            cluster_posts = list(self.posts[self.kmeans_model.labels_ == cluster])
            combined_cluster_list.append(" <SEP> ".join(cluster_posts))

        if self.use_stemmer:
            stemmer = PorterStemmer()
            combined_cluster_list = [stemmer.stem(cluster_post) for cluster_post in combined_cluster_list]
        X, tf_idf_model = tf_idf_matrix(combined_cluster_list, self.n_gram)
        keywords_per_cluster = generate_pseudo_labeling(X, tf_idf_model.get_feature_names_out(), 0.01)
        unique_keywords_per_cluster = self.to_unique_sets(keywords_per_cluster)

    @staticmethod
    def to_unique_sets(kws):
        kws = [set(row) for row in kws]
        new_kws = []
        for i, curr_set in enumerate(kws):
            total_set = functools.reduce(lambda x, y: x | y, kws[:i]+kws[i+1:])
            new_kws.append(curr_set - total_set)
        return new_kws
