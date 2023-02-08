import pandas as pd
from sklearn.cluster import KMeans
from utils.text_handling import *
from nltk.stem import PorterStemmer
import functools
from nltk.tokenize import sent_tokenize, word_tokenize

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from copy import deepcopy
import json


class PseudoLabeler:

    def __init__(self, kmeans_model: KMeans, n_gram: tuple = (1, 3), n_pseudo: float = 0.02, use_stemmer: bool = False):
        self.kmeans_model = kmeans_model
        self.n_gram = n_gram
        self.n_pseudo = n_pseudo
        self.use_stemmer = use_stemmer
        self.posts = None

    def read_posts(self, config):
        df = pd.read_csv(config['DATA']['posts_to_embed'])
        self.posts = df['text'].values

    def pseudo_labeling(self):
        clusters_posts = []
        for cluster in range(self.kmeans_model.n_clusters):
            cluster_posts = list(self.posts[self.kmeans_model.labels_ == cluster])
            clusters_posts.append(cluster_posts)
        return clusters_posts

    def generate_cluster_common_unique_words_dict(self):
        clusters_posts = self.pseudo_labeling()
        combined_cluster_list = []
        for cluster_posts in clusters_posts:
            combined_cluster_list.append(" <SEP> ".join(cluster_posts))

        if self.use_stemmer:
            stemmer = PorterStemmer()
            combined_cluster_list = [stemmer.stem(cluster_post) for cluster_post in combined_cluster_list]

        X, tf_idf_model = tf_idf_matrix(combined_cluster_list, (1, 1))
        keywords_per_cluster, weights = generate_pseudo_labeling(X, tf_idf_model.get_feature_names_out(), 0.1)  # 0.01)
        clusters_dict = self.to_clusters_dicts(keywords_per_cluster, weights)
        new_clusters_dict = self.to_unique_dicts(clusters_dict)
        return new_clusters_dict

    def further_analyze(self, clusters_dicts, save_lists_path=''):
        # plot word clouds
        for cl in clusters_dicts:
            self.plot_wordcloud(cl)

        if save_lists_path:
            words = [list(cd.keys()) for cd in clusters_dicts]
            with open(save_lists_path, 'w+') as f:
                json.dump(words, f)

    def check_clusters_agains_kb(self, clusters, kb, labels_to_leave):
        for k in kb:
            if k not in labels_to_leave:
                del kb[k]

    @staticmethod
    def to_unique_sets(kws):
        kws = [set(row) for row in kws]
        new_kws = []
        for i, curr_set in enumerate(kws):
            total_set = functools.reduce(lambda x, y: x | y, kws[:i] + kws[i + 1:])
            new_kws.append(curr_set - total_set)
        return new_kws

    @staticmethod
    def to_unique_dicts(clusters_dict):
        print(f"Before: {[len(cd) for cd in clusters_dict]}")
        new_clusters_dict = deepcopy(clusters_dict)
        keys = [set(wc.keys()) for wc in clusters_dict]
        for i, curr_dict in enumerate(clusters_dict):
            keys_to_remove = functools.reduce(lambda x, y: x | y, keys[:i] + keys[i + 1:])
            for k in keys_to_remove:
                if k in new_clusters_dict[i]:
                    del new_clusters_dict[i][k]
        print(f"After: {[len(cd) for cd in new_clusters_dict]}")
        return new_clusters_dict

    @staticmethod
    def to_clusters_dicts(keywords_per_cluster, weights):
        total_clusters = []
        for i, row in enumerate(keywords_per_cluster):
            curr_cluster = {w: weights[i, j] for j, w in enumerate(row)}
            total_clusters.append(curr_cluster)
        return total_clusters

    def plot_wordcloud(self, words_and_weights):
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate_from_frequencies(
            words_and_weights)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def stem(self, doc:str):
        tokenized_sentence = []
        for word in doc.split(' '):
            tokenized_sentence.append(PorterStemmer().stem(word))
        return " ".join(tokenized_sentence)
