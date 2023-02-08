from configparser import ConfigParser
import numpy as np
from sklearn.cluster import KMeans
from src.k_means import BestKMeans
from src.pseudo_labeler import PseudoLabeler


def kmeans_find_best_k(embeddings):
    number_of_ks = 4
    best_k_means = BestKMeans(embeddings)
    best_ks = best_k_means.best_n_k(number_of_ks, (2, 10))
    return best_ks


def pseudo_labeler(embeddings, k, config):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)

    pl = PseudoLabeler(kmeans)
    pl.read_posts(config)
    clusters_posts = pl.pseudo_labeling()
    clusters_dict = pl.generate_cluster_common_unique_words_dict()
    pl.further_analyze(clusters_dict, save_lists_path='output/words_per_cluster.json')


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')

    # print("No training:")
    # embeddings = np.load('output/embeddings/baseline.npy')
    # kmeans_find_best_k(embeddings)

    # print("\nWith training:")
    embeddings = np.load('output/embeddings/trained_embeddings.npy')

    pseudo_labeler(embeddings, 4, conf)

    # kmeans_find_best_k(embeddings)
    # print(1)
