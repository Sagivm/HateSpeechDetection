from configparser import ConfigParser
import numpy as np
from sklearn.cluster import KMeans
from src.k_means import BestKMeans
from src.pseudo_labeler import PseudoLabeler
from typing import Tuple
import warnings

warnings.simplefilter("ignore", FutureWarning)


def kmeans_find_best_k(embeddings, number_of_ks: int = 4, range: Tuple[int] = (2, 10)):
    """
    Run BestKMeans to find the number_of_ks ks which divide the embeddings into clusters with the highest
    silhouette_score.
    :param embeddings: The embeddings vectors to cluster
    :param number_of_ks: number of top ks to retunr
    :param range: range of ks to search
    :return: best_kst - the KMeans models with the highest silhouette_score
    """
    best_k_means = BestKMeans(embeddings)
    best_ks = best_k_means.best_n_k(number_of_ks, range)
    return best_ks


def pseudo_labeler(embeddings, k, config):
    """
    Divide the posts into their clusters, preform analysis over the posts
    :param embeddings:
    :param k:
    :param config:
    :return:
    """
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

    embeddings = np.load('output/embeddings/trained_embeddings.npy')
    pseudo_labeler(embeddings, 5, conf)
