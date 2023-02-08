import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_matrix(corpus: list, ngram_range):
    tf_idf_model = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    X = tf_idf_model.fit_transform(corpus)
    return X, tf_idf_model


def compute_doc_similarity_matrix(X: np.ndarray):
    """
    Return similarity matrix of docs between document
    :param X:
    :type X:
    :return:
    :rtype:
    """
    return X * np.transpose(X)


def compute_word_similarity_matrix(X: np.ndarray):
    """
    Return similarity matrix of docs
    :param X:
    :type X:
    :return:
    :rtype:
    """
    return X * np.transpose(X)


def compute_document_similarity_matrix(X: np.ndarray):
    """
    Return similarity matrix of words
    :param X:
    :type X:
    :return:
    :rtype:
    """
    return X * np.transpose(X) * X


def generate_pseudo_labeling(X, terms: list, threshold: float):
    """

    :param X:
    :type X:
    :param terms:
    :type terms:
    :param threshold: if a number the number of pseudo labels, if float the percentage of pseudo labels to draw
    :type threshold:
    :return:
    :rtype:
    """
    if isinstance(threshold, float):
        threshold = threshold * len(terms)+1
    indeces = np.argsort(X.toarray())[::-1][:, :int(threshold)]
    weights = np.sort(X.toarray())[:, ::-1][:, :int(threshold)]
    return terms[indeces], weights


def calculate_cluster_variance(X: np.ndarray):
    """
    return variance along documents cluster with doc matrix
    :param X:
    :type X:
    :return:
    :rtype:
    """
    return X.var(axis=0)
