import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_matrix(corpus: list, ngram_range):
    model = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    X = model.fit_transform(corpus)
    return X, model


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
    :param threshold: if a number the number of pseudo labels, if float the percentage og pseudo labels to draw
    :type threshold:
    :return:
    :rtype:
    """
    if isinstance(threshold, float):
        threshold = threshold * len(terms)+1
    indeces = np.argsort(X.toarray())[::-1][:, :int(threshold)]
    return terms[indeces]


def calculate_cluster_variance(X:np.ndarray):
    """
    return variance along documents cluster with doc matrix
    :param X:
    :type X:
    :return:
    :rtype:
    """
    return X.var(axis=0)