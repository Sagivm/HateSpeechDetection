from configparser import ConfigParser
import pandas as pd
import numpy as np
from models import PseudoLabelingBERT
from k_means import BestKMeans
from pseudo_labeler import PseudoLabeler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def categorical_to_onehot(labels, values=None):
    if values is None:
        values = np.unique(labels)
    oh_labels = np.zeros((labels.shape[0], len(values)))
    for i, v in enumerate(values):
        oh_labels[labels == v, i] = 1
    return oh_labels


def train_model(config):
    df = pd.read_csv(config['DATA']['labeled_data_path']).iloc[:200]
    posts = df['text'].values
    labels = df['fake_label'].values
    labels = categorical_to_onehot(labels)

    model = PseudoLabelingBERT(use_tqdm=False, local_model_path='local_model')
    model.init_model(model_path=config['MODEL']['model_path'],
                     tokenizer_path=config['MODEL']['tokenizer_path'],
                     num_labels=config['MODEL'].getint('num_labels'))

    model.fit(posts=posts, labels=labels, model_name='bert_model_0', verbose=2, val_ratio=0.15, epochs=2, batch_size=1)


def load_model(config):
    model = PseudoLabelingBERT(local_model_path='local_model')
    model.load_model(model_path=config['MODEL']['model_path'],
                     tokenizer_path=config['MODEL']['tokenizer_path'],
                     num_labels=config['MODEL'].getint('num_labels'),
                     model_name='bert_model_0')

    df = pd.read_csv(config['DATA']['labeled_data_path']).iloc[50:250]
    posts = df['text'].values
    # pred = model.predict(X=posts, batch_size=8)
    embeddings = model.embeddings(X=posts, batch_size=8)
    return embeddings


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    #train_model(conf)
    embeddings = load_model(conf)
    post_embeddings = np.mean(embeddings,axis=1)  # Word axis

    number_of_ks = 4
    best_k_means = BestKMeans(post_embeddings)
    best_ks = best_k_means.best_n_k(number_of_ks, (2, 20))

    for kmean in best_ks:
        pl = PseudoLabeler(kmean)
        pl.read_posts(conf)
        pl.generate_pseudo_label()

