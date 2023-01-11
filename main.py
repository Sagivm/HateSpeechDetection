from configparser import ConfigParser
import pandas as pd
import numpy as np
from models import PseudoLabelingBERT


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

    model.fit(posts=posts, labels=labels, model_name='bert_model_0', verbose=2, val_ratio=0.15, epochs=2, batch_size=8)


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
    print(1)


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    # train_model(conf)
    load_model(conf)
