from configparser import ConfigParser
import pandas as pd
import numpy as np
from src.models import PseudoLabelingBERT
import warnings
import os


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def categorical_to_onehot(labels, n_labels=None):
    if n_labels is None:
        values = np.unique(labels)
    else:
        values = np.arange(n_labels)

    oh_labels = np.zeros((labels.shape[0], len(values)))
    for i, v in enumerate(values):
        oh_labels[labels == v, i] = 1
    return oh_labels


def train_model(config: dict):
    df = pd.read_csv(config['DATA']['train_labeled_data_path'])
    posts = df['text'].values
    labels = df['label'].values
    labels = categorical_to_onehot(labels, n_labels=config['MODEL'].getint('num_labels'))

    model = PseudoLabelingBERT(use_tqdm=False, local_model_dir=config['MODEL']['local_models_dir'])
    model.init_model(model_path=config['MODEL']['base_model_path'],
                     tokenizer_path=config['MODEL']['tokenizer_path'],
                     num_labels=config['MODEL'].getint('num_labels'))

    model.fit(posts=posts, labels=labels, model_name=config['MODEL']['trained_model_name'],
              verbose=2, val_ratio=0.15, epochs=10, batch_size=8, lr=5e-05)
    return model


def load_model(config: dict):
    model = PseudoLabelingBERT(local_model_dir=config['MODEL']['local_models_dir'])
    model.load_model(model_path=config['MODEL']['base_model_path'],
                     tokenizer_path=config['MODEL']['tokenizer_path'],
                     num_labels=config['MODEL'].getint('num_labels'),
                     model_name=config['MODEL']['trained_model_name'])
    return model


def extract_embeddings(model: PseudoLabelingBERT, config: dict, save_file_name: str = ''):
    df = pd.read_csv(config['DATA']['posts_to_embed'])
    posts = df['text']
    tokens_embeddings = model.embeddings(posts, batch_size=8)
    post_embeddings = np.mean(tokens_embeddings, axis=1)

    save_dir_path = config['OUTPUT']['embeddings_dir_path']
    if save_file_name:
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        np.save(os.path.join(save_dir_path, save_file_name), post_embeddings)
    return post_embeddings


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    # model = load_model(conf)
    model = train_model(conf)

    post_embeddings = extract_embeddings(model, conf, save_file_name='random_labels_embeddings.npy')
