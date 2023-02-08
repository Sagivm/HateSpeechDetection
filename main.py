from configparser import ConfigParser
import pandas as pd
import numpy as np
from src.models import ClassifyEmbedBERT
import warnings
import os
import json


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def categorical_to_onehot(labels: np.ndarray, n_labels=None):
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

    df = pd.read_csv(config['DATA']['test_labeled_data_path'])
    test_posts = df['text'].values
    test_labels = df['label'].values
    test_labels = categorical_to_onehot(test_labels, n_labels=config['MODEL'].getint('num_labels'))

    model = ClassifyEmbedBERT(use_tqdm=False, local_model_dir=config['MODEL']['local_models_dir'])
    model.init_model(model_path=config['MODEL']['base_model_path'],
                     tokenizer_path=config['MODEL']['tokenizer_path'],
                     num_labels=config['MODEL'].getint('num_labels'))

    train_summary = model.fit(posts=posts, labels=labels, test_posts=test_posts, test_labels=test_labels, model_name=config['MODEL']['trained_model_name'],
                              verbose=2, val_ratio=0.15, epochs=15, batch_size=8, lr=5e-05)
    return model, train_summary


def load_model(config: dict):
    model = ClassifyEmbedBERT(local_model_dir=config['MODEL']['local_models_dir'])
    model.load_model(model_path=config['MODEL']['base_model_path'],
                     tokenizer_path=config['MODEL']['tokenizer_path'],
                     num_labels=config['MODEL'].getint('num_labels'),
                     model_name=config['MODEL']['trained_model_name'])
    return model


def extract_embeddings(model: ClassifyEmbedBERT, config: dict):
    df = pd.read_csv(config['DATA']['posts_to_embed'])
    posts = df['text']
    tokens_embeddings = model.embeddings(posts, batch_size=8)
    post_embeddings = np.mean(tokens_embeddings, axis=1)

    save_dir_path = config['OUTPUT']['embeddings_dir_path']
    save_file_name = config['OUTPUT'].get('embeddings_file_name', '')
    if save_file_name:
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        np.save(os.path.join(save_dir_path, save_file_name), post_embeddings)
    return post_embeddings


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    # model = load_model(conf)
    model, train_summary = train_model(conf)

    post_embeddings = extract_embeddings(model, conf)
    with open('output/train_summaries/t1.json', 'w+') as f:
        json.dump(train_summary, f)
