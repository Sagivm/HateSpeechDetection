import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import json
from copy import deepcopy


def plot_loss(losses, smooth_rate=50):
    losses_, indexes = smooth(losses, smooth_rate)

    plt.plot(indexes, losses_)
    plt.title('Train Loss over Training Steps')
    plt.xlabel('Training steps')
    plt.ylabel('Model\'s loss')
    plt.show()


def plot_accuracies(train_acc, val_acc):
    plt.plot(np.arange(1, len(train_acc)+1), train_acc, label='Train set')
    plt.plot(np.arange(1, len(val_acc)+1), val_acc, label='Validation set')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy scores')
    plt.legend()
    plt.show()


def plot_conf_matrix(conf_matrix):
    labels = ['Black & African-American', 'LGBTQI+', 'Jewish', 'Women & Girls']

    df = pd.DataFrame(conf_matrix, columns=labels, index=labels, dtype=int)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df, annot=True, fmt='g')
    plt.title('Validation Set Confusion Matrix')
    plt.show()


def smooth(a, rate):
    diff = rate - (len(a) % rate)
    res = a[-(len(a) % rate):]
    res_mean = np.mean(res)
    b = a + (list(np.ones((diff)) * res_mean))
    b = np.array(b)

    b = b.reshape(-1, rate)
    b = np.mean(b, axis=1)

    indexes = np.arange(1, len(a)+1 + diff)
    indexes = indexes.reshape(-1, rate)
    indexes = indexes[:, 0]
    return b, indexes


def leave_k_epochs(train_summary, k_epochs):
    total_steps = len(train_summary['losses'])
    total_epochs = len(train_summary['train_accuracy_scores'])
    steps_per_epoch = int(total_steps/total_epochs)

    new_train_summary = deepcopy(train_summary)
    new_train_summary['losses'] = train_summary['losses'][:steps_per_epoch*k_epochs]
    new_train_summary['train_accuracy_scores'] = train_summary['train_accuracy_scores'][:k_epochs]
    new_train_summary['val_accuracy_scores'] = train_summary['val_accuracy_scores'][:k_epochs]

    return new_train_summary


if __name__ == '__main__':
    with open('../output/train_summaries/t1.json') as f:
        train_summary = json.load(f)
    train_summary = leave_k_epochs(train_summary, 5)
    plot_loss(train_summary['losses'])
    plot_accuracies(train_summary['train_accuracy_scores'], train_summary['val_accuracy_scores'])
    plot_conf_matrix(train_summary['val_conf_matrix'])
