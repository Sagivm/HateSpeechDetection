import os
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


class PseudoLabelingBERT:
    def __init__(self, use_tqdm=True, local_model_path=''):
        self.tokenizer = None
        self.model_path = None
        self.local_model_path = local_model_path
        self.num_labels = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tqdm = use_tqdm

    def init_model(self, model_path, tokenizer_path, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_path = model_path
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def load_model(self, model_path, tokenizer_path, num_labels, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.num_labels = num_labels
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        curr_model_path = os.path.join(self.local_model_path, 'trained_models', f"{model_name}.pt")
        self.model.load_state_dict(torch.load(curr_model_path))
        print(f"Model loaded from {curr_model_path}")
        self.model.cuda()

    def preprocess_post(self, post):
        return self.tokenizer.encode_plus(
            post,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

    def forward(self, ids, masks, batch_size=0, return_embeddings=False):
        if batch_size == 0:
            with torch.no_grad():
                # Forward pass
                eval_output = self.model(ids,
                                         token_type_ids=None,
                                         attention_mask=masks,
                                         return_dict=return_embeddings,
                                         output_hidden_states=return_embeddings)
            logits = eval_output.logits.detach().cpu().numpy()
            if return_embeddings:
                embs = eval_output.hidden_states[-1].detach().cpu().numpy()
        else:
            total_logits = []
            total_embs = []
            with torch.no_grad():
                for b_input_ids, b_input_mask in tqdm_iter(iter_batches(ids, masks, batch_size),
                                                           total=math.ceil(ids.shape[0] / batch_size),
                                                           desc='Eval', use_tqdm=self.use_tqdm):
                    eval_output = self.model(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             return_dict=return_embeddings,
                                             output_hidden_states=return_embeddings)

                    logits = eval_output.logits.detach().cpu().numpy()
                    total_logits.append(logits)
                    if return_embeddings:
                        curr_emb = eval_output.hidden_states[-1].detach().cpu().numpy()
                        total_embs.append(curr_emb)

            logits = np.concatenate(total_logits)
            embs = np.concatenate(total_embs)

        if not return_embeddings:
            return logits
        else:
            return logits, embs

    def forward_with_true_labels(self, X, batch_size=0):
        input_ids, input_mask, labels = [t.to(self.device) for t in X.tensors]
        labels = labels.to('cpu').numpy()
        logits = self.forward(input_ids, input_mask, batch_size=batch_size)
        return logits, labels

    def fit(self, model_name, posts, labels, val_ratio=0.2, epochs=7, batch_size=8, plot_confusion_matrix=False,
            verbose=0):
        if not os.path.exists(os.path.join(self.local_model_path, 'temp_models')):
            os.makedirs(os.path.join(self.local_model_path, 'temp_models'))

        token_id = []
        attention_masks = []
        for post in posts:
            encoding_dict = self.preprocess_post(post)
            token_id.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        token_id = torch.cat(token_id, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # Indices of the train and validation splits stratified by labels
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=val_ratio,
            shuffle=True,
            stratify=labels)

        print(f"BERT - Train set of size - {train_idx.shape[0]:,}, validation - {val_idx.shape[0]:,}")
        # Train and validation sets
        train_set = TensorDataset(token_id[train_idx],
                                  attention_masks[train_idx],
                                  labels[train_idx])

        val_set = TensorDataset(token_id[val_idx],
                                attention_masks[val_idx],
                                labels[val_idx])

        # Prepare DataLoader
        train_dataloader = DataLoader(
            train_set,
            sampler=RandomSampler(train_set),
            batch_size=batch_size
        )
        val_dataloader = DataLoader(
            val_set,
            sampler=RandomSampler(val_set),
            batch_size=batch_size
        )

        # Load the BertForSequenceClassification model

        # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=5e-5,
                                      eps=1e-08
                                      )

        # Run on GPU
        self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        train_accuracy = []
        val_accuracy = []
        best_val_acc = 0
        best_epoch = 0

        for epoch in tqdm(range(epochs), desc='BERT training epochs'):
            # ========== Training ==========

            # Set model to training mode
            self.model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in tqdm_iter(enumerate(train_dataloader), total=len(train_dataloader), desc='Train',
                                         use_tqdm=self.use_tqdm):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                # Forward pass
                train_output = self.model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)
                # Backward pass
                train_output.loss.backward()
                optimizer.step()
                # Update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            # ========== Validation ==========

            # Set model to evaluation mode
            self.model.eval()

            logits, label_ids = self.forward_with_true_labels(train_set, batch_size)
            curr_train_acc = accuracy_score(y_pred=np.argmax(logits, axis=1), y_true=np.argmax(label_ids, axis=1))
            train_accuracy.append(curr_train_acc)

            logits, label_ids = self.forward_with_true_labels(val_set, batch_size)
            curr_val_acc = accuracy_score(y_pred=np.argmax(logits, axis=1), y_true=np.argmax(label_ids, axis=1))
            val_accuracy.append(curr_val_acc)

            if curr_val_acc > best_val_acc:
                torch.save(self.model.state_dict(), os.path.join(self.local_model_path, 'temp_models', f"{model_name}.tmp"))
                best_epoch = epoch
                best_val_acc = curr_val_acc

            if verbose > 1:
                print(f"Epoch {epoch}: Train acc - {curr_train_acc}, Validation acc - {curr_val_acc}")

        if plot_confusion_matrix:
            conf_mat = confusion_matrix(y_pred=np.argmax(logits, axis=1), y_true=np.argmax(label_ids, axis=1))
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                                          display_labels=['Pro-Ukraine', 'Neutral', 'Pro-Russia'])
            disp.plot()
            plt.title('Others - test set')
            plt.show()

        if verbose > 0:
            print(f"Best epoch - {best_epoch}: Validation acc - {best_val_acc}")

        self.model.load_state_dict(torch.load(os.path.join(self.local_model_path, 'temp_models', f"{model_name}.tmp")))
        self.save_model(model_name)

    def save_model(self, model_name):
        if not os.path.exists(os.path.join(self.local_model_path, 'trained_models')):
            os.makedirs(os.path.join(self.local_model_path, 'trained_models'))

        curr_model_path = os.path.join(self.local_model_path, 'trained_models', f"{model_name}.pt")
        torch.save(self.model.state_dict(), curr_model_path)

        print(f"Model saved at {curr_model_path}")

    def pseudo_label(self, X):
        pass

    def predict(self, X, batch_size=0):
        token_id = []
        attention_masks = []
        for post in X:
            encoding_dict = self.preprocess_post(post)
            token_id.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        token_id = torch.cat(token_id, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        X = TensorDataset(token_id,
                          attention_masks)
        input_ids, input_mask = [t.to(self.device) for t in X.tensors]

        logits = self.forward(input_ids, input_mask, batch_size=batch_size)
        return np.argmax(logits, axis=1)

    def embeddings(self, X, batch_size=0):
        token_id = []
        attention_masks = []
        for post in X:
            encoding_dict = self.preprocess_post(post)
            token_id.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        token_id = torch.cat(token_id, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        X = TensorDataset(token_id,
                          attention_masks)
        input_ids, input_mask = [t.to(self.device) for t in X.tensors]

        _, embs = self.forward(input_ids, input_mask, batch_size=batch_size, return_embeddings=True)
        return embs


def tqdm_iter(iter, desc, total, use_tqdm=True):
    if use_tqdm:
        return tqdm(iter, total=total, desc=desc)
    else:
        return iter


def iter_batches(input_ids, input_mask, batch_size):
    """
    X - Matrix of samples and features. Shape - (samples_dimension, number_of_samples).
    Y - Labeling of samples. Shape - (number_of_classes, number_of_samples).
    batch_size - size of wanted batch.
    idx - number of batch wanted.
    returns -
        X_curr - current batch of samples.
        Y_curr - current batch labels.
    """
    m = input_ids.shape[0]
    for j in range(math.ceil(m / batch_size)):
        l_idx = j * batch_size
        u_idx = (j + 1) * batch_size
        if u_idx > m:
            u_idx = m
        yield input_ids[l_idx:u_idx], input_mask[l_idx:u_idx]
