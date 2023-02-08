import os
import math

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class ClassifyEmbedBERT:
    def __init__(self, use_tqdm: bool = True, local_model_dir: str = ''):
        self.tokenizer = None
        self.model_path = None
        self.local_model_dir = local_model_dir
        self.num_labels = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tqdm = use_tqdm

    def init_model(self, model_path: str, tokenizer_path: str, num_labels: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_path = model_path
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def load_model(self, model_path: str, tokenizer_path: str, num_labels: int, model_name: str):
        """
        Load model and tokenizer from a given path. If not exists initiate new model.
        :param model_path:
        :param tokenizer_path:
        :param num_labels:
        :param model_name:
        """
        self.init_model(model_path, tokenizer_path, num_labels)
        curr_model_path = os.path.join(self.local_model_dir, 'trained_models', f"{model_name}.pt")
        if not os.path.exists(curr_model_path):
            print(f"Model not found, saving base model in {curr_model_path}")
            self.save_model(model_name)
        else:
            self.model.load_state_dict(torch.load(curr_model_path))
            print(f"Model loaded from {curr_model_path}")
        self.model.cuda()

    def preprocess_post(self, post: str):
        """
        Tokenize a post using initiated tokenizer
        :param post:
        :return: Tokenized post
        """
        return self.tokenizer.encode_plus(
            post,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

    def forward(self, ids: torch.Tensor, masks: torch.Tensor, batch_size: int = 0, return_embeddings: bool = False):
        """
        Preform a forward pass over the model, given the token-ids and masks
        :param ids: Token IDs
        :param masks: Masked vector to pass the model
        :param batch_size: Batch size to use to reduce memory usage
        :param return_embeddings: Whether to return embedding vectors along with the model output, used to generate
                                  posts embeddings
        :return: model output and possibly embeddings
        """
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
                for b_input_ids, b_input_mask in tqdm(iter_batches(ids, masks, batch_size),
                                                      total=math.ceil(ids.shape[0] / batch_size),
                                                      desc='Eval', disable=(not self.use_tqdm)):

                    eval_output = self.model(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             output_hidden_states=return_embeddings)

                    logits = eval_output.logits.detach().cpu().numpy()
                    total_logits.append(logits)
                    if return_embeddings:
                        curr_emb = eval_output.hidden_states[-1].detach().cpu().numpy()
                        total_embs.append(curr_emb)

            logits = np.concatenate(total_logits)
            if return_embeddings:
                embs = np.concatenate(total_embs)

        if not return_embeddings:
            return logits
        else:
            return logits, embs

    def forward_with_true_labels(self, X: TensorDataset, batch_size: int = 0):
        """
        Run forward pass over the model, return true-labels of the given dataset
        :param X:
        :param batch_size:
        :return:
        """
        input_ids, input_mask, labels = [t.to(self.device) for t in X.tensors]
        labels = labels.to('cpu').numpy()
        logits = self.forward(input_ids, input_mask, batch_size=batch_size)
        return logits, labels

    def fit(self, model_name: str, posts: np.ndarray, labels: np.ndarray, test_posts: np.ndarray,
            test_labels: np.ndarray, val_ratio: float = 0.15, epochs: int = 7, batch_size: int = 8, verbose: int = 0,
            lr: float = 5e-5):
        """
        Train the model over the given train dataset, with the given parameters.
        Keep results of the training and testing sets along the training process.
        :param model_name:
        :param posts:
        :param labels:
        :param test_posts:
        :param test_labels:
        :param val_ratio:
        :param epochs:
        :param batch_size:
        :param verbose:
        :param lr:
        :return: A summary of statistics and metrics from the training process.
        """
        if not os.path.exists(os.path.join(self.local_model_dir, 'temp_models')):
            os.makedirs(os.path.join(self.local_model_dir, 'temp_models'))
        train_summary = {'losses': []}

        token_id, attention_masks, labels = self.preprocess_data_set(posts, labels)
        test_token_id, test_attention_masks, test_labels = self.preprocess_data_set(test_posts, test_labels)

        # Indices of the train and validation splits stratified by labels
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=val_ratio,
            shuffle=True,
            stratify=labels)

        print(
            f"BERT - Train set of size - {train_idx.shape[0]:,}, validation - {val_idx.shape[0]:,}, test, - {test_token_id.shape[0]:,}")
        # Train and validation sets
        train_set = TensorDataset(token_id[train_idx],
                                  attention_masks[train_idx],
                                  labels[train_idx])

        val_set = TensorDataset(token_id[val_idx],
                                attention_masks[val_idx],
                                labels[val_idx])

        test_set = TensorDataset(test_token_id,
                                 test_attention_masks,
                                 test_labels)

        # Prepare DataLoader
        train_dataloader = DataLoader(
            train_set,
            sampler=RandomSampler(train_set),
            batch_size=batch_size
        )

        # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      eps=1e-08)

        # Run on GPU
        self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        train_accuracy = []
        val_accuracy = []
        test_accuracy = []
        train_f1 = []
        val_f1 = []
        test_f1 = []

        train_loss = []
        best_val_acc = 0
        best_epoch = 0

        for epoch in tqdm(range(epochs), desc='BERT training epochs', disable=False):
            # ========== Training ==========

            # Set model to training mode
            self.model.train()

            # Tracking variables
            tr_loss = 0

            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train',
                                    disable=(not self.use_tqdm)):
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

                batch_loss = train_output.loss.item()
                tr_loss += batch_loss
                train_summary['losses'].append(batch_loss)

            train_loss.append(tr_loss)
            # ========== Validation ==========

            # Set model to evaluation mode
            self.model.eval()

            curr_train_acc, train_conf_matrix, curr_train_f1 = self.get_metrics(train_set, batch_size)
            train_accuracy.append(curr_train_acc)
            train_f1.append(curr_train_f1)

            curr_val_acc, val_conf_matrix, curr_val_f1 = self.get_metrics(val_set, batch_size)
            val_accuracy.append(curr_val_acc)
            val_f1.append(curr_val_f1)

            curr_test_acc, test_conf_matrix, curr_test_f1 = self.get_metrics(test_set, batch_size)
            test_accuracy.append(curr_test_acc)
            test_f1.append(curr_test_f1)

            if curr_val_acc > best_val_acc:
                torch.save(self.model.state_dict(),
                           os.path.join(self.local_model_dir, 'temp_models', f"{model_name}.tmp"))
                best_epoch = epoch
                best_val_acc = curr_val_acc
                train_summary['train_conf_matrix'] = train_conf_matrix.tolist()
                train_summary['val_conf_matrix'] = val_conf_matrix.tolist()
                train_summary['test_conf_matrix'] = test_conf_matrix.tolist()

            if verbose > 1:
                print(
                    f"Epoch {epoch}: Train acc - {curr_train_acc}, Validation acc - {curr_val_acc}, Test acc - {curr_test_acc}, Loss - {tr_loss}\n"
                    f"Epoch {epoch}: Train f1 - {curr_train_f1}, Validation f1 - {curr_val_f1}, Test f1 - {curr_test_f1}\n"
                    f"Train: {train_conf_matrix}\n"
                    f"Val: {val_conf_matrix}\n"
                    f"Test: {test_conf_matrix}\n")

        if verbose > 0:
            print(f"Best epoch - {best_epoch}: Validation acc - {best_val_acc}")
            print(f"Train losses: {train_loss}")
        self.model.load_state_dict(torch.load(os.path.join(self.local_model_dir, 'temp_models', f"{model_name}.tmp")))
        self.save_model(model_name)

        train_summary['train_accuracy_scores'] = train_accuracy
        train_summary['val_accuracy_scores'] = val_accuracy
        train_summary['test_accuracy_scores'] = test_accuracy

        train_summary['train_f1_scores'] = train_f1
        train_summary['val_f1_scores'] = val_f1
        train_summary['test_f1_scores'] = test_f1

        return train_summary

    def preprocess_data_set(self, posts, labels=None):
        """
        Process posts into tokenized vectors, create attentions masks.
        :param posts:
        :param labels:
        :return:
        """
        token_id = []
        attention_masks = []
        for post in posts:
            encoding_dict = self.preprocess_post(post)
            token_id.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        token_id = torch.cat(token_id, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        if labels is not None:
            labels = torch.tensor(labels)
            return token_id, attention_masks, labels
        else:
            return token_id, attention_masks

    def get_metrics(self, data_set, batch_size):
        """
        Calculate accuracy, Mean-F1 score and confusion matrix of the given dataset, with the trained model.
        :param data_set:
        :param batch_size:
        :return:
        """
        logits, label_ids = self.forward_with_true_labels(data_set, batch_size)
        true_labels, pred_labels = np.argmax(logits, axis=1), np.argmax(label_ids, axis=1)

        curr_acc = accuracy_score(y_pred=pred_labels, y_true=true_labels)
        conf_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)
        curr_f1 = f1_score(y_pred=pred_labels, y_true=true_labels, average='macro')

        return curr_acc, conf_matrix, curr_f1

    def save_model(self, model_name: str):
        """
        Save model current state locally.
        :param model_name:
        :return:
        """
        if not os.path.exists(os.path.join(self.local_model_dir, 'trained_models')):
            os.makedirs(os.path.join(self.local_model_dir, 'trained_models'))

        curr_model_path = os.path.join(self.local_model_dir, 'trained_models', f"{model_name}.pt")
        torch.save(self.model.state_dict(), curr_model_path)

        print(f"Model saved at {curr_model_path}")

    def predict(self, X: np.ndarray, batch_size=0):
        """
        Over the given dataset X, return classify and return labels.
        :param X:
        :param batch_size:
        :return:
        """
        token_id, attention_masks = self.preprocess_data_set(X)
        X = TensorDataset(token_id,
                          attention_masks)
        input_ids, input_mask = [t.to(self.device) for t in X.tensors]

        logits = self.forward(input_ids, input_mask, batch_size=batch_size)
        return np.argmax(logits, axis=1)

    def embeddings(self, X: np.ndarray, batch_size=0):
        """
        Over the given dataset X, calculate embedding vectors from model
        :param X:
        :param batch_size:
        :return:
        """
        token_id, attention_masks = self.preprocess_data_set(X)

        X = TensorDataset(token_id,
                          attention_masks)
        input_ids, input_mask = [t.to(self.device) for t in X.tensors]

        _, embs = self.forward(input_ids, input_mask, batch_size=batch_size, return_embeddings=True)
        return embs


def iter_batches(input_ids, input_mask, batch_size: int):
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
