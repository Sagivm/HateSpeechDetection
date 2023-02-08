Uses configuration file located in `config.ini`.\
### main.py - Model Training and Embeddings Extraction
**train_model()** - \
Training the model to classify the posts into classes based on their target community. 
The training dataset, test dataset, base model and directories to save model and results are configurable using the 
configuration file.\
The method load the data, process the label into one-hot encoded vectors, initiate model and fine-tune over the train dataset.\
A summarization of the training process's statistics and performance over the train and test data set is returned to further analysis.

**load_model()** - \
Loading a trained model according to the location specified in the configuration file.

**extract_embeddings()** - \
Using the given model to extract embedding over the dataset specified in the configuration file. The function extract token 
embeddings from the model, and calculate their mean values to get a vector embedding for each post.
Saves the results in `.npy` file which specified in the configuration file.


### analyze_embeddings.py - Cluster Posts and Assign them Pseudo-labels 
**kmeans_find_best_k** -\
Find the ks which divide the embedding vectors into clusters with the highest
silhouette_score out of a given range of ks.

**pseudo_labeler** - \
For a given k and embedding vectors, train a KMeans model to divide the posts into clusters. For each cluster, find its
unique set of words which its TF-IDF score is the highest, for a certian amount of tokens wanted.

