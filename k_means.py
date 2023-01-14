from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

class BestKMeans():
    def __init__(self,X):
        self.X = X

    def fit(self, k:int):
        self.model = KMeans(n_clusters=k)
        self.model.fit(self.X)
        return self.model

    def best_n_k(self, n: int, k_range: tuple):
        scores = list()
        for k in range(*k_range):
            model = self.fit(k)
            scores.append(silhouette_score(self.X,model.labels_))
            print(f"KMeans {k},Score - {silhouette_score(self.X,model.labels_)}")

        #  Get best n score indexes
        # df = pd.DataFrame(scores, columns=["kmeans score"])
        # df.to_csv('kmeans_scores.csv', index=False)

        best_k_score = sorted(range(len(scores)), key=lambda i: scores[i],reverse=True)[:n]
        return [self.fit(k+k_range[0]) for k in best_k_score]