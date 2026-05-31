import numpy as np
import pandas as pd


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def assign_clusters(X, centroids):
    labels = []

    for x in X:
        distances = [euclidean(x, c) for c in centroids]
        labels.append(np.argmin(distances))

    return np.array(labels)


def recompute_centroids(X, labels, k):
    new_centroids = []

    for i in range(k):
        points = X[labels == i]

        if len(points) == 0:
            new_centroids.append(X[np.random.randint(0, len(X))])
        else:
            new_centroids.append(np.mean(points, axis=0))

    return np.array(new_centroids)


def kmeans(k, df, max_iter=100):

    X = df[['x', 'y']].values

    # init aleatoria
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):

        labels = assign_clusters(X, centroids)
        new_centroids = recompute_centroids(X, labels, k)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    df = df.copy()
    df['centroid'] = labels

    return df, centroids