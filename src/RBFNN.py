import numpy as np

def compute_sigmas(self, X, labels, centroids):

    sigmas = []

    for j, c in enumerate(centroids):

        cluster_points = X[labels == j]

        if len(cluster_points) == 0:
            sigmas.append(1.0)
        else:
            distances = np.linalg.norm(cluster_points - c, axis=1)
            sigmas.append(np.mean(distances))

    return np.array(sigmas)