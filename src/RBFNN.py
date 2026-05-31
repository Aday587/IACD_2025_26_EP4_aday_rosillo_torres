import numpy as np
from Adaline import Adaline


class RBFNN:

    def __init__(self):
        self.centroids = None
        self.sigmas = None
        self.adaline = None
        self.has_trained = False

    def compute_sigmas(self, X, labels, centroids):

        sigmas = []

        for j, c in enumerate(centroids):

            cluster_points = X[labels == j]

            if len(cluster_points) == 0:
                sigmas.append(1.0)
                continue

            distances = np.linalg.norm(cluster_points - c, axis=1)
            sigma = np.mean(distances)

            if sigma < 1e-6:
                sigma = 1.0

            sigmas.append(sigma)

        return np.array(sigmas)

    def build_Z(self, X, centroids, sigmas):

        Z = []

        for x in X:

            row = []

            for j, c in enumerate(centroids):

                dist2 = np.sum((x - c) ** 2)

                value = np.exp(
                    -dist2 / (2 * (sigmas[j] ** 2))
                )

                row.append(value)

            Z.append(row)

        return np.array(Z)

    def fit(self, X, Y, k, epochs, kmeans):

        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)

        import pandas as pd
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['label'] = Y

        df, centroids = kmeans(k, df)

        X_np = df[['x', 'y']].values.astype(np.float64)
        labels = df['centroid'].values

        sigmas = self.compute_sigmas(X_np, labels, centroids)

        Z = self.build_Z(X_np, centroids, sigmas)

        self.adaline = Adaline(eta=0.01)
        self.adaline.fit(Z, Y, epochs)

        self.centroids = centroids
        self.sigmas = sigmas
        self.has_trained = True

    def predict(self, X):

        if not self.has_trained:
            raise Exception("Modelo no entrenado")

        X = np.array(X, dtype=np.float64)

        Z = self.build_Z(
            X,
            self.centroids,
            self.sigmas
        )

        return self.adaline.predict(Z)