def compute_sigmas(self, X, labels, centroids):

    sigmas = []

    for j, c in enumerate(centroids):

        # puntos del cluster j
        cluster_points = X[labels == j]

        # si el cluster está vacío (caso raro de KMeans)
        if len(cluster_points) == 0:
            sigmas.append(1.0)
            continue

        # distancia euclídea de cada punto al centroide
        distances = np.linalg.norm(cluster_points - c, axis=1)

        # sigma = dispersión media del cluster
        sigma = np.mean(distances)

        # evitar sigma demasiado pequeño (estabilidad numérica)
        if sigma < 1e-6:
            sigma = 1.0

        sigmas.append(sigma)

    return np.array(sigmas)