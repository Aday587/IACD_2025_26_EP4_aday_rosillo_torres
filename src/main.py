from DataGenerator import generateSamples
from RBFNN import RBFNN
from KMeans import kmeans
from Plot3d import plot_3d

import numpy as np


def main():

    data = generateSamples(500)

    X = data[['x', 'y']]
    Y = data['label']

    print("Datos generados correctamente")
    print(data.head())

    rbf = RBFNN()

    print("\nEntrenando RBFNN...")


    rbf.fit(
        X=X,
        Y=Y,
        k=10,              # neuronas ocultas
        epochs=100,
        kmeans=kmeans
    )

    print("Entrenamiento finalizado")


    pred = rbf.predict(X)

    print("\nPredicciones (primeras 10):")
    print(pred[:10])

    print("\nValores reales (primeros 10):")
    print(np.array(Y)[:10])

    mse = np.mean((np.array(Y) - pred) ** 2)

    print("\nMSE:", mse)
    plot_3d(rbf)


if __name__ == "__main__":
    main()