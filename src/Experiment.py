import numpy as np
import matplotlib.pyplot as plt
import os
from DataGenerator import generateSamples
from RBFNN import RBFNN
from KMeans import kmeans


def run_experiment():

    data = generateSamples(500)

    X = data[['x', 'y']]
    Y = np.sqrt(data['x']**2 + data['y']**2)

    ks = [2, 5, 10, 15, 20]
    mses = []

    for k in ks:

        model = RBFNN()

        model.fit(X, Y, 100, k, kmeans)

        pred = model.predict(X)

        pred = np.array(pred).flatten()
        Y_np = np.array(Y).flatten()

        mse = np.mean((Y_np - pred) ** 2)

        mses.append(mse)

        print(k, mse)

    os.makedirs("../Results", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, mses, marker='o')
    plt.xlabel("Neuronas ocultas (K)")
    plt.ylabel("MSE")
    plt.title("RBFNN: Neuronas ocultas vs MSE")
    plt.grid(True)

    plt.savefig("../Results/MSE_vs_Neuronas.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()