import numpy as np
import matplotlib.pyplot as plt
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

        model = RBFNN(2, k, 1, 0.01)

        model.fit(X, Y, 100, kmeans)

        pred = model.predict(X)

        mse = np.mean((Y - pred.flatten()) ** 2)

        mses.append(mse)

        print(k, mse)

    plt.plot(ks, mses, marker='o')
    plt.xlabel("Neuronas ocultas (K)")
    plt.ylabel("MSE")
    plt.title("RBFNN: Neuronas vs Error")
    plt.grid(True)
    plt.savefig("../Results/MSE_vs_Neuronas.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()