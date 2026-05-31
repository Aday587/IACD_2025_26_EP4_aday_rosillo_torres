import numpy as np
import matplotlib.pyplot as plt
from DataGenerator import generateSamples
from RBFNN import RBFNN
from KMeans import kmeans


def run_experiment():

    data = generateSamples(500)

    X = data[['x', 'y']]
    Y = data['label']

    ks = [2, 5, 10, 15, 20]
    mses = []

    for k in ks:

        model = RBFNN()

        model.fit(
            X=X,
            Y=Y,
            k=k,
            epochs=100,
            kmeans=kmeans
        )

        pred = model.predict(X)

        mse = np.mean((np.array(Y) - pred) ** 2)

        mses.append(mse)

        print(k, mse)

    plt.plot(ks, mses)
    plt.xlabel("Neurons (K)")
    plt.ylabel("MSE")
    plt.title("RBFNN: K vs MSE")
    plt.show()


if __name__ == "__main__":
    run_experiment()