import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from RBFNN import RBFNN
from KMeans import kmeans


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


def plot_3d(model, n=50):

    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, y)

    X_flat = np.column_stack([X.ravel(), Y.ravel()])

    Z_real = f(X_flat[:, 0], X_flat[:, 1])

    Z_pred = model.predict(X_flat)

    Z_real = Z_real.reshape(X.shape)
    Z_pred = Z_pred.reshape(X.shape)

    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z_real)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z_pred)

    plt.show()