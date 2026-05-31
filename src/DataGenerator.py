import numpy as np
import pandas as pd


def generateSamples(nSamples=2000):


    x = np.random.uniform(-5, 5, nSamples)
    y = np.random.uniform(-5, 5, nSamples)

    label = np.sin(np.sqrt(x**2 + y**2))

    df = pd.DataFrame({
        'x': x,
        'y': y,
        'label': label
    })

    return df