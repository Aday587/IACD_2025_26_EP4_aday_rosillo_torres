import numpy as np


class Adaline:

    def __init__(self, eta):
        self.eta = eta
        self.W_ = list()
        self.errorForEachEpoch_ = list()

    def _initW(self, trainingX):

        W_ = list()

        totalMean_ = np.mean(trainingX)
        totalStd_ = np.std(trainingX)

        means_ = np.mean(trainingX, axis=0)
        stds_ = np.std(trainingX, axis=0)

        random_seed = np.random.RandomState()

        W_.append(
            random_seed.normal(
                loc=totalMean_,
                scale=totalStd_
            )
        )

        for mean_, std_ in zip(means_, stds_):

            W_.append(
                random_seed.normal(
                    loc=mean_,
                    scale=std_
                )
            )

        return np.array(W_)

    def fit(self, trainingX, trainingY, epoch=50):

        if not self.errorForEachEpoch_:

            bestError = np.inf
            W_ = self._initW(trainingX)

        else:

            bestError = min(self.errorForEachEpoch_)
            W_ = self.W_

        for _ in range(epoch):

            randomIndices_ = np.array(
                range(trainingX.shape[0])
            )

            np.random.shuffle(randomIndices_)

            for iTX, iTY in zip(
                    trainingX[randomIndices_],
                    trainingY[randomIndices_]
            ):

                # Regla Widrow-Hoff (ADALINE)

                individualUpdate_ = self.eta * (
                        iTY - self._net_input(iTX, W_)
                )

                W_[1:] += individualUpdate_ * iTX
                W_[0] += individualUpdate_

            error_ = self.calculateError(
                trainingX,
                trainingY,
                W_
            )

            self.errorForEachEpoch_.append(error_)

            if error_ < bestError:

                bestError = error_
                self.W_ = W_

    def predict(self, X):

        return self._net_input(
            X,
            self.W_
        )

    def _net_input(self, X, W):

        return np.dot(
            X,
            W[1:]
        ) + W[0]

    def calculateError(self, X, Y, W_):

        error_ = 0

        for iTX, iTY in zip(X, Y):

            error_ += (
                    (
                            iTY -
                            self._net_input(iTX, W_)
                    ) ** 2
            )

        return error_ / X.shape[0]