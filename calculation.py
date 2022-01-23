import numpy as np

class VariableCalculation:
    # def __init__(self, X):
    #     self.X = X

    def return_time_series(self, X):
        """
        Parameters
        ----------
        X: array, shape = [m, t]
            Price time series

        Returns
        -------
        G: array, shape = [m, t-1]
        """
        rp = np.diff(X) / X[:, 1:]
        return rp

    def ewm_time_series(self, X, alpha):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        """
        alpha_rev = 1 - alpha
        n = X.shape[0]

        pows = alpha_rev ** (np.arange(n + 1))

        scale_arr = 1 / pows[:-1]
        offset = X[0] * pows[1:]
        pw0 = alpha * alpha_rev ** (n - 1)

        mult = X * pw0 * scale_arr
        cumsums = np.cumsum(mult, axis=1)
        out = offset + cumsums * scale_arr[::-1]
        return out

    def variance_time_series(self, X):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        """

    def variance_price(self, X):
        X = np.concatenate((X, np.broadcast_to(np.array([1]), X.shape[0])), axis=0)
        return X


