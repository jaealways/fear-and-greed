import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VolatilityVolume(object):
    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        Y: list(array), shape = [m, (n, t)], dtype=object
            Volume time series.
        """
        self.X = X.astype(np.float64)
        self.Y = Y.astype(np.float64)

    def compute(self):
        """
        Compute distance matrix.
        Returns
        -------
        D: array, shape = [m, (n, t)]
            Distance matrix.
        """
        return euclidean_distances(self.X, self.Y, squared=True)


class Momentum(object):
    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X: array, shape = [m, (n, t)]
            Price time series
        Y: array, shape = [m, (n, t)]
            Volume time series.
        """
        self.X = X.astype(np.float64)
        self.Y = Y.astype(np.float64)

