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
        ewm = np.zeros(X.shape)
        ewm[:,0] = X[:,0]
        for i in range(1, X.shape[1]):
            ewm[:,i] = alpha * X[:,i] + (1.-alpha) * ewm[:,i-1]
        return ewm

    def std_time_series(self, X):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        """

    def variance_price(self, rp):
        """start with date 0 vp는 rp 보다 1 시계열 큼"""
        lam = 0.94
        vp = np.zeros((rp.shape[0], rp.shape[1]+1))

        for i, x in enumerate(vp.T[:-1, :]):
            vp[:, i+1] = lam * vp[:, i] + (1 - lam) * (rp[:, i]**2)

        return vp

    def index_nan(self, arr_price, arr_volume):
        nan_price, nan_volume = np.isnan(arr_price), np.isnan(arr_volume)

        return nan_price, nan_volume

    def log_vp(self, vp, duration=365):
        log_mu, log_std = np.zeros((vp.shape[0], vp.shape[1]-duration)), np.zeros((vp.shape[0], vp.shape[1]-duration))
        stdp = np.sqrt(vp)

        for idx in range(stdp.shape[1]-duration):
            arr_365 = np.ma.masked_invalid(np.log(stdp[:, idx:duration+idx]))
            log_mu[:, idx], log_std[:, idx] = np.mean(arr_365, axis=1), np.std(arr_365, axis=1)

        return log_mu, log_std

    # 이격도, min max 함수, 정규식으로 c 변환

