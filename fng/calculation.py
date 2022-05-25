import numpy as np

class VariableCalculation:
    def get_return_time_series(self, X):
        """
        Parameters
        ----------
        X: array, shape = [m, t]
            Price time series

        Returns
        -------
        G: array, shape = [m, t-1]
        """
        rp = np.true_divide(np.diff(X), X[:, 1:]) * 100
        rp[rp == 0] = 0.00000001

        return rp

    def get_ewm_time_series(self, X, alpha):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        """
        ewm = np.zeros((X.shape[0], X.shape[1]+1))
        X = np.nan_to_num(X, copy=True, nan=0)

        for i in range(1, X.shape[1]+1):
            ewm[:,i] = alpha * X[:, i-1] + (1.-alpha) * ewm[:,i-1]
        ewm[ewm == 0] = np.nan

        return ewm[:, 1:]

    def get_std_time_series(self, X):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        """

    def get_variance_price(self, rp):
        lam = 0.94
        rp = np.nan_to_num(rp, copy=True, nan=0)
        vp = np.zeros((rp.shape[0], rp.shape[1]+1))

        for i, x in enumerate(vp.T[:-1, :]):
            vp[:, i+1] = lam * vp[:, i] + (1 - lam) * (rp[:, i]**2)
        vp[vp==0] = np.nan

        return vp[:, 1:]

    def get_index_nan(self, arr_price, arr_volume):
        nan_price, nan_volume = np.isnan(arr_price), np.isnan(arr_volume)

        return nan_price, nan_volume

    def get_log_vp(self, vp, duration=365):
        log_mu, log_std = np.zeros((vp.shape[0], vp.shape[1]-duration)), np.zeros((vp.shape[0], vp.shape[1]-duration))
        stdp = np.sqrt(vp)

        for idx in range(stdp.shape[1]-duration):
            arr_365 = np.log(stdp[:, idx:duration + idx])
            log_mu[:, idx], log_std[:, idx] = np.nanmean(arr_365, axis=1), np.nanstd(arr_365, axis=1)

        return log_mu, log_std

    def get_disparity(self, X, alpha):
        ewm_X = self.get_ewm_time_series(X, alpha)
        x_X = np.true_divide(X-ewm_X, ewm_X) * 100

        return x_X

    def get_weight_vv_long_short(self, score_vv):
        alpha = 9*score_vv + 1
        l_l, l_s = 10 - alpha, alpha

        return l_l, l_s

    def get_minmax(self, X, thd, method='min'):
        X = X.reshape((-1, 1))
        if method=="min":
            min_array = np.full(X.shape, thd)
            array = np.min(np.concatenate((min_array, X), axis=1), axis=1)
        elif method=="max":
            max_array = np.full(X.shape, thd)
            array = np.max(np.concatenate((max_array, X), axis=1), axis=1)

        return array
