import math
import pandas as pd
import numpy as np

from calculation import VariableCalculation as vc


class VolatilityVolume(object):
    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        """
        nan_x, nan_y = vc().index_nan(X, Y)
        self.X, self.Y = X.astype(np.float64), Y.astype(np.float64)
        np.nan_to_num(self.X, copy=False), np.nan_to_num(self.Y, copy=False)


    def compute_volatility(self):
        """
        Compute volatility score
        Returns
        -------
        D: array, shape = [m, (n, t)]
            Distance matrix.
        """
        rp = vc().return_time_series(self.X)
        vp = vc().variance_price(rp)
        log_mu, log_std = vc().log_vp(vp)
        m, n = log_mu.shape

        score_volatility = np.zeros((m, n))

        for idx in range(n):
            ln_sigma = np.ma.masked_invalid(np.log(vp[:, -n+idx]))
            ln_sigma = np.nan_to_num(ln_sigma, copy=True, posinf=0, neginf=0, nan=0)
            val_max = np.true_divide((ln_sigma-log_mu[:,idx].reshape(-1)), log_std[:,idx].reshape(-1)).reshape(-1,1)
            array_max = np.max(np.concatenate((np.full(val_max.shape, -4), val_max), axis=1), axis=1).reshape(-1,1)
            val_min = np.concatenate((np.full(array_max.shape, 4), array_max), axis=1)
            score_volatility[:, idx] = np.min(val_min, axis=1).reshape(-1)

        return score_volatility

    def compute_volume(self):
        """
        Compute volume score
        """
        ewm_vlm_s, ewm_vlm_l = vc().ewm_time_series(self.Y, 1-1/20), vc().ewm_time_series(self.Y, 1-1/60)
        ln_vlm_s = np.log(np.true_divide(self.Y, ewm_vlm_s))
        ln_vlm_l = np.log(np.true_divide(self.Y, ewm_vlm_l))
        ln_vlm_s = np.nan_to_num(ln_vlm_s, copy=True, posinf=0, neginf=0, nan=0)
        ln_vlm_l = np.nan_to_num(ln_vlm_l, copy=True, posinf=0, neginf=0, nan=0)

        score_volume = np.zeros(ewm_vlm_s.shape)

        val_max = np.true_divide((ln_vlm_s + ln_vlm_l), 2)
        for idx in range(ewm_vlm_s.shape[1]):
            array_max = np.max(np.concatenate((np.full((val_max.shape[0], 1), -4), val_max[:,idx].reshape(-1,1)), axis=1), axis=1).reshape(-1,1)
            val_min = np.concatenate((np.full(array_max.shape, 4), array_max), axis=1)
            score_volume[:,idx] = np.min(val_min, axis=1)

        return score_volume

    def compute_volatility_volume(self):
        """
        Compute volatility volume score
        Returns
        -------
        D: array, shape = [m, (n, t)]
            Distance matrix.
        """
        s_volatility, s_volume = self.compute_volatility(), self.compute_volume()
        s_volume = s_volume[:, -s_volatility.shape[1]:]
        score_vv = np.zeros(s_volume.shape)

        for idx in range(s_volatility.shape[1]):
            val_max = s_volatility + s_volume
            array_max = np.max(
                np.concatenate((np.full((val_max.shape[0], 1), -4), val_max[:, idx].reshape(-1, 1)), axis=1),
                axis=1).reshape(-1, 1)
            val_min = np.concatenate((np.full(array_max.shape, 4), array_max), axis=1)
            score_vv[:, idx] = np.min(val_min, axis=1)/8 + 0.5

        return score_vv

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
        nan_x, nan_y = vc().index_nan(X, Y)
        self.X, self.Y = X.astype(np.float64), Y.astype(np.float64)
        np.nan_to_num(self.X, copy=False), np.nan_to_num(self.Y, copy=False)

    def compute_momentum(self):
        score_vv = VolatilityVolume(self.X, self.Y).compute_volatility_volume()
        alpha = 9*score_vv + 1
        l_s, l_l = alpha, 10-alpha

        ewm_p_l, ewm_p_s = vc().ewm_time_series(self.X, 1-1/30), vc().ewm_time_series(self.X, 1-1/7)
        x_l, x_s = np.true_divide(self.X-ewm_p_s, ewm_p_s), np.true_divide(self.X-ewm_p_s, ewm_p_s)
        x_l = np.nan_to_num(x_l, copy=True, posinf=0, neginf=0, nan=0)
        x_s = np.nan_to_num(x_s, copy=True, posinf=0, neginf=0, nan=0)

        x_l, x_s = x_l[:, -l_l.shape[1]:], x_s[:, -l_s.shape[1]:]

        c = 400
        s_momentum = c*(np.multiply(l_s, x_s)+np.multiply(l_l, x_l))/10

        return s_momentum, score_vv

    def compute_compensation(self):
        s_momentum, score_vv = self.compute_momentum()
        ewm_w_s, ewm_w_l = vc().ewm_time_series(s_momentum, 1-1/2), vc().ewm_time_series(s_momentum, 1-1/7)
        ewm_w = (ewm_w_s + ewm_w_l)/2

        beta = 2 + np.absolute(ewm_w) - 4/(1+np.exp(np.absolute(-ewm_w)))

        list_ewm_w_neg = list(np.where(ewm_w < 0))
        idx_ewm_w_neg = tuple(zip(*list_ewm_w_neg))

        beta_com = beta.copy()

        for idx in idx_ewm_w_neg:
            beta_com[idx] = -beta[idx]

        s_momentum_com = s_momentum - beta_com

        s_fng = 1/(1+np.exp(-(np.multiply(score_vv, s_momentum_com) )))

        return s_fng




