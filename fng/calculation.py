import numpy as np

class VariableCalculation:
    def get_return_time_series(self, P):
        """
        Get price return of P.

        Parameters
        ----------
        P: time-series data of price with m variables and t times

        Returns
        -------
        rp: price return of P with m variables and t-1 times
        """
        rp = np.true_divide(np.diff(P), P[:, 1:]) * 100
        rp[rp == 0] = 0.00000001

        return rp

    def get_ewm_time_series(self, P, alpha):
        """
        Get price return of exponential moving average with duration alpha.

        Parameters
        ----------
        P: time-series data of price with m variables and t times
        alpha: parameter determines the duration of past dates in exponential weighted moving average

        Returns
        -------
        ewm: exponential weighted moving average of P with m variables and t-1 times
        """
        ewm = np.zeros((P.shape[0], P.shape[1]+1))
        P = np.nan_to_num(P, copy=True, nan=0)

        for i in range(1, P.shape[1]+1):
            ewm[:,i] = alpha * P[:, i-1] + (1.-alpha) * ewm[:,i-1]
        ewm[ewm == 0] = np.nan

        return ewm[:, 1:]

    def get_variance_price(self, rp):
        """
        Get variance of price return matrix rp.

        Parameters
        ----------
        rp: return of price P data of price with m variables and t-1 times

        Returns
        ---------
        vp: variance of price return rp with m variables and t-1 times
        """
        lam = 0.94
        rp = np.nan_to_num(rp, copy=True, nan=0)
        vp = np.zeros((rp.shape[0], rp.shape[1]+1))

        for i, x in enumerate(vp.T[:-1, :]):
            vp[:, i+1] = lam * vp[:, i] + (1 - lam) * (rp[:, i]**2)
        vp[vp==0] = np.nan

        return vp[:, 1:]

    def get_ewm_volume(self, dur_l=60, dur_s=20):
        """
        Parameters
        ----------
        dur_l: time-series data of price with m variables and t times
        dur_s:

        Returns
        -------
        ewm_vlm_l: return of P with m variables and t-1 times
        ewm_vlm_s: return of P with m variables and t-1 times
        """
        ewm_vlm_l = self.get_ewm_time_series(X=self.V, alpha=1-1/dur_l)
        ewm_vlm_s = self.get_ewm_time_series(X=self.V, alpha=1-1/dur_s)

        return ewm_vlm_l, ewm_vlm_s


    def get_log_vp(self, vp, duration=365):
        """
        Get log of vp.

        Parameters
        ----------
        vp: time-series data of variance of price return with m variables and t-1 times
        duration:

        Returns
        -------
        log_mu: return of P with m variables and t-1 times
        log_std: return of P with m variables and t-1 times
        """
        log_mu, log_std = np.zeros((vp.shape[0], vp.shape[1]-duration)), np.zeros((vp.shape[0], vp.shape[1]-duration))
        stdp = np.sqrt(vp)

        for idx in range(stdp.shape[1]-duration):
            arr_365 = np.log(stdp[:, idx:duration + idx])
            log_mu[:, idx], log_std[:, idx] = np.nanmean(arr_365, axis=1), np.nanstd(arr_365, axis=1)

        return log_mu, log_std

    def get_disparity(self, P, alpha):
        """
        Get disparity index of P with duration of alpha.

        Parameters
        ----------
        P: time-series data of price with m variables and t times
        alpha:

        Returns
        -------
        x_P: return of P with m variables and t-1 times
        """
        ewm_P = self.get_ewm_time_series(P, alpha)
        x_P = np.true_divide(P-ewm_P, ewm_P) * 100

        return x_P

    def get_ewm_score_momentum(self, score_momentum, dur_s=2, dur_l=7):
        """
        Parameters
        ----------
        score_momentum: time-series data of price with m variables and t times
        dur_s: return of P with m variables and t-1 times
        dur_l: return of P with m variables and t-1 times

        Returns
        -------
        ewm_w: return of P with m variables and t-1 times
        """
        ewm_w_s, ewm_w_l = self.get_ewm_time_series(score_momentum, 1-1/dur_s), self.get_ewm_time_series(score_momentum, 1-1/dur_l)
        ewm_w = (ewm_w_s + ewm_w_l)/2

        return ewm_w

    def get_weight_vv_long_short(self, score_vv):
        """
        Get

        Parameters
        ----------
        score_vv: time-series data of price with m variables and t times

        Returns
        -------
        l_l: return of P with m variables and t-1 times
        l_s: return of P with m variables and t-1 times
        """
        alpha = 9*score_vv + 1
        l_l, l_s = 10 - alpha, alpha

        return l_l, l_s

    def get_minmax(self, P, thd, method='min'):
        """
        Parameters
        ----------
        P: time-series data of price with m variables and t times
        thd: threshold of number
        method: time-series data of price with m variables and t times


        Returns
        -------
        array: return of P with m variables and t-1 times
        """
        P = P.reshape((-1, 1))
        if method=="min":
            min_array = np.full(P.shape, thd)
            array = np.min(np.concatenate((min_array, P), axis=1), axis=1)
        elif method=="max":
            max_array = np.full(P.shape, thd)
            array = np.max(np.concatenate((max_array, P), axis=1), axis=1)

        return array
