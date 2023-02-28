import numpy as np

from .calculation import VariableCalculation as vc

# Index 부분이랑 Stock 부분 코드 재사용성 고민.
# 겹치지 않는 지표는 merge


class scoreIndex(object):
    def __init__(self, P, V):
        """
        Object of scoreIndex to calculate fear and greed index of financial indexes with data of price and volume

        :param P: dataframe or np.array of price with m variables and t times
        :param V: dataframe or np.array of volume with m variables and t times
        """
        self.P, self.V = P.astype(np.float64), V.astype(np.float64)
        self.V[self.V == 0] = 0.1

    def volatility_score(self, duration):
        """
        Parameters
        ----------
        duration: time-series data of price with m variables and t times

        Returns
        -------
        score_volatility: return of P with m variables and t-1 times
        """
        rp = vc().get_return_time_series(self.P)
        vp = vc().get_variance_price(rp)
        log_mu, log_std = vc().get_log_vp(vp, duration=duration)
        log_std[log_std == 0] = 0.1
        m, n = log_mu.shape

        score_volatility = np.zeros((m, n))

        for idx in range(n):
            ln_sigma = np.log(np.sqrt(vp[:, -n+idx]))
            val_max = np.true_divide((ln_sigma-log_mu[:,idx].reshape(-1)), log_std[:,idx].reshape(-1)).reshape(-1,1)
            array_max = vc().get_minmax(val_max, -4, "max")
            score_volatility[:, idx] = vc().get_minmax(array_max, 4, "min")

        return score_volatility



    def volume_score(self, ewm_vlm_l, ewm_vlm_s):
        """
        Parameters
        ----------
        ewm_vlm_l: time-series data of price with m variables and t times
        ewm_vlm_s: time-series data of price with m variables and t times


        Returns
        -------
        score_volume: return of P with m variables and t-1 times
        """
        ln_vlm_s = np.log(np.true_divide(self.V[:, 1:], ewm_vlm_s[:, :-1]))
        ln_vlm_l = np.log(np.true_divide(self.V[:, 1:], ewm_vlm_l[:, :-1]))

        score_volume = np.zeros((ewm_vlm_s.shape[0], ewm_vlm_s.shape[1]-1))

        val_max = np.true_divide((ln_vlm_s + ln_vlm_l), 2)
        for idx in range(ewm_vlm_s.shape[1]-1):
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_volume[:, idx] = vc().get_minmax(array_max, 4, "min")

        return score_volume

    def volatility_volume_score(self, score_volatility, score_volume):
        """
        Parameters
        ----------
        score_volatility: time-series data of price with m variables and t times
        score_volume:

        Returns
        -------
        score_vv: return of P with m variables and t-1 times
        """
        score_volume = score_volume[:, -score_volatility.shape[1]:]
        score_vv = np.zeros(score_volume.shape)

        for idx in range(score_volatility.shape[1]):
            val_max = score_volatility + score_volume
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_vv[:, idx] = vc().get_minmax(array_max, 4, "min")/8 + 0.5

        return score_vv

    def weight_long_short(self, score_vv):
        """
        Parameters
        ----------
        score_vv: time-series data of price with m variables and t times

        Returns
        -------
        l_l: return of P with m variables and t-1 times
        l_s: return of P with m variables and t-1 times
        """
        l_l, l_s = vc().get_weight_vv_long_short(score_vv)

        return l_l, l_s

    def disparity(self, l_l, l_s, dur_l=30, dur_s=7):
        """
        Parameters
        ----------
        l_l: time-series data of price with m variables and t times
        l_s: time-series data of price with m variables and t times

        dur_l: time-series data of price with m variables and t times
        dur_s: time-series data of price with m variables and t times


        Returns
        -------
        x_l: return of P with m variables and t-1 times
        x_s: return of P with m variables and t-1 times

        """
        x_l, x_s = vc().get_disparity(self.P, 1-1/dur_l), vc().get_disparity(self.P, 1-1/dur_s)
        x_l, x_s = x_l[:, -l_l.shape[1]:], x_s[:, -l_s.shape[1]:]

        return x_l, x_s

    def score_momentum(self, x_l, x_s, l_l, l_s, c=15):
        """
        Parameters
        ----------
        x_l: return of P with m variables and t-1 times
        x_s: return of P with m variables and t-1 times

        l_l: return of P with m variables and t-1 times
        l_s: return of P with m variables and t-1 times

        c: return of P with m variables and t-1 times


        Returns
        -------
        score_momentum: return of P with m variables and t-1 times
        """
        score_momentum = c*(np.multiply(x_l, l_l)+np.multiply(x_s, l_s))/10

        return score_momentum

    def score_c(self, duration):
        """
        Parameters
        ----------
        duration: time-series data of price with m variables and t times

        Returns
        -------
        score_c_comp: return of P with m variables and t-1 times
        """
        score_c = np.zeros((self.P.shape[0], self.P.shape[1]-duration))

        for i in range(self.P.shape[1]-duration):
            std_price = np.nanstd(self.P[:, i:i+duration], axis=1)/np.nanmean(self.P[:, i:i+duration])
            std_volume = np.nanstd(self.V[:, i:i+duration], axis=1)/np.nanmean(self.V[:, i:i+duration])
            score_c[:, i] = ((std_price + std_volume)/2).T
        score_c[score_c == 0] = 0.01
        score_c_comp = 5+15/(np.exp(1/score_c-2)+1)

        return score_c_comp

    def beta_compensated(self, ewm_w):
        """
        Parameters
        ----------
        ewm_w: time-series data of price with m variables and t times

        Returns
        -------
        beta_com: return of P with m variables and t-1 times
        """
        beta = 2 + np.absolute(ewm_w) - 4/(1+np.exp(np.absolute(-ewm_w)))

        list_ewm_w_neg = list(np.where(ewm_w < 0))
        idx_ewm_w_neg = tuple(zip(*list_ewm_w_neg))

        beta_com = beta.copy()

        for idx in idx_ewm_w_neg:
            beta_com[idx] = -beta[idx]

        return beta_com

    def score_compensation(self, score_momentum, score_vv, beta_com):
        """
        Calculate score Volatility & Volume with compensated.

        Parameters
        ----------
        score_momentum: time-series data of price with m variables and t times
        score_vv:
        beta_com:

        Returns
        -------
        score_fng: return of P with m variables and t-1 times
        """
        s_momentum_com = score_momentum - beta_com
        score_fng = 1/(1+np.exp(-(np.multiply(score_vv, s_momentum_com)))) * 100

        return score_fng


class scoreStock(object):
    def __init__(self, P, H, L, V):
        """
        Object of scoreStock to calculate fear and greed index of individual stocks with data of price and volume

        :param P: dataframe or np.array of end price with m variables and t times
        :param H: dataframe or np.array of highest price with m variables and t times
        :param L: dataframe or np.array of lowest price with m variables and t times
        :param V: dataframe or np.array of volume with m variables and t times
        """
        self.P, self.H, self.L, self.V = P.astype(np.float64), H.astype(np.float64), L.astype(np.float64), V.astype(np.float64)
        self.V[self.V == 0] = 0.1

    def volatility_score(self, duration):
        """
        Parameters
        ----------
        duration: time-series data of price with m variables and t times

        Returns
        -------
        score_volatility: return of P with m variables and t-1 times
        """
        rp = vc().get_return_time_series(self.P)
        vp = vc().get_variance_price(rp)
        log_mu, log_std = vc().get_log_vp(vp, duration=duration)
        log_std[log_std == 0] = 0.1
        m, n = log_mu.shape

        score_volatility = np.zeros((m, n))

        for idx in range(n):
            ln_sigma = np.log(np.sqrt(vp[:, -n+idx]))
            val_max = np.true_divide((ln_sigma-log_mu[:,idx].reshape(-1)), log_std[:,idx].reshape(-1)).reshape(-1,1)
            array_max = vc().get_minmax(val_max, -4, "max")
            score_volatility[:, idx] = vc().get_minmax(array_max, 4, "min")

        return score_volatility

    def volume_score(self, ewm_vlm_l, ewm_vlm_s):
        """
        Parameters
        ----------
        ewm_vlm_l: time-series data of price with m variables and t times
        ewm_vlm_s: time-series data of price with m variables and t times

        Returns
        -------
        score_volume: return of P with m variables and t-1 times
        """
        ln_vlm_s = np.log(np.true_divide(self.V, ewm_vlm_s)) + 1/(500*self.V)
        ln_vlm_l = np.log(np.true_divide(self.V, ewm_vlm_l)) + 1/(500*self.V)

        score_volume = np.zeros(ewm_vlm_s.shape)

        val_max = np.true_divide((ln_vlm_s + ln_vlm_l), 2)
        for idx in range(ewm_vlm_s.shape[1]):
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_volume[:,idx] = vc().get_minmax(array_max, 4, "min")

        return score_volume

    def volatility_volume_score(self, score_volatility, score_volume):
        """
        Parameters
        ----------
        score_volatility: time-series data of price with m variables and t times
        score_volume: time-series data of price with m variables and t times


        Returns
        -------
        score_vv: return of P with m variables and t-1 times
        """
        score_volume = score_volume[:, -score_volatility.shape[1]:]
        score_vv = np.zeros(score_volume.shape)

        for idx in range(score_volatility.shape[1]):
            val_max = score_volatility + score_volume
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_vv[:, idx] = vc().get_minmax(array_max, 4, "min")/8 + 0.5

        return score_vv

    def weight_long_short(self, score_vv):
        """
        Parameters
        ----------
        score_vv: time-series data of price with m variables and t times

        Returns
        -------
        l_l: return of P with m variables and t-1 times
        l_s:
        """
        l_l, l_s = vc().get_weight_vv_long_short(score_vv)
        return l_l, l_s

    def disparity(self, l_l, l_s, dur_s=7, dur_l=30):
        """
        Parameters
        ----------
        l_l: time-series data of price with m variables and t times
        l_s: time-series data of
        dur_s:
        dur_l:

        Returns
        -------
        x_l: return of P with m variables and t-1 times
        x_s:
        """
        x_l, x_s = vc().get_disparity(self.P, 1-1/dur_l), vc().get_disparity(self.P, 1-1/dur_s)
        x_l, x_s = x_l[:, -l_l.shape[1]:], x_s[:, -l_s.shape[1]:]

        return x_l, x_s

    def score_momentum(self, x_l, x_s, l_l, l_s, c=15):
        """
        Calculate Momentum score.

        Parameters
        ----------
        x_l: time-series data of price with m variables and t times
        x_s:
        l_l:
        l_s:
        c: parameter of

        Returns
        -------
        score_momentum: return of P with m variables and t-1 times
        """
        score_momentum = c*(np.multiply(x_l, l_l)+np.multiply(x_s, l_s))/10

        return score_momentum



    def beta_compensated(self, ewm_w):
        """
        Parameters
        ----------
        ewm_w: time-series data of price with m variables and t times

        Returns
        -------
        beta_com: return of P with m variables and t-1 times
        """
        beta = 2 + np.absolute(ewm_w) - 4/(1+np.exp(np.absolute(-ewm_w)))

        list_ewm_w_neg = list(np.where(ewm_w < 0))
        idx_ewm_w_neg = tuple(zip(*list_ewm_w_neg))

        beta_com = beta.copy()

        for idx in idx_ewm_w_neg:
            beta_com[idx] = -beta[idx]

        return beta_com

    def score_c(self, duration):
        """
        Parameters
        ----------
        duration: time-series data of price with m variables and t times

        Returns
        -------
        score_c_comp: return of P with m variables and t-1 times
        """
        score_c = np.zeros((self.P.shape[0], self.P.shape[1]-duration))

        for i in range(self.P.shape[1]-duration):
            std_price = np.nanstd(self.P[:, i:i+duration], axis=1)/np.nanmean(self.P[:, i:i+duration])
            std_volume = np.nanstd(self.V[:, i:i+duration], axis=1)/np.nanmean(self.V[:, i:i+duration])
            score_c[:, i] = ((std_price + std_volume)/2).T
        score_c[score_c <= 0] = 0.01
        score_c_comp = 5+15/(np.exp(1/score_c-2)+1)

        return score_c_comp

    def score_compensation(self, score_momentum, score_vv, beta_com):
        """
        Parameters
        ----------
        score_momentum: time-series data of price with m variables and t times
        score_vv:
        beta_com:

        Returns
        -------
        score_fng: return of P with m variables and t-1 times
        """
        s_momentum_com = score_momentum - beta_com
        score_fng = 1/(1+np.exp(-(np.multiply(score_vv, s_momentum_com)))) * 100

        return score_fng



class FearGreed(object):
    def __init__(self, score: object):
        self.score = score

    def compute_index(self, duration=120):
        """
        Parameters
        ----------
        duration: time-series data of price with m variables and t times

        Returns
        -------
        score_compensation: return of P with m variables and t-1 times
        """
        score_volatility = self.score.volatility_score(duration=duration)
        ewm_vlm_l, ewm_vlm_s = vc().get_ewm_volume()
        score_volume = self.score.volume_score(ewm_vlm_l, ewm_vlm_s)
        score_vv = self.score.volatility_volume_score(score_volatility, score_volume)

        l_l, l_s = self.score.weight_long_short(score_vv)
        x_l, x_s = self.score.disparity(l_l, l_s)

        score_c = self.score.score_c(duration=duration)

        score_momentum = self.score.score_momentum(x_l, x_s, l_l, l_s, c=score_c[:, 1:])
        ewm_w = vc().get_ewm_score_momentum(score_momentum)
        beta_com = self.score.beta_compensated(ewm_w)
        score_compensation = self.score.score_compensation(score_momentum, score_vv, beta_com)

        return score_compensation

    def compute_stock(self, duration=120):
        """
        Parameters
        ----------
        duration: time-series data of price with m variables and t times

        Returns
        -------
        score_compensation: return of P with m variables and t-1 times
        """
        score_volatility = self.score.volatility_score(duration=duration)
        ewm_vlm_l, ewm_vlm_s = vc().get_ewm_volume()
        score_volume = self.score.volume_score(ewm_vlm_l, ewm_vlm_s)
        score_vv = self.score.volatility_volume_score(score_volatility, score_volume)

        l_l, l_s = self.score.weight_long_short(score_vv)
        x_l, x_s = self.score.disparity(l_l, l_s)

        score_c = self.score.score_c(duration=duration)

        score_momentum = self.score.score_momentum(x_l, x_s, l_l, l_s, c=score_c[:, 1:])
        ewm_w = vc().get_ewm_score_momentum(score_momentum)
        beta_com = self.score.beta_compensated(ewm_w)
        score_compensation = self.score.score_compensation(score_momentum, score_vv, beta_com)

        return score_compensation

