import numpy as np
from scipy import stats

from .calculation import VariableCalculation as vc


class Score(object):
    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series
        """
        self.X, self.Y = X.astype(np.float64), Y.astype(np.float64)
        self.Y[self.Y == 0] = 0.00000001

    def volatility_score(self, duration):
        """
        Compute volatility score
        Returns
        -------
        D: array, shape = [m, (n, t)]
            Distance matrix.
        """
        rp = vc().get_return_time_series(self.X)
        vp = vc().get_variance_price(rp)
        log_mu, log_std = vc().get_log_vp(vp, duration=duration)
        log_std[log_std == 0] = 0.00000001
        m, n = log_mu.shape

        score_volatility = np.zeros((m, n))

        for idx in range(n):
            ln_sigma = np.log(np.sqrt(vp[:, -n+idx]))
            val_max = np.true_divide((ln_sigma-log_mu[:,idx].reshape(-1)), log_std[:,idx].reshape(-1)).reshape(-1,1)
            array_max = vc().get_minmax(val_max, -4, "max")
            score_volatility[:, idx] = vc().get_minmax(array_max, 4, "min")

        return score_volatility

    def ewm_volume(self, dur_l=60, dur_s=20):
        ewm_vlm_l = vc().get_ewm_time_series(X=self.Y, alpha=1-1/dur_l)
        ewm_vlm_s = vc().get_ewm_time_series(X=self.Y, alpha=1-1/dur_s)
        return ewm_vlm_l, ewm_vlm_s

    def volume_score(self, ewm_vlm_l, ewm_vlm_s):
        """
        Compute volume score
        """
        ln_vlm_s = np.log(np.true_divide(self.Y, ewm_vlm_s))
        ln_vlm_l = np.log(np.true_divide(self.Y, ewm_vlm_l))

        score_volume = np.zeros(ewm_vlm_s.shape)

        val_max = np.true_divide((ln_vlm_s + ln_vlm_l), 2)
        for idx in range(ewm_vlm_s.shape[1]):
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_volume[:,idx] = vc().get_minmax(array_max, 4, "min")

        return score_volume

    def volatility_volume_score(self, score_volatility, score_volume):
        """
        Compute volatility volume score
        Returns
        -------
        D: array, shape = [m, (n, t)]
            Distance matrix.
        """
        score_volume = score_volume[:, -score_volatility.shape[1]:]
        score_vv = np.zeros(score_volume.shape)

        for idx in range(score_volatility.shape[1]):
            val_max = score_volatility + score_volume
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_vv[:, idx] = vc().get_minmax(array_max, 4, "min")/8 + 0.5

        return score_vv

    def weight_long_short(self, score_vv):
        l_l, l_s = vc().get_weight_vv_long_short(score_vv)
        return l_l, l_s

    def disparity(self, l_l, l_s, dur_s=7, dur_l=30):
        x_l, x_s = vc().get_disparity(self.X, 1-1/dur_l), vc().get_disparity(self.X, 1-1/dur_s)
        x_l, x_s = x_l[:, -l_l.shape[1]:], x_s[:, -l_s.shape[1]:]

        return x_l, x_s

    def score_momentum(self, x_l, x_s, l_l, l_s, c=15):
        score_momentum = c*(np.multiply(x_l, l_l)+np.multiply(x_s, l_s))/10

        return score_momentum

    def ewm_score_momentum(self, score_momentum, dur_s=2, dur_l=7):
        ewm_w_s, ewm_w_l = vc().get_ewm_time_series(score_momentum, 1-1/dur_s), vc().get_ewm_time_series(score_momentum, 1-1/dur_l)
        ewm_w = (ewm_w_s + ewm_w_l)/2

        return ewm_w

    def beta_compensated(self, ewm_w):
        beta = 2 + np.absolute(ewm_w) - 4/(1+np.exp(np.absolute(-ewm_w)))

        list_ewm_w_neg = list(np.where(ewm_w < 0))
        idx_ewm_w_neg = tuple(zip(*list_ewm_w_neg))

        # 어디선가 음수값도 양수로 바뀜, 0 변환해주면서
        beta_com = beta.copy()

        for idx in idx_ewm_w_neg:
            beta_com[idx] = -beta[idx]

        return beta_com

    def score_compensation(self, score_momentum, score_vv, beta_com):
        s_momentum_com = score_momentum - beta_com
        score_fng = 1/(1+np.exp(-(np.multiply(score_vv, s_momentum_com)))) * 100

        return score_fng


class FearGreed(object):
    def __init__(self, X, Y, score):
        """
        Parameters
        ----------
        X: list(array), shape = [m, (n, t)], dtype=object
            Price time series

        조절가능한 변수?
        변동성 점수: 과거 365일 가중평균, 람다 0.94?
        거래량 점수: 단기, 장기 20일, 60일
        모멘텀 점수: 장단기 가중시 변수 9, 이격도 장기 단기 30일, 7일, 파라미터 C
        공포탐욕 점수: 모멘텀 W 장기, 단기 7일, 2일
        """
        self.X, self.Y = X.astype(np.float64), Y.astype(np.float64)
        self.Y[self.Y == 0] = 0.00000001

        self.score = score

    def compute(self):
        """"""
        score_volatility = self.score.volatility_score(duration=120)
        ewm_vlm_l, ewm_vlm_s = self.score.ewm_volume()
        score_volume = self.score.volume_score(ewm_vlm_l, ewm_vlm_s)
        score_vv = self.score.volatility_volume_score(score_volatility, score_volume)

        l_l, l_s = self.score.weight_long_short(score_vv)
        x_l, x_s = self.score.disparity(l_l, l_s)

        score_momentum = self.score.score_momentum(x_l, x_s, l_l, l_s)
        ewm_w = self.score.ewm_score_momentum(score_momentum)
        beta_com = self.score.beta_compensated(ewm_w)
        score_compensation = self.score.score_compensation(score_momentum, score_vv, beta_com)

        return score_compensation

    def compute_grad(self):
        """"""
        score_volatility = self.score.volatility_score(duration=120)
        ewm_vlm_l, ewm_vlm_s = self.score.ewm_volume()
        score_volume = self.score.volume_score(ewm_vlm_l, ewm_vlm_s)
        score_vv = self.score.volatility_volume_score(score_volatility, score_volume)

        l_l, l_s = self.score.weight_long_short(score_vv)
        x_l, x_s = self.score.disparity(l_l, l_s)

        score_fng = np.full(score_vv.shape, np.nan)

        for idx, val in enumerate(zip(x_l, x_s, l_l, l_s)):
            p_val, c = 0, 100000
            while p_val <= 0.05:
                score_momentum = self.score.score_momentum(val[0], val[1], val[2], val[3], c=c).reshape(-1, 1)
                ewm_w = self.score.ewm_score_momentum(score_momentum)
                beta_com = self.score.beta_compensated(ewm_w)
                score_compensation = self.score.score_compensation(score_momentum, score_vv[idx].reshape(-1, 1), beta_com)
                score_comp_droped = score_compensation[~np.isnan(score_compensation)].tolist()

                p_val = stats.shapiro(score_comp_droped).pvalue
                err = 0.05 - p_val
                c += err*100000
                print(idx, p_val, c)

            score_fng[:, idx] = score_compensation

        return score_fng


# 왜 이격도 첫 시작값 다 같나?
# score_comp 중간에 nan??
