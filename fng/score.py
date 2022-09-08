import numpy as np
from scipy import stats

from .calculation import VariableCalculation as vc


class scoreIndex(object):
    def __init__(self, X, Y):
        #log0 값 몇으로 치환??

        self.X, self.Y = X.astype(np.float64), Y.astype(np.float64)
        self.Y[self.Y == 0] = 0.1

    def volatility_score(self, duration):
        rp = vc().get_return_time_series(self.X)
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

    def ewm_volume(self, dur_l=60, dur_s=20):
        ewm_vlm_l = vc().get_ewm_time_series(X=self.Y, alpha=1-1/dur_l)
        ewm_vlm_s = vc().get_ewm_time_series(X=self.Y, alpha=1-1/dur_s)

        return ewm_vlm_l, ewm_vlm_s

    def volume_score(self, ewm_vlm_l, ewm_vlm_s):
        ln_vlm_s = np.log(np.true_divide(self.Y[:, 1:], ewm_vlm_s[:, :-1]))
        ln_vlm_l = np.log(np.true_divide(self.Y[:, 1:], ewm_vlm_l[:, :-1]))

        score_volume = np.zeros((ewm_vlm_s.shape[0], ewm_vlm_s.shape[1]-1))

        val_max = np.true_divide((ln_vlm_s + ln_vlm_l), 2)
        for idx in range(ewm_vlm_s.shape[1]-1):
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_volume[:, idx] = vc().get_minmax(array_max, 4, "min")

        return score_volume

    def volatility_volume_score(self, score_volatility, score_volume):
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

    def score_c(self, duration):
        score_c = np.zeros((self.X.shape[0], self.X.shape[1]-duration))
        std_price_all = np.zeros((self.X.shape[0], self.X.shape[1]-duration))
        std_volume_all = np.zeros((self.X.shape[0], self.X.shape[1] - duration))

        # 핵심은 시그마와 가격변수의 분포에 따라 C 파라미터의 값 다르게 도출하기
        # 시그마가 시계열의 양극단 값으로 몰릴수록(즉 유동성이 너무 적거나 가격 변화량 자체가 너무 큰 경우
        # C 파라미터가 변동성을 더 잘 경고해줘야 함. C를 20내외로 설정)
        # 그 외에 변동성이 거의 없는 경우 5 내외의 파라미터를 설정
        # 시그마의 움직임에 따라(단 평균이 높으면 시그마가 높으니 정규화과정이 필요) 측정가능한 메트릭 개발하기
        # 자산의 움직임을 가우시안 분포로 가정할 수 있는가
        # riskmetric은 시그마 기반으로 측정해야함

        for i in range(self.X.shape[1]-duration):
            std_price = np.nanstd(self.X[:, i:i+duration], axis=1)/np.nanmean(self.X[:, i:i+duration])
            std_volume = np.nanstd(self.Y[:, i:i+duration], axis=1)/np.nanmean(self.Y[:, i:i+duration])
            std_price_all[:, i] = np.nanstd(self.X[:, i:i+duration], axis=1)
            std_volume_all[:, i] = np.nanstd(self.Y[:, i:i+duration], axis=1)
            score_c[:, i] = (std_price * std_volume).T
        score_c[score_c == 0] = 0.01
        score_c_comp = 10 * np.log(score_c+2)

        return score_c_comp

    def beta_compensated(self, ewm_w):
        beta = 2 + np.absolute(ewm_w) - 4/(1+np.exp(np.absolute(-ewm_w)))

        list_ewm_w_neg = list(np.where(ewm_w < 0))
        idx_ewm_w_neg = tuple(zip(*list_ewm_w_neg))

        beta_com = beta.copy()

        for idx in idx_ewm_w_neg:
            beta_com[idx] = -beta[idx]

        return beta_com

    def score_compensation(self, score_momentum, score_vv, beta_com):
        s_momentum_com = score_momentum - beta_com
        score_fng = 1/(1+np.exp(-(np.multiply(score_vv, s_momentum_com)))) * 100

        return score_fng


class scoreStock(object):
    def __init__(self, A, B, C, Y):
        # scoreIndex와의 차이
        # 개별종목은 유동성 부족으로 시가, 종가까지 고려
        # 거래량, 모멘텀은 지수와 동일, 변동성 점수 계산시 시가와 종가의 위치 고려
        self.A, self.B, self.C, self.Y = A.astype(np.float64), B.astype(np.float64), C.astype(np.float64), Y.astype(np.float64)
        self.Y[self.Y == 0] = 0.1

    def volatility_score(self, duration):
        rp = vc().get_return_time_series(self.A)
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

    def ewm_volume(self, dur_l=60, dur_s=20):
        ewm_vlm_l = vc().get_ewm_time_series(X=self.Y, alpha=1-1/dur_l)
        ewm_vlm_s = vc().get_ewm_time_series(X=self.Y, alpha=1-1/dur_s)
        return ewm_vlm_l, ewm_vlm_s

    def volume_score(self, ewm_vlm_l, ewm_vlm_s):
        # 거래량 점수 뒤에 왜 penalty 구간???
        ln_vlm_s = np.log(np.true_divide(self.Y, ewm_vlm_s)) + 1/(500*self.Y)
        ln_vlm_l = np.log(np.true_divide(self.Y, ewm_vlm_l)) + 1/(500*self.Y)

        score_volume = np.zeros(ewm_vlm_s.shape)

        val_max = np.true_divide((ln_vlm_s + ln_vlm_l), 2)
        for idx in range(ewm_vlm_s.shape[1]):
            array_max = vc().get_minmax(val_max[:, idx], -4, "max")
            score_volume[:,idx] = vc().get_minmax(array_max, 4, "min")

        return score_volume

    def volatility_volume_score(self, score_volatility, score_volume):
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
        x_l, x_s = vc().get_disparity(self.A, 1-1/dur_l), vc().get_disparity(self.A, 1-1/dur_s)
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
        # 보정함수는 극단값 잡기 위함. MCPI 지수에 맞는 metric 적용
        beta = 2 + np.absolute(ewm_w) - 4/(1+np.exp(np.absolute(-ewm_w)))

        # 왜 list 사용???
        list_ewm_w_neg = list(np.where(ewm_w < 0))
        idx_ewm_w_neg = tuple(zip(*list_ewm_w_neg))

        beta_com = beta.copy()

        for idx in idx_ewm_w_neg:
            beta_com[idx] = -beta[idx]

        return beta_com

    def score_compensation(self, score_momentum, score_vv, beta_com):
        s_momentum_com = score_momentum - beta_com
        score_fng = 1/(1+np.exp(-(np.multiply(score_vv, s_momentum_com)))) * 100

        return score_fng



class FearGreed(object):
    def __init__(self, score):
        self.score = score

    def compute_index(self, duration=120):
        """"""
        score_volatility = self.score.volatility_score(duration=duration)
        ewm_vlm_l, ewm_vlm_s = self.score.ewm_volume()
        score_volume = self.score.volume_score(ewm_vlm_l, ewm_vlm_s)
        score_vv = self.score.volatility_volume_score(score_volatility, score_volume)

        l_l, l_s = self.score.weight_long_short(score_vv)
        x_l, x_s = self.score.disparity(l_l, l_s)

        score_c = self.score.score_c(duration=duration)

        score_momentum = self.score.score_momentum(x_l, x_s, l_l, l_s, c=score_c[:, 1:])
        ewm_w = self.score.ewm_score_momentum(score_momentum)
        beta_com = self.score.beta_compensated(ewm_w)
        score_compensation = self.score.score_compensation(score_momentum, score_vv, beta_com)

        return score_compensation

    def compute_stock(self, duration=120):
        """"""
        score_volatility = self.score.volatility_score(duration=duration)
        ewm_vlm_l, ewm_vlm_s = self.score.ewm_volume()
        score_volume = self.score.volume_score(ewm_vlm_l, ewm_vlm_s)
        score_vv = self.score.volatility_volume_score(score_volatility, score_volume)

        l_l, l_s = self.score.weight_long_short(score_vv)
        x_l, x_s = self.score.disparity(l_l, l_s)

        score_c = self.score.score_c(duration=duration)

        score_momentum = self.score.score_momentum(x_l, x_s, l_l, l_s, c=score_c[:, 1:])
        ewm_w = self.score.ewm_score_momentum(score_momentum)
        beta_com = self.score.beta_compensated(ewm_w)
        score_compensation = self.score.score_compensation(score_momentum, score_vv, beta_com)

        return score_compensation

