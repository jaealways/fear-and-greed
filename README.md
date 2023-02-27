# 공포탐욕지수
<br/>

두나무 Datavalue 팀에서 만든 공포탐욕지수를 비교적 유동성이 적은 자산에 적용하기 위해 개선한 모델입니다.
자세한 내용은 [다음 아티클](https://jaealways.tistory.com/100?category=977505)을 참고하시기 바랍니다.
<br/><br/><br/>

## Requirements
* python (version 3.8 이상)
<br/><br/><br/>


## Example
<br/><br/>

```python
pip install git+https://github.com/jaealways/fear-and-greed.git
```
<br/><br/>
터미널에서 위 코드를 실행해서 라이브러리를 다운받습니다.
<br/><br/>


### Index 계산
<br/><br/>
코스피 같은 자본시장 지수의 공포탐욕지수를 계산할 때 사용합니다. 종가데이터와 거래량데이터를 입력으로 넣어줍니다. 기간은 365일을 권장합니다.
<br/><br/>

```python
from fng.score import scoreIndex, FearGreed

x, y = df_price.to_numpy(), df_price_volume.to_numpy()
score = scoreIndex(x, y)
score_fng = FearGreed(score).compute_index(duration=365)
```

<br/>
x(가격데이터), y(거래량데이터)는 여러 개의 시계열데이터로 to_numpy 변환을 권장합니다. 이 때 column이 시간축이 되도록 합니다.

<br/><br/><br/>


### Stock 계산
<br/><br/>
개별 종목의 공포탐욕지수를 할 때 사용합니다. 종가데이터, 고가데이터, 저가데이터와 거래량데이터를 입력으로 넣어줍니다. 기간은 120일을 권장합니다.
<br/><br/>

```python
from fng.score import scoreStock, FearGreed

a, b, c, y = df_price.to_numpy(), df_price_high.to_numpy(), df_price_low.to_numpy(), df_price_volume.to_numpy()
score = scoreStock(a,b,c,y)
score_fng = FearGreed(score).compute_stock(duration=120)
```

<br/><br/>
a(가격 종가 데이터), b(가격 고가 데이터), c(가격 저가 데이터), y(거래량데이터)는 여러 개의 시계열데이터로 to_numpy 변환을 권장합니다. 이 때 column이 시간축이 되도록 합니다.
<br/><br/>


```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3)
axs[0].plot(x[num, -score_fng.shape[1]:])
axs[1].plot(y[num, -score_fng.shape[1]:])
axs[2].plot(score_fng[num, :])
```

<br/><br/>

![Figure_1](https://user-images.githubusercontent.com/71856506/197672082-cb628989-03ee-405d-a14e-8735b42fbc0d.png)

이와 같이 가격, 거래량, 공포탐욕점수 시계열을 비교할 수 있습니다.

<br/><br/>



## Reference
<br/>

[디지털자산 공포-탐욕 지수 Methodology Book version 2.0](https://datavalue.dunamu.com/static/pdf/%EB%91%90%EB%82%98%EB%AC%B4%20%EB%94%94%EC%A7%80%ED%84%B8%EC%9E%90%EC%82%B0%20%EA%B3%B5%ED%8F%AC-%ED%83%90%EC%9A%95%20%EC%A7%80%EC%88%98%20Methodology%20Book%202.0.pdf)

[J.P.Morgan/Reuters RiskMetrics —Technical Document](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a)
