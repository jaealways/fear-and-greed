from score import Score, FearGreed

import numpy as np
import pandas as pd

# 어떤 지점에서 nan으로 치환??

df_mcpi_volume = pd.read_pickle("df_song_volume_17-01-01_23-12-31.pkl")
df_mcpi_index = pd.read_pickle("df_price_17-01-01_23-12-31.pkl")
x, y = df_mcpi_index.to_numpy(), df_mcpi_volume.to_numpy()

score = Score(x, y)
score_fng = FearGreed(x, y, score).compute_grad()
score_fng
