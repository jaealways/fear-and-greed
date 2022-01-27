from score import Score, FearGreed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 어떤 지점에서 nan으로 치환??

df_mcpi_volume = pd.read_pickle("df_song_volume_17-01-01_23-12-31.pkl")
df_mcpi_index = pd.read_pickle("df_price_17-01-01_23-12-31.pkl")
x, y = df_mcpi_index.to_numpy(), df_mcpi_volume.to_numpy()


score = Score(x, y)
score_fng = FearGreed(x, y, score).compute()

date = df_mcpi_index.columns[-len(score_fng):]
index_list = df_mcpi_volume.index.tolist()
num = index_list.index(1016)
print(num)


fig, axs = plt.subplots(2)
axs[0].plot(x[num, -score_fng.shape[1]:])
axs[1].plot(score_fng[num, :])

plt.show()

