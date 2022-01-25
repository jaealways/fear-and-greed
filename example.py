from score import VolatilityVolume, Momentum

import numpy as np
import pandas as pd

# 어떤 지점에서 nan으로 치환??

df_mcpi_volume = pd.read_pickle("df_song_volume_17-01-01_23-12-31.pkl")
df_mcpi_index = pd.read_pickle("df_price_17-01-01_23-12-31.pkl")
mcpi_volume, mcpi_index = df_mcpi_volume.to_numpy(), df_mcpi_index.iloc[:,:-1].to_numpy()

score_vv = Momentum(mcpi_index, mcpi_volume).compute_compensation()
score_vv
