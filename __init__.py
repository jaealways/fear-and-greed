import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FearAndGreedIndex:
    def __init__(self, df, c, lam):
        self.df_price, self.df_vol = df.loc[0], df.loc[1]
        self.array_price, self.array_vol = np.array(self.df_price), np.array(self.df_vol)
        self.c, self.lam = c, lam

    def __getattribute__(self, df):
        return df.colunms

