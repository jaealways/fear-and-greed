from score import VolatilityVolume
from calculation import VariableCalculation

import numpy as np


a = np.array([[3, 1], [2, np.nan]])
b = np.array([[4, 1], [2, np.nan]])

s = VariableCalculation().variance_price(a)
s

# S_vv = VolatilityVolume(a, b)

