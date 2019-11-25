# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:12:38 2019

@author: abc33
"""
#%%
import numpy as np
import pandas as pd
from pandas_datareader import data
import datetime as dt


import scipy.stats as si
def euro_vanilla_call(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

s=3128.65
T_t = 1
r = 0.0158
sigma = 0.1
#%%
options = pd.read_excel("options.xlsx")
c = options.iloc[:, 1].values # traded call options price
K = options.iloc[:, 0].values/100 # strike price


# suppose maturity = 1 
bs_c = []
for i in K:
    bs_c.append(euro_vanilla_call(s, i, T_t, r, sigma))
    
bs_c = np.asarray(bs_c, dtype=np.float64).reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model_1 = regressor.fit(bs_c, c)

print(model_1.coef_)
print(model_1.intercept_)

# suppose maturity = 0.5 
bs_c2 = []
T_t = 0.5
for i in K:
    bs_c2.append(euro_vanilla_call(s, i, T_t, r, sigma))
    
bs_c2 = np.asarray(bs_c2, dtype=np.float64).reshape(-1,1)

regressor2 = LinearRegression()
model_2 = regressor2.fit(bs_c2, c)

print(model_2.coef_)
print(model_2.intercept_)
