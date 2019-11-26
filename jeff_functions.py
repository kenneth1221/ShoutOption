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
import os
import matplotlib.pyplot as plt
import scipy.stats as si

import ShoutOption as so;
#%%
def euro_vanilla_call(S, K, T, r, d, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    d1 = (np.log(S / K) + (r - d + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - d - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-(r-d) * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

s=so.get_latest_price()
T_t = 1
r = 0.0158
d = 0.0185
sigma = so.get_sigma()
#%%
os.chdir(os.path.dirname(__file__))
print(os.getcwd())
#%%
options = pd.read_excel("options.xlsx")
#options = pd.read_csv("options.csv")
c1yr = options.loc[:, 'Dec 2020 call'].values # traded call options price
chalfyr = options.loc[:, 'Jun 2020 call'].values
K = options.loc[:, 'strike'].values/100 # strike price


# suppose maturity = 1 
bs_c = []

for i in K:
    bs_c.append(euro_vanilla_call(s, i, T_t, r, d,sigma))
#%%    
bs_c = np.asarray(bs_c, dtype=np.float64).reshape(-1,1)
Xs = np.hstack((bs_c, K.reshape(-1,1)))
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()
regressorb = LinearRegression()
model_1 = regressor.fit(bs_c, c1yr)

print(model_1.coef_)
print(model_1.intercept_)
print(r2_score(model_1.coef_*bs_c+model_1.intercept_, c1yr))

multlin_1 = regressorb.fit(Xs,c1yr)
print(multlin_1.coef_)
print(multlin_1.intercept_)
print(r2_score( np.sum(multlin_1.coef_*Xs, axis = 1)+multlin_1.intercept_, c1yr))

plt.plot(K, model_1.coef_*bs_c+model_1.intercept_, '.')
plt.plot(K, np.sum(multlin_1.coef_*Xs, axis = 1)+multlin_1.intercept_, '.')
plt.plot(K,c1yr, '.')

#%%
# suppose maturity = 0.5 
bs_c2 = []
T_t = 0.5
for i in K:
    bs_c2.append(euro_vanilla_call(s, i, T_t, r, d,sigma))
    
bs_c2 = np.asarray(bs_c2, dtype=np.float64).reshape(-1,1)

regressor2 = LinearRegression()
model_2 = regressor2.fit(bs_c2, chalfyr)

print(model_2.coef_)
print(model_2.intercept_)
print(r2_score(model_2.coef_*bs_c2+model_2.intercept_, chalfyr))

plt.plot(K, model_2.coef_*bs_c2+model_2.intercept_, '.')
plt.plot(K, chalfyr, '.')