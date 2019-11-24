# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:09:47 2019

@author: steve
"""
#%%
import numpy as np
import pandas as pd
from pandas_datareader import data
import datetime as dt
import numpy.random as rd
import matplotlib.pyplot as plt
#%%
def load_financial_data(name, output_file):
    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading',name,'data')
    except FileNotFoundError:
        print('File not found...downloading', name, 'data')
        df = data.DataReader(name, 'yahoo', '2001-01-01', '2019-11-24')
        df.to_pickle(output_file)
    return df
#%%
SP500 = load_financial_data('^GSPC', '^GSPC_data.pkl')
#%%
def get_logreturn(yahoo_dataframe):
    out = pd.DataFrame(index = yahoo_dataframe.index)
    prices = yahoo_dataframe.Close
    days = pd.Series(yahoo_dataframe.index).dt.day
    daydelta = days.iloc[1:].values - days.iloc[:-1].values
    out['LogReturn'] = np.log(prices.shift(-1)/prices).dropna()/np.sqrt(daydelta)
    return out
#%%
lreturns = get_logreturn(SP500)
#%%
daybasis = 252
n_years = 3
pastyears = lreturns.iloc[-n_years*daybasis:]
dailyvol = pastyears.std()

yearlyvol = dailyvol*np.sqrt(daybasis)
dailyalpha = pastyears.mean() + dailyvol**2/2
yearlyalpha = dailyalpha*daybasis
#%%
Z1 = rd.randn(10000,1)
Z2 = rd.randn(10000,1)

sigma = yearlyvol
T = 1
trig = .5
r = .0158
d = .02
S = SP500.iloc[-1].Close
F = 10
K = 3100

def TriggerPayoff(Q):
    Payoff = np.zeros((100,1))
    
    Shalf = S*np.exp( ( r - d - sigma**2/2)*trig + sigma*Z1 * np.sqrt(trig)) 
    
    S1 = Shalf*np.exp( ( r - d - sigma**2/2)*(T-trig) + sigma*Z2 * np.sqrt(T-trig)) 
    Payoff = np.maximum(S1-K, 0)
    Payoff[Shalf < Q] = F
    meanPayoff = np.mean(Payoff)

    return meanPayoff*np.exp(-r*T)

def TwoPeriodEuroCall():
    Shalf = S*np.exp( ( r - d - sigma**2/2)*trig + sigma*Z1 * np.sqrt(trig)) 
    S1 = Shalf*np.exp( ( r - d - sigma**2/2)*(T-trig) + sigma*Z2 * np.sqrt(T-trig))
    Payoff = np.maximum(S1-K,0)
    return np.mean(Payoff)*np.exp(-r*T)

#%%
    
payoffs = []
eurocall = []
strikes = []
minrange = 50
maxrange = 150
step = .01
#%%
for i in np.arange(minrange,maxrange,step):
    payoffs.append(TriggerPayoff(i))
    eurocall.append(TwoPeriodEuroCall())
    strikes.append(i)
#%%
plt.plot(np.arange(minrange,maxrange,step), payoffs)
plt.plot(np.arange(minrange,maxrange,step), eurocall)
print('value: ',max(payoffs), 'vanilla: ', max(eurocall), 'best Q level: ', strikes[payoffs.index(max(payoffs))])
#%% testing code
#a=SP500.index.shift(1, 'd')
#
#b=SP500.index
#a = pd.Series(a)
#b = pd.Series(b).dt.day
#c = b.iloc[1:].values-b.iloc[:-1].values