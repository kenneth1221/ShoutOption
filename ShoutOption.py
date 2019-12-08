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
import os
import scipy.stats as si
#from jeff_functions import euro_vanilla_call
#%%
os.chdir(os.path.dirname(__file__))
print(os.getcwd())
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

def euro_vanilla_call(S, K, T, r, d, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity/
    #r: interest rate
    #sigma: volatility of underlying asset
    d1 = (np.log(S / K) + (r - d + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - d - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-(r-d) * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call
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

def get_sigma():
    daybasis = 252
    SP500 = load_financial_data('^GSPC', '^GSPC_data.pkl')
    lreturns = get_logreturn(SP500)
    pastyears = lreturns.iloc[-n_years*daybasis:]
    dailyvol = pastyears.std()[0]
    yearlyvol = dailyvol*np.sqrt(daybasis)
    return yearlyvol

def get_latest_price():
    return load_financial_data('^GSPC', '^GSPC_data.pkl').iloc[-1].Close
#%%
lreturns = get_logreturn(SP500)
#%%
daybasis = 252
n_years = 3
pastyears = lreturns.iloc[-n_years*daybasis:]
dailyvol = pastyears.std()[0]

yearlyvol = dailyvol*np.sqrt(daybasis)
dailyalpha = pastyears.mean() + dailyvol**2/2
yearlyalpha = dailyalpha*daybasis


#%%
# generate  set of random numbers
num_random = 5000
Z1 = rd.randn(num_random,1)
Z2 = rd.randn(num_random,1)
Z1 = (Z1-Z1.mean())/Z1.std()
Z2 = (Z2-Z2.mean())/Z2.std()
def RegenerateRandomNumbers():
    global Z1, Z2
    Z1 = rd.randn(num_random,1)
    Z2 = rd.randn(num_random,1)
    #control variate method
    Z1 = (Z1-Z1.mean())/Z1.std()
    Z2 = (Z2-Z2.mean())/Z2.std()
    
def TriggerPayoff(Q, F):
    """ calculates trigger payoff for a given exercise level Q via Monte Carlo simulation"""
    
    Shalf = S*np.exp( ( r - d - sigma**2/2)*trig + sigma*Z1 * np.sqrt(trig)) 
    
    S1 = Shalf*np.exp( ( r - d - sigma**2/2)*(T-trig) + sigma*Z2 * np.sqrt(T-trig)) 
    
    Payoff = np.maximum(S1-K, 0)
    Payoff[Shalf < Q] = F
    meanPayoff = np.mean(Payoff)
#    return np.hstack((Shalf, Payoff))
    return meanPayoff*np.exp(-r*T)

#theoretically, at T=.5, the option is either an option to get a fixed payment or a call.
def TwoPeriodEuroCall():
    """calculates the value of a vanilla european call using common random numbers of the shout option"""
    Shalf = S*np.exp( ( r - d - sigma**2/2)*trig + sigma*Z1 * np.sqrt(trig)) 
    S1 = Shalf*np.exp( ( r - d - sigma**2/2)*(T-trig) + sigma*Z2 * np.sqrt(T-trig))
    Payoff = np.maximum(S1-K,0)
    return np.mean(Payoff)*np.exp(-r*T)

#%% sets parameters
sigma = yearlyvol
T = 1
trig = .5
r = .0158
d = .0185
S = SP500.iloc[-1].Close
F = 10
K = 3150

#%% reads option data
options = pd.read_excel("options.xlsx")
#options = pd.read_csv("options.csv")
c1yr = options.loc[:, 'Dec 2020 call'].values # traded call options price
chalfyr = options.loc[:, 'Jun 2020 call'].values
Ks = options.loc[:, 'strike'].values/100 # strike price
# suppose maturity = 1 
bs_c = []

for i in Ks:
    bs_c.append(euro_vanilla_call(S, i, T, r, d,sigma))
    
bs_c = np.asarray(bs_c, dtype=np.float64).reshape(-1,1)
#Xs is a matrix of theoretical black scholes price and strike. 
Xs = np.hstack((bs_c, Ks.reshape(-1,1)))
#%%
#linear regression to connect the black scholes call price to the 1 year call price
# establishes a link between the theoretical black scholes and market observed prices
# so, using a given result from the black scholes formula, we can fill in the blanks for what the market would say
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()
#regressorb = LinearRegression()
model_1 = regressor.fit(bs_c, c1yr)

b_year, a_year = model_1.coef_[0], model_1.intercept_

print('slope: ',b_year)
print('intercept: ',a_year)
print('r2 score: ',r2_score(model_1.coef_*bs_c+model_1.intercept_, c1yr))

#multlin_1 = regressorb.fit(Xs,c1yr)
#print(multlin_1.coef_)
#print(multlin_1.intercept_)
#print(r2_score( np.sum(multlin_1.coef_*Xs, axis = 1)+multlin_1.intercept_, c1yr))

plt.plot(Ks, model_1.coef_*bs_c+model_1.intercept_, '.')
#plt.plot(Ks, np.sum(multlin_1.coef_*Xs, axis = 1)+multlin_1.intercept_, '.')
plt.plot(Ks,c1yr, '.')


#%%
# suppose maturity = 0.5 
# regresses half year black scholes theoretical call price to real call price
bs_c2 = []
for i in Ks:
    bs_c2.append(euro_vanilla_call(S, i, trig, r, d,sigma))
    
bs_c2 = np.asarray(bs_c2, dtype=np.float64).reshape(-1,1)

regressor2 = LinearRegression()
model_2 = regressor2.fit(bs_c2, chalfyr)

b_half, a_half = model_2.coef_[0], model_2.intercept_
print('slope: ', b_half)
print('intercept: ', a_half)
print('r2 score: ',r2_score(model_2.coef_*bs_c2+model_2.intercept_, chalfyr))

plt.plot(Ks, model_2.coef_*bs_c2+model_2.intercept_, '.')
plt.plot(Ks, chalfyr, '.')
#%% 
# these coefficients relate black scholes prices to market observed prices for a given strike
# for strikes between  3075 and 3150, for options of 1 year and half a year
# of course, this might not be necessary -- it might be simpler to get straight to the point
# which means regress directly: (what to what?)
# simulate shout values, let those be Y
# xs are then... 
# I guess it could be with different strike prices to empirically determine a payoff function for a shout
# theoretically with a shout F of 0, the damn thing should converge to a european call
# anyways, what we want is an equation of form a + b1 O1 + b2 O2,
# where a is a fixed cost, b1 is the relation to the half year option, and b2 is the relation to the year option
# how do we benchmark this? mentally I'm stuck
# information we have: simulated half year stock prices, full year stock prices
# market data: half year option, full year option
print(a_year, b_year, a_half, b_half)

def scaled_eurocall(bsprice, alpha, beta):
    return bsprice*beta + alpha
    
#%%
def main(k):
    sigma = yearlyvol
    T = 1
    trig = .5
    r = .0158
    d = .0185
    S = SP500.iloc[-1].Close
    F = 100 #we want some F that makes the shout more valuable than the vanilla
    K = k
    RegenerateRandomNumbers()
    payoffs = []
    basepays = []
    eurocall = []
    trueeurocall = []
    strikes = []
    
    minrange = round((S+K)/2 -750)
    maxrange = round((S+K)/2 +750)
    step = .2
    steprange = np.arange(minrange, maxrange, step)
    
    simeurcall = scaled_eurocall(TwoPeriodEuroCall(),a_year, b_year )#common random number eurocall
    bseurcall = scaled_eurocall(euro_vanilla_call(S,K,T,r,d,sigma), a_year, b_year )#analytical eurocall
    
    for i in steprange:
        # this payoff takes the control variate technique and applies it to the trigger payoff
        # we know what the analytical european call price should be
        # we can simulate the european call price using the same common random numbers as the trigger
        # thus, we can correct the effect of the randomnuess on the trigger payoff via:
        # simulated_trigger - simulated_european + analytical_european
        basepay = TriggerPayoff(i,F)

        j = basepay - simeurcall + bseurcall
        
        
        payoffs.append(j)
        basepays.append(basepay)
        eurocall.append(simeurcall)
        trueeurocall.append(bseurcall)
        strikes.append(i)
    plt.plot(steprange, payoffs)
    plt.plot(steprange, eurocall)
    plt.plot(steprange, trueeurocall)
    bestq = strikes[payoffs.index(max(payoffs))]
    value = max(payoffs)
    print('value: ',value, 'sim-vanilla: ', max(eurocall), 'true-vanilla: ', trueeurocall[0],'best Q level: ', bestq)
    return value, bestq

#%%
k = 3150
values = []
optimalqs = []
for j in range(5):        
    v, q = main(k)
    values.append(v)
    optimalqs.append(q)
#%%

#values = np.array(values)
#optimalqs = np.array(optimalqs)
#%% testing code
#a=SP500.index.shift(1, 'd')
#
#b=SP500.index
#a = pd.Series(a)
#b = pd.Series(b).dt.day
#c = b.iloc[1:].values-b.iloc[:-1].values