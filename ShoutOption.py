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
    out['LogReturn'] = np.log(prices.shift(-1)/prices)
    np.sqrt( (out.index.shift(1, 'd')-out.index).days )
    return out
#%%
get_logreturn(SP500)
#%%
a=SP500.index.shift(1, 'd')

b=SP500.index
a = pd.Series(a)
b = pd.Series(b)
b.iloc[1:].values-b.iloc[:-1].values