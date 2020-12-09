# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:48:40 2020

@author: leona
Plotting results

"""

path_input = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from env.BitcoinTradingEnv import BitcoinTradingEnv

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def calculate_AR(series,init_value):        
    return ((((np.sum(series)+init_value)/init_value+1)**(1/len(series))))**365-1
    
def calculate_SR(series,init_value):
    risk_free = 0.0001
    return (((((np.sum(series)+init_value)/init_value+1)**(1/len(series))))-1-risk_free)/np.std(series)*np.sqrt(365)


############################################################
# Load first datas
############################################################


assets = ['btc','eth','ltc','xmr','dash']
######################################################################################################################
# Create table for cumlateive returns
######################################################################################################################
df = pd.DataFrame(columns = ['Asset','BH','RRL','DQL','A2C','BSTS','RSLSTM-A'])
df['Asset']= assets
df = df.set_index('Asset')

df_SR = pd.DataFrame(columns = ['Asset','BH','RRL','DQL','A2C','BSTS','RSLSTM-A'])
df_SR['Asset']= assets
df_SR = df_SR.set_index('Asset')

df_AR = pd.DataFrame(columns = ['Asset','BH','RRL','DQL','A2C','BSTS','RSLSTM-A'])
df_AR['Asset']= assets
df_AR = df_AR.set_index('Asset')

alpha = 0.05
k_fold = 15
min_df_size = 3000
comission = 0.005
gamma = 0.99
M  = 51
mu = 1
n_epoch = 600
decay = 200
reward_strategy = 'return'


input_datas = []

input_datas.append(path_input+'/data/Poloniex_BTCUSD_1h.csv')
input_datas.append(path_input+'/data/Poloniex_ETHUSD_1h.csv')
input_datas.append(path_input+'/data/Poloniex_LTCUSD_1h.csv')
input_datas.append(path_input+'/data/Poloniex_XMRUSD_1h.csv')
input_datas.append(path_input+'/data/Poloniex_DASHUSD_1h.csv')

for asset_name, data_name in zip(assets, input_datas):
    data = pd.read_csv(data_name)
    valid_len = int(len(data) * 0.1/2)
    test_len = valid_len
    train_len = int(len(data)) - valid_len*2   
    test_df = data[train_len+M+valid_len:]
    length = len(test_df)
    test_env = BitcoinTradingEnv(test_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length)
    firs_value_test = test_env.df['Close'][0]
    
    df.loc[asset_name,'A2C'] = np.sum(np.cumsum(np.load('A2C_agent_returns__'+asset_name+'.npy')))
    df.loc[asset_name,'DQL'] = np.sum(np.cumsum(np.load('DQN_agent_returns___'+asset_name+'.npy')))
    df.loc[asset_name,'RRL'] = np.sum(np.cumsum(np.load('RRL_model_returns_f__'+asset_name+'.npy')))
    df.loc[asset_name,'BSTS'] = np.sum(np.cumsum(np.load('cbadapt_agent_returns__'+asset_name+'.npy')))
    df.loc[asset_name,'RSLSTM-A'] = np.sum(np.cumsum(np.load('resnet_agent_returns__'+asset_name+'.npy')))
    df.loc[asset_name,'BH'] = np.sum(np.cumsum(np.load('bh___'+asset_name+'.npy')))

    df_SR.loc[asset_name,'A2C'] = calculate_SR(np.load('A2C_agent_returns__'+asset_name+'.npy'),firs_value_test)
    df_SR.loc[asset_name,'DQL'] = calculate_SR(np.load('DQN_agent_returns___'+asset_name+'.npy'),firs_value_test)
    df_SR.loc[asset_name,'RRL'] = calculate_SR(np.load('RRL_model_returns_f__'+asset_name+'.npy'),firs_value_test)
    df_SR.loc[asset_name,'BSTS'] = calculate_SR(np.load('cbadapt_agent_returns__'+asset_name+'.npy'),firs_value_test)
    df_SR.loc[asset_name,'RSLSTM-A'] = calculate_SR(np.load('resnet_agent_returns__'+asset_name+'.npy'),firs_value_test)
    df_SR.loc[asset_name,'BH'] = calculate_SR(np.load('bh___'+asset_name+'.npy'),firs_value_test)

    df_AR.loc[asset_name,'A2C'] = calculate_AR(np.load('A2C_agent_returns__'+asset_name+'.npy'),firs_value_test)
    df_AR.loc[asset_name,'DQL'] = calculate_AR(np.load('DQN_agent_returns___'+asset_name+'.npy'),firs_value_test)
    df_AR.loc[asset_name,'RRL'] = calculate_AR(np.load('RRL_model_returns_f__'+asset_name+'.npy'),firs_value_test)
    df_AR.loc[asset_name,'BSTS'] = calculate_AR(np.load('cbadapt_agent_returns__'+asset_name+'.npy'),firs_value_test)
    df_AR.loc[asset_name,'RSLSTM-A'] = calculate_AR(np.load('resnet_agent_returns__'+asset_name+'.npy'),firs_value_test)    
    df_AR.loc[asset_name,'BH'] = calculate_AR(np.load('bh___'+asset_name+'.npy'),firs_value_test)

df.to_csv('CR.csv')
df_SR.to_csv('SR.csv')
df_AR.to_csv('AR.csv')



